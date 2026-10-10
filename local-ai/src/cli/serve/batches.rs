//! `/v1/batches`: the HTTP adapter for [`local_services::batches`] and the one
//! worker that runs batch lines on the loaded model.
//!
//! | method | path                        | answer                |
//! |--------|-----------------------------|-----------------------|
//! | `POST` | `/v1/batches`               | `batch` (in progress) |
//! | `GET`  | `/v1/batches`               | `list` of batches     |
//! | `GET`  | `/v1/batches/{id}`          | `batch`               |
//! | `POST` | `/v1/batches/{id}/cancel`   | `batch` (cancelling)  |
//!
//! Creating a batch validates its whole input file and stores it, or stores
//! nothing (see [`local_services::batches`]); unsupported endpoints are
//! refused then, never queued, and so is a batch whose `body.model` is not
//! the loaded model. A stored line naming another model (a batch resumed by a
//! server with a different model) fails without being generated. Lists are
//! newest first, 20 per page (1 to 100), paged with `after`.
//! `output_expires_after` is not supported and is refused like any other
//! unknown field.
//!
//! # Execution
//!
//! One worker thread runs one line at a time, through the same
//! [`prepare_retaining`] and non-streaming body collector as the HTTP
//! endpoints, so each output `body` is the document `POST {url}` answers.
//! Lines run statelessly: no Responses store, temporary store or conversation
//! is read or written, so `store: true`, `previous_response_id` and
//! `conversation` fail their line as they do on a server without those
//! stores. Batch lines share the engine queue with interactive requests and
//! count toward its depth; a full queue is retried with a capped backoff that
//! cancellation, expiry and shutdown interrupt.
//!
//! The worker never polls the database. It scans for runnable batches when it
//! starts (reclaiming those left by a stopped server whose lease is free),
//! after each create, and after each cancel. A batch whose lease another live
//! server holds is skipped; it is not looked at again until the next scan.
//! Cancelling a batch this server is running cancels its current line at
//! once; one another server runs stops at that server's next line boundary.
//! Expiry cancels a running line at the deadline. Shutdown cancels the
//! current line and leaves it pending, so a later start generates it again.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::request::Parts;
use axum::http::{Method, StatusCode};
use axum::response::Response;
use http_body_util::{BodyExt as _, LengthLimitError, Limited};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::sync::{Semaphore, watch};

use local_engine::{EngineHandle, Event, Signal};
use local_services::batches::{
    BatchLease, CreateBatch, Endpoint, LineOutcome, ListBatches, NextLine, PendingLine,
};
use local_services::{Error, Metadata, Store};

use super::chunked::{Body as Document, Flow, absorb};
use super::reasoning_crypto::ReasoningCipher;
use super::request::GenerationRequest;
use super::response::{
    Protocol, Reply, checked, error_response, error_status, error_status_message, json_response,
    unix_now,
};
use super::store::query_pairs;
use super::{Api, AppState, MAX_REQUEST_BYTES, QueueDepth, prepare_retaining};

/// The route prefix served here.
const PREFIX: &str = "/v1/batches";
/// Store calls running on the blocking pool at once for HTTP requests.
const MAX_BLOCKING: usize = 4;
/// First wait after the engine queue was full.
const QUEUE_RETRY_FIRST: Duration = Duration::from_millis(50);
/// Longest wait between attempts on a full engine queue.
const QUEUE_RETRY_MAX: Duration = Duration::from_secs(2);
/// A line whose generation ended without a result.
const STOPPED: &str = "generation stopped before it produced a result";

/// Why a request failed: its status and message.
type Failure = (StatusCode, String);

/// Cooperatively cancels one generation.
pub(super) type Cancel = Arc<dyn Fn() + Send + Sync>;
/// The next signal of one generation, waiting at most the given time;
/// `Ok(None)` is its end.
pub(super) type Next =
    Box<dyn FnMut(Option<Duration>) -> Result<Option<Signal>, RecvTimeoutError> + Send>;

/// One admitted generation, as the worker drives it.
pub(super) struct Generation {
    pub(super) next: Next,
    pub(super) cancel: Cancel,
}

/// Queues one request on the engine (or, in tests, a script).
pub(super) type Submit = Arc<dyn Fn(GenerationRequest) -> crate::Result<Generation> + Send + Sync>;

/// What the worker needs from the server, cloned from [`AppState`].
pub(super) struct Config {
    pub(super) model: Arc<str>,
    pub(super) thinking: bool,
    pub(super) cipher: Option<Arc<ReasoningCipher>>,
    pub(super) depth: QueueDepth,
}

/// Whether `path` belongs to the Batches API (any method).
pub(super) fn matches(path: &str) -> bool {
    path.strip_prefix(PREFIX)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// Answer a Batches request. Authentication has already passed.
pub(super) async fn handle(state: &AppState, parts: &Parts, body: Body) -> Response {
    let Some(batches) = state.batches.as_ref() else {
        return error_response(
            StatusCode::NOT_FOUND,
            "route not found: this server runs no batches",
            state,
            &parts.headers,
        )
        .await;
    };
    match batches.dispatch(parts, body).await {
        Ok(value) => json_response(StatusCode::OK, value, state, &parts.headers).await,
        Err((status, message)) => error_response(status, &message, state, &parts.headers).await,
    }
}

/// The Batches service: its store, its worker and the bound on HTTP work.
pub(super) struct Batches {
    store: Store,
    worker: Arc<Worker>,
    blocking: Arc<Semaphore>,
    finished: watch::Receiver<bool>,
}

impl Batches {
    /// Start the worker on the server's engine and native store. It begins
    /// with a scan, so batches a stopped server left are resumed.
    pub(super) fn start(state: &AppState) -> crate::Result<Arc<Self>> {
        let services = state.services.as_ref().ok_or_else(|| {
            crate::Error::InvalidArgument("batches need the native services store".into())
        })?;
        let config = Config {
            model: Arc::clone(&state.model),
            thinking: state.thinking,
            cipher: state.cipher.clone(),
            depth: state.depth.clone(),
        };
        Self::spawn(
            services.store().clone(),
            engine_submit(state.engine.clone()),
            config,
        )
    }

    /// Start the worker on `store`, generating through `submit`.
    pub(super) fn spawn(store: Store, submit: Submit, config: Config) -> crate::Result<Arc<Self>> {
        let worker = Arc::new(Worker {
            store: store.clone(),
            submit,
            config,
            control: Mutex::new(Control {
                pending: true,
                ..Control::default()
            }),
            wake: Condvar::new(),
        });
        let (done, finished) = watch::channel(false);
        let running = Arc::clone(&worker);
        std::thread::Builder::new()
            .name("batches".into())
            .spawn(move || {
                running.run();
                let _ = done.send(true);
            })?;
        Ok(Arc::new(Self {
            store,
            worker,
            blocking: Arc::new(Semaphore::new(MAX_BLOCKING)),
            finished,
        }))
    }

    /// Stop taking lines and cancel the current one, which stays pending.
    /// Idempotent; never blocks on the engine.
    pub(super) fn close(&self) {
        self.worker.close();
    }

    /// Wait until the worker has let go of the engine, its lease and the
    /// store.
    pub(super) async fn drained(&self) {
        let mut finished = self.finished.clone();
        let _ = finished.wait_for(|done| *done).await;
    }

    /// Answer one request, independent of the server's other state.
    pub(super) async fn dispatch(&self, parts: &Parts, body: Body) -> Result<Value, Failure> {
        let rest = parts.uri.path().strip_prefix(PREFIX).unwrap_or_default();
        let segments: Vec<&str> = rest.split('/').skip(1).collect();
        if segments.iter().any(|segment| segment.is_empty()) {
            return Err(route_not_found());
        }
        let query = parts.uri.query();
        match (parts.method.clone(), segments.as_slice()) {
            (Method::POST, []) => {
                no_query(query)?;
                self.create(body).await
            }
            (Method::GET, []) => {
                let list = list_query(query)?;
                self.call(move |store| store.list_batches(&list).map(|page| to_json(&page)))
                    .await
            }
            (Method::GET, [id]) => {
                no_query(query)?;
                let id = (*id).to_owned();
                self.call(move |store| store.get_batch(&id).map(|batch| to_json(&batch)))
                    .await
            }
            (Method::POST, [id, "cancel"]) => {
                no_query(query)?;
                let id = (*id).to_owned();
                let worker = Arc::clone(&self.worker);
                self.call(move |store| {
                    let batch = store.cancel_batch(&id)?;
                    worker.cancelled(&id);
                    Ok(to_json(&batch))
                })
                .await
            }
            _ => Err(route_not_found()),
        }
    }

    async fn create(&self, body: Body) -> Result<Value, Failure> {
        let request: CreateBody = read_json(body).await?;
        let create = CreateBatch {
            input_file_id: request.input_file_id,
            endpoint: request.endpoint,
            completion_window: request.completion_window,
            metadata: request.metadata.unwrap_or_default(),
        };
        let model = Arc::clone(&self.worker.config.model);
        let batch = self
            .call(move |store| store.create_batch(&create, &model))
            .await?;
        self.worker.notify();
        Ok(to_json(&batch))
    }

    /// Run `work` against the store on the blocking pool, at most
    /// [`MAX_BLOCKING`] at once.
    async fn call<T, F>(&self, work: F) -> Result<T, Failure>
    where
        T: Send + 'static,
        F: FnOnce(&Store) -> local_services::Result<T> + Send + 'static,
    {
        let permit = Arc::clone(&self.blocking)
            .acquire_owned()
            .await
            .map_err(|_| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "the batch service is shutting down".to_owned(),
                )
            })?;
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            work(&store)
        })
        .await
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("batch store task failed: {error}"),
            )
        })?
        .map_err(failure)
    }
}

/// The body of `POST /v1/batches`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateBody {
    input_file_id: String,
    endpoint: String,
    completion_window: String,
    #[serde(default)]
    metadata: Option<Metadata>,
}

/// The worker and what it shares with the HTTP side.
struct Worker {
    store: Store,
    submit: Submit,
    config: Config,
    control: Mutex<Control>,
    /// Signalled on every change to `control`.
    wake: Condvar,
}

#[derive(Default)]
struct Control {
    /// A scan for runnable batches is due.
    pending: bool,
    closing: bool,
    /// Batches cancelled through this server, so a line about to start is
    /// not started.
    cancelled: HashSet<String>,
    current: Option<Current>,
}

/// The line being generated.
struct Current {
    batch: String,
    interrupted: Arc<AtomicBool>,
    /// Set once the engine has admitted the line.
    cancel: Option<Cancel>,
}

/// Interrupt the current line, if it belongs to `batch` (any with `None`).
fn interrupt(control: &Control, batch: Option<&str>) {
    if let Some(current) = &control.current
        && batch.is_none_or(|batch| batch == current.batch)
    {
        current.interrupted.store(true, Ordering::SeqCst);
        if let Some(cancel) = &current.cancel {
            cancel();
        }
    }
}

impl Worker {
    fn lock(&self) -> MutexGuard<'_, Control> {
        self.control.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn notify(&self) {
        self.lock().pending = true;
        self.wake.notify_all();
    }

    fn close(&self) {
        let mut control = self.lock();
        control.closing = true;
        interrupt(&control, None);
        drop(control);
        self.wake.notify_all();
    }

    fn cancelled(&self, batch: &str) {
        {
            let mut control = self.lock();
            control.cancelled.insert(batch.to_owned());
            interrupt(&control, Some(batch));
            control.pending = true;
        }
        self.wake.notify_all();
    }

    fn closing(&self) -> bool {
        self.lock().closing
    }

    /// Scan whenever asked, until closed.
    fn run(&self) {
        loop {
            {
                let mut control = self.lock();
                while !control.pending && !control.closing {
                    control = self
                        .wake
                        .wait(control)
                        .unwrap_or_else(PoisonError::into_inner);
                }
                if control.closing {
                    return;
                }
                control.pending = false;
            }
            let ids = match self.store.active_batches() {
                Ok(ids) => ids,
                Err(error) => {
                    eprintln!("warning: batches cannot be listed: {error}");
                    continue;
                }
            };
            for id in ids {
                if self.closing() {
                    return;
                }
                match self.store.lease_batch(&id) {
                    Ok(Some(lease)) => {
                        if let Err(error) = self.run_batch(&lease) {
                            eprintln!(
                                "warning: batch {id} stopped: {error}; it resumes at the next \
                                 scan or server start"
                            );
                        }
                        self.lock().cancelled.remove(&id);
                    }
                    // Another live server owns it.
                    Ok(None) => {}
                    Err(error) => eprintln!("warning: batch {id} cannot be leased: {error}"),
                }
            }
        }
    }

    /// Run the leased batch to its end, or until shutdown.
    fn run_batch(&self, lease: &BatchLease) -> local_services::Result<()> {
        loop {
            if self.closing() {
                return Ok(());
            }
            match self.store.next_batch_line(lease)? {
                NextLine::Done => return Ok(()),
                NextLine::Finish => {
                    self.store.finish_batch(lease)?;
                    return Ok(());
                }
                // A line that was interrupted stays pending; the next round
                // sees why (cancelled, expired or closing).
                NextLine::Line(line) => {
                    if let Some(outcome) = self.execute(lease.batch_id(), &line) {
                        self.store.settle_batch_line(lease, line.line, &outcome)?;
                    }
                }
            }
        }
    }

    /// Generate one line; `None` when it was interrupted.
    fn execute(&self, batch: &str, line: &PendingLine) -> Option<LineOutcome> {
        let api = match line.endpoint {
            Endpoint::Responses => Api::Responses,
            Endpoint::ChatCompletions => Api::Chat,
            Endpoint::Completions => Api::Completion,
        };
        let config = &self.config;
        // A batch stored by a server with another model (resumed here) never
        // runs on this one.
        let model = line.body.get("model").and_then(Value::as_str);
        if model != Some(config.model.as_ref()) {
            return Some(failed(
                StatusCode::BAD_REQUEST,
                &format!(
                    "body.model {:?} is not the loaded model {:?}",
                    model.unwrap_or_default(),
                    config.model
                ),
            ));
        }
        let body = serde_json::to_vec(&line.body).unwrap_or_default();
        let prepared = prepare_retaining(
            api,
            &body,
            config.thinking,
            (None, None),
            None,
            config.cipher.as_ref(),
            &config.model,
        );
        let (stream, request, protocol) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return Some(failed(error_status(&error), &error.to_string())),
        };
        if stream {
            return Some(failed(
                StatusCode::BAD_REQUEST,
                "stream must be false in a batch",
            ));
        }
        let interrupted = Arc::new(AtomicBool::new(false));
        {
            let mut control = self.lock();
            if control.closing || control.cancelled.contains(batch) {
                return None;
            }
            control.current = Some(Current {
                batch: batch.to_owned(),
                interrupted: Arc::clone(&interrupted),
                cancel: None,
            });
        }
        let outcome = self.generate(line, &request, protocol, &interrupted);
        self.lock().current = None;
        outcome
    }

    /// Admit `request` and collect its document.
    fn generate(
        &self,
        line: &PendingLine,
        request: &GenerationRequest,
        protocol: Protocol,
        interrupted: &AtomicBool,
    ) -> Option<LineOutcome> {
        let deadline = deadline(line.expires_at);
        let mut generation = match self.admit(request, interrupted, deadline) {
            Ok(generation) => generation,
            Err(outcome) => return outcome,
        };
        let _admitted = self.config.depth.admit();
        let responses = matches!(protocol, Protocol::Responses(_));
        let (mut document, _) =
            Document::open(Reply::new(protocol, Arc::clone(&self.config.model)), false);
        let mut bytes = Vec::new();
        let unless_interrupted =
            |outcome: LineOutcome| (!interrupted.load(Ordering::SeqCst)).then_some(outcome);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let signal = match (generation.next)(Some(left)) {
                Ok(Some(signal)) => signal,
                Ok(None) | Err(RecvTimeoutError::Disconnected) => {
                    return unless_interrupted(failed(StatusCode::INTERNAL_SERVER_ERROR, STOPPED));
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Expired: stop the engine and hold the line until it has.
                    (generation.cancel)();
                    while let Ok(Some(_)) = (generation.next)(None) {}
                    return None;
                }
            };
            let signal = match signal {
                Signal::Event(event) => {
                    let event = if responses { event } else { checked(event) };
                    if let Event::Error(message) = &event {
                        return unless_interrupted(failed(error_status_message(message), message));
                    }
                    Signal::Event(event)
                }
                progress @ Signal::Progress(_) => progress,
            };
            let flow = absorb(&mut document, signal);
            if let Some(chunk) = document.take(flow == Flow::Last) {
                bytes.extend_from_slice(&chunk);
            }
            match flow {
                Flow::Last => {
                    let outcome = serde_json::from_slice::<Value>(&bytes).map_or_else(
                        |error| {
                            failed(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                &format!("the response document could not be read: {error}"),
                            )
                        },
                        |body| LineOutcome::Response {
                            status_code: StatusCode::OK.as_u16(),
                            body,
                        },
                    );
                    return unless_interrupted(outcome);
                }
                Flow::Stop => {
                    return unless_interrupted(failed(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "the response document could not be completed",
                    ));
                }
                Flow::Keep | Flow::Flush => {}
            }
        }
    }

    /// Queue `request` on the engine, retrying a full queue with a capped
    /// backoff. `Err(None)` when interrupted (or expired) first, `Err(Some)`
    /// when the engine refused the request.
    fn admit(
        &self,
        request: &GenerationRequest,
        interrupted: &AtomicBool,
        deadline: Instant,
    ) -> Result<Generation, Option<LineOutcome>> {
        let mut wait = QUEUE_RETRY_FIRST;
        loop {
            match (self.submit)(duplicate(request)) {
                Ok(generation) => {
                    {
                        let mut control = self.lock();
                        if let Some(current) = control.current.as_mut() {
                            current.cancel = Some(Arc::clone(&generation.cancel));
                        }
                        drop(control);
                    }
                    // An interrupt that came before the handle was registered.
                    if interrupted.load(Ordering::SeqCst) {
                        (generation.cancel)();
                    }
                    return Ok(generation);
                }
                Err(crate::Error::QueueFull) => {
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Err(None);
                    }
                    // Checked under the lock before waiting, so a wake-up
                    // that came first is not lost.
                    let closing = self
                        .wake
                        .wait_timeout_while(self.lock(), wait.min(left), |control| {
                            !control.closing && !interrupted.load(Ordering::SeqCst)
                        })
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                        .closing;
                    if closing || interrupted.load(Ordering::SeqCst) {
                        return Err(None);
                    }
                    wait = wait.saturating_mul(2).min(QUEUE_RETRY_MAX);
                }
                Err(error) => return Err(Some(failed(error_status(&error), &error.to_string()))),
            }
        }
    }
}

/// The engine as the worker drives it.
fn engine_submit(engine: EngineHandle) -> Submit {
    Arc::new(
        move |request: GenerationRequest| -> crate::Result<Generation> {
            let mut events = match request {
                GenerationRequest::Chat(request) => engine.chat(request),
                GenerationRequest::Completion(request) => engine.complete(request),
            }?;
            let handle = events.cancel_handle();
            Ok(Generation {
                next: Box::new(move |timeout| events.next_signal(timeout)),
                cancel: Arc::new(move || handle.cancel()),
            })
        },
    )
}

fn duplicate(request: &GenerationRequest) -> GenerationRequest {
    match request {
        GenerationRequest::Chat(request) => GenerationRequest::Chat(request.clone()),
        GenerationRequest::Completion(request) => GenerationRequest::Completion(request.clone()),
    }
}

/// When generation for a batch expiring at `expires_at` must stop: just
/// after it, so the next line boundary sees the batch expired.
fn deadline(expires_at: u64) -> Instant {
    let left = expires_at.saturating_add(1).saturating_sub(unix_now());
    Instant::now()
        .checked_add(Duration::from_secs(left))
        .unwrap_or_else(Instant::now)
}

/// A line answered with an error status, in the HTTP error shape.
fn failed(status: StatusCode, message: &str) -> LineOutcome {
    let kind = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    LineOutcome::Response {
        status_code: status.as_u16(),
        body: json!({"error":{"message":message,"type":kind,"param":null,"code":null}}),
    }
}

async fn read_json<T: DeserializeOwned>(body: Body) -> Result<T, Failure> {
    let bytes = match Limited::new(body, MAX_REQUEST_BYTES).collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) if error.downcast_ref::<LengthLimitError>().is_some() => {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                "HTTP request is too large".to_owned(),
            ));
        }
        Err(error) => return Err(bad_request(format!("invalid HTTP body: {error}"))),
    };
    serde_json::from_slice(&bytes)
        .map_err(|error| bad_request(format!("invalid request body: {error}")))
}

fn no_query(query: Option<&str>) -> Result<(), Failure> {
    match query_pairs(query).first() {
        Some((key, _)) => Err(bad_request(format!(
            "query parameter {key:?} is not supported"
        ))),
        None => Ok(()),
    }
}

/// `after` and `limit`, each at most once.
fn list_query(query: Option<&str>) -> Result<ListBatches, Failure> {
    let mut list = ListBatches::default();
    let mut seen: Vec<String> = Vec::new();
    for (key, value) in query_pairs(query) {
        if seen.contains(&key) {
            return Err(bad_request(format!("query parameter {key:?} is repeated")));
        }
        match key.as_str() {
            "after" => list.after = Some(value),
            "limit" => {
                list.limit = Some(value.parse().map_err(|_| {
                    bad_request(format!("limit {value:?} is not a positive integer"))
                })?);
            }
            _ => {
                return Err(bad_request(format!(
                    "query parameter {key:?} is not supported"
                )));
            }
        }
        seen.push(key);
    }
    Ok(list)
}

fn to_json(value: &impl serde::Serialize) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// The status a native error is answered with.
fn failure(error: Error) -> Failure {
    match error {
        Error::InvalidArgument(message) => (StatusCode::BAD_REQUEST, message),
        Error::NotFound(message) => (StatusCode::NOT_FOUND, message),
        Error::Conflict(message) => (StatusCode::CONFLICT, message),
        other => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("batch store failed: {other}"),
        ),
    }
}

fn bad_request(message: impl Into<String>) -> Failure {
    (StatusCode::BAD_REQUEST, message.into())
}

fn route_not_found() -> Failure {
    (StatusCode::NOT_FOUND, "route not found".to_owned())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
#[path = "batch_tests.rs"]
mod tests;
