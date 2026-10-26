use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{Method, StatusCode, header};
use axum::response::Response;
use axum::{Router, routing::any};
use http_body_util::{BodyExt, LengthLimitError, Limited};
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;

use local_engine::{CancelHandle, ChatRequest, Engine, EngineHandle};

use crate::resources::Resources;

mod admin;
mod background;
mod backpressure;
mod batches;
mod browser;
mod chunked;
mod compaction;
mod connection;
mod decisions;
mod embeddings;
mod file_search;
mod hosted;
mod http3;
mod images;
mod journal;
mod models;
mod moderations;
mod options;
mod reasoning_crypto;
mod request;
mod response;
mod responses;
mod resume;
mod services;
mod speech;
mod sse;
mod store;
mod tool_loop;
mod transcription;
mod vector_stores;

#[cfg(test)]
use self::request::message_parts;
#[cfg(test)]
use axum::http::HeaderMap;
#[cfg(test)]
use local_engine::Event;

use self::background::{Background, Failure, Resumed, Temporary};
use self::chunked::start_chunked;
use self::http3::serve_h3;
use self::options::{parse, usage};
use self::reasoning_crypto::ReasoningCipher;
#[cfg(test)]
use self::request::prepare_generation;
use self::response::{
    Protocol, Reply, add_alt_svc, error_response, error_status, error_status_message,
    json_response, queue_full_response, wants_zstd,
};
use self::responses::prepare_responses_conversing;
use self::resume::Retrieval;
use self::sse::start_stream;
use self::store::{ItemPage, PageError, ResponseStore};

const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;
/// Chat Completions and Responses bodies, which may carry base64 images.
const MAX_GENERATION_REQUEST_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug)]
struct Args {
    host: IpAddr,
    port: u16,
    thinking: bool,
    api_key: Option<String>,
    /// How long a client may stop consuming before its generation is dropped.
    stall: Duration,
    /// Where Responses are kept: the user's data directory by default.
    response_store: options::ResponseStoreArg,
    reasoning_key: Option<PathBuf>,
    /// `--decisions-artifact`: the evaluated head, gate and calibration that
    /// enable `POST /v1/decisions`. `None` refuses that route.
    decisions_artifact: Option<PathBuf>,
    /// Hosted tool permissions and bounds; none by default.
    tools: options::ToolArgs,
    /// `--allow-origin` values: browser origins admitted while
    /// authentication is off (see [`browser`]).
    allowed_origins: Vec<String>,
    /// `--allow-host` names: `Host` values admitted besides IP addresses and
    /// local names (see [`browser`]).
    allowed_hosts: Vec<String>,
}

#[derive(Clone)]
struct AppState {
    engine: EngineHandle,
    model: Arc<str>,
    /// Registration time of the loaded model resource, stable for this server.
    created: u64,
    /// Tokens one request may hold, prompt and output together, as the engine
    /// admitted them at load (`BonsaiInfo::context`). Requests over it fail
    /// with a context-overflow error rather than being truncated.
    context: usize,
    thinking: bool,
    /// Who may call: the `--api-key` key and the opt-in admin store's keys.
    auth: Arc<admin::Gate>,
    /// Which browser requests are admitted (DNS rebinding and cross-site
    /// protection).
    browsers: Arc<browser::Browsers>,
    alt_svc: Option<Arc<str>>,
    depth: QueueDepth,
    /// How long a client may stop consuming before its generation is dropped.
    stall: Duration,
    /// The opt-in Responses store; `None` serves Responses statelessly.
    responses: Option<Arc<ResponseStore>>,
    cipher: Option<Arc<ReasoningCipher>>,
    /// Background Responses running on this server.
    background: Arc<Background>,
    /// Places for background-response streams ([`resume::MAX_SUBSCRIBERS`]).
    subscribers: Arc<tokio::sync::Semaphore>,
    /// Foreground generations, cancelled at the shutdown drain deadline.
    foreground: Arc<Foreground>,
    /// Where background `store: false` Responses are kept briefly; `None`
    /// when no private temporary store could be created.
    temporary: Option<Arc<Temporary>>,
    /// The calibrated `POST /v1/decisions` service; `None` without an
    /// artifact (`--decisions-artifact` or an installed `models/decisions-v2`).
    decisions: Option<Arc<decisions::Decisions>>,
    /// Native durable services, available automatically in the user's data directory.
    services: Option<Arc<services::Services>>,
    batches: Option<Arc<batches::Batches>>,
    vector_stores: Option<Arc<vector_stores::VectorStores>>,
    /// Hosted tools this server may run for Responses: the toolbox grants
    /// (none without `--tools-*` flags) and the tool loop's bounds.
    tools: Arc<tool_loop::Tools>,
    /// Native text embeddings, when the embedding model is installed.
    embeddings: Option<Arc<embeddings::Embeddings>>,
    /// Native text to speech, when the speech model is installed.
    speech: Option<Arc<speech::Speech>>,
    /// `POST /v1/moderations`, when the moderation artifact is installed too.
    moderations: Option<Arc<moderations::Moderations>>,
    /// Native speech recognition and translation, when Whisper is installed.
    transcription: Option<Arc<transcription::Transcriber>>,
}

/// Requests the engine has accepted and not finished yet: the queue depth as
/// this server sees it.
///
/// It cannot be read off the engine, because `try_send` reports only that the
/// queue was full, and it is only ever read to explain a rejection, so it is
/// tracked here. Every job reaching the queue comes through this module, so it
/// counts queued and running requests together.
#[derive(Clone, Default)]
struct QueueDepth(Arc<Depth>);

#[derive(Default)]
struct Depth {
    count: AtomicUsize,
    /// Told each time a slot is released, for resumptions the full queue
    /// deferred (see [`resume_interrupted`]).
    freed: tokio::sync::Notify,
}

impl QueueDepth {
    fn load(&self) -> usize {
        self.0.count.load(Ordering::SeqCst)
    }

    /// Take a queue slot. The returned guard releases it on drop, so an early
    /// return, a failed generation and a disconnected client all release it.
    fn admit(&self) -> Admitted {
        self.0.count.fetch_add(1, Ordering::SeqCst);
        Admitted(Arc::clone(&self.0))
    }

    /// Wait until a slot is released. A release with nobody waiting is kept
    /// for the next wait, so none is missed between a full queue and the wait.
    async fn freed(&self) {
        self.0.freed.notified().await;
    }
}

struct Admitted(Arc<Depth>);

/// The foreground generations running on this server (Chat Completions and
/// Responses requests a client is waiting on), so shutdown can cancel those
/// still running at its drain deadline.
#[derive(Default)]
struct Foreground {
    next: std::sync::atomic::AtomicU64,
    live: std::sync::Mutex<std::collections::HashMap<u64, CancelHandle>>,
}

impl Foreground {
    /// Track the generation `cancel` controls until the returned guard drops.
    fn track(self: &Arc<Self>, cancel: CancelHandle) -> Generation {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.lock().insert(id, cancel.clone());
        Generation {
            foreground: Arc::clone(self),
            id,
            cancel,
        }
    }

    /// Cancel every tracked generation, returning how many there were.
    fn cancel_all(&self) -> usize {
        let live = self.lock();
        for cancel in live.values() {
            cancel.cancel();
        }
        live.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, std::collections::HashMap<u64, CancelHandle>> {
        self.live
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// One tracked foreground generation, cancelled when this guard drops.
///
/// The guard is what ties a generation to the request waiting for it. The
/// request's handler holds it while the first signal is awaited on the
/// blocking pool, so a client that leaves while its request is queued or
/// prefilling drops the handler, and with it the guard, which cancels the job
/// that thread is still waiting on. The body pump then holds it until the
/// body ends. Cancelling a generation that already ended does nothing.
struct Generation {
    foreground: Arc<Foreground>,
    id: u64,
    cancel: CancelHandle,
}

/// Cancels a submission still running on the blocking pool when the request
/// handler waiting for it is dropped; emptied once the submission returns.
struct Submitting(Option<CancelHandle>);

impl Drop for Submitting {
    fn drop(&mut self) {
        if let Some(cancel) = self.0.take() {
            cancel.cancel();
        }
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.foreground.lock().remove(&self.id);
    }
}

impl Drop for Admitted {
    fn drop(&mut self) {
        self.0.count.fetch_sub(1, Ordering::SeqCst);
        self.0.freed.notify_one();
    }
}

/// Every request, on every HTTP version: browser protection (see
/// [`browser`]), authentication, then [`dispatch`].
async fn route(State(state): State<AppState>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body = connection::paced(body, state.stall);
    if let Some(refused) = state.browsers.host_refusal(&parts)
        && !(refused.unless_authenticated && admin::enabled(&state).await)
    {
        let message = refused.message;
        return error_response(StatusCode::FORBIDDEN, &message, &state, &parts.headers).await;
    }
    let origin = state.browsers.origin(&parts.headers);
    if browser::is_preflight(&parts) {
        // A preflight carries no key: with authentication on, every origin
        // may ask, since the request itself still needs one.
        let origin = match origin {
            Ok(origin) => origin,
            Err(foreign) if admin::enabled(&state).await => foreign.origin,
            Err(foreign) => {
                return error_response(
                    StatusCode::FORBIDDEN,
                    &foreign.message,
                    &state,
                    &parts.headers,
                )
                .await;
            }
        };
        return browser::preflight(origin, &parts.headers);
    }
    let caller = match admin::authorize(&state, &parts).await {
        Ok(caller) => caller,
        Err(refused) => return *refused,
    };
    let origin = match origin {
        Ok(origin) => origin,
        Err(foreign) if caller != admin::Caller::Anonymous => foreign.origin,
        Err(foreign) => {
            return error_response(
                StatusCode::FORBIDDEN,
                &foreign.message,
                &state,
                &parts.headers,
            )
            .await;
        }
    };
    let mut response = dispatch(state, parts, body, caller).await;
    if let Some(origin) = origin {
        browser::allow(&mut response, origin);
    }
    response
}

/// Serve an admitted request.
#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)]
async fn dispatch(
    state: AppState,
    parts: axum::http::request::Parts,
    body: Body,
    caller: admin::Caller,
) -> Response {
    if admin::matches(parts.uri.path()) {
        return admin::handle(&state, &parts, body, &caller).await;
    }
    // Stored objects are the caller's project's (see `admin::Caller::project`).
    let project = caller.project();
    if services::matches(parts.uri.path()) {
        let response = services::handle(&state, &parts, body, project).await;
        if parts.method == Method::DELETE
            && parts.uri.path().starts_with("/v1/files/")
            && response.status().is_success()
            && let Some(vector_stores) = &state.vector_stores
        {
            // The deleted File left every vector store; drop its documents.
            vector_stores.notify();
        }
        return response;
    }
    if batches::matches(parts.uri.path()) {
        return batches::handle(&state, &parts, body, &caller).await;
    }
    if vector_stores::matches(parts.uri.path()) {
        return vector_stores::handle(&state, &parts, body, project).await;
    }
    if parts.method == Method::GET && parts.uri.path() == "/health" {
        return json_response(
            StatusCode::OK,
            json!({"status":"ok"}),
            &state,
            &parts.headers,
        )
        .await;
    }
    if parts.method == Method::GET && parts.uri.path() == "/v1/models" {
        return json_response(
            StatusCode::OK,
            state.catalog().list(),
            &state,
            &parts.headers,
        )
        .await;
    }
    if let Some(model) = parts.uri.path().strip_prefix("/v1/models/")
        && matches!(parts.method, Method::GET | Method::DELETE)
    {
        return match state.catalog().answer(&parts.method, model) {
            Ok(value) => json_response(StatusCode::OK, value, &state, &parts.headers).await,
            Err((status, message)) => {
                error_response(status, &message, &state, &parts.headers).await
            }
        };
    }
    if parts.method == Method::POST && parts.uri.path() == moderations::ROUTE {
        return moderations::handle(&state, &parts.headers, body, &caller).await;
    }
    if parts.method == Method::POST
        && matches!(
            parts.uri.path(),
            transcription::TRANSCRIPTIONS | transcription::TRANSLATIONS
        )
    {
        return transcription::handle(&state, &parts, body, &caller).await;
    }
    if parts.method == Method::POST && parts.uri.path() == embeddings::ROUTE {
        return embeddings::handle(&state, &parts.headers, body, &caller).await;
    }
    if parts.method == Method::POST && parts.uri.path() == speech::ROUTE {
        return speech::handle(&state, &parts.headers, body, &caller).await;
    }
    if parts.method == Method::POST && parts.uri.path() == compaction::ROUTE {
        let meter = admin::Meter::completions(&state, &caller);
        return compaction::handle(&state, &parts.headers, body, (&meter, project)).await;
    }
    if parts.method == Method::POST && parts.uri.path() == decisions::ROUTE {
        return decisions::handle(&state, &parts.headers, body).await;
    }
    // Authentication has already passed, so a stored response is never read or
    // deleted for a caller without the key, nor one of another project.
    if let Some(rest) = parts.uri.path().strip_prefix("/v1/responses/")
        && (matches!(parts.method, Method::GET | Method::DELETE)
            || (parts.method == Method::POST && rest.ends_with("/cancel")))
    {
        return stored_response(
            &state,
            &parts.method,
            rest,
            parts.uri.query(),
            project,
            &parts.headers,
        )
        .await;
    }
    let count_tokens =
        parts.method == Method::POST && parts.uri.path() == "/v1/responses/input_tokens";
    let Some(api) = generation_api(&parts.method, parts.uri.path()) else {
        return error_response(
            StatusCode::NOT_FOUND,
            "route not found",
            &state,
            &parts.headers,
        )
        .await;
    };
    let body = match Limited::new(body, MAX_GENERATION_REQUEST_BYTES)
        .collect()
        .await
    {
        // `Vec::from` takes over the collected buffer instead of copying up
        // to 64 MiB again.
        Ok(value) => Vec::from(value.to_bytes()),
        Err(error) => {
            let status = if error.is::<LengthLimitError>() {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            return error_response(
                status,
                &format!("invalid HTTP body: {error}"),
                &state,
                &parts.headers,
            )
            .await;
        }
    };
    // Off the runtime: resolving `previous_response_id` reads the store.
    let (thinking, store) = (state.thinking, state.responses.clone());
    let (cipher, model) = (state.cipher.clone(), Arc::clone(&state.model));
    let temporary = state.temporary.clone();
    let services = state
        .services
        .as_ref()
        .map(|services| services.store().scoped(project));
    let owner = project.map(str::to_owned);
    if count_tokens {
        let engine = state.engine.clone();
        let counted = tokio::task::spawn_blocking(move || {
            let request = responses::prepare_input_tokens_conversing(
                &body,
                thinking,
                store.as_ref(),
                (services.as_ref(), owner.as_deref()),
                cipher.as_ref(),
                &model,
            )?;
            engine.count_chat_tokens(&request)
        })
        .await
        .unwrap_or_else(|error| Err(crate::Error::Generation(error.to_string())));
        return match counted {
            Ok(tokens) => {
                json_response(
                    StatusCode::OK,
                    json!({"object":"response.input_tokens","input_tokens":tokens}),
                    &state,
                    &parts.headers,
                )
                .await
            }
            Err(error) => {
                error_response(
                    error_status(&error),
                    &error.to_string(),
                    &state,
                    &parts.headers,
                )
                .await
            }
        };
    }
    let prepared = if matches!(api, Api::Responses) {
        // Compaction and reasoning summaries run engine work of their own.
        let meter = admin::Meter::completions(&state, &caller);
        compaction::prepare_responses(&state, body, (&meter, project)).await
    } else {
        tokio::task::spawn_blocking(move || {
            prepare_retaining(
                api,
                &body,
                thinking,
                (store.as_ref(), temporary.as_ref()),
                (services.as_ref(), owner.as_deref()),
                cipher.as_ref(),
                &model,
            )
        })
        .await
        .unwrap_or_else(|error| {
            Err(crate::Error::Generation(format!(
                "request preparation failed: {error}"
            )))
        })
    };
    let prepared = match prepared {
        Ok(prepared) => prepared,
        Err(error) => {
            return error_response(
                error_status(&error),
                &error.to_string(),
                &state,
                &parts.headers,
            )
            .await;
        }
    };
    let (stream, mut request, mut protocol) = prepared;
    if let Protocol::Responses(echo) = &mut protocol
        && let Some(echo) = Arc::get_mut(echo)
    {
        // A resumable background Response keeps its creator, so a restarted
        // server attributes the regenerated usage to them.
        echo.set_creator(caller.creator());
    }
    let reply = Reply::new(protocol, Arc::clone(&state.model));
    if let Protocol::Responses(echo) = &reply.protocol
        && let (Some(config), Some(vector_stores)) =
            (echo.kinds().file_search(), &state.vector_stores)
        && let Err(message) = vector_stores
            .check_searchable(config.vector_store_ids.clone(), project)
            .await
    {
        let message = format!("file_search: {message}");
        return error_response(StatusCode::BAD_REQUEST, &message, &state, &parts.headers).await;
    }
    let tool_loop = match hosted_loop(&state, &reply, &mut request) {
        Ok(tool_loop) => tool_loop,
        Err(message) => {
            return error_response(StatusCode::BAD_REQUEST, &message, &state, &parts.headers).await;
        }
    };
    let background = matches!(&reply.protocol, Protocol::Responses(echo) if echo.background());
    if background && state.background.closing() {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "the server is shutting down",
            &state,
            &parts.headers,
        )
        .await;
    }
    // A streamed background request takes its stream's place before any
    // engine work, so a full house refuses it rather than starting a job
    // whose stream it then refuses. (The place is held until the stream
    // starts, which is what the lint cannot see.)
    let slot =
        match (background && stream).then(|| Arc::clone(&state.subscribers).try_acquire_owned()) {
            Some(Err(_)) => return subscribers_full(&state, &parts.headers).await,
            slot => slot.and_then(Result::ok),
        };
    let responses = matches!(&reply.protocol, Protocol::Responses(_));
    // Submission compiles a structured response format synchronously before
    // queueing (about half a second the first time, building the tokenizer's
    // grammar tables), so it runs on the blocking pool, never on a runtime
    // worker. A dropped request drops the stream, which cancels the job.
    let compaction_turn = match &reply.protocol {
        Protocol::Responses(echo) => echo.compaction_turn().cloned(),
        Protocol::Chat { .. } => None,
    };
    let events = if let Some(stats) = compaction_turn {
        // A `compaction_trigger` turn already ran its one generation.
        Ok(compaction::finished(stats).await)
    } else {
        let engine = state.engine.clone();
        // Submission also decodes and encodes the request's images, which can
        // take seconds; a client that leaves meanwhile drops this handler, and
        // the guard stops that work at its next video sample or GPU submission.
        let cancel = CancelHandle::new();
        let mut submitting = Submitting(Some(cancel.clone()));
        let events = tokio::task::spawn_blocking(move || engine.chat_with_cancel(request, &cancel))
            .await
            .unwrap_or_else(|error| {
                Err(crate::Error::Generation(format!(
                    "request submission failed: {error}"
                )))
            });
        // From here the stream's own guards own cancellation.
        submitting.0 = None;
        events
    };
    let events = match events {
        Ok(events) => match tool_loop {
            // The loop records its own rounds' usage.
            Some(tool_loop) => admin::Metered::unmetered(tool_loop::start(
                tool_loop,
                events,
                admin::Meter::completions(&state, &caller),
            )),
            None => admin::meter(&state, &caller, events),
        },
        Err(error) => {
            if matches!(error, crate::Error::QueueFull) {
                return queue_full_response(&state, &parts.headers).await;
            }
            // The native API names its field response_format; Responses
            // clients need the corresponding field in their own request.
            let error = match error {
                crate::Error::InvalidArgument(message) if responses => {
                    crate::Error::InvalidArgument(message.replacen(
                        "invalid response_format:",
                        "invalid text.format:",
                        1,
                    ))
                }
                other => other,
            };
            return error_response(
                error_status(&error),
                &error.to_string(),
                &state,
                &parts.headers,
            )
            .await;
        }
    };
    if background {
        let admitted = state.depth.admit();
        // Not waiting for the first event: a pre-generation failure becomes a
        // `failed` status on the stored response, not this request's status.
        let started = background::start(
            Arc::clone(&state.background),
            events,
            reply,
            admitted,
            stream,
        )
        .await;
        return match (started, slot) {
            (Ok(queued), Some(slot)) => {
                // The client follows the journal from its first event; going
                // away only ends this stream, never the job.
                let id = queued["id"].as_str().unwrap_or_default().to_owned();
                stream_events(&state, id, None, project, slot, &parts.headers).await
            }
            (Ok(queued), None) => {
                json_response(StatusCode::OK, queued, &state, &parts.headers).await
            }
            (Err((status, message)), _) => {
                error_response(status, &message, &state, &parts.headers).await
            }
        };
    }
    let admitted = state.depth.admit();
    if stream {
        match start_stream(events, reply, admitted, &state.foreground, state.stall).await {
            Ok(events) => event_stream(events, &state),
            Err(error) => {
                error_response(error_status_message(&error), &error, &state, &parts.headers).await
            }
        }
    } else {
        // The body streams as the generation produces it, so the first byte is on
        // the wire while prefill is still running and a client that has gone away
        // fails a write instead of being waited out. `zstd` is decided from the
        // request here and reported back from the body, because the response may
        // only claim an encoding it is really using.
        let zstd = wants_zstd(&parts.headers);
        let started = start_chunked(
            events,
            reply,
            admitted,
            &state.foreground,
            state.stall,
            zstd,
        );
        match started.await {
            Ok((frames, compressed)) => {
                let mut response = Response::new(Body::from_stream(ReceiverStream::new(frames)));
                response.headers_mut().insert(
                    header::CONTENT_TYPE,
                    header::HeaderValue::from_static("application/json"),
                );
                if compressed {
                    response.headers_mut().insert(
                        header::CONTENT_ENCODING,
                        header::HeaderValue::from_static("zstd"),
                    );
                }
                add_alt_svc(&mut response, &state);
                response
            }
            Err(error) => {
                error_response(error_status_message(&error), &error, &state, &parts.headers).await
            }
        }
    }
}

/// The tool loop for a Response that declared hosted tools, refusing one
/// whose tools this server was not permitted to run.
fn hosted_loop(
    state: &AppState,
    reply: &Reply,
    request: &mut ChatRequest,
) -> Result<Option<tool_loop::Loop>, String> {
    let Protocol::Responses(echo) = &reply.protocol else {
        return Ok(None);
    };
    responses_loop(state, echo, &reply.id, request)
}

/// [`hosted_loop`] for Response `id`, as `echo` describes it. A resumable
/// background Response marks, durably, that a hosted call may run before the
/// first one does, so a restarted server never regenerates one whose calls
/// may already have taken effect.
fn responses_loop(
    state: &AppState,
    echo: &responses::Echo,
    id: &str,
    request: &mut ChatRequest,
) -> Result<Option<tool_loop::Loop>, String> {
    if echo.kinds().all_hosted().next().is_none() {
        return Ok(None);
    }
    let tools = &state.tools;
    tool_loop::check(tools, echo.kinds(), state.vector_stores.as_deref())?;
    let engine = state.engine.clone();
    // Every round renders the same growing conversation, so one prompt-cache
    // affinity lets each round reuse the previous one's prefix.
    request
        .session
        .get_or_insert_with(|| response::new_id("tools_"));
    Ok(Some(tool_loop::Loop {
        submit: Box::new(move |request| engine.chat(request)),
        request: request.clone(),
        toolbox: tools.toolbox.clone(),
        vector_stores: state.vector_stores.clone(),
        kinds: echo.kinds().clone(),
        bounds: tools.bounds,
        max_calls: echo.max_tool_calls(),
        before_tools: echo.resumable_store().map(|store| {
            let (store, id) = (Arc::clone(store), id.to_owned());
            Box::new(move || {
                store.mark_tools_started(&id).map_err(|error| {
                    format!(
                        "no hosted tool call was run: whether one may have run could not be \
                         recorded for resuming this response after a restart: {error}"
                    )
                })
            }) as tool_loop::BeforeTools
        }),
    }))
}

/// A `text/event-stream` response whose frames arrive on `frames`.
fn event_stream(
    frames: tokio::sync::mpsc::Receiver<Result<bytes::Bytes, std::io::Error>>,
    state: &AppState,
) -> Response {
    let mut response = Response::new(Body::from_stream(ReceiverStream::new(frames)));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache"),
    );
    add_alt_svc(&mut response, state);
    response
}

/// Stream background response `id`'s events after `after` in `slot`, to a
/// caller acting for `project`, or answer with the status that refuses it
/// before any stream starts.
async fn stream_events(
    state: &AppState,
    id: String,
    after: Option<u64>,
    project: Option<&str>,
    slot: tokio::sync::OwnedSemaphorePermit,
    headers: &axum::http::HeaderMap,
) -> Response {
    let not_found = format!("response with id {id:?} not found");
    let resuming = state.clone();
    let project = project.map(str::to_owned);
    let subscribed = tokio::task::spawn_blocking(move || {
        let state = resuming;
        let background = &state.background;
        let located = locate(
            state.temporary.as_ref(),
            state.responses.as_ref(),
            background,
            &id,
            project.as_deref(),
        )?;
        if adopt_orphan(&state, &located, &id)? {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "response {id} was interrupted by a restart and is waiting for an engine \
                     queue slot to be resumed; retry shortly"
                ),
            ));
        }
        resume::subscribe(background, &located.store, &id, after)
    })
    .await
    .unwrap_or_else(|error| {
        Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("response store task failed: {error}"),
        ))
    });
    match subscribed {
        Ok(Some(subscription)) => {
            event_stream(resume::start(subscription, state.stall, slot), state)
        }
        Ok(None) => error_response(StatusCode::NOT_FOUND, &not_found, state, headers).await,
        Err((status, message)) => error_response(status, &message, state, headers).await,
    }
}

/// The `503` refusing a background-response stream past
/// [`resume::MAX_SUBSCRIBERS`].
async fn subscribers_full(state: &AppState, headers: &axum::http::HeaderMap) -> Response {
    let message = format!(
        "{} background response streams are already open; retry after 1s with your own backoff",
        resume::MAX_SUBSCRIBERS
    );
    let mut response =
        error_response(StatusCode::SERVICE_UNAVAILABLE, &message, state, headers).await;
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, header::HeaderValue::from_static("1"));
    response
}

#[derive(Clone, Copy)]
enum Api {
    Chat,
    Responses,
}

/// The generation API `method` and `path` name, or `None` for any other
/// route, which is a 404. Only current `OpenAI` APIs are served: the legacy
/// Completions API (`POST /v1/completions`) is deliberately absent.
fn generation_api(method: &Method, path: &str) -> Option<Api> {
    match (method, path) {
        (&Method::POST, "/v1/chat/completions") => Some(Api::Chat),
        (&Method::POST, "/v1/responses" | "/v1/responses/input_tokens") => Some(Api::Responses),
        _ => None,
    }
}

/// [`prepare_retaining`] on a server with no temporary store.
#[cfg(test)]
fn prepare(
    api: Api,
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<(bool, ChatRequest, Protocol)> {
    prepare_retaining(
        api,
        body,
        thinking,
        (store, None),
        (None, None),
        cipher,
        model,
    )
}

/// Parse a request body for `api` into the engine request and how to answer
/// it, keeping background `store: false` Responses in `temporary`, for a
/// caller acting for `project` (see [`prepare_responses_conversing`];
/// `conversations` is scoped to it).
fn prepare_retaining(
    api: Api,
    body: &[u8],
    thinking: bool,
    (store, temporary): (Option<&Arc<ResponseStore>>, Option<&Arc<Temporary>>),
    (conversations, project): (Option<&local_services::Store>, Option<&str>),
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<(bool, ChatRequest, Protocol)> {
    match api {
        Api::Chat => {
            let prepared = request::prepare_generation_with(body, thinking, conversations)?;
            let protocol = Protocol::Chat {
                include_usage: prepared.include_usage,
                logprobs: prepared.request.sampling.0.logprobs.is_some(),
            };
            Ok((prepared.stream, prepared.request, protocol))
        }
        Api::Responses => {
            let prepared = prepare_responses_conversing(
                body,
                thinking,
                store,
                temporary,
                (conversations, project),
                cipher,
                model,
            )?;
            Ok((
                prepared.stream,
                prepared.request,
                Protocol::Responses(Arc::new(prepared.echo)),
            ))
        }
    }
}

/// `GET` and `DELETE /v1/responses/{id}`, `GET .../{id}/input_items` and
/// `POST .../{id}/cancel`, for a caller acting for `project`.
///
/// `rest` is the path after `/v1/responses/`. Unknown, malformed, deleted,
/// expired and never-stored IDs are all the same 404, and so is every ID but
/// a retained temporary one when no store is configured: none of them names a
/// response this server can produce. So is a Response `project` may not see.
/// See [`locate`] for which store answers.
async fn stored_response(
    state: &AppState,
    method: &Method,
    rest: &str,
    query: Option<&str>,
    project: Option<&str>,
    headers: &axum::http::HeaderMap,
) -> Response {
    let (id, items, cancel) = match rest.split_once('/') {
        None if *method != Method::POST => (rest, false, false),
        Some((id, "input_items")) if *method == Method::GET => (id, true, false),
        Some((id, "cancel")) if *method == Method::POST => (id, false, true),
        _ => {
            return error_response(StatusCode::NOT_FOUND, "route not found", state, headers).await;
        }
    };
    let not_found = format!("response with id {id:?} not found");
    if state.responses.is_none() && state.temporary.is_none() {
        let message = format!("{not_found}: this server was started without --response-store");
        return error_response(StatusCode::NOT_FOUND, &message, state, headers).await;
    }
    let outcome = match stored_request(method, items, query) {
        Ok(Retrieval::Stream(after)) => {
            let Ok(slot) = state.subscribers.clone().try_acquire_owned() else {
                return subscribers_full(state, headers).await;
            };
            return stream_events(state, id.to_owned(), after, project, slot, headers).await;
        }
        Ok(retrieval) => {
            let page = retrieval.into_page();
            let (id, method) = (id.to_owned(), method.clone());
            let background = Arc::clone(&state.background);
            let (temporary, durable) = (state.temporary.clone(), state.responses.clone());
            let resuming = state.clone();
            let project = project.map(str::to_owned);
            tokio::task::spawn_blocking(move || {
                let located = locate(
                    temporary.as_ref(),
                    durable.as_ref(),
                    &background,
                    &id,
                    project.as_deref(),
                )?;
                let store = &located.store;
                if method == Method::GET && page.is_none() {
                    // Retrieving an orphaned job resumes it; one waiting for a
                    // queue slot is reported as it stands.
                    adopt_orphan(&resuming, &located, &id)?;
                }
                if cancel {
                    background.cancel(store, &id)
                } else if method == Method::DELETE {
                    let found = background.delete(store, &id)?;
                    if let Some(temporary) = &located.temporary {
                        temporary.forget(&id);
                    }
                    Ok(found.then(|| deleted(&id)))
                } else {
                    stored_answer(store, &method, &id, page.as_ref())
                }
            })
            .await
            .unwrap_or_else(|error| {
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("response store task failed: {error}"),
                ))
            })
        }
        Err(message) => Err((StatusCode::BAD_REQUEST, message)),
    };
    match outcome {
        Ok(Some(value)) => json_response(StatusCode::OK, value, state, headers).await,
        Ok(None) => error_response(StatusCode::NOT_FOUND, &not_found, state, headers).await,
        Err((status, message)) => error_response(status, &message, state, headers).await,
    }
}

/// Resume interrupted background response `id` in `store` here and start its
/// pump: `Ok(true)` when the full engine queue (or shutdown) leaves it
/// pending with no writer. Blocks on the store and on submission.
fn resume_background(
    state: &AppState,
    store: &Arc<ResponseStore>,
    id: &str,
) -> Result<bool, Failure> {
    let (engine, depth) = (state.engine.clone(), state.depth.clone());
    // Resumed after a restart: the generations are attributed to the caller
    // the record keeps. The
    // creator never changes, so reading it before the lease is taken is safe.
    let creator = store
        .load(id)
        .ok()
        .flatten()
        .and_then(|stored| stored.creator);
    let caller = admin::Caller::from_creator(creator.as_ref());
    let meter = admin::Meter::completions(state, &caller);
    let summarizer = compaction::summarizer(engine.clone(), meter);
    let resumed = state.background.resume(
        store,
        id,
        (state.thinking, state.cipher.as_ref(), Some(&summarizer)),
        move |mut request, reply| {
            let tool_loop = match &reply.protocol {
                Protocol::Responses(echo) => responses_loop(state, echo, &reply.id, &mut request)
                    .map_err(crate::Error::InvalidArgument)?,
                Protocol::Chat { .. } => None,
            };
            let events = engine.chat(request)?;
            let mut events = match tool_loop {
                // The loop records its own rounds' usage.
                Some(tool_loop) => admin::Metered::unmetered(tool_loop::start(
                    tool_loop,
                    events,
                    admin::Meter::completions(state, &caller),
                )),
                None => admin::meter(state, &caller, events),
            };
            let cancel = events.cancel_handle();
            // No deadline: the job has no client to stall, and a closed
            // channel is `None`.
            let next = move || events.next_signal(None).ok().flatten();
            Ok((next, move || cancel.cancel(), Some(depth.admit())))
        },
    )?;
    Ok(match resumed {
        Resumed::Started(pump) => {
            tokio::task::spawn_blocking(move || pump.run());
            false
        }
        Resumed::Settled | Resumed::Owned => false,
        Resumed::Deferred => true,
    })
}

/// Resume response `id` if it is a durable background Response left pending
/// by a writer that is gone (another server sharing the store died): `true`
/// when it waits for a queue slot. Nothing for any other response, so
/// reading one never creates a lease sidecar. Blocks on the store.
fn adopt_orphan(state: &AppState, located: &Located, id: &str) -> Result<bool, Failure> {
    if located.temporary.is_some() || state.background.watch(id).is_some() {
        return Ok(false);
    }
    let orphan = matches!(
        located.store.load(id),
        Ok(Some(stored)) if stored.request.is_some() && background::pending(&stored.response)
    );
    if orphan {
        resume_background(state, &located.store, id)
    } else {
        Ok(false)
    }
}

/// Resume the interrupted background Responses `ids` in order, returning
/// those the full engine queue deferred.
async fn resume_each(
    state: &AppState,
    store: &Arc<ResponseStore>,
    mut ids: std::collections::VecDeque<String>,
) -> std::collections::VecDeque<String> {
    while let Some(id) = ids.front().cloned() {
        let (resuming, store) = (state.clone(), Arc::clone(store));
        let resumed =
            tokio::task::spawn_blocking(move || resume_background(&resuming, &store, &id)).await;
        match resumed {
            Ok(Ok(true)) => return ids,
            Ok(Ok(false)) => {}
            Ok(Err((_, message))) => eprintln!("background response not resumed: {message}"),
            Err(error) => eprintln!("background response not resumed: {error}"),
        }
        ids.pop_front();
    }
    ids
}

/// Resume every background Response a crash or restart interrupted in
/// `store`: as many as the engine queue admits now, before the server
/// serves, and each deferred one once a queue slot is released, on a task of
/// its own that waits for those releases and stops at shutdown.
async fn resume_interrupted(state: &AppState, store: &Arc<ResponseStore>) {
    let listed = Arc::clone(store);
    let ids = match tokio::task::spawn_blocking(move || listed.resumable()).await {
        Ok(Ok(ids)) => ids,
        Ok(Err(error)) => {
            eprintln!("warning: interrupted background responses not listed: {error}");
            return;
        }
        Err(error) => {
            eprintln!("warning: interrupted background responses not listed: {error}");
            return;
        }
    };
    let mut deferred = resume_each(state, store, ids.into()).await;
    if deferred.is_empty() {
        return;
    }
    let (state, store) = (state.clone(), Arc::clone(store));
    tokio::spawn(async move {
        while !deferred.is_empty() && !state.background.closing() {
            state.depth.freed().await;
            deferred = resume_each(&state, &store, deferred).await;
        }
    });
}

/// Why an ID that is not a temporary response is never found without a store.
const WITHOUT_STORE: &str = "this server was started without --response-store, so it keeps only \
     background store=false responses, and only briefly";

/// The store that answers for response `id`.
#[derive(Debug)]
struct Located {
    store: Arc<ResponseStore>,
    /// Set when that is the temporary store.
    temporary: Option<Arc<Temporary>>,
}

/// Which store answers for response `id`: the temporary one while it keeps
/// `id`, otherwise the configured one. Never both, so a temporary response is
/// never looked up in, or confused with, the durable store.
///
/// A temporary response whose retention has ended is deleted here and is
/// then the same 404 as any unknown ID, and so is a response a caller acting
/// for `project` may not see ([`responses::visible`]). Blocks on the store.
fn locate(
    temporary: Option<&Arc<Temporary>>,
    durable: Option<&Arc<ResponseStore>>,
    background: &Background,
    id: &str,
    project: Option<&str>,
) -> Result<Located, Failure> {
    let not_found = format!("response with id {id:?} not found");
    let located = match temporary.map(|temporary| (temporary, temporary.holds(background, id))) {
        Some((temporary, Some(true))) => Located {
            store: Arc::clone(temporary.store()),
            temporary: Some(Arc::clone(temporary)),
        },
        Some((_, Some(false))) => return Err((StatusCode::NOT_FOUND, not_found)),
        _ => match durable {
            Some(store) => Located {
                store: Arc::clone(store),
                temporary: None,
            },
            None => {
                return Err((
                    StatusCode::NOT_FOUND,
                    format!("{not_found}: {WITHOUT_STORE}"),
                ));
            }
        },
    };
    if responses::visible(&located.store, id, project).map_err(|error| store_failure(&error))? {
        Ok(located)
    } else {
        Err((StatusCode::NOT_FOUND, not_found))
    }
}

/// Validate the query of a stored-response request.
///
/// `GET /v1/responses/{id}` streams with `stream=true`, resuming after
/// `starting_after` (see [`resume::retrieval`]); `include` has nothing to
/// add. Other parameters are ignored.
fn stored_request(method: &Method, items: bool, query: Option<&str>) -> Result<Retrieval, String> {
    if items {
        return ItemPage::parse(query)
            .map(Retrieval::Items)
            .map_err(|error| match error {
                PageError::Invalid(message) | PageError::AfterNotFound(message) => message,
            });
    }
    if *method == Method::GET {
        return resume::retrieval(query);
    }
    Ok(Retrieval::Document)
}

/// The answer to a successful `DELETE /v1/responses/{id}`.
fn deleted(id: &str) -> serde_json::Value {
    json!({"id":id,"object":"response.deleted","deleted":true})
}

/// The `500` for a response store that failed: the error, which can name
/// files on this machine, goes to the server log, not to the client.
fn store_failure(error: &std::io::Error) -> Failure {
    eprintln!("response store failed: {error}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        "the response store failed; the server log has the details".to_owned(),
    )
}

/// Read, list or delete stored response `id`; `Ok(None)` is a 404.
fn stored_answer(
    store: &ResponseStore,
    method: &Method,
    id: &str,
    page: Option<&ItemPage>,
) -> Result<Option<serde_json::Value>, (StatusCode, String)> {
    let failed = |error: std::io::Error| store_failure(&error);
    if *method == Method::DELETE {
        return Ok(background::delete_unowned(store, id)?.then(|| deleted(id)));
    }
    let Some(stored) = store.load(id).map_err(failed)? else {
        return Ok(None);
    };
    let Some(page) = page else {
        return Ok(Some(stored.response));
    };
    match page.list(&stored.input_items) {
        Ok(list) => Ok(Some(list)),
        Err(PageError::AfterNotFound(after)) => Err((
            StatusCode::NOT_FOUND,
            format!("input item with id {after:?} not found in response {id}"),
        )),
        Err(PageError::Invalid(message)) => Err((StatusCode::BAD_REQUEST, message)),
    }
}

impl models::Installed {
    const fn of(state: &AppState) -> Self {
        Self {
            embeddings: state.embeddings.is_some(),
            moderation: state.moderations.is_some(),
            speech: state.speech.is_some(),
            transcription: state.transcription.is_some(),
        }
    }
}

impl AppState {
    /// The models this server serves.
    fn catalog(&self) -> models::Catalog<'_> {
        models::Catalog {
            bonsai: &self.model,
            context: self.context,
            created: self.created,
            installed: models::Installed::of(self),
        }
    }
}

// Keep service startup and shutdown ownership together.
#[allow(clippy::too_many_lines)]
async fn run_async(args: Args) -> crate::Result<()> {
    let services = services::Services::open_default()?;
    let vector_stores = match vector_stores::VectorStores::start(services.store()) {
        Ok(vector_stores) => Some(vector_stores),
        // Another server on this data directory owns the index: serve
        // everything else.
        Err(local_services::Error::Conflict(message)) => {
            eprintln!("warning: vector stores are unavailable: {message}");
            None
        }
        Err(error) => {
            return Err(crate::Error::InvalidArgument(format!(
                "cannot start vector stores: {error}"
            )));
        }
    };
    // The key is created on first use; by default it lives beside the
    // services store, so encrypted reasoning and compaction work out of the box.
    let key = match &args.reasoning_key {
        Some(path) => path.clone(),
        None => services::data_dir()?.join("reasoning.key"),
    };
    let cipher =
        Some(Arc::new(ReasoningCipher::open(&key).map_err(|error| {
            crate::Error::InvalidArgument(error.to_string())
        })?));
    // Opened before the model loads, so a bad directory fails in milliseconds.
    let responses = match &args.response_store {
        options::ResponseStoreArg::Off => None,
        options::ResponseStoreArg::Dir(dir) => Some(open_responses(dir)?),
        options::ResponseStoreArg::Default => {
            let store = open_responses(&services::data_dir()?.join("responses"))?;
            match store.prune(RESPONSE_RETENTION) {
                Ok(0) => {}
                Ok(removed) => eprintln!(
                    "removed {removed} stored responses older than {} days",
                    RESPONSE_RETENTION.as_secs() / 86_400
                ),
                Err(error) => eprintln!("warning: old stored responses were not pruned: {error}"),
            }
            Some(store)
        }
    };
    // Hosted tools are opened before the model loads too, so a bad root or
    // domain fails in milliseconds; nothing on the machine is granted without
    // a flag (file_search, which only reads the vector stores, needs none).
    let tools = open_tools(&args.tools)?;
    // Background `store: false` Responses are kept only in a private store of
    // this process; without one they are refused, and nothing else changes.
    let temporary = Temporary::create()
        .map_err(|error| {
            eprintln!(
                "warning: background requests with store=false are unavailable: cannot create a \
                 private temporary response store: {error}"
            );
        })
        .ok();
    // One discovery serves both the engine and the TLS lookup: each one probes
    // the disk's write rate with a 16 MiB file, so a second is wasted startup.
    let resources = Resources::discover(None, true)?;
    // Beside the Bonsai model's own folder: `models/embeddinggemma2-text`.
    let models_dir = resources.model.parent().and_then(std::path::Path::parent);
    // The calibrated route's artifact is verified (hashes, acceptance, scope)
    // before the model loads: `--decisions-artifact`, or one installed as
    // `models/decisions-v2`.
    let decisions_artifact = args.decisions_artifact.clone().or_else(|| {
        models_dir
            .map(|models_dir| models_dir.join("decisions-v2"))
            .filter(|dir| dir.is_dir())
    });
    let decisions = decisions::Decisions::start(decisions_artifact.as_deref())?;
    let model: Arc<str> = models::BONSAI_MODEL_ID.into();
    let engine = Engine::from_resources(&resources)?;
    eprintln!("{}", engine.info().json);
    let context = engine.info().model.context;
    let engine = engine.into_handle();
    let prompt_cache = engine.clone();
    let embeddings = match models_dir {
        Some(models_dir) => embeddings::Embeddings::start(models_dir, engine.activity())?,
        None => None,
    };
    if let (Some(vector_stores), Some(embeddings)) = (&vector_stores, &embeddings) {
        vector_stores.set_embedder(embeddings.chunk_embedder());
    }
    // Beside it: `models/kokoro-82m`.
    let speech = match models_dir {
        Some(models_dir) => speech::Speech::start(models_dir, engine.activity())?,
        None => None,
    };
    // With the embedding model: `models/moderation`.
    let moderations = match models_dir {
        Some(models_dir) => moderations::Moderations::start(models_dir, embeddings.as_ref())?,
        None => None,
    };
    // And `models/whisper-large-v3`.
    let transcription = match models_dir {
        Some(models_dir) => transcription::Transcriber::start(models_dir, engine.activity())?,
        None => None,
    };
    let mut state = AppState {
        engine,
        model,
        created: response::unix_now(),
        context,
        thinking: args.thinking,
        auth: Arc::new(admin::Gate::new(
            Some(Arc::new(local_services::admin::Admin::new(
                services.store().clone(),
            ))),
            args.api_key.as_deref(),
        )?),
        browsers: Arc::new(browser::Browsers::new(
            args.host,
            args.allowed_origins,
            args.allowed_hosts,
        )),
        alt_svc: resources
            .tls
            .as_ref()
            .map(|_| Arc::from(format!("h3=\":{}\"; ma=86400", args.port))),
        depth: QueueDepth::default(),
        stall: args.stall,
        responses,
        cipher,
        background: Arc::default(),
        subscribers: Arc::new(tokio::sync::Semaphore::new(resume::MAX_SUBSCRIBERS)),
        foreground: Arc::default(),
        temporary: temporary.clone(),
        decisions: decisions.clone(),
        services: Some(services),
        batches: None,
        vector_stores: vector_stores.clone(),
        tools,
        embeddings,
        speech,
        moderations,
        transcription,
    };
    let address = SocketAddr::new(args.host, args.port);
    let listener = tokio::net::TcpListener::bind(address).await?;
    let batches = batches::Batches::start(&state)?;
    state.batches = Some(Arc::clone(&batches));
    let h3_task = if let Some((cert, key)) = resources.tls {
        Some(tokio::spawn(serve_h3(address, state.clone(), cert, key)))
    } else {
        None
    };
    // Started once nothing can fail early, since it holds the store until
    // shutdown closes it.
    if let Some(temporary) = &temporary
        && let Err(error) = temporary.reap(Arc::clone(&state.background))
    {
        // Expiry still happens on access; unread responses wait for shutdown.
        eprintln!("warning: temporary responses expire only when accessed: {error}");
    }
    // Before serving, so a client finds every resumed job already running.
    if let Some(store) = state.responses.clone() {
        resume_interrupted(&state, &store).await;
    }
    admin::announce(&state.auth, args.host);
    eprintln!("Bonsai server listening on http://{address}");
    let background = Arc::clone(&state.background);
    let app_foreground = Arc::clone(&state.foreground);
    let app = Router::new().fallback(any(route)).with_state(state);
    // Background jobs are settled and cancelled as soon as shutdown begins,
    // while axum is still draining foreground connections.
    let closing = Arc::clone(&background);
    let closing_decisions = decisions.clone();
    let closing_batches = Arc::clone(&batches);
    // Foreground generations still running at the drain deadline are
    // cancelled, so a long one cannot hold shutdown until a SIGKILL that
    // might cut the prompt-cache writes below.
    let foreground = Arc::clone(&app_foreground);
    connection::serve(
        listener,
        app,
        connection::Limits::default(),
        async move {
            shutdown().await;
            closing_batches.close();
            // Waiting decisions end at once, cancelling their engine work,
            // so they never hold up the drain.
            decisions::close(closing_decisions.as_deref());
            let _ = tokio::task::spawn_blocking(move || closing.close()).await;
        },
        move || foreground.cancel_all(),
    )
    .await;
    // Again, for a server that stopped without the signal; settling is idempotent.
    decisions::close(decisions.as_deref());
    batches.close();
    batches.drained().await;
    // The indexer finishes its current file; an unfinished one resumes on start.
    if let Some(vector_stores) = vector_stores {
        let _ = tokio::task::spawn_blocking(move || vector_stores.close()).await;
    }
    let closing = Arc::clone(&background);
    let _ = tokio::task::spawn_blocking(move || closing.close()).await;
    // Each cancelled job still holds its engine slot and lease until the worker
    // lets go of it, which is at most one prefill chunk or decode step away.
    background.drained().await;
    // No job writes any more: temporary responses are discarded, never kept
    // across a restart.
    if let Some(temporary) = temporary {
        let _ = tokio::task::spawn_blocking(move || temporary.close()).await;
    }
    if let Some(task) = h3_task {
        task.abort();
    }
    // Last, once no request can queue another snapshot: the engine's worker
    // thread is never joined, so exiting now would kill a disk write partway.
    let flushed = tokio::task::spawn_blocking(move || {
        prompt_cache.close_prompt_cache(PROMPT_CACHE_CLOSE_DEADLINE)
    })
    .await
    .unwrap_or(false);
    if !flushed {
        eprintln!(
            "warning: prompt-cache disk writes did not finish within {} s of shutdown; \
             the unfinished snapshot was discarded",
            PROMPT_CACHE_CLOSE_DEADLINE.as_secs()
        );
    }
    Ok(())
}

/// How long the default response store keeps a settled response, as
/// `OpenAI` retains stored responses for 30 days.
const RESPONSE_RETENTION: Duration = Duration::from_hours(30 * 24);

fn open_responses(dir: &std::path::Path) -> crate::Result<Arc<ResponseStore>> {
    ResponseStore::open(dir).map(Arc::new).map_err(|error| {
        crate::Error::InvalidArgument(format!(
            "cannot open the response store {}: {error}",
            dir.display()
        ))
    })
}

/// How long shutdown waits for accepted prompt-cache snapshots to commit.
///
/// At most two are pending (one writing, one queued). The largest, ~10 GB at
/// a ~250K-token prompt, hashes and writes in about 15 s on an M4 Pro
/// (~1.7 GB/s measured write rate), so two minutes covers both with room for
/// a slower disk while still bounding the stop; past it the write in flight
/// is aborted and its temporary removed.
const PROMPT_CACHE_CLOSE_DEADLINE: std::time::Duration = std::time::Duration::from_mins(2);

/// How long the runtime waits for its remaining tasks once the server has
/// shut down.
const RUNTIME_EXIT_DEADLINE: Duration = Duration::from_secs(1);

/// The hosted tools `args` grant (none without flags) and the tool loop's
/// bounds, which tools needing no permission (`file_search`) run within.
fn open_tools(args: &options::ToolArgs) -> crate::Result<Arc<tool_loop::Tools>> {
    let toolbox = local_services::tools::Toolbox::new(
        local_services::tools::Permissions {
            root: args.root.clone(),
            write: args.write,
            shell: args.shell,
            fetch_domains: args.fetch_domains.clone(),
        },
        local_services::tools::Limits {
            call_timeout: args.call_timeout,
            ..local_services::tools::Limits::default()
        },
    )
    .map_err(|error| crate::Error::InvalidArgument(format!("hosted tools: {error}")))?;
    if toolbox.any() {
        eprintln!(
            "hosted tools enabled: root={} write={} shell={} fetch={:?}",
            toolbox
                .root()
                .map_or_else(|| "none".to_owned(), |root| root.display().to_string()),
            toolbox.permissions().write,
            toolbox.permissions().shell,
            toolbox.permissions().fetch_domains
        );
    }
    Ok(Arc::new(tool_loop::Tools {
        toolbox,
        bounds: tool_loop::Bounds {
            max_iterations: args.max_iterations,
            budget: args.budget,
            call_timeout: args.call_timeout,
        },
    }))
}

/// Resolve on SIGINT or SIGTERM: both stop the server gracefully, so a
/// supervisor's SIGTERM no longer loses queued prompt-cache writes.
async fn shutdown() {
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut terminate) => {
                terminate.recv().await;
            }
            Err(_) => std::future::pending().await,
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        () = terminate => {}
    }
}

pub fn main_with_args(args: &[String]) -> ExitCode {
    match parse(args) {
        Ok(args) => match tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => {
                let served = runtime.block_on(run_async(args));
                // Everything that writes has finished by now; a blocking task
                // still waiting on a cancelled job must not hold the exit
                // (dropping the runtime would wait for it indefinitely).
                runtime.shutdown_timeout(RUNTIME_EXIT_DEADLINE);
                match served {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(error) => {
                        eprintln!("error: {error}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(error) => {
                eprintln!("error: could not start Tokio runtime: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            if !error.is_empty() {
                eprintln!("error: {error}");
            }
            usage();
            if error.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod protocol_tests;

#[cfg(test)]
mod store_tests;

#[cfg(test)]
mod background_tests;

#[cfg(test)]
mod resume_tests;

#[cfg(test)]
mod recovery_tests;

#[cfg(test)]
mod combos_tests;

#[cfg(test)]
mod projects_tests;
