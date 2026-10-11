//! `/v1/evals`: the HTTP adapter for [`local_services::evals`] and the worker
//! that drives eval runs.
//!
//! | method   | path                                                   | answer                  |
//! |----------|--------------------------------------------------------|-------------------------|
//! | `POST`   | `/v1/evals`                                            | `eval` (`201`)          |
//! | `GET`    | `/v1/evals`                                            | `list` of evals         |
//! | `GET`    | `/v1/evals/{id}`                                       | `eval`                  |
//! | `POST`   | `/v1/evals/{id}`                                       | `eval` (name, metadata) |
//! | `DELETE` | `/v1/evals/{id}`                                       | `eval.deleted`          |
//! | `POST`   | `/v1/evals/{id}/runs`                                  | `eval.run` (`201`)      |
//! | `GET`    | `/v1/evals/{id}/runs`                                  | `list` of runs          |
//! | `GET`    | `/v1/evals/{id}/runs/{run}`                            | `eval.run`              |
//! | `POST`   | `/v1/evals/{id}/runs/{run}/cancel`                     | `eval.run` (canceled)   |
//! | `DELETE` | `/v1/evals/{id}/runs/{run}`                            | `eval.run.deleted`      |
//! | `GET`    | `/v1/evals/{id}/runs/{run}/output_items`               | `list` of output items  |
//! | `GET`    | `/v1/evals/{id}/runs/{run}/output_items/{item}`        | `eval.run.output_item`  |
//!
//! Bodies are strict (unknown fields are a `400`) and so are query strings:
//! lists take `after`, `limit` (1 to 100, default 20) and `order` (`asc`, the
//! default, or `desc`), plus `order_by` for evals and `status` for runs and
//! output items. Invalid input is a `400` (including deleting an eval or run
//! that still has unfinished work), an unknown object a `404`, and
//! cancelling a finished run a `409`. A run's `report_url` is its own URL on
//! this server.
//!
//! # Execution
//!
//! One worker thread drives runs with
//! [`Store::advance_eval_run`](local_services::Store::advance_eval_run);
//! every generation is a line of a batch that the Batches worker runs on the
//! loaded model, so evals share its queue, resume and cancellation. The eval
//! worker never polls: it scans the runs still to drive when it starts
//! (resuming those a stopped server left), after each run is created or
//! cancelled, and whenever the batch worker has finished with a batch.
//! Cancelling a run cancels its batch's current line at once.

use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};

use axum::body::Body;
use axum::http::request::Parts;
use axum::http::{Method, StatusCode, header};
use axum::response::Response;
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::{Semaphore, watch};

use local_services::conversations::Order;
use local_services::evals::{
    Advance, CreateEval, CreateEvalRun, EvalOrderBy, EvalRun, ListEvalRuns, ListEvals,
    ListOutputItems, OutputItemStatus, RunStatus, UpdateEval,
};
use local_services::{Error, Metadata, Store};

use super::AppState;
use super::batches::{Batches, read_json};
use super::response::{error_response, json_response};
use super::store::query_pairs;

/// The route prefix served here.
const PREFIX: &str = "/v1/evals";
/// Store calls running on the blocking pool at once for HTTP requests.
const MAX_BLOCKING: usize = 4;

/// Why a request failed: its status and message.
type Failure = (StatusCode, String);

/// Whether `path` belongs to the Evals API (any method).
pub(super) fn matches(path: &str) -> bool {
    path.strip_prefix(PREFIX)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// Answer an Evals request. Authentication has already passed.
pub(super) async fn handle(state: &AppState, parts: &Parts, body: Body) -> Response {
    let Some(evals) = state.evals.as_ref() else {
        return error_response(
            StatusCode::NOT_FOUND,
            "route not found: this server runs no evals",
            state,
            &parts.headers,
        )
        .await;
    };
    match evals.dispatch(parts, body).await {
        Ok((status, value)) => json_response(status, value, state, &parts.headers).await,
        Err((status, message)) => error_response(status, &message, state, &parts.headers).await,
    }
}

/// The Evals service: its store, its worker and the bound on HTTP work.
pub(super) struct Evals {
    store: Store,
    worker: Arc<Worker>,
    blocking: Arc<Semaphore>,
    finished: watch::Receiver<bool>,
}

impl Evals {
    /// Start the worker on the server's native store and batch service. It
    /// begins with a scan, so runs a stopped server left are resumed.
    pub(super) fn start(state: &AppState, batches: &Arc<Batches>) -> crate::Result<Arc<Self>> {
        let services = state.services.as_ref().ok_or_else(|| {
            crate::Error::InvalidArgument("evals need the native services store".into())
        })?;
        Self::spawn(services.store().clone(), batches, Arc::clone(&state.model))
    }

    /// Start the worker on `store`, generating through `batches`.
    pub(super) fn spawn(
        store: Store,
        batches: &Arc<Batches>,
        model: Arc<str>,
    ) -> crate::Result<Arc<Self>> {
        let worker = Arc::new(Worker {
            store: store.clone(),
            model,
            batches: Arc::clone(batches),
            control: Mutex::new(Control {
                pending: true,
                closing: false,
            }),
            wake: Condvar::new(),
        });
        let observer: Weak<Worker> = Arc::downgrade(&worker);
        batches.on_finish(Arc::new(move || {
            if let Some(worker) = observer.upgrade() {
                worker.notify();
            }
        }));
        let (done, finished) = watch::channel(false);
        let running = Arc::clone(&worker);
        std::thread::Builder::new()
            .name("evals".into())
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

    /// Stop driving runs. Idempotent; never blocks.
    pub(super) fn close(&self) {
        self.worker.close();
    }

    /// Wait until the worker has let go of its leases and the store.
    pub(super) async fn drained(&self) {
        let mut finished = self.finished.clone();
        let _ = finished.wait_for(|done| *done).await;
    }

    /// Answer one request, independent of the server's other state.
    #[allow(clippy::too_many_lines)]
    pub(super) async fn dispatch(
        &self,
        parts: &Parts,
        body: Body,
    ) -> Result<(StatusCode, Value), Failure> {
        let rest = parts.uri.path().strip_prefix(PREFIX).unwrap_or_default();
        let segments: Vec<String> = rest.split('/').skip(1).map(str::to_owned).collect();
        if segments.iter().any(String::is_empty) {
            return Err(route_not_found());
        }
        let query = parts.uri.query();
        let base = report_base(parts);
        let ok = |value: Value| (StatusCode::OK, value);
        let segments: Vec<&str> = segments.iter().map(String::as_str).collect();
        match (parts.method.clone(), segments.as_slice()) {
            (Method::POST, []) => {
                no_query(query)?;
                let request: CreateEvalBody = read_json(body).await?;
                let create = CreateEval {
                    name: request.name,
                    metadata: request.metadata,
                    data_source_config: request.data_source_config,
                    testing_criteria: request.testing_criteria,
                };
                let model = Arc::clone(&self.worker.model);
                let eval = self
                    .call(move |store| store.create_eval(&create, &model))
                    .await?;
                Ok((StatusCode::CREATED, to_json(&eval)))
            }
            (Method::GET, []) => {
                let list = list_evals_query(query)?;
                self.call(move |store| store.list_evals(&list).map(|page| to_json(&page)))
                    .await
                    .map(ok)
            }
            (Method::GET, [id]) => {
                no_query(query)?;
                let id = (*id).to_owned();
                self.call(move |store| store.get_eval(&id).map(|eval| to_json(&eval)))
                    .await
                    .map(ok)
            }
            (Method::POST, [id]) => {
                no_query(query)?;
                let request: UpdateEvalBody = read_json(body).await?;
                let update = UpdateEval {
                    name: request.name,
                    metadata: request.metadata,
                };
                let id = (*id).to_owned();
                self.call(move |store| store.update_eval(&id, &update).map(|eval| to_json(&eval)))
                    .await
                    .map(ok)
            }
            (Method::DELETE, [id]) => {
                no_query(query)?;
                let id = (*id).to_owned();
                self.call(move |store| store.delete_eval(&id).map(|deleted| to_json(&deleted)))
                    .await
                    .map(ok)
            }
            (Method::POST, [id, "runs"]) => {
                no_query(query)?;
                let request: CreateRunBody = read_json(body).await?;
                let create = CreateEvalRun {
                    name: request.name,
                    metadata: request.metadata,
                    data_source: request.data_source,
                };
                let (id, model) = ((*id).to_owned(), Arc::clone(&self.worker.model));
                let run = self
                    .call(move |store| store.create_eval_run(&id, &create, &model))
                    .await?;
                self.worker.notify();
                Ok((StatusCode::CREATED, run_json(run, &base)))
            }
            (Method::GET, [id, "runs"]) => {
                let list = list_runs_query(query)?;
                let id = (*id).to_owned();
                let page = self
                    .call(move |store| store.list_eval_runs(&id, &list))
                    .await?;
                let mut value = to_json(&page);
                if let Some(Value::Array(runs)) = value.get_mut("data") {
                    for (slot, run) in runs.iter_mut().zip(page.data) {
                        *slot = run_json(run, &base);
                    }
                }
                Ok(ok(value))
            }
            (Method::GET, [id, "runs", run]) => {
                no_query(query)?;
                let (id, run) = ((*id).to_owned(), (*run).to_owned());
                let run = self
                    .call(move |store| store.get_eval_run(&id, &run))
                    .await?;
                Ok(ok(run_json(run, &base)))
            }
            (Method::POST, [id, "runs", run, "cancel"]) => {
                no_query(query)?;
                let (id, run) = ((*id).to_owned(), (*run).to_owned());
                let canceled = self
                    .call(move |store| store.cancel_eval_run(&id, &run))
                    .await?;
                if let Some(batch) = &canceled.batch {
                    self.worker.batches.interrupt(batch);
                }
                self.worker.notify();
                Ok(ok(run_json(canceled.run, &base)))
            }
            (Method::DELETE, [id, "runs", run]) => {
                no_query(query)?;
                let (id, run) = ((*id).to_owned(), (*run).to_owned());
                self.call(move |store| {
                    store
                        .delete_eval_run(&id, &run)
                        .map(|deleted| to_json(&deleted))
                })
                .await
                .map(ok)
            }
            (Method::GET, [id, "runs", run, "output_items"]) => {
                let list = list_items_query(query)?;
                let (id, run) = ((*id).to_owned(), (*run).to_owned());
                self.call(move |store| {
                    store
                        .list_eval_output_items(&id, &run, &list)
                        .map(|page| to_json(&page))
                })
                .await
                .map(ok)
            }
            (Method::GET, [id, "runs", run, "output_items", item]) => {
                no_query(query)?;
                let (id, run, item) = ((*id).to_owned(), (*run).to_owned(), (*item).to_owned());
                self.call(move |store| store.get_eval_output_item(&id, &run, &item))
                    .await
                    .map(ok)
            }
            _ => Err(route_not_found()),
        }
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
                    "the eval service is shutting down".to_owned(),
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
                format!("eval store task failed: {error}"),
            )
        })?
        .map_err(failure)
    }
}

/// The body of `POST /v1/evals`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateEvalBody {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    metadata: Option<Metadata>,
    data_source_config: Value,
    testing_criteria: Vec<Value>,
}

/// The body of `POST /v1/evals/{id}`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateEvalBody {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    metadata: Option<Metadata>,
}

/// The body of `POST /v1/evals/{id}/runs`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateRunBody {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    metadata: Option<Metadata>,
    data_source: Value,
}

/// The worker and what it shares with the HTTP side.
struct Worker {
    store: Store,
    model: Arc<str>,
    batches: Arc<Batches>,
    control: Mutex<Control>,
    /// Signalled on every change to `control`.
    wake: Condvar,
}

struct Control {
    /// A scan for runs to drive is due.
    pending: bool,
    closing: bool,
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
        self.lock().closing = true;
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
            let ids = match self.store.active_eval_runs() {
                Ok(ids) => ids,
                Err(error) => {
                    eprintln!("warning: eval runs cannot be listed: {error}");
                    continue;
                }
            };
            for id in ids {
                if self.closing() {
                    return;
                }
                match self.store.lease_eval_run(&id) {
                    Ok(Some(lease)) => match self.store.advance_eval_run(&lease, &self.model) {
                        Ok(Advance::Started(_)) => self.batches.notify(),
                        Ok(Advance::Waiting(_) | Advance::Finished) => {}
                        Err(error) => eprintln!(
                            "warning: eval run {id} stopped: {error}; it resumes at the next \
                             scan or server start"
                        ),
                    },
                    // Another live server drives it.
                    Ok(None) => {}
                    Err(error) => eprintln!("warning: eval run {id} cannot be leased: {error}"),
                }
            }
        }
    }
}

/// `http://<host>/v1/evals`, from the request's `Host` when it is a plain
/// authority.
fn report_base(parts: &Parts) -> String {
    let host = parts
        .headers
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .filter(|host| {
            !host.is_empty()
                && host
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b".-:[]".contains(&byte))
        })
        .unwrap_or("localhost");
    format!("http://{host}{PREFIX}")
}

fn run_json(mut run: EvalRun, base: &str) -> Value {
    run.report_url = format!("{base}/{}/runs/{}", run.eval_id, run.id);
    to_json(&run)
}

fn no_query(query: Option<&str>) -> Result<(), Failure> {
    match query_pairs(query).first() {
        Some((key, _)) => Err(bad_request(format!(
            "query parameter {key:?} is not supported"
        ))),
        None => Ok(()),
    }
}

/// The documented parameters of a list, each at most once.
fn list_pairs(query: Option<&str>, allowed: &[&str]) -> Result<Vec<(String, String)>, Failure> {
    let pairs = query_pairs(query);
    for (index, (key, _)) in pairs.iter().enumerate() {
        if !allowed.contains(&key.as_str()) {
            return Err(bad_request(format!(
                "query parameter {key:?} is not supported"
            )));
        }
        if pairs[..index].iter().any(|(seen, _)| seen == key) {
            return Err(bad_request(format!("query parameter {key:?} is repeated")));
        }
    }
    Ok(pairs)
}

fn limit(value: &str) -> Result<u32, Failure> {
    value
        .parse()
        .map_err(|_| bad_request(format!("limit {value:?} is not a positive integer")))
}

fn order(value: &str) -> Result<Order, Failure> {
    match value {
        "asc" => Ok(Order::Asc),
        "desc" => Ok(Order::Desc),
        _ => Err(bad_request(format!("order {value:?} must be asc or desc"))),
    }
}

fn list_evals_query(query: Option<&str>) -> Result<ListEvals, Failure> {
    let mut list = ListEvals::default();
    for (key, value) in list_pairs(query, &["after", "limit", "order", "order_by"])? {
        match key.as_str() {
            "after" => list.after = Some(value),
            "limit" => list.limit = Some(limit(&value)?),
            "order" => list.order = Some(order(&value)?),
            _ => {
                list.order_by = Some(match value.as_str() {
                    "created_at" => EvalOrderBy::CreatedAt,
                    "updated_at" => EvalOrderBy::UpdatedAt,
                    _ => {
                        return Err(bad_request(format!(
                            "order_by {value:?} must be created_at or updated_at"
                        )));
                    }
                });
            }
        }
    }
    Ok(list)
}

fn list_runs_query(query: Option<&str>) -> Result<ListEvalRuns, Failure> {
    let mut list = ListEvalRuns::default();
    for (key, value) in list_pairs(query, &["after", "limit", "order", "status"])? {
        match key.as_str() {
            "after" => list.after = Some(value),
            "limit" => list.limit = Some(limit(&value)?),
            "order" => list.order = Some(order(&value)?),
            _ => {
                list.status = Some(match value.as_str() {
                    "queued" => RunStatus::Queued,
                    "in_progress" => RunStatus::InProgress,
                    "completed" => RunStatus::Completed,
                    "canceled" => RunStatus::Canceled,
                    "failed" => RunStatus::Failed,
                    _ => {
                        return Err(bad_request(format!(
                            "status {value:?} must be queued, in_progress, completed, canceled \
                             or failed"
                        )));
                    }
                });
            }
        }
    }
    Ok(list)
}

fn list_items_query(query: Option<&str>) -> Result<ListOutputItems, Failure> {
    let mut list = ListOutputItems::default();
    for (key, value) in list_pairs(query, &["after", "limit", "order", "status"])? {
        match key.as_str() {
            "after" => list.after = Some(value),
            "limit" => list.limit = Some(limit(&value)?),
            "order" => list.order = Some(order(&value)?),
            _ => {
                list.status = Some(match value.as_str() {
                    "pass" => OutputItemStatus::Pass,
                    "fail" => OutputItemStatus::Fail,
                    _ => {
                        return Err(bad_request(format!(
                            "status {value:?} must be pass or fail"
                        )));
                    }
                });
            }
        }
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
            format!("eval store failed: {other}"),
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
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#[path = "eval_tests.rs"]
mod tests;
