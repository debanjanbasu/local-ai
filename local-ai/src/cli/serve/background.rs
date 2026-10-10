//! Background Responses: `POST /v1/responses` with `background: true`.
//!
//! The supported subset is non-streaming and stored. The request returns the
//! `queued` Response once the engine has admitted the job and that Response is
//! durably in the store; a pump thread then drains the generation, independent
//! of any HTTP client, and persists `in_progress` on the first progress or
//! event and one terminal state: `completed`, `incomplete`, `failed` or
//! `cancelled`. `GET /v1/responses/{id}` reads the store, so it reports the
//! latest durable status, from this server or another sharing the directory.
//!
//! Every write and every state change of one job happens under that job's
//! lock, so the pump, `cancel`, `DELETE` and shutdown settle it exactly once:
//! whichever takes the lock first decides, and the rest see a terminal or
//! deleted job. Nothing waits on a timer: the pump parks on the engine's
//! channel, cancellation is the engine's cancel token, and the end of the
//! native work is the channel closing, which a cancelled request does without
//! sending `Finished`. The job's store lease and engine queue slot are held
//! until that close, so neither is released while the worker still runs.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use axum::http::StatusCode;
use serde_json::{Value, json};
use tokio::sync::watch;

use local_engine::{EventStream, Signal};

use super::Admitted;
use super::response::{Protocol, Reply};
use super::responses::ResponsesState;
use super::store::{ResponseLease, ResponseStore};

/// What a stopped server reports for a job it did not finish.
const SHUTDOWN_MESSAGE: &str =
    "the server shut down before this background response finished; it is not resumed";

/// What a job reports when the engine closed its stream without a result.
const STOPPED_MESSAGE: &str = "generation stopped before it produced a result";

/// An HTTP status and message for a refused or failed request.
pub(super) type Failure = (StatusCode, String);

fn store_failed(error: &std::io::Error) -> Failure {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("response store failed: {error}"),
    )
}

/// The background jobs this server is running.
pub(super) struct Background {
    /// Watched so shutdown can wait for the map to empty without polling.
    registry: watch::Sender<Registry>,
}

#[derive(Default)]
struct Registry {
    jobs: HashMap<String, Arc<Job>>,
    /// Set once by [`Background::close`]; refuses new jobs from then on.
    closing: bool,
}

impl Default for Background {
    fn default() -> Self {
        Self {
            registry: watch::channel(Registry::default()).0,
        }
    }
}

/// One background Response and the lock that serialises its settlement.
struct Job {
    state: Mutex<Inner>,
    /// Cooperatively cancels the native work.
    stop: Box<dyn Fn() + Send + Sync>,
}

struct Inner {
    doc: ResponsesState,
    phase: Phase,
}

enum Phase {
    /// The durable record says `queued`.
    Queued,
    /// The durable record says `in_progress`.
    InProgress,
    /// Settled once, with this final Response; anything later is drained.
    Settled(Value),
    /// No terminal record could be persisted; retries must still report failure.
    StoreFailed(String),
    /// The record was deleted; nothing is written for it again.
    Deleted,
}

impl Inner {
    const fn pending(&self) -> bool {
        matches!(self.phase, Phase::Queued | Phase::InProgress)
    }

    /// Persist the terminal `response` and settle on it.
    ///
    /// When it cannot be written the job settles `failed` instead, and that is
    /// written if it can; if neither can be, the record is removed rather than
    /// left claiming work that is no longer running. `Err` carries why the
    /// intended state is not durable.
    fn settle(&mut self, response: Value) -> Result<Value, String> {
        match self.doc.save(&response) {
            Ok(()) => {
                self.phase = Phase::Settled(response.clone());
                Ok(response)
            }
            Err(error) => {
                let id = self.doc.id().to_owned();
                let message = format!("the response could not be stored: {error}");
                eprintln!("background response {id}: {message}");
                let failed = self.doc.fail(&message);
                if let Err(error) = self.doc.save(&failed) {
                    eprintln!("background response {id}: failure not stored either: {error}");
                    if let Some(store) = self.doc.store() {
                        let _ = store.delete(&id);
                    }
                    self.phase = Phase::StoreFailed(message.clone());
                } else {
                    self.phase = Phase::Settled(failed);
                }
                Err(message)
            }
        }
    }
}

impl Job {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Fold one engine signal in, persisting `in_progress` on the first.
    fn advance(&self, signal: Signal) {
        let mut inner = self.lock();
        if !inner.pending() {
            // Settled or deleted: the rest is drained only to release the worker.
            return;
        }
        if matches!(inner.phase, Phase::Queued) {
            let progress = inner.doc.snapshot("in_progress");
            if let Err(error) = inner.doc.save(&progress) {
                let failed = inner
                    .doc
                    .fail(&format!("the response could not be stored: {error}"));
                let _ = inner.settle(failed);
                drop(inner);
                (self.stop)();
                return;
            }
            inner.phase = Phase::InProgress;
        }
        if let Signal::Event(event) = signal
            && let Some(done) = inner.doc.event(event)
            && inner.settle(done).is_err()
        {
            drop(inner);
            (self.stop)();
        }
    }

    /// Settle a job still pending as `failed` with `message`, and stop it.
    fn abandon(&self, message: &str) {
        let mut inner = self.lock();
        if inner.pending() {
            let failed = inner.doc.fail(message);
            let _ = inner.settle(failed);
        }
        drop(inner);
        (self.stop)();
    }

    fn cancel(&self) -> Result<Option<Value>, Failure> {
        let mut inner = self.lock();
        let settled = match &inner.phase {
            Phase::Settled(response) => return Ok(Some(response.clone())),
            Phase::StoreFailed(message) => {
                return Err((StatusCode::INTERNAL_SERVER_ERROR, message.clone()));
            }
            Phase::Deleted => return Ok(None),
            Phase::Queued | Phase::InProgress => {
                let cancelled = inner.doc.cancelled();
                inner.settle(cancelled)
            }
        };
        drop(inner);
        (self.stop)();
        settled
            .map(Some)
            .map_err(|message| (StatusCode::INTERNAL_SERVER_ERROR, message))
    }

    fn delete(&self, store: &ResponseStore, id: &str) -> Result<bool, Failure> {
        let mut inner = self.lock();
        if matches!(inner.phase, Phase::Deleted) {
            return Ok(false);
        }
        // Removed under the lock, so no write of this job can follow it.
        let found = store.delete(id).map_err(|error| store_failed(&error))?;
        let pending = inner.pending();
        inner.phase = Phase::Deleted;
        drop(inner);
        if pending {
            (self.stop)();
        }
        Ok(found)
    }
}

/// The detached half of a started job: drains the engine stream and then
/// releases everything the job held.
pub(super) struct Pump {
    background: Arc<Background>,
    id: String,
    job: Arc<Job>,
    next: Box<dyn FnMut() -> Option<Signal> + Send>,
    lease: ResponseLease,
    admitted: Option<Admitted>,
}

impl Pump {
    /// Run until the engine closes the stream. Blocks the calling thread.
    pub(super) fn run(mut self) {
        while let Some(signal) = (self.next)() {
            self.job.advance(signal);
        }
        let mut inner = self.job.lock();
        if inner.pending() {
            let failed = inner.doc.fail(STOPPED_MESSAGE);
            let _ = inner.settle(failed);
        }
        drop(inner);
        // The worker has let go of the request: release the native slot and
        // the lease, then stop answering for the job here.
        drop(self.next);
        drop(self.admitted);
        drop(self.lease);
        let id = self.id;
        self.background.registry.send_modify(|registry| {
            registry.jobs.remove(&id);
        });
    }
}

impl Background {
    /// Whether shutdown has begun, so new background work is refused.
    pub(super) fn closing(&self) -> bool {
        self.registry.borrow().closing
    }

    fn job(&self, id: &str) -> Option<Arc<Job>> {
        self.registry.borrow().jobs.get(id).cloned()
    }

    /// Take the lease, durably store `doc` as `queued` and register the job.
    ///
    /// `next` yields the engine's signals and `None` once the native work has
    /// released the request; `stop` cancels it. On failure both are dropped,
    /// which cancels the native work. Blocks on the store.
    pub(super) fn admit(
        self: &Arc<Self>,
        doc: ResponsesState,
        next: impl FnMut() -> Option<Signal> + Send + 'static,
        stop: impl Fn() + Send + Sync + 'static,
        admitted: Option<Admitted>,
    ) -> Result<(Value, Pump), Failure> {
        let id = doc.id().to_owned();
        let store = doc
            .store()
            .ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "a background response must be stored".to_owned(),
                )
            })?
            .clone();
        let lease = store
            .lease(&id)
            .map_err(|error| store_failed(&error))?
            .ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("response {id} is already leased"),
                )
            })?;
        let queued = doc.snapshot("queued");
        if let Err(error) = doc.save(&queued) {
            stop();
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("the background response could not be stored: {error}"),
            ));
        }
        let job = Arc::new(Job {
            state: Mutex::new(Inner {
                doc,
                phase: Phase::Queued,
            }),
            stop: Box::new(stop),
        });
        let mut registered = false;
        self.registry.send_modify(|registry| {
            if !registry.closing {
                registry.jobs.insert(id.clone(), Arc::clone(&job));
                registered = true;
            }
        });
        if !registered {
            // Shutdown began after the queued write: leave no pending record.
            job.abandon(SHUTDOWN_MESSAGE);
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "the server is shutting down".to_owned(),
            ));
        }
        let pump = Pump {
            background: Arc::clone(self),
            id,
            job,
            next: Box::new(next),
            lease,
            admitted,
        };
        Ok((queued, pump))
    }

    /// `POST /v1/responses/{id}/cancel`: the final Response, `Ok(None)` for 404.
    ///
    /// Idempotent: a response that already ended is returned as it ended. Only
    /// background responses can be cancelled, and one still pending under
    /// another server's lease is refused rather than overwritten.
    pub(super) fn cancel(&self, store: &ResponseStore, id: &str) -> Result<Option<Value>, Failure> {
        if let Some(job) = self.job(id) {
            return job.cancel();
        }
        let Some(stored) = store.load(id).map_err(|error| store_failed(&error))? else {
            return Ok(None);
        };
        if stored.response.get("background").and_then(Value::as_bool) != Some(true) {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "response {id} was not created with background=true, so it cannot be cancelled"
                ),
            ));
        }
        if !pending(&stored.response) {
            return Ok(Some(stored.response));
        }
        let _lease = lease_or_conflict(store, id)?;
        // Re-read under the lease: its writer may have finished meanwhile.
        let Some(mut stored) = store.load(id).map_err(|error| store_failed(&error))? else {
            return Ok(None);
        };
        if pending(&stored.response) {
            // Its writer is gone, and a pending record has produced no output
            // yet, so cancelling it is only its status.
            stored.response["status"] = json!("cancelled");
            store
                .save(&stored.response, &stored.input_items)
                .map_err(|error| store_failed(&error))?;
        }
        Ok(Some(stored.response))
    }

    /// `DELETE /v1/responses/{id}`: whether there was one to delete.
    ///
    /// A job running here is deleted under its lock and cancelled, so it never
    /// writes the record back.
    pub(super) fn delete(&self, store: &ResponseStore, id: &str) -> Result<bool, Failure> {
        self.job(id)
            .map_or_else(|| delete_unowned(store, id), |job| job.delete(store, id))
    }

    /// Begin shutdown: refuse new jobs, settle every pending one as `failed`
    /// and cancel its native work. Blocks on the store.
    pub(super) fn close(&self) {
        let mut jobs = Vec::new();
        self.registry.send_modify(|registry| {
            registry.closing = true;
            jobs.extend(registry.jobs.values().cloned());
        });
        for job in jobs {
            job.abandon(SHUTDOWN_MESSAGE);
        }
    }

    /// Wait until every job's native work has released its request.
    pub(super) async fn drained(&self) {
        let mut registry = self.registry.subscribe();
        let _ = registry.wait_for(|registry| registry.jobs.is_empty()).await;
    }
}

/// Delete a response no job of this server is writing. A background record
/// still pending is only deleted under its lease, so another live server's
/// work is refused with a conflict instead of being deleted under it.
pub(super) fn delete_unowned(store: &ResponseStore, id: &str) -> Result<bool, Failure> {
    let _lease = match store.load(id) {
        Ok(Some(stored)) if pending(&stored.response) => Some(lease_or_conflict(store, id)?),
        _ => None,
    };
    store.delete(id).map_err(|error| store_failed(&error))
}

fn pending(response: &Value) -> bool {
    matches!(
        response.get("status").and_then(Value::as_str),
        Some("queued" | "in_progress")
    )
}

fn lease_or_conflict(store: &ResponseStore, id: &str) -> Result<ResponseLease, Failure> {
    store
        .lease(id)
        .map_err(|error| store_failed(&error))?
        .ok_or_else(|| {
            (
                StatusCode::CONFLICT,
                format!(
                    "response {id} is still being generated by another server sharing this \
                     response store; cancel or delete it there"
                ),
            )
        })
}

/// Start `events` as a background job and return its `queued` Response.
///
/// The pump runs on the blocking pool, where it parks on the engine channel
/// between signals; it owns `events`, the lease and `admitted` until the
/// worker closes the channel.
pub(super) async fn start(
    background: Arc<Background>,
    mut events: EventStream,
    reply: Reply,
    admitted: Admitted,
) -> Result<Value, Failure> {
    let Protocol::Responses(echo) = &reply.protocol else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "only Responses run in the background".to_owned(),
        ));
    };
    let doc = ResponsesState::new(&reply, Arc::clone(echo), false);
    let cancel = events.cancel_handle();
    // No deadline: the job has no client to stall, and a closed channel is `None`.
    let next = move || events.next_signal(None).ok().flatten();
    tokio::task::spawn_blocking(move || {
        let (queued, pump) =
            background.admit(doc, next, move || cancel.cancel(), Some(admitted))?;
        tokio::task::spawn_blocking(move || pump.run());
        Ok(queued)
    })
    .await
    .unwrap_or_else(|error| {
        Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("background task failed: {error}"),
        ))
    })
}
