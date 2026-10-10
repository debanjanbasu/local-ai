//! Background Responses: `POST /v1/responses` with `background: true`.
//!
//! Background responses are stored. The request returns the `queued` Response
//! once the engine has admitted the job and that Response is durably in the
//! store; a pump thread then drains the generation, independent of any HTTP
//! client, and persists `in_progress` on the first progress or event and one
//! terminal state: `completed`, `incomplete`, `failed` or `cancelled`.
//! `GET /v1/responses/{id}` reads the store, so it reports the latest durable
//! status, from this server or another sharing the directory.
//!
//! With `stream: true` the job also keeps a numbered event journal (see
//! [`super::journal`]): `response.created` and `response.queued` with the
//! `queued` Response, `response.in_progress` once the engine signals, the
//! output events, and the events that end the stream. Every batch is appended,
//! under the job's lock, before [`Published`] tells subscribers it exists, and
//! the end is appended only once the terminal record is durable, so no
//! subscriber ever sees an end the store does not hold. Subscribers (see
//! [`super::resume`]) only read; one that disconnects or stalls never cancels
//! or slows the job.
//!
//! Every write and every state change of one job happens under that job's
//! lock, so the pump, `cancel`, `DELETE` and shutdown settle it exactly once:
//! whichever takes the lock first decides, and the rest see a terminal or
//! deleted job. Nothing waits on a timer: the pump parks on the engine's
//! channel, cancellation is the engine's cancel token, and the end of the
//! native work is the channel closing, which a cancelled request does without
//! sending `Finished`. The job's store lease and engine queue slot are held
//! until that close, so neither is released while the worker still runs.
//!
//! A background Response created with `store: false` (or with `store`
//! omitted on a server without `--response-store`) runs exactly the same
//! way, but in [`Temporary`]: a private, owner-only store this process
//! creates under the system temporary directory and removes when it stops.
//! It reports `store: false`, never touches the configured store, and stays
//! retrievable, cancellable and replayable until [`TEMPORARY_RETENTION`]
//! after it ends; a running job is never expired. Nothing about it is
//! durable: a restart discards it.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use serde_json::{Value, json};
use tokio::sync::watch;

use local_engine::{EventStream, Signal};

use super::Admitted;
use super::journal::{JournalWriter, terminal_events};
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

/// How much of a streamed job's journal subscribers may read, and whether
/// any more is coming.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Published {
    /// Bytes of whole, flushed event batches in the journal.
    pub(super) bytes: u64,
    /// The job has settled: nothing more will be appended.
    pub(super) settled: bool,
    /// The response was deleted: subscribers stop at once.
    pub(super) deleted: bool,
    /// A failed write left the journal past `bytes` unknown: subscribers send
    /// what was published and stop without deriving an end, which a replay
    /// after the job is gone numbers from the journal as it is.
    pub(super) torn: bool,
}

/// One background Response and the lock that serialises its settlement.
struct Job {
    state: Mutex<Inner>,
    /// Cooperatively cancels the native work.
    stop: Box<dyn Fn() + Send + Sync>,
    /// What subscribers clone to follow the journal; it never waits itself.
    watcher: watch::Receiver<Published>,
}

struct Inner {
    doc: ResponsesState,
    phase: Phase,
    /// The event journal, for a job created with `stream: true`.
    journal: Option<JournalWriter>,
    published: watch::Sender<Published>,
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

    /// Append `events` to the journal and tell subscribers. Nothing without
    /// a journal. `Err` once the journal cannot be written.
    fn append(&mut self, events: Vec<Value>) -> Result<(), String> {
        let Some(journal) = self.journal.as_mut() else {
            return Ok(());
        };
        if let Err(error) = journal.append(events) {
            if journal.torn() {
                self.published
                    .send_if_modified(|published| !std::mem::replace(&mut published.torn, true));
            }
            let message = format!("the response's events could not be stored: {error}");
            eprintln!("background response {}: {message}", self.doc.id());
            return Err(message);
        }
        let bytes = journal.bytes();
        self.published
            .send_modify(|published| published.bytes = bytes);
        Ok(())
    }

    /// Journal the events the document has produced since the last call.
    fn publish(&mut self) -> Result<(), String> {
        let events = self.doc.take_events();
        self.append(events)
    }

    /// Journal the end of `response`, which is durable, fully flushed.
    fn publish_end(&mut self, response: &Value) {
        if self.append(terminal_events(response)).is_ok()
            && let Some(journal) = &self.journal
            && let Err(error) = journal.sync()
        {
            eprintln!(
                "background response {}: event journal not flushed: {error}",
                self.doc.id()
            );
        }
    }

    /// Persist the terminal `response` and settle on it.
    ///
    /// The output events before the end are journaled first, the record is
    /// written, and only then the end, derived from that record. When the
    /// record cannot be written the job settles `failed` instead, and that is
    /// written if it can; if neither can be, the record is removed rather than
    /// left claiming work that is no longer running. `Err` carries why the
    /// intended state is not durable.
    fn settle(&mut self, response: Value) -> Result<Value, String> {
        self.doc.retract_end();
        // A journal failure here leaves the record to decide; readers then
        // derive the end from it.
        let _ = self.publish();
        let settled = match self.doc.save(&response) {
            Ok(()) => {
                self.publish_end(&response);
                self.phase = Phase::Settled(response.clone());
                Ok(response)
            }
            Err(error) => {
                let id = self.doc.id().to_owned();
                let message = format!("the response could not be stored: {error}");
                eprintln!("background response {id}: {message}");
                let failed = self.doc.fail(&message);
                self.doc.retract_end();
                let _ = self.publish();
                if let Err(error) = self.doc.save(&failed) {
                    eprintln!("background response {id}: failure not stored either: {error}");
                    if let Some(store) = self.doc.store() {
                        let _ = store.delete(&id);
                    }
                    self.phase = Phase::StoreFailed(message.clone());
                } else {
                    self.publish_end(&failed);
                    self.phase = Phase::Settled(failed);
                }
                Err(message)
            }
        };
        self.published
            .send_modify(|published| published.settled = true);
        // Retention of a temporary response counts from here, its end.
        if let Some(temporary) = self.doc.temporary() {
            temporary.retain(self.doc.id());
        }
        settled
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
            inner.doc.started();
            inner.phase = Phase::InProgress;
        }
        let done = match signal {
            Signal::Event(event) => inner.doc.event(event),
            Signal::Progress(_) => None,
        };
        let failed = match done {
            Some(done) => inner.settle(done).is_err(),
            None => inner.publish().is_err_and(|message| {
                // Subscribers can no longer be told what was generated, so the
                // job ends here rather than run on unobserved.
                let failed = inner.doc.fail(&message);
                let _ = inner.settle(failed);
                true
            }),
        };
        if failed {
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
        inner.journal = None;
        inner
            .published
            .send_modify(|published| published.deleted = true);
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
        inner.journal = None;
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

    /// Follow the journal of job `id`, when this server is running it.
    pub(super) fn watch(&self, id: &str) -> Option<watch::Receiver<Published>> {
        self.job(id).map(|job| job.watcher.clone())
    }

    /// Make every later journal write of job `id` fail.
    #[cfg(test)]
    pub(super) fn break_journal(&self, store: &ResponseStore, id: &str) {
        if let (Some(job), Some(path)) = (self.job(id), store.journal_file(id))
            && let Some(journal) = job.lock().journal.as_mut()
        {
            let _ = journal.break_for_test(&path);
        }
    }

    /// Make the next journal write of job `id` leave whole lines behind and
    /// fail to roll them back.
    #[cfg(test)]
    pub(super) fn tear_journal(&self, store: &ResponseStore, id: &str) {
        if let (Some(job), Some(path)) = (self.job(id), store.journal_file(id))
            && let Some(journal) = job.lock().journal.as_mut()
        {
            journal.tear_for_test(&path);
        }
    }

    /// Take the lease, durably store `doc` as `queued` and register the job.
    ///
    /// `next` yields the engine's signals and `None` once the native work has
    /// released the request; `stop` cancels it. On failure both are dropped,
    /// which cancels the native work. A streaming `doc` gets its journal,
    /// holding its opening events, before the record is written. Blocks on
    /// the store.
    pub(super) fn admit(
        self: &Arc<Self>,
        mut doc: ResponsesState,
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
        // A temporary response is answered for from the temporary store from
        // here on, and forgotten again if it is not admitted.
        let temporary = doc.temporary().cloned();
        if let Some(temporary) = &temporary {
            temporary.track(&id);
        }
        let not_stored = |error: std::io::Error| {
            if let Some(temporary) = &temporary {
                temporary.forget(&id);
            }
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("the background response could not be stored: {error}"),
            )
        };
        let journal = if doc.streaming() {
            doc.start_queued();
            match store.create_journal(&id, doc.take_events()) {
                Ok(journal) => Some(journal),
                Err(error) => {
                    stop();
                    return Err(not_stored(error));
                }
            }
        } else {
            None
        };
        let queued = doc.snapshot("queued");
        if let Err(error) = doc.save(&queued) {
            stop();
            let _ = store.delete(&id);
            return Err(not_stored(error));
        }
        let (published, watcher) = watch::channel(Published {
            bytes: journal.as_ref().map_or(0, JournalWriter::bytes),
            ..Published::default()
        });
        let job = Arc::new(Job {
            state: Mutex::new(Inner {
                doc,
                phase: Phase::Queued,
                journal,
                published,
            }),
            stop: Box::new(stop),
            watcher,
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
            if let Err(error) = store.seal_journal(id, &stored.response) {
                // Readers derive the same end from the record.
                eprintln!("response {id}: event journal not sealed: {error}");
            }
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

pub(super) fn pending(response: &Value) -> bool {
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

/// How long a background `store: false` Response stays retrievable after it
/// ends, as `OpenAI` documents its temporary retention ("roughly 10 minutes").
pub(super) const TEMPORARY_RETENTION: Duration = Duration::from_mins(10);

/// What [`Temporary`] reads the time from.
pub(super) type Clock = Arc<dyn Fn() -> Instant + Send + Sync>;

/// The private store of background `store: false` Responses, and when each
/// expires.
///
/// Its records and journals are the ordinary [`ResponseStore`] files, so
/// every lifecycle path (admission, the pump, cancellation, deletion,
/// streaming and replay) is the durable one; only where they live and how
/// long differ. An ID is answered for here exactly while this index holds
/// it, and never looked up in the configured store meanwhile, so the two
/// never answer for the same ID.
///
/// Expiry needs no scan: a response gets its deadline when it settles, an
/// access past the deadline expires it on the spot, and [`Self::reap`] parks
/// until the nearest deadline or a new one, deleting only what is due.
/// Expiring a response deletes it through [`Background::delete`], under its
/// job's lock; it has already settled, so no native work is cancelled.
pub(super) struct Temporary {
    store: Arc<ResponseStore>,
    ttl: Duration,
    clock: Clock,
    index: Mutex<Index>,
    /// Wakes [`Self::reap`] for an earlier deadline or for shutdown.
    wake: Condvar,
    /// The private directory this process created, removed by
    /// [`Self::close`] or on drop; `None` when the store is borrowed.
    dir: Option<PathBuf>,
}

#[derive(Default)]
struct Index {
    /// Every response kept here: `None` while it runs, its expiry once it
    /// has ended.
    ids: HashMap<String, Option<Instant>>,
    /// The ended ones, earliest expiry first.
    deadlines: BTreeSet<(Instant, String)>,
    closed: bool,
}

impl std::fmt::Debug for Temporary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Temporary")
            .field("store", &self.store)
            .field("ttl", &self.ttl)
            .finish_non_exhaustive()
    }
}

/// Create a fresh owner-only directory under the system temporary directory.
///
/// The name is random and the directory is created, not reused, so nothing
/// another user prepared can be adopted; on Unix it must then be a real
/// directory owned by this user.
fn private_dir() -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join(super::response::new_id("local-ai-temporary-"));
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(0o700);
    }
    builder.create(&dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        let metadata = std::fs::symlink_metadata(&dir)?;
        if !metadata.is_dir() || metadata.uid() != rustix::process::geteuid().as_raw() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("{} is not a private directory", dir.display()),
            ));
        }
    }
    Ok(dir)
}

impl Temporary {
    /// Create the private store, in a new owner-only directory that
    /// [`Self::close`] removes.
    pub(super) fn create() -> std::io::Result<Arc<Self>> {
        let dir = private_dir()?;
        match ResponseStore::open(&dir) {
            Ok(store) => {
                let mut temporary =
                    Self::with(Arc::new(store), TEMPORARY_RETENTION, Arc::new(Instant::now));
                temporary.dir = Some(dir);
                Ok(Arc::new(temporary))
            }
            Err(error) => {
                let _ = std::fs::remove_dir_all(&dir);
                Err(error)
            }
        }
    }

    /// Keep temporary responses in `store` for `ttl` after they end, by
    /// `clock`. The directory is the caller's.
    pub(super) fn with(store: Arc<ResponseStore>, ttl: Duration, clock: Clock) -> Self {
        Self {
            store,
            ttl,
            clock,
            index: Mutex::default(),
            wake: Condvar::new(),
            dir: None,
        }
    }

    fn lock(&self) -> MutexGuard<'_, Index> {
        self.index.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(super) const fn store(&self) -> &Arc<ResponseStore> {
        &self.store
    }

    /// Answer for response `id`, which is being admitted, from here.
    pub(super) fn track(&self, id: &str) {
        self.lock().ids.insert(id.to_owned(), None);
    }

    /// Start the retention of response `id`, which has just ended. Once only:
    /// a later settlement attempt does not extend it.
    pub(super) fn retain(&self, id: &str) {
        let mut index = self.lock();
        let deadline = (self.clock)() + self.ttl;
        if let Some(slot @ None) = index.ids.get_mut(id) {
            *slot = Some(deadline);
            index.deadlines.insert((deadline, id.to_owned()));
            drop(index);
            self.wake.notify_all();
        }
    }

    /// Stop answering for response `id`.
    pub(super) fn forget(&self, id: &str) {
        let mut index = self.lock();
        if let Some(Some(deadline)) = index.ids.remove(id) {
            index.deadlines.remove(&(deadline, id.to_owned()));
        }
    }

    /// Whether response `id` is kept here, expired or not.
    pub(super) fn knows(&self, id: &str) -> bool {
        self.lock().ids.contains_key(id)
    }

    /// Whether response `id` is kept here: `None` when it is not this
    /// store's, `Some(false)` when its retention had ended, and it has now
    /// been deleted. Blocks on the store.
    pub(super) fn holds(&self, background: &Background, id: &str) -> Option<bool> {
        let deadline = *self.lock().ids.get(id)?;
        let expired = deadline.is_some_and(|deadline| deadline <= (self.clock)());
        if expired {
            self.expire(background, id);
        }
        Some(!expired)
    }

    /// Delete response `id` and stop answering for it. Its job, if it is
    /// still draining, sees a deletion; it has ended, so nothing is cancelled.
    fn expire(&self, background: &Background, id: &str) {
        if let Err((_, message)) = background.delete(&self.store, id) {
            // Unreachable from now on all the same; the directory goes when
            // the server stops.
            eprintln!("temporary response {id} not deleted on expiry: {message}");
        }
        self.forget(id);
    }

    /// Delete every response whose retention has ended, returning how many.
    /// Blocks on the store.
    pub(super) fn expire_due(&self, background: &Background) -> usize {
        let now = (self.clock)();
        let due: Vec<String> = {
            let index = self.lock();
            index
                .deadlines
                .iter()
                .take_while(|(deadline, _)| *deadline <= now)
                .map(|(_, id)| id.clone())
                .collect()
        };
        for id in &due {
            self.expire(background, id);
        }
        due.len()
    }

    /// The private directory this process created for the store.
    #[cfg(test)]
    pub(super) fn dir(&self) -> Option<&std::path::Path> {
        self.dir.as_deref()
    }

    /// The earliest pending expiry.
    #[cfg(test)]
    pub(super) fn next_deadline(&self) -> Option<Instant> {
        self.lock().deadlines.first().map(|(deadline, _)| *deadline)
    }

    /// Expire responses as they fall due, on a thread of its own that parks
    /// until the nearest deadline or a new one, until [`Self::close`].
    pub(super) fn reap(
        self: &Arc<Self>,
        background: Arc<Background>,
    ) -> std::io::Result<std::thread::JoinHandle<()>> {
        let temporary = Arc::clone(self);
        std::thread::Builder::new()
            .name("local-ai-temporary-expiry".into())
            .spawn(move || {
                loop {
                    let mut index = temporary.lock();
                    loop {
                        if index.closed {
                            return;
                        }
                        let now = (temporary.clock)();
                        index = match index.deadlines.first() {
                            Some((deadline, _)) if *deadline <= now => break,
                            Some((deadline, _)) => {
                                let wait = *deadline - now;
                                temporary
                                    .wake
                                    .wait_timeout(index, wait)
                                    .unwrap_or_else(PoisonError::into_inner)
                                    .0
                            }
                            None => temporary
                                .wake
                                .wait(index)
                                .unwrap_or_else(PoisonError::into_inner),
                        };
                    }
                    drop(index);
                    temporary.expire_due(&background);
                }
            })
    }

    /// Stop expiring, forget every response and remove the private
    /// directory, records and journals with it. For shutdown, once no job
    /// writes any more.
    pub(super) fn close(&self) {
        {
            let mut index = self.lock();
            index.closed = true;
            index.ids.clear();
            index.deadlines.clear();
        }
        self.wake.notify_all();
        if let Some(dir) = &self.dir
            && let Err(error) = std::fs::remove_dir_all(dir)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "temporary response directory {} not removed: {error}",
                dir.display()
            );
        }
    }
}

impl Drop for Temporary {
    fn drop(&mut self) {
        if let Some(dir) = &self.dir {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

/// Start `events` as a background job and return its `queued` Response;
/// with `stream`, the job journals its events for [`super::resume`].
///
/// The pump runs on the blocking pool, where it parks on the engine channel
/// between signals; it owns `events`, the lease and `admitted` until the
/// worker closes the channel.
pub(super) async fn start(
    background: Arc<Background>,
    mut events: EventStream,
    reply: Reply,
    admitted: Admitted,
    stream: bool,
) -> Result<Value, Failure> {
    let Protocol::Responses(echo) = &reply.protocol else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "only Responses run in the background".to_owned(),
        ));
    };
    let doc = ResponsesState::new(&reply, Arc::clone(echo), stream);
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
