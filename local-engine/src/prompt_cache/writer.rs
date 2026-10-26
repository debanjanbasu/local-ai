//! Background persistence for the disk tier.
//!
//! Persisting a snapshot hashes the whole payload with SHA-256, writes a
//! temporary, `fsync`s it, renames it into place and trims the directory to the
//! budget. Measured on an M4 Pro for a ~152 MiB snapshot that is ~56–69 ms of
//! hashing, ~15 ms of writing and ~30 ms of `fsync`, all of which used to run
//! on the engine thread: in the middle of prefill for a shared-prefix boundary,
//! where it delayed the first token, and after decode for the request tail,
//! where it delayed the response and the next queued request. None of it needs
//! the GPU or the model, so it runs here, on one thread the engine owns.
//! Measured end to end with ~204 MB snapshots (MTP on), a cold 696-token
//! prompt whose 678-token system boundary is persisted mid-prefill reached its
//! first token in 8.469 s instead of 8.611 s (median of six), and a 17-token
//! request that persists its tail finished in 1.847 s instead of 1.931 s.
//!
//! The engine still captures the snapshot itself (`prompt_snapshot` reads the
//! GPU state back into anonymous mappings) and hands the owned bytes over.
//!
//! Guarantees:
//!
//! - **Only complete snapshots are visible.** A store reports its
//!   [`DiskEntry`] after the rename, through [`Writer::take_completed`]; the
//!   engine indexes nothing it has not been told is complete, so it can never
//!   load a file that is still being written.
//! - **Bounded memory.** At most one job is in flight and one is queued, so at
//!   most two snapshots (~150–200 MB each) are held here. A newer job replaces a
//!   queued one, except that a request-tail snapshot never displaces a queued
//!   shared-prefix boundary, the snapshot most likely to be reused.
//! - **Writes are flushed on drop.** Dropping the [`Writer`] finishes the
//!   in-flight and queued jobs before joining the thread, so a process that
//!   drops its engine finds every accepted snapshot on the next start.
//! - **Bounded close.** A process that cannot drop its engine (the server,
//!   whose engine lives on a worker thread) calls [`Closer::close`] when it
//!   shuts down: new jobs are refused, accepted ones get a deadline to commit,
//!   and past it the in-flight write is aborted within one piece and its
//!   temporary removed. A process that exits without either loses at most
//!   those two jobs and leaves at worst an unlocked `.tmp`, which the next
//!   sweep (at the latest the next start's) reclaims.
//! - **Admission.** A job whose snapshot the trim after it would delete again
//!   (see `DiskIndex::admits`) is not written at all, so a near-duplicate
//!   boundary neither costs a multi-gigabyte write nor evicts the shared
//!   prefix it extends.
//! - **Serial order.** Jobs run one at a time in submission order, so `trim`
//!   never races another `store` from the same engine.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use super::{
    DiskEntry, DiskIndex, PromptSnapshot, store_abortable, stored_bytes,
    sweep_abandoned_temporaries,
};

/// How long [`Closer::close`] waits, after aborting, for the in-flight write
/// to stop and remove its temporary. An abort is seen within one piece
/// (64 MiB); only a final `fsync` already under way can take longer, and a
/// write still unfinished past this is left to the next sweep.
const ABORT_GRACE: Duration = Duration::from_secs(10);

/// One snapshot to persist, with everything `store` and `trim` need.
pub struct StoreJob {
    pub root: PathBuf,
    pub model_key: String,
    pub tokens: Vec<u32>,
    pub session_id: Option<String>,
    pub snapshot: PromptSnapshot,
    pub reusable_boundary: bool,
    /// Disk budget the directory is trimmed to after this store.
    pub budget: u64,
}

/// The outcome of one job, reported once the store and trim have finished.
pub struct Completion {
    /// The committed snapshot, `None` when it was not admitted (the trim
    /// would have deleted it again), or why it was not written.
    pub stored: crate::Result<Option<DiskEntry>>,
    /// Snapshots `trim` deleted to stay within the budget.
    pub evicted: Vec<PathBuf>,
}

/// What [`Writer::submit`] did with a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Submitted {
    /// Queued behind nothing, or behind only the in-flight job.
    Queued,
    /// Queued in place of an older queued job, which will not be written.
    Replaced,
    /// Not queued: a shared-prefix boundary is already waiting, the writer
    /// is closed, or the thread is gone.
    Dropped,
}

#[derive(Default)]
struct State {
    queued: Option<StoreJob>,
    /// Tokens of the job being written, so a caller can wait for it.
    in_flight: Option<Vec<u32>>,
    completed: Vec<Completion>,
    shutdown: bool,
    /// [`Closer::close`] was called: no new job is accepted.
    closed: bool,
    /// The thread is gone (it panicked); nothing pending will ever finish.
    dead: bool,
}

impl State {
    fn pending(&self, mut matches: impl FnMut(&[u32]) -> bool) -> bool {
        !self.dead
            && (self.in_flight.as_deref().is_some_and(&mut matches)
                || self.queued.as_ref().is_some_and(|job| matches(&job.tokens)))
    }
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
    /// Set by a [`Closer::close`] past its deadline; read by the store
    /// between pieces.
    abort: AtomicBool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Single background thread that persists snapshots in submission order.
pub struct Writer {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

/// What [`Closer::close`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Closed {
    /// Every accepted job finished (committed, skipped or failed) in time.
    Flushed,
    /// The deadline passed: the queued job was discarded and the in-flight
    /// write aborted, its temporary removed.
    Aborted,
}

/// Shuts a [`Writer`] down from another thread with a deadline, whether or
/// not its thread was ever spawned.
#[derive(Clone, Default)]
pub struct Closer {
    shared: Arc<Shared>,
}

impl Closer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Refuse new jobs and wait up to `deadline` for the accepted ones; past
    /// it, discard the queued job and abort the in-flight write, then wait
    /// up to [`ABORT_GRACE`] for it to remove its temporary.
    ///
    /// A blocking wait on the writer's own completion signal, not a poll.
    pub fn close(&self, deadline: Duration) -> Closed {
        if self.wait(true, deadline) {
            return Closed::Flushed;
        }
        self.shared.lock().queued = None;
        self.shared.abort.store(true, Ordering::Release);
        self.shared.changed.notify_all();
        let _ = self.wait(false, ABORT_GRACE);
        Closed::Aborted
    }

    /// Wait up to `timeout` until nothing is pending, first refusing new jobs
    /// when `close`; whether nothing is.
    // The guard moves into the wait, which the lint does not follow.
    #[expect(clippy::significant_drop_tightening)]
    fn wait(&self, close: bool, timeout: Duration) -> bool {
        let mut state = self.shared.lock();
        state.closed |= close;
        let waited = self
            .shared
            .changed
            .wait_timeout_while(state, timeout, |state| state.pending(|_| true))
            .unwrap_or_else(PoisonError::into_inner)
            .1;
        !waited.timed_out()
    }
}

impl Writer {
    #[cfg(test)]
    pub fn spawn() -> std::io::Result<Self> {
        Self::spawn_with(&Closer::new())
    }

    /// Spawn the thread behind `closer`, so a [`Closer`] handed out before
    /// the first write closes this writer.
    pub fn spawn_with(closer: &Closer) -> std::io::Result<Self> {
        let shared = Arc::clone(&closer.shared);
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("prompt-cache-writer".into())
            .spawn(move || run(&worker))?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    /// Hand a job to the thread without waiting for any of it.
    pub fn submit(&self, job: StoreJob) -> Submitted {
        let mut state = self.shared.lock();
        if state.dead || state.closed {
            return Submitted::Dropped;
        }
        let submitted = enqueue(&mut state.queued, job);
        drop(state);
        self.shared.changed.notify_all();
        submitted
    }

    /// Outcomes finished since the last call, oldest first.
    #[must_use]
    pub fn take_completed(&self) -> Vec<Completion> {
        std::mem::take(&mut self.shared.lock().completed)
    }

    /// Block while a pending job's tokens satisfy `matches`.
    ///
    /// Returns at once when nothing pending matches, so a caller pays for the
    /// wait only when the snapshot it is about to look up is still on its way.
    pub fn wait_for(&self, mut matches: impl FnMut(&[u32]) -> bool) {
        let state = self.shared.lock();
        let _state = self
            .shared
            .changed
            .wait_while(state, |state| state.pending(&mut matches))
            .unwrap_or_else(PoisonError::into_inner);
    }

    /// The length of the longest pending job whose tokens satisfy `matches`,
    /// without waiting for it.
    pub fn longest_pending(&self, mut matches: impl FnMut(&[u32]) -> bool) -> Option<usize> {
        let state = self.shared.lock();
        if state.dead {
            return None;
        }
        state
            .in_flight
            .iter()
            .chain(state.queued.as_ref().map(|job| &job.tokens))
            .filter(|tokens| matches(tokens))
            .map(Vec::len)
            .max()
    }

    /// Block until every accepted job has finished.
    pub fn flush(&self) {
        self.wait_for(|_| true);
    }
}

/// The one-slot queue: a newer job replaces the waiting one, unless that would
/// trade a shared-prefix boundary for a request tail.
pub(super) fn enqueue(queued: &mut Option<StoreJob>, job: StoreJob) -> Submitted {
    let submitted = match queued {
        None => Submitted::Queued,
        Some(waiting) if waiting.reusable_boundary && !job.reusable_boundary => {
            return Submitted::Dropped;
        }
        Some(_) => Submitted::Replaced,
    };
    *queued = Some(job);
    submitted
}

impl Drop for Writer {
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.changed.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Marks the thread dead on any exit, including an unwind, so `flush` and
/// `wait_for` can never wait on a thread that no longer exists.
struct Exit<'a>(&'a Shared);

impl Drop for Exit<'_> {
    fn drop(&mut self) {
        let mut state = self.0.lock();
        state.dead = true;
        state.queued = None;
        state.in_flight = None;
        drop(state);
        self.0.changed.notify_all();
    }
}

fn run(shared: &Shared) {
    let _exit = Exit(shared);
    let mut index: Option<DiskIndex> = None;
    loop {
        let job = {
            let state = shared.lock();
            let mut state = shared
                .changed
                .wait_while(state, |state| state.queued.is_none() && !state.shutdown)
                .unwrap_or_else(PoisonError::into_inner);
            let Some(job) = state.queued.take() else {
                // Shut down with nothing left to write.
                return;
            };
            state.in_flight = Some(job.tokens.clone());
            job
        };
        let completion = persist(job, &mut index, &shared.abort);
        let mut state = shared.lock();
        state.in_flight = None;
        state.completed.push(completion);
        drop(state);
        shared.changed.notify_all();
    }
}

/// Admission, `store` then `trim`, against an in-memory index of the
/// directory.
///
/// The index is read from disk once per directory, with `discover`, and then
/// kept current from this thread's own stores and evictions, so trimming no
/// longer re-opens every `.bpc` to read its header and tokens after each store.
fn persist(job: StoreJob, index: &mut Option<DiskIndex>, abort: &AtomicBool) -> Completion {
    let index = match index {
        Some(index) if index.is_for(&job.root, &job.model_key) => index,
        _ => index.insert(DiskIndex::discover(&job.root, &job.model_key)),
    };
    let bytes = stored_bytes(
        &job.model_key,
        &job.tokens,
        job.session_id.as_deref(),
        &job.snapshot,
        job.reusable_boundary,
    );
    if !index.admits(&job.tokens, bytes, job.reusable_boundary, job.budget) {
        return Completion {
            stored: Ok(None),
            evicted: Vec::new(),
        };
    }
    let stored = store_abortable(
        &job.root,
        &job.model_key,
        &job.tokens,
        job.session_id.as_deref(),
        &job.snapshot,
        job.reusable_boundary,
        abort,
    );
    // The payload is no longer needed; release it before the trim.
    drop(job.snapshot);
    let stored = stored.map(|path| {
        index.insert(&path, &job.tokens, job.reusable_boundary);
        Some(DiskEntry {
            path,
            tokens: job.tokens,
            session_id: job.session_id,
            reusable_boundary: job.reusable_boundary,
        })
    });
    sweep_abandoned_temporaries(&job.root, &job.model_key);
    let evicted = index.evict_over(job.budget);
    Completion { stored, evicted }
}
