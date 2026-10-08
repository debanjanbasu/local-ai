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
//!   drops its engine finds every accepted snapshot on the next start. A
//!   process that exits without dropping it loses at most those two jobs and
//!   leaves at worst a `.tmp`, which `trim` reclaims once it is abandoned.
//! - **Serial order.** Jobs run one at a time in submission order, so `trim`
//!   never races another `store` from the same engine.

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use super::{DiskEntry, DiskIndex, PromptSnapshot, store, sweep_abandoned_temporaries};

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
    /// The committed snapshot, or why it was not written.
    pub stored: crate::Result<DiskEntry>,
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
    /// Not queued: a shared-prefix boundary is already waiting, or the thread
    /// is gone.
    Dropped,
}

#[derive(Default)]
struct State {
    queued: Option<StoreJob>,
    /// Tokens of the job being written, so a caller can wait for it.
    in_flight: Option<Vec<u32>>,
    completed: Vec<Completion>,
    shutdown: bool,
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

impl Writer {
    pub fn spawn() -> std::io::Result<Self> {
        let shared = Arc::new(Shared::default());
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
        if state.dead {
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
        let completion = persist(job, &mut index);
        let mut state = shared.lock();
        state.in_flight = None;
        state.completed.push(completion);
        drop(state);
        shared.changed.notify_all();
    }
}

/// `store` then `trim`, against an in-memory index of the directory.
///
/// The index is read from disk once per directory, with `discover`, and then
/// kept current from this thread's own stores and evictions, so trimming no
/// longer re-opens every `.bpc` to read its header and tokens after each store.
fn persist(job: StoreJob, index: &mut Option<DiskIndex>) -> Completion {
    let index = match index {
        Some(index) if index.is_for(&job.root, &job.model_key) => index,
        _ => index.insert(DiskIndex::discover(&job.root, &job.model_key)),
    };
    let stored = store(
        &job.root,
        &job.model_key,
        &job.tokens,
        job.session_id.as_deref(),
        &job.snapshot,
        job.reusable_boundary,
    );
    // The payload is no longer needed; release it before the trim.
    drop(job.snapshot);
    let stored = stored.map(|path| {
        index.insert(&path, job.reusable_boundary);
        DiskEntry {
            path,
            tokens: job.tokens,
            session_id: job.session_id,
            reusable_boundary: job.reusable_boundary,
        }
    });
    sweep_abandoned_temporaries(&job.root, &job.model_key);
    let evicted = index.evict_over(job.budget);
    Completion { stored, evicted }
}
