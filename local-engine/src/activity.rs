//! Decode activity of the engine worker, for low-priority GPU work elsewhere.
//!
//! Metal offers no command-queue priorities, so work submitted from another
//! queue (embeddings) competes with Bonsai's decode steps on equal terms. The
//! worker publishes here whether generations are running and counts finished
//! steps; low-priority work submits short command buffers and, between them,
//! calls [`EngineActivity::yield_to_decode`], which returns at once when no
//! generation runs and otherwise blocks until the worker finishes a step (or
//! a deadline passes, so the low-priority work is slowed, never starved).
//! Each decode step is therefore delayed by at most one low-priority command
//! buffer. The wait is a condition variable woken by the worker, not a poll.

use std::sync::{Condvar, Mutex, PoisonError};
use std::time::Duration;

#[derive(Default)]
struct State {
    /// A generation is admitted and the worker is stepping.
    busy: bool,
    /// Steps finished since the worker started.
    steps: u64,
}

/// Shared between the engine worker and low-priority GPU users.
#[derive(Default)]
pub struct EngineActivity {
    state: Mutex<State>,
    changed: Condvar,
}

impl EngineActivity {
    /// The worker is about to step running generations.
    pub(crate) fn stepping(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.busy = true;
    }

    /// The worker finished a step; `busy` is whether generations remain.
    pub(crate) fn stepped(&self, busy: bool) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.steps = state.steps.wrapping_add(1);
        state.busy = busy;
        drop(state);
        self.changed.notify_all();
    }

    /// Whether generations are running.
    #[must_use]
    pub fn busy(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .busy
    }

    /// Steps finished so far.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .steps
    }

    /// Return at once when no generation runs; otherwise block until the
    /// worker finishes its next step, generations stop, or `limit` passes.
    /// Returns whether it waited.
    pub fn yield_to_decode(&self, limit: Duration) -> bool {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let (busy, start) = (state.busy, state.steps);
        if !busy {
            drop(state);
            return false;
        }
        drop(
            self.changed
                .wait_timeout_while(state, limit, |state| state.busy && state.steps == start)
                .unwrap_or_else(PoisonError::into_inner),
        );
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use super::*;

    #[test]
    fn idle_engine_does_not_block() {
        let activity = EngineActivity::default();
        assert!(!activity.yield_to_decode(Duration::from_secs(5)));
    }

    #[test]
    fn busy_engine_releases_after_one_step() {
        let activity = Arc::new(EngineActivity::default());
        activity.stepping();
        let worker = Arc::clone(&activity);
        let started = Instant::now();
        let stepper = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            worker.stepped(true);
        });
        assert!(activity.yield_to_decode(Duration::from_secs(5)));
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(activity.steps(), 1);
        assert!(stepper.join().is_ok());
    }

    #[test]
    fn deadline_bounds_the_wait() {
        let activity = EngineActivity::default();
        activity.stepping();
        let started = Instant::now();
        assert!(activity.yield_to_decode(Duration::from_millis(20)));
        assert!(started.elapsed() >= Duration::from_millis(20));
        assert!(activity.busy());
    }
}
