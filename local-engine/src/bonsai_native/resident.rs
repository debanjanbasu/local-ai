//! Keeps the model's residency sets resident between requests.
//!
//! The target's weights (and the drafter's, see `kflash`) sit in residency
//! sets on the model's command queue, but Metal lets them go after 2-3 s
//! with the GPU unused (see [`QueueResidency`]): through `serve` a request's
//! first four command buffers waited 59-76 ms in the driver after a 2 s
//! pause and 85-120 ms after 5-30 s, against 7-17 ms with this. Within
//! [`WINDOW`] of the model's last command buffer a side thread commits a
//! minimal one ([`QueueResidency::ping`]) [`PERIOD`] after the later of that
//! and its previous one. Past the window, or while the kernel reports memory
//! pressure (read when a ping is due, as the thread wakes then anyway), it
//! takes the sets off the queue so Metal may evict and unwire them; the
//! model's next command buffer puts them back. The sets keep the weights
//! wired meanwhile (8.1 GB above idle with the drafter's). Metal has no
//! notification for its idle release, so this is a bounded periodic task
//! like the Neural Engine's keepalive.

use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use local_metal::context::QueueResidency;

/// Below the 2-3 s after which Metal releases the residency.
const PERIOD: Duration = Duration::from_secs(1);
/// How long after the last command buffer the sets stay resident: the
/// Neural Engine keepalive's window.
const WINDOW: Duration = Duration::from_secs(60);

struct State {
    inner: Mutex<Inner>,
    changed: Condvar,
}

struct Inner {
    stop: bool,
    last_use: Instant,
    /// Whether the thread waits untimed for [`Resident::touch`].
    idle: bool,
}

fn lock(state: &State) -> MutexGuard<'_, Inner> {
    state.inner.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The side thread; dropping it stops the thread and leaves the sets as
/// they are.
pub(super) struct Resident {
    state: Arc<State>,
    residency: QueueResidency,
    thread: Option<JoinHandle<()>>,
}

impl Resident {
    pub(super) fn start(residency: QueueResidency) -> crate::Result<Self> {
        let state = Arc::new(State {
            inner: Mutex::new(Inner {
                stop: false,
                last_use: Instant::now(),
                idle: false,
            }),
            changed: Condvar::new(),
        });
        let shared = Arc::clone(&state);
        let thread_residency = residency.clone();
        let thread = std::thread::Builder::new()
            .name("gpu-resident".into())
            .spawn(move || run(&shared, &thread_residency))
            .map_err(|error| crate::Error::Generation(format!("residency thread: {error}")))?;
        Ok(Self {
            state,
            residency,
            thread: Some(thread),
        })
    }

    /// After a command buffer of the model: restart the window, and put
    /// suspended sets back on the queue.
    pub(super) fn touch(&self) {
        let mut inner = lock(&self.state);
        inner.last_use = Instant::now();
        // Under the state lock, so the thread cannot suspend in between.
        self.residency.resume();
        if inner.idle {
            inner.idle = false;
            drop(inner);
            self.state.changed.notify_all();
        }
    }
}

impl Drop for Resident {
    fn drop(&mut self) {
        lock(&self.state).stop = true;
        self.state.changed.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Ping [`PERIOD`] after the later of the last use and the last ping while
/// within [`WINDOW`] of the last use and free of memory pressure; otherwise
/// suspend the sets and wait untimed for [`Resident::touch`] (or the drop).
#[expect(clippy::significant_drop_tightening)]
fn run(state: &State, residency: &QueueResidency) {
    let mut inner = lock(state);
    let mut last_ping: Option<Instant> = None;
    loop {
        if inner.stop {
            return;
        }
        let due = last_ping.map_or(inner.last_use, |ping| ping.max(inner.last_use)) + PERIOD;
        let now = Instant::now();
        if now < due {
            // A touch meanwhile only moves the deadline: read it again then.
            inner = state
                .changed
                .wait_timeout(inner, due - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
            continue;
        }
        if inner.last_use.elapsed() > WINDOW || memory_pressure() {
            residency.suspend();
            inner.idle = true;
            while inner.idle && !inner.stop {
                inner = state
                    .changed
                    .wait(inner)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            continue;
        }
        drop(inner);
        let result = residency.ping();
        last_ping = Some(Instant::now());
        inner = lock(state);
        if result.is_err() {
            // The model's own next command buffer meets whatever went wrong.
            residency.suspend();
            return;
        }
    }
}

/// Whether the kernel reports memory pressure (warning or critical).
#[expect(unsafe_code)]
fn memory_pressure() -> bool {
    let mut level: libc::c_int = 0;
    let mut size = size_of::<libc::c_int>();
    // SAFETY: the name is NUL-terminated and `level` holds `size` bytes.
    let status = unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&raw mut level).cast(),
            &raw mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    // 1 is normal; 2 warning and 4 critical.
    status == 0 && level > 1
}
