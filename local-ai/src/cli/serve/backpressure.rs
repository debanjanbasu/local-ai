//! Bounded, event-driven handoff of one body frame to a slow HTTP client.
//!
//! Both streaming bodies, SSE and chunked, push frames from a blocking pump
//! thread into a capacity-1 channel that axum drains. This is the one place
//! that thread waits for the client.

use std::future::Future as _;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;

/// What became of a frame offered to the client.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FrameOutcome {
    /// The frame is in the channel, or never needed sending.
    Sent,
    /// The body was dropped, so the client is gone. The normal end.
    Closed,
    /// The client stopped taking frames for the whole stall budget.
    Stalled,
}

/// Offer one frame to `sender`, abandoning it if the client stops reading.
///
/// A bare `blocking_send` is what wedges a server: it parks this thread until
/// the slot frees, and a client that stops reading with its socket open never
/// frees it — no TCP event distinguishes that client from a slow one. The wait
/// here is bounded by a single deadline instead.
///
/// `try_send` first is what makes the healthy case free: a live reader leaves
/// the capacity-1 slot free, the first `try_send` succeeds, and the frame is on
/// its way without a clock read, a waker or a park. The deadline is therefore
/// started only once that attempt finds the slot full.
///
/// The slow path waits on the channel itself rather than on a clock: a pinned
/// [`mpsc::Sender::reserve`] future is polled with a waker that unparks this
/// thread, so the thread wakes the moment the receiver frees the slot or is
/// dropped, and otherwise sleeps once, until the deadline. No Tokio runtime is
/// needed, because nothing here spawns or uses a timer; the receiver's own
/// `recv` or drop is what calls the waker. Returning drops the future, which
/// deregisters the waiter, so an abandoned frame holds no capacity.
pub(super) fn send_frame(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    frame: Bytes,
    stall: Duration,
) -> FrameOutcome {
    let frame = match sender.try_send(Ok(frame)) {
        Ok(()) => return FrameOutcome::Sent,
        Err(TrySendError::Closed(_)) => return FrameOutcome::Closed,
        Err(TrySendError::Full(returned)) => returned,
    };
    if stall.is_zero() {
        return FrameOutcome::Stalled;
    }
    // `None` is a budget too large to represent as an instant, which is a wait
    // with no practical end, the same as the old `elapsed() >= stall` never
    // becoming true.
    let deadline = Instant::now().checked_add(stall);
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut reserve = pin!(sender.reserve());
    loop {
        match reserve.as_mut().poll(&mut context) {
            Poll::Ready(Ok(permit)) => {
                permit.send(frame);
                return FrameOutcome::Sent;
            }
            Poll::Ready(Err(_)) => return FrameOutcome::Closed,
            Poll::Pending => {}
        }
        // Parking can return spuriously or on a stale unpark token, so the
        // remaining time is recomputed from the one deadline on every turn
        // rather than restarted.
        let Some(deadline) = deadline else {
            thread::park();
            continue;
        };
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return FrameOutcome::Stalled;
        }
        thread::park_timeout(left);
    }
}

/// Run `future` to completion on this blocking thread, parked between polls.
///
/// For waits on Tokio's synchronisation primitives from a pump thread, the
/// same way [`send_frame`] waits on its channel: whatever completes the future
/// calls the waker, which unparks this thread, so nothing polls on a timer and
/// no runtime is needed.
pub(super) fn park_until<F: std::future::Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        thread::park();
    }
}

/// Unparks the pump thread waiting in [`send_frame`].
///
/// Called from whichever thread frees the slot or drops the receiver, so the
/// wait ends at that moment instead of at the next poll.
struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
