//! The TCP listener: HTTP/1.1 and HTTP/2 connections, the deadlines on what a
//! client sends, and graceful shutdown with a drain deadline.
//!
//! `axum::serve` builds its hyper connections without a timer, and hyper then
//! silently drops its HTTP/1 header-read timeout, so a client that opened a
//! connection and sent half a request line held it forever. This listener
//! builds them itself, with a timer:
//!
//! - **Headers.** An HTTP/1 request's headers must arrive within
//!   [`HEADER_READ_TIMEOUT`] of the server starting to read them, which also
//!   bounds an idle keep-alive connection. Before hyper knows the protocol it
//!   reads the connection's first bytes with no deadline at all, so a
//!   connection that has not delivered a whole request within the same time
//!   is closed too; that covers one that sends nothing, or a partial HTTP/2
//!   preface.
//! - **Bodies.** A request body is [`paced`]: it must deliver at least
//!   [`PACE_BYTES`] (or end) within every stall budget (`--stall-timeout`)
//!   the handler spends waiting for it, or the read fails. A trickling upload
//!   cannot hold an upload, transcription or generation slot indefinitely,
//!   while a slow but moving one is never cut off.
//! - **Shutdown.** The listener stops accepting, every connection finishes
//!   its requests and closes, and the server waits up to the drain deadline
//!   ([`DRAIN_DEADLINE`]) for them. Generations still running then are
//!   cancelled through their [`super::Foreground`] guards, so their bodies end
//!   at the next engine step and the engine settles its own writes; whatever
//!   still has not closed [`CANCELLED_DRAIN`] later is dropped.

use std::convert::Infallible;
use std::future::Future;
use std::pin::{Pin, pin};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use bytes::Bytes;
use http_body::{Frame, SizeHint};
use hyper::body::Incoming;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::time::{Instant, Sleep};
use tower_service::Service as _;

/// How long a client may take to send a request's headers, and how long a
/// connection may stay open without having sent a request.
///
/// hyper's own default for the header read, which it applies only with a
/// timer; generous for any real client, which sends its headers in one write.
pub(super) const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// How long shutdown waits for foreground requests to finish before it
/// cancels the generations still running.
///
/// launchd sends SIGKILL 20 s after SIGTERM by default; ten seconds of drain
/// and at most [`CANCELLED_DRAIN`] after it leave the rest of shutdown (the
/// prompt-cache writes, which run last) time inside that.
pub(super) const DRAIN_DEADLINE: Duration = Duration::from_secs(10);

/// How long a cancelled generation's connection gets to end its response
/// before it is dropped: an engine step, with room to spare.
pub(super) const CANCELLED_DRAIN: Duration = Duration::from_secs(5);

/// Body bytes a request must deliver within each stall budget: 64 KiB, about
/// 2 KiB/s at the default 30 s, so only a trickle is refused.
pub(super) const PACE_BYTES: usize = 64 * 1024;

/// The deadlines [`serve`] applies; [`Limits::default`] is the server's.
#[derive(Clone, Copy, Debug)]
pub(super) struct Limits {
    pub(super) header_read: Duration,
    pub(super) drain: Duration,
    pub(super) cancelled_drain: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            header_read: HEADER_READ_TIMEOUT,
            drain: DRAIN_DEADLINE,
            cancelled_drain: CANCELLED_DRAIN,
        }
    }
}

/// Serve `app` on `listener` until `shutdown` resolves, then drain.
///
/// `cancel_remaining` runs once the drain deadline passes with connections
/// still open, and returns how many generations it cancelled.
pub(super) async fn serve(
    mut listener: TcpListener,
    app: Router,
    limits: Limits,
    shutdown: impl Future<Output = ()>,
    cancel_remaining: impl FnOnce() -> usize,
) {
    let mut builder = auto::Builder::new(TokioExecutor::new());
    builder
        .http1()
        .timer(TokioTimer::new())
        .header_read_timeout(limits.header_read);
    let builder = Arc::new(builder);
    // `draining` tells connections to finish and close; `forced` drops them.
    // Every connection task holds a `closed` receiver, so the sender's
    // `closed()` is the moment the last one has ended.
    let (draining, _) = watch::channel(false);
    let (forced, _) = watch::channel(false);
    let (closed, closed_receiver) = watch::channel(());
    let mut shutdown = pin!(shutdown);
    loop {
        let (stream, _) = tokio::select! {
            accepted = axum::serve::Listener::accept(&mut listener) => accepted,
            () = &mut shutdown => break,
        };
        let connection = Connection {
            builder: Arc::clone(&builder),
            app: app.clone(),
            header_read: limits.header_read,
            draining: draining.subscribe(),
            forced: forced.subscribe(),
            closed: closed_receiver.clone(),
        };
        tokio::spawn(connection.run(stream));
    }
    drop(listener);
    drop(closed_receiver);
    let _ = draining.send(true);
    if tokio::time::timeout(limits.drain, closed.closed())
        .await
        .is_ok()
    {
        return;
    }
    let cancelled = cancel_remaining();
    eprintln!(
        "shutdown: connections were still open {} s after shutdown began; cancelled {cancelled} \
         running generations",
        limits.drain.as_secs()
    );
    if tokio::time::timeout(limits.cancelled_drain, closed.closed())
        .await
        .is_err()
    {
        eprintln!(
            "shutdown: closing connections that did not finish {} s after their generations \
             were cancelled",
            limits.cancelled_drain.as_secs()
        );
        let _ = forced.send(true);
        closed.closed().await;
    }
}

/// One accepted TCP connection and what it needs to serve and close.
struct Connection {
    builder: Arc<auto::Builder<TokioExecutor>>,
    app: Router,
    header_read: Duration,
    draining: watch::Receiver<bool>,
    forced: watch::Receiver<bool>,
    /// Held until the connection ends; see [`serve`].
    closed: watch::Receiver<()>,
}

impl Connection {
    async fn run(mut self, stream: tokio::net::TcpStream) {
        let requested = Arc::new(AtomicBool::new(false));
        let service = {
            let (app, requested) = (self.app.clone(), Arc::clone(&requested));
            hyper::service::service_fn(move |request: hyper::Request<Incoming>| {
                requested.store(true, Ordering::Relaxed);
                let mut app = app.clone();
                async move {
                    let response = app.call(request.map(Body::new)).await;
                    Ok::<_, Infallible>(response.unwrap_or_else(|never| match never {}))
                }
            })
        };
        let mut connection = pin!(
            self.builder
                .serve_connection_with_upgrades(TokioIo::new(stream), service)
        );
        let first_request = tokio::time::sleep(self.header_read);
        let mut first_request = pin!(first_request);
        let (mut waiting_first, mut draining) = (true, false);
        loop {
            tokio::select! {
                _ = connection.as_mut() => break,
                () = &mut first_request, if waiting_first => {
                    waiting_first = false;
                    if !requested.load(Ordering::Relaxed) {
                        // Nothing was asked in time: an idle connection, or
                        // one still in hyper's protocol detection, which has
                        // no deadline of its own.
                        connection.as_mut().graceful_shutdown();
                    }
                }
                changed = self.draining.changed(), if !draining => {
                    draining = true;
                    if changed.is_ok() {
                        connection.as_mut().graceful_shutdown();
                    }
                }
                _ = self.forced.changed() => break,
            }
        }
        drop(self.closed);
    }
}

/// `body`, failing its read when it delivers fewer than [`PACE_BYTES`], and
/// does not end, within any `window` the reader spends waiting for it.
///
/// Time the reader spends not asking for the body (validating a field,
/// writing a chunk to disk) is not charged to the client.
pub(super) fn paced(body: Body, window: Duration) -> Body {
    let now = Instant::now();
    Body::new(Paced {
        inner: body,
        window,
        deadline: Box::pin(tokio::time::sleep_until(now + window)),
        received: 0,
        returned: Some(now),
    })
}

struct Paced {
    inner: Body,
    window: Duration,
    deadline: Pin<Box<Sleep>>,
    /// Bytes delivered since the window last restarted.
    received: usize,
    /// When the last frame was handed to the reader, until it asks again.
    returned: Option<Instant>,
}

impl http_body::Body for Paced {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let this = self.get_mut();
        if let Some(returned) = this.returned.take() {
            let deadline = this.deadline.deadline() + returned.elapsed();
            this.deadline.as_mut().reset(deadline);
        }
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    this.received += data.len();
                    if this.received >= PACE_BYTES {
                        this.received = 0;
                        this.deadline.as_mut().reset(Instant::now() + this.window);
                    }
                }
                this.returned = Some(Instant::now());
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(other) => {
                this.returned = Some(Instant::now());
                Poll::Ready(other)
            }
            Poll::Pending => {
                if this.deadline.as_mut().poll(cx).is_pending() {
                    return Poll::Pending;
                }
                Poll::Ready(Some(Err(axum::Error::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!(
                        "the request body stalled: fewer than {PACE_BYTES} bytes arrived in {} s",
                        this.window.as_secs()
                    ),
                )))))
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
#[path = "connection_tests.rs"]
mod tests;
