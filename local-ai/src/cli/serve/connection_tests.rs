//! The TCP listener over real sockets on loopback: header and first-request
//! deadlines, paced request bodies, and the shutdown drain deadline, each
//! with short test deadlines and bounded reads.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::{get, post};
use http_body_util::{BodyExt as _, Limited};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use super::{Limits, paced, serve};

/// Far longer than any deadline below, so a missed one fails instead of
/// passing slowly.
const PATIENCE: Duration = Duration::from_secs(5);
const SHORT: Duration = Duration::from_millis(300);

struct Server {
    address: std::net::SocketAddr,
    shutdown: oneshot::Sender<()>,
    served: tokio::task::JoinHandle<()>,
    cancelled: Arc<AtomicUsize>,
}

/// Serve `app` with `limits`; the drain-deadline callback drops `on_cancel`.
async fn start(app: Router, limits: Limits, on_cancel: Option<mpsc::Sender<()>>) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let (shutdown, signal) = oneshot::channel::<()>();
    let cancelled = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&cancelled);
    let served = tokio::spawn(serve(
        listener,
        app,
        limits,
        async move {
            let _ = signal.await;
        },
        move || {
            drop(on_cancel);
            counter.fetch_add(1, Ordering::SeqCst);
            1
        },
    ));
    Server {
        address,
        shutdown,
        served,
        cancelled,
    }
}

fn limits(header_read: Duration) -> Limits {
    Limits {
        header_read,
        drain: SHORT,
        cancelled_drain: SHORT,
    }
}

/// Read until the server closes the connection; `None` if it stays open.
async fn closed_by_server(stream: &mut TcpStream) -> Option<Vec<u8>> {
    let mut seen = Vec::new();
    tokio::time::timeout(PATIENCE, async {
        let mut buffer = [0_u8; 1024];
        loop {
            match stream.read(&mut buffer).await {
                Ok(0) | Err(_) => return,
                Ok(read) => seen.extend_from_slice(&buffer[..read]),
            }
        }
    })
    .await
    .ok()
    .map(|()| seen)
}

/// Read until `needle` appears, within [`PATIENCE`].
async fn read_until(stream: &mut TcpStream, needle: &[u8]) -> Vec<u8> {
    let mut seen = Vec::new();
    tokio::time::timeout(PATIENCE, async {
        let mut buffer = [0_u8; 1024];
        while !seen.windows(needle.len()).any(|window| window == needle) {
            let read = stream.read(&mut buffer).await.expect("read");
            assert!(read > 0, "closed before {needle:?}");
            seen.extend_from_slice(&buffer[..read]);
        }
    })
    .await
    .expect("in time");
    seen
}

fn hello() -> Router {
    Router::new().route("/", get(|| async { "hello" }))
}

#[tokio::test]
async fn a_request_whose_headers_stall_is_closed_after_the_header_timeout() {
    let server = start(hello(), limits(SHORT), None).await;
    let mut stalled = TcpStream::connect(server.address).await.expect("connect");
    stalled
        .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n")
        .await
        .expect("write");
    assert!(
        closed_by_server(&mut stalled).await.is_some(),
        "a connection with half its headers stayed open"
    );
    // A connection that sends nothing at all is still in hyper's protocol
    // detection, which has no timer of its own.
    let mut silent = TcpStream::connect(server.address).await.expect("connect");
    assert!(
        closed_by_server(&mut silent).await.is_some(),
        "a silent connection stayed open"
    );
    // A whole request is served as before, and the idle keep-alive
    // connection after it is closed on the same deadline.
    let mut whole = TcpStream::connect(server.address).await.expect("connect");
    whole
        .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .expect("write");
    let answer = closed_by_server(&mut whole).await.expect("idle closed");
    let answer = String::from_utf8_lossy(&answer);
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    assert!(answer.ends_with("hello"), "{answer}");
    let _ = server.shutdown.send(());
    tokio::time::timeout(PATIENCE, server.served)
        .await
        .expect("shut down")
        .expect("served");
    assert_eq!(
        server.cancelled.load(Ordering::SeqCst),
        0,
        "nothing to cancel"
    );
}

/// Collect a paced body as the routes do, answering its length or why not.
async fn read_paced(request: Request) -> Response {
    let body = paced(request.into_body(), SHORT);
    match Limited::new(body, 1 << 20).collect().await {
        Ok(collected) => Response::new(Body::from(collected.to_bytes().len().to_string())),
        Err(error) => {
            let mut response = Response::new(Body::from(error.to_string()));
            *response.status_mut() = StatusCode::BAD_REQUEST;
            response
        }
    }
}

#[tokio::test]
async fn a_trickling_body_fails_its_read_and_a_moving_one_does_not() {
    let app = Router::new().route("/", post(read_paced));
    let server = start(app, limits(Duration::from_secs(30)), None).await;
    let mut trickle = TcpStream::connect(server.address).await.expect("connect");
    trickle
        .write_all(
            b"POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1000000\r\n\r\n0123456789",
        )
        .await
        .expect("write");
    let answer = read_until(&mut trickle, b"stalled").await;
    let answer = String::from_utf8_lossy(&answer);
    assert!(answer.starts_with("HTTP/1.1 400"), "{answer}");

    // 256 KiB in 64 KiB pieces, each well inside the window: never refused,
    // however long the whole body takes.
    let mut moving = TcpStream::connect(server.address).await.expect("connect");
    let size = 4 * super::PACE_BYTES;
    moving
        .write_all(
            format!("POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: {size}\r\n\r\n")
                .as_bytes(),
        )
        .await
        .expect("write");
    for _ in 0..4 {
        tokio::time::sleep(SHORT / 2).await;
        moving
            .write_all(&vec![b'x'; super::PACE_BYTES])
            .await
            .expect("write");
    }
    let answer = read_until(&mut moving, size.to_string().as_bytes()).await;
    let answer = String::from_utf8_lossy(&answer);
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    let _ = server.shutdown.send(());
}

#[tokio::test]
async fn shutdown_cancels_what_is_still_running_at_the_drain_deadline() {
    // A response that runs until it is cancelled, as a long generation does.
    let (cancel, cancelled) = mpsc::channel::<()>(1);
    let running = Arc::new(std::sync::Mutex::new(Some(cancelled)));
    let app = Router::new().route(
        "/generate",
        get(move || {
            let running = Arc::clone(&running);
            async move {
                let mut cancelled = running.lock().expect("lock").take().expect("one request");
                let (frames, receiver) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(1);
                tokio::spawn(async move {
                    let _ = frames.send(Ok(bytes::Bytes::from_static(b"token "))).await;
                    // The engine stops at its next step, and the body ends.
                    let _ = cancelled.recv().await;
                    let _ = frames
                        .send(Ok(bytes::Bytes::from_static(b"cancelled")))
                        .await;
                });
                Body::from_stream(ReceiverStream::new(receiver))
            }
        }),
    );
    let server = start(app, limits(Duration::from_secs(30)), Some(cancel)).await;
    let mut client = TcpStream::connect(server.address).await.expect("connect");
    client
        .write_all(b"GET /generate HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .expect("write");
    read_until(&mut client, b"token ").await;
    let begun = std::time::Instant::now();
    let _ = server.shutdown.send(());
    tokio::time::timeout(PATIENCE, server.served)
        .await
        .expect("shutdown is bounded by the drain deadline")
        .expect("served");
    assert!(begun.elapsed() >= SHORT, "waited for the drain deadline");
    assert_eq!(server.cancelled.load(Ordering::SeqCst), 1);
    // The cancelled response ended properly: its last chunk and the chunked
    // terminator, then the connection closed.
    let rest = closed_by_server(&mut client).await.expect("closed");
    let rest = String::from_utf8_lossy(&rest);
    assert!(rest.contains("cancelled"), "{rest:?}");
    assert!(rest.ends_with("0\r\n\r\n"), "{rest:?}");
}
