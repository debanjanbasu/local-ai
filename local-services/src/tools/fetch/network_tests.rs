//! Web fetch against local HTTP/1.1, HTTP/2 and HTTP/3 servers.
//!
//! The servers use a test CA (`testdata/`, valid until 2126) and listen on
//! loopback. The client under test is built with that CA instead of the
//! platform verifier, with `fetch.test` resolving to 127.0.0.1 and with
//! loopback addresses let through the address check: both test-only
//! (`#[cfg(test)]`) switches, absent from the library itself.

use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use hyper::body::Frame;
use rustls::pki_types::pem::PemObject as _;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

use super::body::Coding;
use super::client::{Client, Key, Protocol, Timeouts};
use super::{Output, Target, ToolError};

const CA: &[u8] = include_bytes!("testdata/ca.pem");
const CERT: &[u8] = include_bytes!("testdata/leaf.pem");
const KEY: &[u8] = include_bytes!("testdata/leaf.key");
const HELLO: &str = "hello from the test server";
const LOCALHOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

fn provider() -> Arc<rustls::crypto::CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

fn sample() -> Vec<u8> {
    (0..8000u32)
        .flat_map(|n| format!("sample line {n}\n").into_bytes())
        .collect()
}

fn bomb(coding: Coding) -> Vec<u8> {
    super::body::tests::encode(coding, &vec![0; 16 << 20])
}

const PAGE: &str = r#"<html><head><title>T</title><script>SCRIPT</script></head>
<body><nav><a href="/">NAV</a></nav><main><h1>Fixture page</h1>
<p>See <a href="docs/next.html">the next page</a>.</p></main></body></html>"#;

/// One canned response.
struct Reply {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: Vec<u8>,
    /// Send a chunk every 20 ms until the client goes away.
    endless: bool,
}

fn reply(path: &str, alt_svc: Option<u16>) -> Reply {
    let ok = |content_type: &str, body: Vec<u8>| Reply {
        status: 200,
        headers: vec![("content-type", content_type.to_owned())],
        body,
        endless: false,
    };
    let coding = |name: &str| match name {
        "gzip" => Coding::Gzip,
        "deflate" => Coding::Deflate,
        "br" => Coding::Brotli,
        "zstd" => Coding::Zstd,
        _ => Coding::Identity,
    };
    let encoded = path
        .strip_prefix("/enc/")
        .map(|name| (name, super::body::tests::encode(coding(name), &sample())))
        .or_else(|| {
            path.strip_prefix("/bomb/")
                .map(|name| (name, bomb(coding(name))))
        });
    let mut reply = if let Some((name, body)) = encoded {
        let mut reply = ok("text/plain", body);
        reply.headers.push(("content-encoding", name.to_owned()));
        reply
    } else {
        match path {
            "/text" => ok("text/plain; charset=utf-8", HELLO.into()),
            "/page" => ok("text/html", PAGE.into()),
            "/big-page" => ok(
                "text/html",
                format!("<p>{}</p>", "word ".repeat(4000)).into(),
            ),
            "/latin1" => ok(
                "text/html; charset=windows-1252",
                b"<p>caf\xe9</p>".to_vec(),
            ),
            "/image" => ok("image/png", vec![0x89, b'P', b'N', b'G']),
            "/skippable" => {
                // zstd skippable frames: megabytes on the wire, nothing decoded.
                let mut frames = Vec::new();
                for _ in 0..64 {
                    frames.extend_from_slice(&0x184D_2A50u32.to_le_bytes());
                    frames.extend_from_slice(&(32_768u32).to_le_bytes());
                    frames.resize(frames.len() + 32_768, 0);
                }
                let mut reply = ok("text/plain", frames);
                reply.headers.push(("content-encoding", "zstd".into()));
                reply
            }
            "/slow" => Reply {
                endless: true,
                ..ok("text/plain", Vec::new())
            },
            "/redirect" => Reply {
                status: 302,
                headers: vec![("location", "/image".into())],
                body: Vec::new(),
                endless: false,
            },
            _ => Reply {
                status: 404,
                ..ok("text/plain", b"missing".to_vec())
            },
        }
    };
    if let Some(port) = alt_svc {
        reply
            .headers
            .push(("alt-svc", format!(r#"h3=":{port}"; ma=3600"#)));
    }
    reply
}

/// A body fed through a channel, so the server notices when the client
/// stops reading.
struct Chunks(tokio::sync::mpsc::Receiver<Bytes>);

impl hyper::body::Body for Chunks {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        self.0
            .poll_recv(cx)
            .map(|chunk| chunk.map(|data| Ok(Frame::data(data))))
    }
}

/// What a test server reports.
struct Server {
    port: u16,
    /// Connections accepted (TCP connections, or QUIC ones).
    connections: Arc<AtomicUsize>,
    /// One message each time an endless body's client went away.
    gone: tokio::sync::mpsc::UnboundedReceiver<()>,
}

fn server_tls(alpn: &[&[u8]]) -> rustls::ServerConfig {
    let certs = CertificateDer::pem_slice_iter(CERT)
        .collect::<Result<Vec<_>, _>>()
        .expect("certificate");
    let key = PrivateKeyDer::from_pem_slice(KEY).expect("key");
    let mut config = rustls::ServerConfig::builder_with_provider(provider())
        .with_safe_default_protocol_versions()
        .expect("versions")
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .expect("server TLS");
    config.alpn_protocols = alpn.iter().map(|protocol| protocol.to_vec()).collect();
    config
}

/// Feed `reply`'s body into `sender`, then (for an endless body) report
/// when the receiver is gone.
async fn feed(
    reply: &Reply,
    sender: tokio::sync::mpsc::Sender<Bytes>,
    gone: tokio::sync::mpsc::UnboundedSender<()>,
) {
    if reply.endless {
        loop {
            if sender.send(Bytes::from_static(b"tick\n")).await.is_err() {
                let _ = gone.send(());
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
    for chunk in reply.body.chunks(16 * 1024) {
        if sender.send(Bytes::copy_from_slice(chunk)).await.is_err() {
            return;
        }
    }
}

/// An HTTP/1.1 and HTTP/2 server over TCP (TLS with `alpn`, or plain).
async fn tcp_server(alpn: Option<&[&[u8]]>, alt_svc: Option<u16>) -> Server {
    let listener = tokio::net::TcpListener::bind((LOCALHOST, 0))
        .await
        .expect("bind");
    let port = listener.local_addr().expect("address").port();
    let connections = Arc::new(AtomicUsize::new(0));
    let (gone_sender, gone) = tokio::sync::mpsc::unbounded_channel();
    let acceptor = alpn.map(|alpn| tokio_rustls::TlsAcceptor::from(Arc::new(server_tls(alpn))));
    let counter = Arc::clone(&connections);
    tokio::spawn(async move {
        loop {
            let Ok((tcp, _)) = listener.accept().await else {
                return;
            };
            counter.fetch_add(1, Ordering::SeqCst);
            let acceptor = acceptor.clone();
            let gone = gone_sender.clone();
            tokio::spawn(async move {
                let service = hyper::service::service_fn(
                    move |request: http::Request<hyper::body::Incoming>| {
                        let reply = reply(request.uri().path(), alt_svc);
                        let gone = gone.clone();
                        async move {
                            let (sender, receiver) = tokio::sync::mpsc::channel(4);
                            let mut response = http::Response::builder().status(reply.status);
                            for (name, value) in &reply.headers {
                                response = response.header(*name, value);
                            }
                            tokio::spawn(async move {
                                feed(&reply, sender, gone).await;
                            });
                            Ok::<_, Infallible>(response.body(Chunks(receiver)).expect("response"))
                        }
                    },
                );
                let builder = hyper_util::server::conn::auto::Builder::new(
                    hyper_util::rt::TokioExecutor::new(),
                );
                match acceptor {
                    Some(acceptor) => {
                        let Ok(tls) = acceptor.accept(tcp).await else {
                            return;
                        };
                        let _ = builder
                            .serve_connection(hyper_util::rt::TokioIo::new(tls), service)
                            .await;
                    }
                    None => {
                        let _ = builder
                            .serve_connection(hyper_util::rt::TokioIo::new(tcp), service)
                            .await;
                    }
                }
            });
        }
    });
    Server {
        port,
        connections,
        gone,
    }
}

/// An HTTP/3 server on a loopback UDP port.
fn h3_server() -> Server {
    let mut tls = server_tls(&[b"h3"]);
    tls.max_early_data_size = u32::MAX;
    let quic = quinn::crypto::rustls::QuicServerConfig::try_from(tls).expect("QUIC TLS");
    let endpoint = quinn::Endpoint::server(
        quinn::ServerConfig::with_crypto(Arc::new(quic)),
        SocketAddr::new(LOCALHOST, 0),
    )
    .expect("endpoint");
    let port = endpoint.local_addr().expect("address").port();
    let connections = Arc::new(AtomicUsize::new(0));
    let (gone_sender, gone) = tokio::sync::mpsc::unbounded_channel();
    let counter = Arc::clone(&connections);
    tokio::spawn(async move {
        while let Some(incoming) = endpoint.accept().await {
            counter.fetch_add(1, Ordering::SeqCst);
            let gone = gone_sender.clone();
            tokio::spawn(async move {
                let Ok(connection) = incoming.await else {
                    return;
                };
                let Ok(mut connection) = h3::server::builder()
                    .build::<_, Bytes>(h3_quinn::Connection::new(connection))
                    .await
                else {
                    return;
                };
                while let Ok(Some(resolver)) = connection.accept().await {
                    let gone = gone.clone();
                    tokio::spawn(async move {
                        let Ok((request, mut stream)) = resolver.resolve_request().await else {
                            return;
                        };
                        let reply = reply(request.uri().path(), None);
                        let mut response = http::Response::builder().status(reply.status);
                        for (name, value) in &reply.headers {
                            response = response.header(*name, value);
                        }
                        if stream
                            .send_response(response.body(()).expect("response"))
                            .await
                            .is_err()
                        {
                            return;
                        }
                        let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
                        let endless = reply.endless;
                        tokio::spawn(async move {
                            feed(&reply, sender, tokio::sync::mpsc::unbounded_channel().0).await;
                        });
                        while let Some(chunk) = receiver.recv().await {
                            if stream.send_data(chunk).await.is_err() {
                                if endless {
                                    let _ = gone.send(());
                                }
                                return;
                            }
                        }
                        let _ = stream.finish().await;
                    });
                }
            });
        }
    });
    Server {
        port,
        connections,
        gone,
    }
}

fn client_with(timeouts: Timeouts, loopback: bool) -> Arc<Client> {
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(CertificateDer::from_pem_slice(CA).expect("CA"))
        .expect("root");
    let verifier =
        rustls::client::WebPkiServerVerifier::builder_with_provider(Arc::new(roots), provider())
            .build()
            .expect("verifier");
    let mut client = Client::new(None, provider(), verifier, timeouts).expect("client");
    if loopback {
        client = client.allow_loopback();
    }
    client
        .resolver
        .overrides
        .insert("fetch.test".into(), vec![LOCALHOST]);
    Arc::new(client)
}

fn client() -> Arc<Client> {
    client_with(Timeouts::default(), true)
}

fn target(scheme: &str, port: u16, path: &str) -> Target {
    super::parse(&format!("{scheme}://fetch.test:{port}{path}")).expect("url")
}

/// `GET` and read the whole body; the protocol and the body.
async fn get(client: &Arc<Client>, target: &Target) -> (Protocol, Vec<u8>) {
    let mut response = client.get(target, "*/*").await.expect("response");
    let protocol = response.protocol;
    let read = super::body::read(&mut response, usize::MAX / 2)
        .await
        .expect("body");
    (protocol, read.bytes)
}

fn text(output: Output) -> (String, bool) {
    match output {
        Output::Text { text, truncated } => (text, truncated),
        Output::Shell(_) => panic!("not text"),
    }
}

#[tokio::test]
async fn h1_and_h2_follow_alpn_and_reuse_their_connection() {
    for (alpn, scheme, expected) in [
        (
            Some(&[b"http/1.1".as_slice()][..]),
            "https",
            Protocol::Http1,
        ),
        (
            Some(&[b"h2".as_slice(), b"http/1.1"][..]),
            "https",
            Protocol::Http2,
        ),
        (None, "http", Protocol::Http1),
    ] {
        let server = tcp_server(alpn, None).await;
        let client = client();
        let target = target(scheme, server.port, "/text");
        for _ in 0..3 {
            assert_eq!(get(&client, &target).await, (expected, HELLO.into()));
        }
        assert_eq!(
            server.connections.load(Ordering::SeqCst),
            1,
            "{expected:?} {scheme}"
        );
        let (body, truncated) = text(
            super::fetch_target(&client, &target, 1024)
                .await
                .expect("fetch"),
        );
        assert!(!truncated);
        assert!(
            body.starts_with("HTTP 200\ncontent-type: text/plain; charset=utf-8\n"),
            "{body}"
        );
        assert!(body.ends_with(HELLO), "{body}");
    }
}

#[tokio::test]
async fn every_content_coding_decodes_over_every_protocol() {
    let h3 = h3_server();
    let h2 = tcp_server(Some(&[b"h2", b"http/1.1"]), Some(h3.port)).await;
    let h1 = tcp_server(Some(&[b"http/1.1"]), None).await;
    let client = client();
    // Learn HTTP/3 from Alt-Svc, then drop the TCP connection.
    let _ = get(&client, &target("https", h2.port, "/text")).await;
    for coding in ["identity", "gzip", "deflate", "br", "zstd"] {
        let path = format!("/enc/{coding}");
        for (port, expected) in [(h1.port, Protocol::Http1), (h2.port, Protocol::Http3)] {
            client.clear_pool();
            let (protocol, body) = get(&client, &target("https", port, &path)).await;
            assert_eq!(
                (protocol, body.len()),
                (expected, sample().len()),
                "{coding}"
            );
            assert_eq!(body, sample(), "{coding} over {expected:?}");
        }
    }
    let (body, truncated) = text(
        super::fetch_target(&client, &target("https", h1.port, "/enc/br"), 1 << 20)
            .await
            .expect("fetch"),
    );
    assert!(!truncated && body.ends_with("sample line 7999\n"));
}

#[tokio::test]
async fn bombs_and_endless_encodings_stop_at_the_limits() {
    let server = tcp_server(Some(&[b"h2"]), None).await;
    let client = client();
    for coding in ["gzip", "zstd"] {
        let target = target("https", server.port, &format!("/bomb/{coding}"));
        let started = Instant::now();
        let (body, truncated) = text(
            super::fetch_target(&client, &target, 64 * 1024)
                .await
                .expect("fetch"),
        );
        assert!(truncated, "{coding}");
        assert!(body.len() < 65 * 1024, "{coding}: {}", body.len());
        assert!(body.ends_with("[truncated: the body was cut at 65536 bytes]"));
        let error = super::download_from(&client, &url(&target), 1 << 20, |_| true)
            .await
            .expect_err("too large");
        assert!(matches!(error, ToolError::Invalid(_)), "{error}");
        assert!(started.elapsed() < Duration::from_secs(5), "{coding}");
    }
    // Megabytes that decode to nothing stop at the encoded limit.
    let (body, truncated) = text(
        super::fetch_target(&client, &target("https", server.port, "/skippable"), 4096)
            .await
            .expect("fetch"),
    );
    assert!(
        truncated && body.ends_with("[truncated: the body was cut at 4096 bytes]"),
        "{body}"
    );
}

fn url(target: &Target) -> String {
    super::url_of(target)
}

#[tokio::test]
async fn alt_svc_moves_new_connections_to_h3_and_pools_them() {
    let h3 = h3_server();
    let tcp = tcp_server(Some(&[b"h2", b"http/1.1"]), Some(h3.port)).await;
    let client = client();
    let target = target("https", tcp.port, "/text");
    let key = Key::new(true, "fetch.test", tcp.port);
    assert_eq!(client.h3_port(&key), None);
    assert_eq!(get(&client, &target).await.0, Protocol::Http2);
    assert_eq!(client.h3_port(&key), Some(h3.port));
    // The live HTTP/2 connection is still preferred; a new one is QUIC.
    assert_eq!(get(&client, &target).await.0, Protocol::Http2);
    client.clear_pool();
    for _ in 0..3 {
        assert_eq!(get(&client, &target).await, (Protocol::Http3, HELLO.into()));
    }
    assert_eq!(h3.connections.load(Ordering::SeqCst), 1);
    // A new QUIC connection resumes the TLS session (0-RTT when offered).
    client.clear_pool();
    assert_eq!(get(&client, &target).await, (Protocol::Http3, HELLO.into()));
    assert_eq!(tcp.connections.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn blocked_quic_falls_back_to_tcp_and_marks_the_origin_broken() {
    // A UDP port that swallows every packet, like a firewall dropping QUIC.
    let black_hole = std::net::UdpSocket::bind((LOCALHOST, 0)).expect("udp");
    let blocked = black_hole.local_addr().expect("address").port();
    let tcp = tcp_server(Some(&[b"h2"]), Some(blocked)).await;
    let timeouts = Timeouts {
        head_start: Duration::from_millis(100),
        quic_handshake: Duration::from_millis(500),
    };
    let client = client_with(timeouts, true);
    let target = target("https", tcp.port, "/text");
    let key = Key::new(true, "fetch.test", tcp.port);
    assert_eq!(get(&client, &target).await.0, Protocol::Http2);
    assert_eq!(client.h3_port(&key), Some(blocked));
    client.clear_pool();
    let started = Instant::now();
    assert_eq!(get(&client, &target).await, (Protocol::Http2, HELLO.into()));
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_millis(100) && elapsed < Duration::from_millis(450),
        "{elapsed:?}"
    );
    // The QUIC attempt runs out in the background and marks the origin.
    tokio::time::sleep(Duration::from_millis(700)).await;
    assert_eq!(client.h3_port(&key), None);
    client.clear_pool();
    let started = Instant::now();
    assert_eq!(get(&client, &target).await.0, Protocol::Http2);
    assert!(started.elapsed() < Duration::from_millis(100));
}

#[tokio::test]
async fn cancelling_a_fetch_abandons_its_stream_promptly() {
    let mut h3 = h3_server();
    let mut h2 = tcp_server(Some(&[b"h2"]), Some(h3.port)).await;
    let mut h1 = tcp_server(Some(&[b"http/1.1"]), None).await;
    let client = client();
    let _ = get(&client, &target("https", h2.port, "/text")).await;
    for (port, server_gone, label) in [
        (h1.port, &mut h1.gone, "h1"),
        (h2.port, &mut h2.gone, "h2"),
        (h2.port, &mut h3.gone, "h3"),
    ] {
        if label == "h3" {
            client.clear_pool();
        }
        let slow = target("https", port, "/slow");
        let call = super::fetch_target(&client, &slow, 1 << 20);
        assert!(
            tokio::time::timeout(Duration::from_millis(300), call)
                .await
                .is_err(),
            "{label} should still be streaming"
        );
        tokio::time::timeout(Duration::from_secs(2), server_gone.recv())
            .await
            .unwrap_or_else(|_| panic!("{label}: the server kept sending"))
            .expect("server");
    }
    // The cut HTTP/1.1 connection is not reused; the multiplexed ones are.
    assert_eq!(
        get(&client, &target("https", h1.port, "/text")).await.1,
        HELLO.as_bytes()
    );
    assert_eq!(h1.connections.load(Ordering::SeqCst), 2);
    assert_eq!(
        get(&client, &target("https", h2.port, "/text")).await.0,
        Protocol::Http3
    );
    assert_eq!(h3.connections.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn redirects_are_reported_by_fetch_and_followed_by_download() {
    let server = tcp_server(Some(&[b"h2"]), None).await;
    let client = client();
    let redirect = target("https", server.port, "/redirect");
    let (body, truncated) = text(
        super::fetch_target(&client, &redirect, 1024)
            .await
            .expect("fetch"),
    );
    assert!(!truncated);
    assert_eq!(
        body,
        "HTTP 302\nlocation: /image\n[redirect not followed; fetch the location to continue]\n"
    );
    let download = super::download_from(&client, &url(&redirect), 1024, |content_type| {
        content_type == Some("image/png")
    })
    .await
    .expect("download");
    assert_eq!(download.body, [0x89, b'P', b'N', b'G']);
    let refused = super::download_from(&client, &url(&redirect), 1024, |_| false).await;
    assert!(matches!(refused, Err(ToolError::Denied(_))), "{refused:?}");
    let too_small = super::download_from(&client, &url(&redirect), 3, |_| true).await;
    assert!(
        matches!(too_small, Err(ToolError::Invalid(_))),
        "{too_small:?}"
    );
}

#[tokio::test]
async fn html_arrives_as_markdown_within_the_limits() {
    let server = tcp_server(Some(&[b"h2"]), None).await;
    let client = client();
    let port = server.port;
    let (body, truncated) = text(
        super::fetch_target(&client, &target("https", port, "/page"), 4096)
            .await
            .expect("fetch"),
    );
    assert!(!truncated);
    assert_eq!(
        body,
        format!(
            "HTTP 200\ncontent-type: text/html\n[converted from HTML to Markdown]\n\n# Fixture page\n\n\
             See [the next page](https://fetch.test:{port}/docs/next.html)."
        )
    );
    let (body, _) = text(
        super::fetch_target(&client, &target("https", port, "/latin1"), 4096)
            .await
            .expect("fetch"),
    );
    assert!(body.ends_with("\n\ncafé"), "{body}");
    // 20 KB of HTML: past four times a 1 KiB limit before conversion, and
    // its Markdown past 1 KiB after.
    let (body, truncated) = text(
        super::fetch_target(&client, &target("https", port, "/big-page"), 1024)
            .await
            .expect("fetch"),
    );
    assert!(truncated);
    assert!(
        body.ends_with(
            "\n[truncated: the page was cut at 4096 bytes of HTML before conversion]\
             \n[truncated: the Markdown was cut at 1024 bytes]"
        ),
        "{body}"
    );
}

#[tokio::test]
async fn non_public_addresses_are_refused_without_the_test_switch() {
    let server = tcp_server(Some(&[b"h2"]), None).await;
    let client = client_with(Timeouts::default(), false);
    for url in [
        format!("https://fetch.test:{}/text", server.port),
        format!("https://127.0.0.1:{}/text", server.port),
    ] {
        let error = super::download_from(&client, &url, 1024, |_| true)
            .await
            .expect_err("refused");
        assert!(
            matches!(&error, ToolError::Denied(message) if message.contains("non-public")),
            "{error}"
        );
    }
    assert_eq!(server.connections.load(Ordering::SeqCst), 0);
}
