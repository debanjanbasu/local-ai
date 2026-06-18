//! HTTP/3 (QUIC) front-end for the continuous-batching server.
//!
//! This runs **alongside** the axum HTTP/1.1+HTTP/2 TCP listener (see
//! [`super::serve`]) on the same numeric port, but over UDP/QUIC. It reuses the
//! exact same request parsing and GPU-worker submission path via the shared
//! [`AppState`] — only the transport differs. Cleartext HTTP stays on TCP;
//! HTTP/3 requires TLS, so we terminate QUIC with an ephemeral self-signed
//! certificate (ALPN `h3`).
//!
//! Architecture mirrors the axum side: async QUIC connections fan requests into
//! the single GPU worker over the channel in [`AppState`]; generated
//! [`ServeEvent`]s stream back per request. The non-`Send` pipeline never
//! touches an async task.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use std::io::Write;

use bytes::{Buf, Bytes};
use http::{Response, StatusCode};

use crate::pipeline::ServeEvent;

use super::serve::{
    AppState, SubmitError, authorized, error_body, log_completion, ok_body, parse_request,
    sse_token_chunk, submit,
};

/// The QUIC body buffer type used for all HTTP/3 response data frames.
type Body = Bytes;

/// Only compress non-stream bodies at least this large; below it the framing
/// overhead outweighs any gain (mirrors tower-http's default 32-byte floor).
const MIN_COMPRESS_BYTES: usize = 32;

/// Maximum accepted request body, matching axum's `DefaultBodyLimit` (2 MiB).
/// Caps memory per HTTP/3 request so a client cannot stream an unbounded body.
const MAX_REQUEST_BODY: usize = 2 * 1024 * 1024;

/// Negotiated HTTP response content-encoding for a non-stream body.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Encoding {
    Identity,
    Zstd,
    Brotli,
}

impl Encoding {
    /// HTTP `Content-Encoding` token, or `None` for identity.
    const fn header(self) -> Option<&'static str> {
        match self {
            Self::Identity => None,
            Self::Zstd => Some("zstd"),
            Self::Brotli => Some("br"),
        }
    }
}

/// Pick the best encoding the client advertises in `Accept-Encoding`. Brotli is
/// preferred (best text ratio), then zstd; unknown/absent → identity. (gzip and
/// deflate are negotiated on the TCP/axum side; modern HTTP/3 clients support
/// br/zstd.)
fn negotiate_encoding(req: &http::Request<()>) -> Encoding {
    let accept = req
        .headers()
        .get(http::header::ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let advertises = |name: &str| {
        accept.split(',').any(|part| {
            let token = part.split(';').next().unwrap_or("").trim();
            token.eq_ignore_ascii_case(name)
        })
    };
    if advertises("br") {
        Encoding::Brotli
    } else if advertises("zstd") {
        Encoding::Zstd
    } else {
        Encoding::Identity
    }
}

/// Compress `data` with the negotiated encoding at a high (HTTP-practical)
/// quality. zstd level 19 and brotli quality 11 give near-archive ratios while
/// keeping per-response latency bounded for live traffic. Returns the original
/// bytes on the (very unlikely) compressor error so the response still flows.
fn encode_body(data: &[u8], enc: Encoding) -> Bytes {
    match enc {
        Encoding::Identity => Bytes::copy_from_slice(data),
        Encoding::Zstd => zstd::bulk::compress(data, 19)
            .map_or_else(|_| Bytes::copy_from_slice(data), Bytes::from),
        Encoding::Brotli => {
            let mut out = Vec::with_capacity(data.len() / 2 + 64);
            // params: buffer 4 KiB, quality 11 (max), lgwin 24 (max).
            let mut writer = brotli::CompressorWriter::new(&mut out, 4096, 11, 24);
            if writer.write_all(data).is_ok() && writer.flush().is_ok() {
                drop(writer);
                Bytes::from(out)
            } else {
                Bytes::copy_from_slice(data)
            }
        }
    }
}

/// Build the QUIC endpoint, bind it on `udp://host:port`, and spawn the
/// accept loop on the current tokio runtime. Returns once the socket is bound
/// (so bind errors surface synchronously); the accept loop runs in the
/// background sharing `state` with the axum front-end.
pub(super) fn spawn(state: AppState, host: IpAddr, port: u16) -> Result<(), String> {
    // Install a process-default rustls crypto provider (ring) if none is set.
    // Ignore the error: it only means a provider was already installed.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let server_config = build_server_config()?;
    let addr = SocketAddr::new(host, port);
    let endpoint = quinn::Endpoint::server(server_config, addr)
        .map_err(|e| format!("failed to bind QUIC/UDP {addr}: {e}"))?;

    tokio::spawn(accept_loop(endpoint, state));
    Ok(())
}

/// Build a `quinn::ServerConfig` from an ephemeral self-signed certificate with
/// ALPN restricted to `h3`.
fn build_server_config() -> Result<quinn::ServerConfig, String> {
    let (cert, key) = self_signed_cert()?;
    let mut tls = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert], key)
        .map_err(|e| format!("invalid TLS certificate: {e}"))?;
    tls.alpn_protocols = vec![b"h3".to_vec()];
    let quic_tls = quinn::crypto::rustls::QuicServerConfig::try_from(tls)
        .map_err(|e| format!("failed to build QUIC TLS config: {e}"))?;
    Ok(quinn::ServerConfig::with_crypto(Arc::new(quic_tls)))
}

/// Generate an ephemeral self-signed certificate for `localhost` / loopback.
fn self_signed_cert() -> Result<
    (
        rustls::pki_types::CertificateDer<'static>,
        rustls::pki_types::PrivateKeyDer<'static>,
    ),
    String,
> {
    let names = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "::1".to_string(),
    ];
    let certified = rcgen::generate_simple_self_signed(names)
        .map_err(|e| format!("failed to generate self-signed cert: {e}"))?;
    let cert = certified.cert.der().clone();
    let key = rustls::pki_types::PrivatePkcs8KeyDer::from(certified.signing_key.serialize_der());
    Ok((cert, rustls::pki_types::PrivateKeyDer::Pkcs8(key)))
}

/// Accept QUIC connections forever, spawning a task per connection.
async fn accept_loop(endpoint: quinn::Endpoint, state: AppState) {
    while let Some(incoming) = endpoint.accept().await {
        let state = state.clone();
        tokio::spawn(async move {
            // Connection-level errors are dominated by routine client
            // disconnects (closed control stream); request-level errors are
            // logged inside `handle_connection`.
            let _ = handle_connection(incoming, state).await;
        });
    }
}

/// Drive one QUIC connection: complete the handshake, wrap it in an HTTP/3
/// connection, and serve each request stream concurrently.
async fn handle_connection(incoming: quinn::Incoming, state: AppState) -> Result<(), String> {
    let conn = incoming
        .await
        .map_err(|e| format!("QUIC handshake failed: {e}"))?;
    let mut h3conn = h3::server::Connection::<_, Body>::new(h3_quinn::Connection::new(conn))
        .await
        .map_err(|e| format!("h3 connection setup failed: {e}"))?;

    loop {
        match h3conn.accept().await {
            Ok(Some(resolver)) => {
                let state = state.clone();
                tokio::spawn(async move {
                    match resolver.resolve_request().await {
                        Ok((req, stream)) => {
                            if let Err(e) = handle_request(state, req, stream).await {
                                eprintln!("h3 request error: {e}");
                            }
                        }
                        Err(e) => eprintln!("h3 resolve error: {e}"),
                    }
                });
            }
            // Graceful end of connection.
            Ok(None) => break,
            Err(e) => return Err(format!("h3 accept error: {e}")),
        }
    }
    Ok(())
}

/// Route + serve a single HTTP/3 request, reusing the shared worker path.
#[allow(clippy::too_many_lines)]
async fn handle_request<S>(
    state: AppState,
    req: http::Request<()>,
    mut stream: h3::server::RequestStream<S, Body>,
) -> Result<(), String>
where
    S: h3::quic::BidiStream<Body>,
{
    let path = req.uri().path().to_string();
    let enc = negotiate_encoding(&req);

    if path == "/health" {
        return send_full(
            &mut stream,
            StatusCode::OK,
            "text/plain",
            Bytes::from_static(b"ok"),
            Encoding::Identity,
        )
        .await;
    }

    if path != "/v1/chat/completions" && path != "/v1/completions" {
        let body = error_body("not found");
        return send_full(
            &mut stream,
            StatusCode::NOT_FOUND,
            "application/json",
            Bytes::from(body),
            enc,
        )
        .await;
    }

    // Auth (only enforced when a key is configured).
    let bearer = req
        .headers()
        .get(http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    if !authorized(state.api_key.as_ref(), bearer) {
        eprintln!("[serve] 401 unauthorized (h3)");
        return send_full(
            &mut stream,
            StatusCode::UNAUTHORIZED,
            "application/json",
            Bytes::from(error_body("missing or invalid API key")),
            enc,
        )
        .await;
    }

    // Read the full request body, capped at `MAX_REQUEST_BODY`.
    let mut body = Vec::new();
    while let Some(mut chunk) = stream
        .recv_data()
        .await
        .map_err(|e| format!("h3 body read error: {e}"))?
    {
        let bytes = chunk.copy_to_bytes(chunk.remaining());
        body.extend_from_slice(&bytes);
        if body.len() > MAX_REQUEST_BODY {
            return send_full(
                &mut stream,
                StatusCode::PAYLOAD_TOO_LARGE,
                "application/json",
                Bytes::from(error_body("request body too large")),
                enc,
            )
            .await;
        }
    }

    let parsed = parse_request(&String::from_utf8_lossy(&body));
    if let Some(err) = parsed.media_error.clone() {
        eprintln!("[serve] 400 bad media (h3): {err}");
        return send_full(
            &mut stream,
            StatusCode::BAD_REQUEST,
            "application/json",
            Bytes::from(error_body(&err)),
            enc,
        )
        .await;
    }
    let want_stream = parsed.stream;

    let mut ev_rx = match submit(&state, parsed) {
        Ok(rx) => rx,
        Err(SubmitError::Overloaded) => {
            eprintln!("[serve] 503 overloaded (h3)");
            return send_full(
                &mut stream,
                StatusCode::SERVICE_UNAVAILABLE,
                "application/json",
                Bytes::from(error_body("server overloaded, retry later")),
                enc,
            )
            .await;
        }
        Err(SubmitError::ShuttingDown) => {
            return send_full(
                &mut stream,
                StatusCode::SERVICE_UNAVAILABLE,
                "application/json",
                Bytes::from(error_body("server is shutting down")),
                enc,
            )
            .await;
        }
    };

    let started = std::time::Instant::now();

    if want_stream {
        // SSE-framed body over HTTP/3: one `data:` line per token, `[DONE]` last.
        send_headers(&mut stream, StatusCode::OK, "text/event-stream").await?;
        while let Some(ev) = ev_rx.recv().await {
            match ev {
                ServeEvent::Token(delta) => {
                    stream
                        .send_data(Bytes::from(sse_token_chunk(&delta)))
                        .await
                        .map_err(|e| format!("h3 send error: {e}"))?;
                }
                ServeEvent::Done(resp) => {
                    log_completion("h3-stream", &resp, started);
                    stream
                        .send_data(Bytes::from_static(b"data: [DONE]\n\n"))
                        .await
                        .map_err(|e| format!("h3 send error: {e}"))?;
                    break;
                }
            }
        }
        return stream
            .finish()
            .await
            .map_err(|e| format!("h3 finish error: {e}"));
    }

    // Non-stream: await the terminal Done and reply with the full completion.
    while let Some(ev) = ev_rx.recv().await {
        if let ServeEvent::Done(resp) = ev {
            log_completion("h3-json", &resp, started);
            return send_full(
                &mut stream,
                StatusCode::OK,
                "application/json",
                Bytes::from(ok_body(&resp.text)),
                enc,
            )
            .await;
        }
    }
    send_full(
        &mut stream,
        StatusCode::INTERNAL_SERVER_ERROR,
        "application/json",
        Bytes::from(error_body(
            "request could not be served (prompt too long or worker error)",
        )),
        enc,
    )
    .await
}

/// Send response headers only (for streaming bodies).
async fn send_headers<S>(
    stream: &mut h3::server::RequestStream<S, Body>,
    status: StatusCode,
    content_type: &str,
) -> Result<(), String>
where
    S: h3::quic::BidiStream<Body>,
{
    let resp = Response::builder()
        .status(status)
        .header("content-type", content_type)
        .body(())
        .map_err(|e| format!("h3 response build error: {e}"))?;
    stream
        .send_response(resp)
        .await
        .map_err(|e| format!("h3 send_response error: {e}"))
}

/// Send a complete response: headers + a single body frame + finish. The body
/// is compressed with `enc` when large enough; identity otherwise.
async fn send_full<S>(
    stream: &mut h3::server::RequestStream<S, Body>,
    status: StatusCode,
    content_type: &str,
    body: Bytes,
    enc: Encoding,
) -> Result<(), String>
where
    S: h3::quic::BidiStream<Body>,
{
    let (body, applied) = if enc != Encoding::Identity && body.len() >= MIN_COMPRESS_BYTES {
        (encode_body(&body, enc), enc)
    } else {
        (body, Encoding::Identity)
    };

    let mut builder = Response::builder()
        .status(status)
        .header("content-type", content_type);
    if let Some(token) = applied.header() {
        builder = builder.header("content-encoding", token);
    }
    let resp = builder
        .body(())
        .map_err(|e| format!("h3 response build error: {e}"))?;
    stream
        .send_response(resp)
        .await
        .map_err(|e| format!("h3 send_response error: {e}"))?;
    stream
        .send_data(body)
        .await
        .map_err(|e| format!("h3 send_data error: {e}"))?;
    stream
        .finish()
        .await
        .map_err(|e| format!("h3 finish error: {e}"))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize};
    use std::time::Duration;

    use tokio::sync::mpsc::unbounded_channel;

    use crate::pipeline::{ServeRequest, ServeResponse, StopReason};

    /// A stub "GPU worker": for every request, echo a fixed token stream and a
    /// terminal completion. Lets us verify the HTTP/3 transport end-to-end
    /// without loading a model.
    fn stub_state(api_key: Option<&str>) -> AppState {
        let (tx, mut rx) = unbounded_channel::<ServeRequest>();
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                let mut sink = req.reply;
                let parts = ["Hello", ", ", "world"];
                if req.stream {
                    for p in parts {
                        sink(ServeEvent::Token(p.to_string()));
                    }
                }
                sink(ServeEvent::Done(ServeResponse {
                    text: "Hello, world".to_string(),
                    tokens: vec![1, 2, 3],
                    stop: StopReason::Eos,
                }));
            }
        });
        AppState {
            tx,
            next_id: Arc::new(AtomicU64::new(0)),
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_in_flight: 1024,
            api_key: api_key.map(Arc::from),
        }
    }

    /// rustls verifier that accepts any server certificate (test client only).
    #[derive(Debug)]
    struct AcceptAny;

    impl rustls::client::danger::ServerCertVerifier for AcceptAny {
        fn verify_server_cert(
            &self,
            _end_entity: &rustls::pki_types::CertificateDer<'_>,
            _intermediates: &[rustls::pki_types::CertificateDer<'_>],
            _server_name: &rustls::pki_types::ServerName<'_>,
            _ocsp: &[u8],
            _now: rustls::pki_types::UnixTime,
        ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
            Ok(rustls::client::danger::ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &rustls::pki_types::CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }

        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &rustls::pki_types::CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }

        fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
            rustls::crypto::ring::default_provider()
                .signature_verification_algorithms
                .supported_schemes()
        }
    }

    fn client_endpoint() -> quinn::Endpoint {
        let mut tls = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAny))
            .with_no_client_auth();
        tls.alpn_protocols = vec![b"h3".to_vec()];
        let quic_tls = quinn::crypto::rustls::QuicClientConfig::try_from(tls).unwrap();
        let client_config = quinn::ClientConfig::new(Arc::new(quic_tls));
        let mut endpoint = quinn::Endpoint::client(([0, 0, 0, 0], 0).into()).unwrap();
        endpoint.set_default_client_config(client_config);
        endpoint
    }

    /// Result of an HTTP/3 roundtrip: status, the `content-encoding` header (if
    /// any), and the raw (possibly compressed) response body bytes.
    struct H3Resp {
        status: u16,
        content_encoding: Option<String>,
        body: Vec<u8>,
    }

    impl H3Resp {
        /// Decode the body, transparently decompressing per `content-encoding`.
        fn text(&self) -> String {
            match self.content_encoding.as_deref() {
                Some("zstd") => {
                    let bytes = zstd::stream::decode_all(&self.body[..]).expect("zstd decode");
                    String::from_utf8_lossy(&bytes).into_owned()
                }
                Some("br") => {
                    use std::io::Read;
                    let mut out = Vec::new();
                    brotli::Decompressor::new(&self.body[..], 4096)
                        .read_to_end(&mut out)
                        .expect("brotli decode");
                    String::from_utf8_lossy(&out).into_owned()
                }
                _ => String::from_utf8_lossy(&self.body).into_owned(),
            }
        }
    }

    async fn h3_roundtrip(
        addr: SocketAddr,
        method: &str,
        path: &str,
        body: Option<&str>,
        accept_encoding: Option<&str>,
    ) -> H3Resp {
        let endpoint = client_endpoint();
        let conn = endpoint.connect(addr, "localhost").unwrap().await.unwrap();
        let (mut driver, mut send_req) = h3::client::new(h3_quinn::Connection::new(conn))
            .await
            .unwrap();

        let drive = tokio::spawn(async move {
            let _ = std::future::poll_fn(|cx| driver.poll_close(cx)).await;
        });

        let mut builder = http::Request::builder()
            .method(method)
            .uri(format!("https://localhost{path}"));
        if let Some(ae) = accept_encoding {
            builder = builder.header("accept-encoding", ae);
        }
        let req = builder.body(()).unwrap();
        let mut stream = send_req.send_request(req).await.unwrap();
        if let Some(b) = body {
            stream.send_data(Bytes::from(b.to_string())).await.unwrap();
        }
        stream.finish().await.unwrap();

        let resp = stream.recv_response().await.unwrap();
        let status = resp.status().as_u16();
        let content_encoding = resp
            .headers()
            .get("content-encoding")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let mut out = Vec::new();
        while let Some(mut chunk) = stream.recv_data().await.unwrap() {
            let bytes = chunk.copy_to_bytes(chunk.remaining());
            out.extend_from_slice(&bytes);
        }
        drive.abort();
        endpoint.wait_idle().await;
        H3Resp {
            status,
            content_encoding,
            body: out,
        }
    }

    fn start_server_with(api_key: Option<&str>) -> SocketAddr {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let state = stub_state(api_key);
        let server_config = build_server_config().unwrap();
        let endpoint = quinn::Endpoint::server(server_config, ([127, 0, 0, 1], 0).into()).unwrap();
        let addr = endpoint.local_addr().unwrap();
        tokio::spawn(accept_loop(endpoint, state));
        addr
    }

    fn start_server() -> SocketAddr {
        start_server_with(None)
    }

    #[tokio::test]
    async fn h3_health_and_completions() {
        let addr = start_server();
        // Give the endpoint a beat to be ready.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // /health
        let r = h3_roundtrip(addr, "GET", "/health", None, None).await;
        assert_eq!(r.status, 200, "health status");
        assert_eq!(r.text(), "ok", "health body");

        // non-stream completion (identity)
        let r = h3_roundtrip(
            addr,
            "POST",
            "/v1/chat/completions",
            Some(r#"{"prompt":"hi"}"#),
            None,
        )
        .await;
        assert_eq!(r.status, 200, "completion status");
        assert!(
            r.content_encoding.is_none(),
            "no accept-encoding → identity"
        );
        let body = r.text();
        assert!(
            body.contains("Hello, world"),
            "non-stream body should contain completion, got: {body}"
        );
        assert!(
            body.contains("chat.completion"),
            "should be chat.completion json"
        );

        // non-stream completion with zstd content-encoding
        let r = h3_roundtrip(
            addr,
            "POST",
            "/v1/chat/completions",
            Some(r#"{"prompt":"hi"}"#),
            Some("zstd"),
        )
        .await;
        assert_eq!(r.status, 200, "zstd completion status");
        assert_eq!(
            r.content_encoding.as_deref(),
            Some("zstd"),
            "should advertise zstd content-encoding"
        );
        assert!(
            r.text().contains("Hello, world"),
            "zstd body should decompress to completion"
        );

        // non-stream completion with brotli content-encoding
        let r = h3_roundtrip(
            addr,
            "POST",
            "/v1/chat/completions",
            Some(r#"{"prompt":"hi"}"#),
            Some("gzip, br"),
        )
        .await;
        assert_eq!(r.status, 200, "br completion status");
        assert_eq!(
            r.content_encoding.as_deref(),
            Some("br"),
            "should prefer brotli when offered"
        );
        assert!(
            r.text().contains("Hello, world"),
            "brotli body should decompress to completion"
        );

        // streaming completion (SSE is never compressed)
        let r = h3_roundtrip(
            addr,
            "POST",
            "/v1/chat/completions",
            Some(r#"{"prompt":"hi","stream":true}"#),
            Some("br, zstd"),
        )
        .await;
        assert_eq!(r.status, 200, "stream status");
        assert!(
            r.content_encoding.is_none(),
            "SSE stream must stay uncompressed for low latency"
        );
        let body = r.text();
        assert!(body.contains("Hello"), "stream should contain first token");
        assert!(body.contains("world"), "stream should contain last token");
        assert!(
            body.contains("[DONE]"),
            "stream should terminate with [DONE]"
        );
        let chunks = body.matches("data:").count();
        assert!(
            chunks >= 4,
            "expected >=4 SSE frames (3 tokens + DONE), got {chunks}"
        );

        eprintln!(
            "HTTP/3 verified: /health, non-stream, zstd + brotli content-encoding, and \
             uncompressed SSE streaming all OK over QUIC"
        );
    }

    #[tokio::test]
    async fn h3_api_key_auth() {
        let addr = start_server_with(Some("s3cret"));
        tokio::time::sleep(Duration::from_millis(50)).await;

        // health is exempt from auth
        let r = h3_roundtrip(addr, "GET", "/health", None, None).await;
        assert_eq!(r.status, 200, "health is unauthenticated");

        // missing key → 401
        let r = h3_roundtrip(
            addr,
            "POST",
            "/v1/chat/completions",
            Some(r#"{"prompt":"hi"}"#),
            None,
        )
        .await;
        assert_eq!(r.status, 401, "no bearer token → 401");

        // wrong key → 401
        let r = h3_roundtrip_auth(
            addr,
            "POST",
            "/v1/chat/completions",
            Some(r#"{"prompt":"hi"}"#),
            "Bearer wrong",
        )
        .await;
        assert_eq!(r.status, 401, "wrong bearer token → 401");

        // correct key → 200
        let r = h3_roundtrip_auth(
            addr,
            "POST",
            "/v1/chat/completions",
            Some(r#"{"prompt":"hi"}"#),
            "Bearer s3cret",
        )
        .await;
        assert_eq!(r.status, 200, "correct bearer token → 200");
        assert!(r.text().contains("Hello, world"));

        eprintln!("HTTP/3 auth verified: health open, /v1/* requires valid bearer token");
    }

    /// Roundtrip variant that sets an `Authorization` header.
    async fn h3_roundtrip_auth(
        addr: SocketAddr,
        method: &str,
        path: &str,
        body: Option<&str>,
        authorization: &str,
    ) -> H3Resp {
        let endpoint = client_endpoint();
        let conn = endpoint.connect(addr, "localhost").unwrap().await.unwrap();
        let (mut driver, mut send_req) = h3::client::new(h3_quinn::Connection::new(conn))
            .await
            .unwrap();
        let drive = tokio::spawn(async move {
            let _ = std::future::poll_fn(|cx| driver.poll_close(cx)).await;
        });
        let req = http::Request::builder()
            .method(method)
            .uri(format!("https://localhost{path}"))
            .header("authorization", authorization)
            .body(())
            .unwrap();
        let mut stream = send_req.send_request(req).await.unwrap();
        if let Some(b) = body {
            stream.send_data(Bytes::from(b.to_string())).await.unwrap();
        }
        stream.finish().await.unwrap();
        let resp = stream.recv_response().await.unwrap();
        let status = resp.status().as_u16();
        let content_encoding = resp
            .headers()
            .get("content-encoding")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let mut out = Vec::new();
        while let Some(mut chunk) = stream.recv_data().await.unwrap() {
            let bytes = chunk.copy_to_bytes(chunk.remaining());
            out.extend_from_slice(&bytes);
        }
        drive.abort();
        endpoint.wait_idle().await;
        H3Resp {
            status,
            content_encoding,
            body: out,
        }
    }
}
