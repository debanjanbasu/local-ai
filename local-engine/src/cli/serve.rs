//! Continuous-batching HTTP server (axum + tokio).
//!
//! Architecture: a single **GPU worker thread** owns the (non-`Send`) inference
//! pipeline and runs the continuous-batching loop. The async axum front-end
//! (HTTP/1.1 + HTTP/2, keep-alive, SSE) hands requests to that worker over a
//! channel and forwards generated events back per request — async I/O for the
//! many concurrent connections, one serial owner for the GPU. The engine itself
//! stays runtime-agnostic: requests carry a boxed [`EventSink`] callback, so the
//! worker never depends on tokio.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver as StdReceiver, Sender as StdSender, channel as std_channel};
use std::thread;
use std::time::Instant;

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::header::AUTHORIZATION;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::compression::{CompressionLayer, CompressionLevel};

use crate::multimodal::{DecodedRgbImage, MediaInput};
use crate::pipeline::{EventSink, ServeEvent, ServeRequest};
use crate::request_queue::RequestPriority;
use crate::{DEFAULT_TEMPERATURE, DEFAULT_TOP_P, Engine, EngineConfig};

const DEFAULT_MODEL_DIR: &str = "models/gemma-4-e2b-it";
const DEFAULT_LANES: usize = 8;
const DEFAULT_LANE_CONTEXT: usize = 4096;
const DEFAULT_MAX_NEW_TOKENS: usize = 512;

const DEFAULT_MAX_IN_FLIGHT: usize = 1024;

struct Args {
    model: PathBuf,
    host: std::net::IpAddr,
    port: u16,
    lanes: usize,
    lane_context: usize,
    max_in_flight: usize,
    api_key: Option<String>,
}

#[derive(Clone)]
pub(super) struct AppState {
    /// Forwards requests to the GPU worker. `UnboundedSender` is `Send + Sync +
    /// Clone`, so it can be shared across all connection tasks.
    pub(super) tx: UnboundedSender<ServeRequest>,
    pub(super) next_id: Arc<AtomicU64>,
    /// Number of accepted-but-not-yet-finished requests (active lanes + queued).
    /// Bounds memory/DoS exposure: new requests are rejected once it hits
    /// `max_in_flight`.
    pub(super) in_flight: Arc<AtomicUsize>,
    pub(super) max_in_flight: usize,
    /// Optional bearer token required on `/v1/*` routes. `None` = open (intended
    /// for the loopback default).
    pub(super) api_key: Option<Arc<str>>,
}

/// Decrements the in-flight counter when the request's event sink is dropped by
/// the worker (i.e. when the request completes or is recycled).
pub(super) struct InFlightGuard(Arc<AtomicUsize>);

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Why a request could not be accepted.
pub(super) enum SubmitError {
    /// `max_in_flight` reached — client should retry later (HTTP 503).
    Overloaded,
    /// The GPU worker has shut down (HTTP 503).
    ShuttingDown,
}

/// Constant-time string equality (length leak is acceptable). Used for the API
/// key check so a wrong key cannot be discovered byte-by-byte via timing.
pub(super) fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Validate the `Authorization: Bearer <token>` header against the configured
/// API key. Returns `true` when no key is configured (open server).
pub(super) fn authorized(api_key: Option<&Arc<str>>, bearer: Option<&str>) -> bool {
    api_key.is_none_or(|key| {
        bearer
            .and_then(|h| h.strip_prefix("Bearer "))
            .is_some_and(|token| constant_time_eq(token, key))
    })
}

fn print_usage() {
    eprintln!("Usage: serve [--model PATH] [--port PORT] [--lanes N] [--lane-context N]");
    eprintln!();
    eprintln!("Continuous-batching HTTP server (axum, HTTP/1.1 + HTTP/2, SSE streaming).");
    eprintln!("One GPU worker decodes many concurrent requests together, so throughput");
    eprintln!("scales with the number of in-flight agents instead of one at a time.");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model <path>      Model directory (default: {DEFAULT_MODEL_DIR})");
    eprintln!("  --host <IP>         Bind address (default: 127.0.0.1; use 0.0.0.0 to expose");
    eprintln!("                      on the network — there is NO built-in auth)");
    eprintln!("  --port <PORT>       Port to listen on (default: 8080)");
    eprintln!("  --lanes <N>         Concurrent decode lanes (default: {DEFAULT_LANES}, max 64)");
    eprintln!(
        "  --lane-context <N>  Per-lane KV context window (default: {DEFAULT_LANE_CONTEXT}, 0 = model max)"
    );
    eprintln!(
        "  --max-inflight <N>  Max accepted in-flight requests before 503 (default: {DEFAULT_MAX_IN_FLIGHT})"
    );
    eprintln!(
        "  --api-key <KEY>     Require `Authorization: Bearer KEY` on /v1/* (or set LOCAL_AI_API_KEY)"
    );
    eprintln!("  --help              Show this help message");
    eprintln!();
    eprintln!("Transport: HTTP/1.1 + HTTP/2 (h2c) on TCP, and HTTP/3 (QUIC) on UDP — both");
    eprintln!("always enabled on the same port. HTTP/3 uses an ephemeral self-signed TLS cert.");
    eprintln!();
    eprintln!("Routes:");
    eprintln!("  POST /v1/chat/completions   OpenAI-style; supports \"stream\": true (SSE)");
    eprintln!("  POST /v1/completions        same, raw prompt");
    eprintln!("  GET  /health                liveness probe");
}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|idx| args.get(idx + 1))
        .cloned()
}

fn parse_args(argv: &[String]) -> Option<Args> {
    if argv
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        print_usage();
        return None;
    }
    let model =
        parse_arg(argv, "--model").map_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR), PathBuf::from);
    let host = match parse_arg(argv, "--host") {
        Some(h) => {
            let Ok(ip) = h.parse() else {
                eprintln!("invalid --host {h:?}: expected an IP address like 127.0.0.1 or 0.0.0.0");
                return None;
            };
            ip
        }
        None => std::net::IpAddr::from([127, 0, 0, 1]),
    };
    let port = parse_arg(argv, "--port")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8080);
    let lanes = parse_arg(argv, "--lanes")
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n >= 1)
        .unwrap_or(DEFAULT_LANES);
    let lane_context = parse_arg(argv, "--lane-context")
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_LANE_CONTEXT);
    let max_in_flight = parse_arg(argv, "--max-inflight")
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n >= 1)
        .unwrap_or(DEFAULT_MAX_IN_FLIGHT);
    let api_key = parse_arg(argv, "--api-key")
        .or_else(|| std::env::var("LOCAL_AI_API_KEY").ok())
        .filter(|k| !k.is_empty());
    Some(Args {
        model,
        host,
        port,
        lanes,
        lane_context,
        max_in_flight,
        api_key,
    })
}

pub(super) struct ParsedRequest {
    pub(super) prompt: String,
    /// Decoded media inputs (images/audio) from `OpenAI` content arrays. Empty for
    /// plain text requests.
    pub(super) media: Vec<MediaInput>,
    /// Set when a media part failed to decode; the handler turns this into a
    /// `400 Bad Request` rather than silently dropping the media.
    pub(super) media_error: Option<String>,
    pub(super) temperature: f32,
    pub(super) top_p: f32,
    pub(super) max_new_tokens: usize,
    pub(super) priority: RequestPriority,
    pub(super) stream: bool,
}

/// Parse a request body. Supports `OpenAI`-style `{"messages":[{"content":...}]}`
/// — including multimodal content arrays with `image_url` (data URIs) and
/// `input_audio` parts — as well as `{"prompt":...}` / `{"content":...}`, plus
/// optional `temperature`, `top_p`, `max_tokens`/`max_new_tokens`, `priority`,
/// and `stream`. Falls back to treating the whole body as the prompt when it is
/// not JSON.
pub(super) fn parse_request(body: &str) -> ParsedRequest {
    let json = serde_json::from_str::<serde_json::Value>(body).ok();

    let (prompt, media, media_error) = json.as_ref().map_or_else(
        || (body.to_string(), Vec::new(), None),
        |j| {
            let (prompt, media, err) = extract_content(j);
            (prompt.unwrap_or_else(|| body.to_string()), media, err)
        },
    );
    let temperature = json
        .as_ref()
        .and_then(|j| j.get("temperature").and_then(serde_json::Value::as_f64))
        .map_or(DEFAULT_TEMPERATURE, |v| v as f32);
    let top_p = json
        .as_ref()
        .and_then(|j| j.get("top_p").and_then(serde_json::Value::as_f64))
        .map_or(DEFAULT_TOP_P, |v| v as f32);
    let max_new_tokens = json
        .as_ref()
        .and_then(|j| {
            j.get("max_tokens")
                .or_else(|| j.get("max_new_tokens"))
                .and_then(serde_json::Value::as_u64)
        })
        .map_or(DEFAULT_MAX_NEW_TOKENS, |v| v as usize);
    let priority = json
        .as_ref()
        .and_then(|j| j.get("priority").and_then(serde_json::Value::as_str))
        .map_or(RequestPriority::Interactive, |p| match p {
            "background" | "low" => RequestPriority::Background,
            _ => RequestPriority::Interactive,
        });
    let stream = json
        .as_ref()
        .and_then(|j| j.get("stream").and_then(serde_json::Value::as_bool))
        .unwrap_or(false);

    ParsedRequest {
        prompt,
        media,
        media_error,
        temperature,
        top_p,
        max_new_tokens,
        priority,
        stream,
    }
}

/// Extract the prompt text and any media from a JSON request body. Handles the
/// `OpenAI` chat shape (last message's `content`, which may be a plain string or a
/// multimodal parts array) and the simple `{"prompt"|"content": "..."}` shape.
/// Returns `(text, media, first_decode_error)`.
fn extract_content(json: &serde_json::Value) -> (Option<String>, Vec<MediaInput>, Option<String>) {
    if let Some(messages) = json.get("messages").and_then(serde_json::Value::as_array)
        && let Some(content) = messages
            .iter()
            .rev()
            .find_map(|m| m.get("content").filter(|c| !c.is_null()))
    {
        if let Some(text) = content.as_str() {
            return (Some(text.to_string()), Vec::new(), None);
        }
        if let Some(parts) = content.as_array() {
            return extract_content_parts(parts);
        }
    }
    for key in ["prompt", "content"] {
        if let Some(s) = json.get(key).and_then(serde_json::Value::as_str) {
            return (Some(s.to_string()), Vec::new(), None);
        }
    }
    (None, Vec::new(), None)
}

/// Walk an `OpenAI` multimodal `content` parts array, concatenating `text` parts
/// and decoding `image_url` / `input_audio` parts into [`MediaInput`]s. Media
/// order is preserved; the first decode failure is returned so the caller can
/// reply `400` instead of silently dropping media.
fn extract_content_parts(
    parts: &[serde_json::Value],
) -> (Option<String>, Vec<MediaInput>, Option<String>) {
    let mut text = String::new();
    let mut media = Vec::new();
    let mut error = None;
    for part in parts {
        match part.get("type").and_then(serde_json::Value::as_str) {
            Some("text") => {
                if let Some(s) = part.get("text").and_then(serde_json::Value::as_str) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(s);
                }
            }
            Some("image_url") => {
                let url = part
                    .get("image_url")
                    .and_then(|v| v.get("url"))
                    .and_then(serde_json::Value::as_str);
                match url
                    .ok_or_else(|| "image_url part missing url".to_string())
                    .and_then(decode_image_data_uri)
                {
                    Ok(m) => media.push(m),
                    Err(e) => {
                        error.get_or_insert(e);
                    }
                }
            }
            Some("input_audio") => {
                let audio = part.get("input_audio");
                let data = audio
                    .and_then(|v| v.get("data"))
                    .and_then(serde_json::Value::as_str);
                match data
                    .ok_or_else(|| "input_audio part missing data".to_string())
                    .and_then(decode_audio_base64)
                {
                    Ok(m) => media.push(m),
                    Err(e) => {
                        error.get_or_insert(e);
                    }
                }
            }
            _ => {}
        }
    }
    let text = if text.is_empty() { None } else { Some(text) };
    (text, media, error)
}

/// Decode a base64 `data:` URI (`data:[<mime>][;base64],<payload>`) into raw
/// bytes. Only `data:` URIs are accepted — remote URL fetching is deliberately
/// unsupported to avoid SSRF / local-file access from the server.
fn decode_data_uri(uri: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    let rest = uri
        .strip_prefix("data:")
        .ok_or_else(|| "only data: URIs are supported for media".to_string())?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| "malformed data URI (no comma)".to_string())?;
    if !meta.contains("base64") {
        return Err("only base64-encoded data URIs are supported".to_string());
    }
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|e| format!("base64 decode failed: {e}"))
}

fn decode_image_data_uri(uri: &str) -> Result<MediaInput, String> {
    let bytes = decode_data_uri(uri)?;
    let rgb = image::load_from_memory(&bytes)
        .map_err(|e| format!("image decode failed: {e}"))?
        .to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(MediaInput::DecodedImage {
        image: DecodedRgbImage {
            width,
            height,
            rgb: rgb.into_raw(),
        },
    })
}

fn decode_audio_base64(data: &str) -> Result<MediaInput, String> {
    use base64::Engine as _;
    // `input_audio.data` is bare base64 (no data: prefix); also tolerate a full
    // data URI for convenience.
    let bytes = if data.starts_with("data:") {
        decode_data_uri(data)?
    } else {
        base64::engine::general_purpose::STANDARD
            .decode(data.trim())
            .map_err(|e| format!("base64 decode failed: {e}"))?
    };
    let audio = crate::gemma4_audio::decode_wav_bytes(&bytes).map_err(|e| e.to_string())?;
    Ok(MediaInput::PcmAudio { audio })
}

fn json_response(status: axum::http::StatusCode, body: String) -> Response {
    (
        status,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response()
}

pub(super) fn ok_body(content: &str) -> String {
    format!(
        "{{\"object\":\"chat.completion\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{content:?}}},\"finish_reason\":\"stop\"}}]}}"
    )
}

pub(super) fn error_body(message: &str) -> String {
    format!("{{\"error\":{{\"message\":{message:?}}}}}")
}

/// One streaming SSE chunk (`OpenAI` `chat.completion.chunk` shape) as a `data:` line.
pub(super) fn sse_token_chunk(delta: &str) -> String {
    format!(
        "data: {{\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{delta:?}}}}}]}}\n\n"
    )
}

/// Build the [`ServeRequest`] for `body` and a tokio receiver of its events,
/// reserving an in-flight slot for backpressure.
///
/// # Errors
/// [`SubmitError::Overloaded`] when `max_in_flight` is reached, or
/// [`SubmitError::ShuttingDown`] when the worker has exited.
pub(super) fn submit(
    state: &AppState,
    parsed: ParsedRequest,
) -> Result<tokio::sync::mpsc::UnboundedReceiver<ServeEvent>, SubmitError> {
    // Reserve a slot up front; release it immediately if we cannot enqueue.
    let prev = state.in_flight.fetch_add(1, Ordering::Relaxed);
    if prev >= state.max_in_flight {
        state.in_flight.fetch_sub(1, Ordering::Relaxed);
        return Err(SubmitError::Overloaded);
    }
    // The guard rides inside the event sink; when the worker drops the sink at
    // request completion, the slot is released.
    let guard = InFlightGuard(Arc::clone(&state.in_flight));

    let id = format!("req-{}", state.next_id.fetch_add(1, Ordering::Relaxed));
    let (ev_tx, ev_rx) = unbounded_channel::<ServeEvent>();
    let sink: EventSink = Box::new(move |ev| {
        // Keep the guard owned by (and alive for) the sink's lifetime.
        let _ = &guard;
        let _ = ev_tx.send(ev);
    });
    let req = ServeRequest {
        id,
        prompt: parsed.prompt,
        media: parsed.media,
        temperature: parsed.temperature,
        top_p: parsed.top_p,
        max_new_tokens: parsed.max_new_tokens,
        priority: parsed.priority,
        stream: parsed.stream,
        reply: sink,
    };
    // On send failure the request (and its sink+guard) drops here, releasing the
    // slot automatically.
    match state.tx.send(req) {
        Ok(()) => Ok(ev_rx),
        Err(_) => Err(SubmitError::ShuttingDown),
    }
}

async fn completions(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let bearer = headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok());
    if !authorized(state.api_key.as_ref(), bearer) {
        eprintln!("[serve] 401 unauthorized");
        return json_response(
            axum::http::StatusCode::UNAUTHORIZED,
            error_body("missing or invalid API key"),
        );
    }

    let text = String::from_utf8_lossy(&body);
    let parsed = parse_request(&text);
    if let Some(err) = &parsed.media_error {
        eprintln!("[serve] 400 bad media: {err}");
        return json_response(axum::http::StatusCode::BAD_REQUEST, error_body(err));
    }
    let stream = parsed.stream;

    let mut ev_rx = match submit(&state, parsed) {
        Ok(rx) => rx,
        Err(SubmitError::Overloaded) => {
            eprintln!(
                "[serve] 503 overloaded (in-flight cap {} reached)",
                state.max_in_flight
            );
            return json_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                error_body("server overloaded, retry later"),
            );
        }
        Err(SubmitError::ShuttingDown) => {
            return json_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                error_body("server is shutting down"),
            );
        }
    };

    let started = Instant::now();

    if stream {
        // SSE: one chunk per token delta, terminated by `[DONE]`. The stream
        // ends when the worker drops this request's event sink.
        let sse = UnboundedReceiverStream::new(ev_rx).map(move |ev| {
            let event = match ev {
                ServeEvent::Token(delta) => Event::default().data(format!(
                    "{{\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{delta:?}}}}}]}}"
                )),
                ServeEvent::Done(resp) => {
                    log_completion("stream", &resp, started);
                    Event::default().data("[DONE]")
                }
            };
            Ok::<Event, std::convert::Infallible>(event)
        });
        return Sse::new(sse)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    // Non-stream: await the terminal Done; reply with the full completion.
    while let Some(ev) = ev_rx.recv().await {
        if let ServeEvent::Done(resp) = ev {
            log_completion("json", &resp, started);
            return json_response(axum::http::StatusCode::OK, ok_body(&resp.text));
        }
    }
    json_response(
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        error_body("request could not be served (prompt too long or worker error)"),
    )
}

/// One-line access log for a finished completion.
pub(super) fn log_completion(kind: &str, resp: &crate::pipeline::ServeResponse, started: Instant) {
    let secs = started.elapsed().as_secs_f32();
    let n = resp.tokens.len();
    let tok_s = if secs > 0.0 { n as f32 / secs } else { 0.0 };
    eprintln!(
        "[serve] {kind} done: {n} tokens, stop={:?}, {secs:.2}s ({tok_s:.1} tok/s)",
        resp.stop
    );
}

async fn health() -> &'static str {
    "ok"
}

/// The GPU worker: build the engine on this thread (so the non-`Send` pipeline
/// never crosses threads) and run the continuous-batching loop until the request
/// channel is closed.
fn worker_main(
    model: PathBuf,
    lanes: usize,
    lane_context: usize,
    rx: &StdReceiver<ServeRequest>,
    ready: &StdSender<Result<(), String>>,
) {
    let config = EngineConfig {
        model_dir: model,
        max_context_length: 0,
    };
    let mut engine = match Engine::new(&config) {
        Ok(e) => e,
        Err(e) => {
            let _ = ready.send(Err(format!("Failed to load engine: {e}")));
            return;
        }
    };
    let _ = ready.send(Ok(()));
    if let Err(e) = engine.serve_batched(lanes, lane_context, rx) {
        eprintln!("serve loop error: {e}");
    }
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("\nshutting down...");
}

fn run(args: Args) -> Result<(), String> {
    let lanes = args.lanes.min(64);

    // GPU worker thread + its std channel (the engine reads a std receiver).
    let (tx_std, rx_std) = std_channel::<ServeRequest>();
    let (ready_tx, ready_rx) = std_channel::<Result<(), String>>();
    let model = args.model.clone();
    let lane_context = args.lane_context;
    let worker = thread::spawn(move || worker_main(model, lanes, lane_context, &rx_std, &ready_tx));

    // Block until the engine has loaded (or failed) before binding.
    match ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err("worker thread exited during startup".into()),
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("failed to build async runtime: {e}"))?;

    runtime.block_on(async move {
        // Bridge: forward async-side requests to the sync worker channel.
        let (tx_tok, mut rx_tok) = unbounded_channel::<ServeRequest>();
        tokio::spawn(async move {
            while let Some(req) = rx_tok.recv().await {
                if tx_std.send(req).is_err() {
                    break;
                }
            }
            // tx_std dropped here -> worker's receiver disconnects and drains.
        });

        let state = AppState {
            tx: tx_tok,
            next_id: Arc::new(AtomicU64::new(0)),
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_in_flight: args.max_in_flight,
            api_key: args.api_key.clone().map(Arc::from),
        };
        if state.api_key.is_some() {
            eprintln!("Auth enabled: /v1/* requires `Authorization: Bearer <key>`");
        }

        // HTTP/3 (QUIC) listener on the same host:port (UDP), always on. It
        // shares the GPU worker via `AppState`; cleartext HTTP/1.1+2 stay on TCP.
        // A QUIC bind failure is non-fatal: TCP keeps serving.
        match super::serve_h3::spawn(state.clone(), args.host, args.port) {
            Ok(()) => eprintln!(
                "HTTP/3 (QUIC) listening on udp://{}:{} (TLS, ephemeral self-signed cert, ALPN h3)",
                args.host, args.port
            ),
            Err(e) => eprintln!("warning: HTTP/3 unavailable ({e}); continuing with HTTP/1.1+HTTP/2"),
        }

        // Negotiated response compression (br / zstd / gzip / deflate) at best
        // quality. The default predicate skips bodies <32 B, already-compressed
        // content types, and `text/event-stream` — so non-stream JSON shrinks
        // while SSE token streams stay uncompressed for low latency.
        let compression = CompressionLayer::new()
            .br(true)
            .zstd(true)
            .gzip(true)
            .deflate(true)
            .quality(CompressionLevel::Best);
        let app = Router::new()
            .route("/v1/chat/completions", post(completions))
            .route("/v1/completions", post(completions))
            .route("/health", get(health))
            .layer(compression)
            .with_state(state);

        let listener = tokio::net::TcpListener::bind((args.host, args.port))
            .await
            .map_err(|e| format!("Failed to bind {}:{}: {e}", args.host, args.port))?;
        eprintln!(
            "Server listening on http://{}:{} ({lanes} decode lanes, axum HTTP/1.1+HTTP/2, SSE streaming)",
            args.host, args.port
        );

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| format!("server error: {e}"))
    })?;

    // Runtime returned -> the bridge task's tx_tok is dropped, the worker drains
    // and exits. Wait for it.
    let _ = worker.join();
    Ok(())
}

#[must_use]
pub fn main_with_args(argv: &[String]) -> ExitCode {
    let requested_help = argv
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"));
    let Some(opts) = parse_args(argv) else {
        return if requested_help {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    };
    if let Err(err) = run(opts) {
        eprintln!("{err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::multimodal::MediaInput;
    use base64::Engine as _;

    fn b64(bytes: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn tiny_png() -> Vec<u8> {
        let img = image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30]));
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut out, image::ImageFormat::Png)
            .expect("encode png");
        out.into_inner()
    }

    fn tiny_wav() -> Vec<u8> {
        // 16 kHz mono PCM16, 4 samples.
        let samples: [i16; 4] = [0, 1000, -1000, 500];
        let data_bytes = (samples.len() * 2) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&1u16.to_le_bytes()); // mono
        out.extend_from_slice(&16_000u32.to_le_bytes());
        out.extend_from_slice(&32_000u32.to_le_bytes()); // byte rate
        out.extend_from_slice(&2u16.to_le_bytes()); // block align
        out.extend_from_slice(&16u16.to_le_bytes()); // bits
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_bytes.to_le_bytes());
        for s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    #[test]
    fn plain_string_content_is_text_only() {
        let body = r#"{"messages":[{"role":"user","content":"hello"}]}"#;
        let parsed = parse_request(body);
        assert_eq!(parsed.prompt, "hello");
        assert!(parsed.media.is_empty());
        assert!(parsed.media_error.is_none());
    }

    #[test]
    fn content_array_text_and_image_decodes() {
        let uri = format!("data:image/png;base64,{}", b64(&tiny_png()));
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"what is this?"}},{{"type":"image_url","image_url":{{"url":"{uri}"}}}}]}}]}}"#
        );
        let parsed = parse_request(&body);
        assert_eq!(parsed.prompt, "what is this?");
        assert!(parsed.media_error.is_none(), "{:?}", parsed.media_error);
        assert_eq!(parsed.media.len(), 1);
        match &parsed.media[0] {
            MediaInput::DecodedImage { image } => {
                assert_eq!(image.width, 2);
                assert_eq!(image.height, 2);
                assert_eq!(image.rgb.len(), 2 * 2 * 3);
            }
            other => panic!("expected decoded image, got {other:?}"),
        }
    }

    #[test]
    fn content_array_input_audio_decodes() {
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"transcribe"}},{{"type":"input_audio","input_audio":{{"data":"{}","format":"wav"}}}}]}}]}}"#,
            b64(&tiny_wav())
        );
        let parsed = parse_request(&body);
        assert_eq!(parsed.prompt, "transcribe");
        assert!(parsed.media_error.is_none(), "{:?}", parsed.media_error);
        assert_eq!(parsed.media.len(), 1);
        match &parsed.media[0] {
            MediaInput::PcmAudio { audio } => {
                assert_eq!(audio.sample_rate, 16_000);
                assert_eq!(audio.channels, 1);
                assert_eq!(audio.samples.len(), 4);
            }
            other => panic!("expected pcm audio, got {other:?}"),
        }
    }

    #[test]
    fn bad_image_data_uri_sets_media_error() {
        let body = r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,not-valid-base64!!"}}]}]}"#;
        let parsed = parse_request(body);
        assert!(parsed.media.is_empty());
        assert!(parsed.media_error.is_some());
    }

    #[test]
    fn remote_image_url_is_rejected() {
        let body = r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}]}"#;
        let parsed = parse_request(body);
        assert!(parsed.media.is_empty());
        assert!(
            parsed
                .media_error
                .as_deref()
                .is_some_and(|e| e.contains("data:"))
        );
    }
}
