use std::net::{IpAddr, SocketAddr};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{Method, StatusCode, header};
use axum::response::Response;
use axum::{Router, routing::any};
use http_body_util::BodyExt;
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;

use local_engine::{Engine, EngineHandle};

use crate::resources::Resources;

mod chunked;
mod http3;
mod options;
mod request;
mod response;
mod responses;
mod sse;

#[cfg(test)]
use self::request::message_text;
#[cfg(test)]
use axum::http::HeaderMap;
#[cfg(test)]
use local_engine::Event;

use self::chunked::start_chunked;
use self::http3::serve_h3;
use self::options::{parse, usage};
use self::request::{GenerationRequest, prepare_generation};
use self::response::{
    Protocol, Reply, add_alt_svc, error_response, error_status, error_status_message,
    json_response, queue_full_response, wants_zstd,
};
use self::responses::prepare_responses;
use self::sse::start_stream;

const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug)]
struct Args {
    host: IpAddr,
    port: u16,
    thinking: bool,
    api_key: Option<String>,
    /// How long a client may stop consuming before its generation is dropped.
    stall: Duration,
}

#[derive(Clone)]
struct AppState {
    engine: EngineHandle,
    model: Arc<str>,
    /// Tokens one request may hold, prompt and output together, as the engine
    /// admitted them at load (`BonsaiInfo::context`). Requests over it fail
    /// with a context-overflow error rather than being truncated.
    context: usize,
    thinking: bool,
    api_key: Option<Arc<str>>,
    alt_svc: Option<Arc<str>>,
    depth: QueueDepth,
    /// How long a client may stop consuming before its generation is dropped.
    stall: Duration,
}

/// Requests the engine has accepted and not finished yet: the queue depth as
/// this server sees it.
///
/// It cannot be read off the engine, because `try_send` reports only that the
/// queue was full, and it is only ever read to explain a rejection, so it is
/// tracked here. Every job reaching the queue comes through this module, so it
/// counts queued and running requests together.
#[derive(Clone, Default)]
struct QueueDepth(Arc<AtomicUsize>);

impl QueueDepth {
    fn load(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }

    /// Take a queue slot. The returned guard releases it on drop, so an early
    /// return, a failed generation and a disconnected client all release it.
    fn admit(&self) -> Admitted {
        self.0.fetch_add(1, Ordering::SeqCst);
        Admitted(Arc::clone(&self.0))
    }
}

struct Admitted(Arc<AtomicUsize>);

impl Drop for Admitted {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[allow(clippy::too_many_lines)]
async fn route(State(state): State<AppState>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    if let Some(expected) = &state.api_key {
        let valid = parts
            .headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .is_some_and(|value| value == expected.as_ref());
        if !valid {
            return error_response(
                StatusCode::UNAUTHORIZED,
                "invalid API key",
                &state,
                &parts.headers,
            )
            .await;
        }
    }
    if parts.method == Method::GET && parts.uri.path() == "/health" {
        return json_response(
            StatusCode::OK,
            json!({"status":"ok"}),
            &state,
            &parts.headers,
        )
        .await;
    }
    if parts.method == Method::GET && parts.uri.path() == "/v1/models" {
        return json_response(
            StatusCode::OK,
            models_json(&state.model, state.context),
            &state,
            &parts.headers,
        )
        .await;
    }
    let api = match (parts.method, parts.uri.path()) {
        (Method::POST, "/v1/chat/completions") => Api::Chat,
        (Method::POST, "/v1/completions") => Api::Completion,
        (Method::POST, "/v1/responses") => Api::Responses,
        _ => {
            return error_response(
                StatusCode::NOT_FOUND,
                "route not found",
                &state,
                &parts.headers,
            )
            .await;
        }
    };
    let body = match body.collect().await {
        Ok(value) => {
            let value = value.to_bytes();
            if value.len() <= MAX_REQUEST_BYTES {
                value.to_vec()
            } else {
                return error_response(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "HTTP request is too large",
                    &state,
                    &parts.headers,
                )
                .await;
            }
        }
        Err(error) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("invalid HTTP body: {error}"),
                &state,
                &parts.headers,
            )
            .await;
        }
    };
    let prepared = match prepare(api, &body, state.thinking) {
        Ok(prepared) => prepared,
        Err(error) => {
            return error_response(
                error_status(&error),
                &error.to_string(),
                &state,
                &parts.headers,
            )
            .await;
        }
    };
    let (stream, request, protocol) = prepared;
    let reply = Reply::new(protocol, Arc::clone(&state.model));
    let events = match request {
        GenerationRequest::Chat(request) => state.engine.chat(request),
        GenerationRequest::Completion(request) => state.engine.complete(request),
    };
    let events = match events {
        Ok(events) => events,
        Err(error) => {
            if matches!(error, crate::Error::QueueFull) {
                return queue_full_response(&state, &parts.headers).await;
            }
            return error_response(
                error_status(&error),
                &error.to_string(),
                &state,
                &parts.headers,
            )
            .await;
        }
    };
    let admitted = state.depth.admit();
    if stream {
        match start_stream(events, reply, admitted, state.stall).await {
            Ok(events) => {
                let mut response = Response::new(Body::from_stream(ReceiverStream::new(events)));
                response.headers_mut().insert(
                    header::CONTENT_TYPE,
                    header::HeaderValue::from_static("text/event-stream"),
                );
                response.headers_mut().insert(
                    header::CACHE_CONTROL,
                    header::HeaderValue::from_static("no-cache"),
                );
                add_alt_svc(&mut response, &state);
                response
            }
            Err(error) => {
                error_response(error_status_message(&error), &error, &state, &parts.headers).await
            }
        }
    } else {
        // The body streams as the generation produces it, so the first byte is on
        // the wire while prefill is still running and a client that has gone away
        // fails a write instead of being waited out. `zstd` is decided from the
        // request here and reported back from the body, because the response may
        // only claim an encoding it is really using.
        let zstd = wants_zstd(&parts.headers);
        match start_chunked(events, reply, admitted, state.stall, zstd).await {
            Ok((frames, compressed)) => {
                let mut response = Response::new(Body::from_stream(ReceiverStream::new(frames)));
                response.headers_mut().insert(
                    header::CONTENT_TYPE,
                    header::HeaderValue::from_static("application/json"),
                );
                if compressed {
                    response.headers_mut().insert(
                        header::CONTENT_ENCODING,
                        header::HeaderValue::from_static("zstd"),
                    );
                }
                add_alt_svc(&mut response, &state);
                response
            }
            Err(error) => {
                error_response(error_status_message(&error), &error, &state, &parts.headers).await
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Api {
    Completion,
    Chat,
    Responses,
}

/// Parse a request body for `api` into the engine request and how to answer it.
fn prepare(
    api: Api,
    body: &[u8],
    thinking: bool,
) -> crate::Result<(bool, GenerationRequest, Protocol)> {
    match api {
        Api::Completion | Api::Chat => {
            let chat = matches!(api, Api::Chat);
            let prepared = prepare_generation(body, chat, thinking)?;
            let protocol = if chat {
                Protocol::Chat {
                    include_usage: prepared.include_usage,
                }
            } else {
                Protocol::Completion
            };
            Ok((prepared.stream, prepared.request, protocol))
        }
        Api::Responses => {
            let prepared = prepare_responses(body, thinking)?;
            Ok((
                prepared.stream,
                GenerationRequest::Chat(prepared.request),
                Protocol::Responses(Arc::new(prepared.echo)),
            ))
        }
    }
}

/// `GET /v1/models`: the one loaded model and the context the engine admitted
/// for it.
///
/// `context_length` (the name `OpenRouter` and LM Studio read) and
/// `max_model_len` (vLLM's) are both the per-request admission limit from
/// `BonsaiInfo::context`, prompt plus output, not the checkpoint's training
/// length: it is what a request can actually use on this machine.
fn models_json(model: &str, context: usize) -> serde_json::Value {
    json!({"object":"list","data":[{"id":model,"object":"model","owned_by":"local","context_length":context,"max_model_len":context}]})
}

async fn run_async(args: Args) -> crate::Result<()> {
    // One discovery serves both the engine and the TLS lookup: each one probes
    // the disk's write rate with a 16 MiB file, so a second is wasted startup.
    let resources = Resources::discover(None, true)?;
    let model: Arc<str> = resources.model.to_string_lossy().into_owned().into();
    let engine = Engine::from_resources(&resources)?;
    eprintln!("{}", engine.info().json);
    let context = engine.info().model.context;
    let state = AppState {
        engine: engine.into_handle(),
        model,
        context,
        thinking: args.thinking,
        api_key: args.api_key.map(Into::into),
        alt_svc: resources
            .tls
            .as_ref()
            .map(|_| Arc::from(format!("h3=\":{}\"; ma=86400", args.port))),
        depth: QueueDepth::default(),
        stall: args.stall,
    };
    let address = SocketAddr::new(args.host, args.port);
    let h3_task = if let Some((cert, key)) = resources.tls {
        Some(tokio::spawn(serve_h3(address, state.clone(), cert, key)))
    } else {
        None
    };
    let listener = tokio::net::TcpListener::bind(address).await?;
    eprintln!("Bonsai server listening on http://{address}");
    let app = Router::new().fallback(any(route)).with_state(state);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await
        .map_err(crate::Error::Io)?;
    if let Some(task) = h3_task {
        task.abort();
    }
    Ok(())
}

async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
}

pub fn main_with_args(args: &[String]) -> ExitCode {
    match parse(args) {
        Ok(args) => match tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => match runtime.block_on(run_async(args)) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("error: {error}");
                    ExitCode::FAILURE
                }
            },
            Err(error) => {
                eprintln!("error: could not start Tokio runtime: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            if !error.is_empty() {
                eprintln!("error: {error}");
            }
            usage();
            if error.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;

#[cfg(test)]
#[allow(clippy::expect_used)]
mod protocol_tests;
