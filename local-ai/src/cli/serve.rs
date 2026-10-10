use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
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

mod background;
mod backpressure;
mod chunked;
mod decisions;
mod http3;
mod journal;
mod options;
mod reasoning_crypto;
mod request;
mod response;
mod responses;
mod resume;
mod sse;
mod store;

#[cfg(test)]
use self::request::message_text;
#[cfg(test)]
use axum::http::HeaderMap;
#[cfg(test)]
use local_engine::Event;

use self::background::{Background, Failure, Temporary};
use self::chunked::start_chunked;
use self::http3::serve_h3;
use self::options::{parse, usage};
use self::reasoning_crypto::ReasoningCipher;
use self::request::{GenerationRequest, prepare_generation};
use self::response::{
    Protocol, Reply, add_alt_svc, error_response, error_status, error_status_message,
    json_response, queue_full_response, wants_zstd,
};
use self::responses::prepare_responses_retaining;
use self::resume::Retrieval;
use self::sse::start_stream;
use self::store::{ItemPage, PageError, ResponseStore};

const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug)]
struct Args {
    host: IpAddr,
    port: u16,
    thinking: bool,
    api_key: Option<String>,
    /// How long a client may stop consuming before its generation is dropped.
    stall: Duration,
    /// `--response-store`: where Responses are kept. `None` keeps nothing.
    response_store: Option<PathBuf>,
    reasoning_key: Option<PathBuf>,
    /// `--experimental-decision-head`: the judgment head that enables
    /// `POST /v1/experimental/decisions`. `None` serves no decisions.
    decision_head: Option<PathBuf>,
}

#[derive(Clone)]
struct AppState {
    engine: EngineHandle,
    model: Arc<str>,
    /// Registration time of the loaded model resource, stable for this server.
    created: u64,
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
    /// The opt-in Responses store; `None` serves Responses statelessly.
    responses: Option<Arc<ResponseStore>>,
    cipher: Option<Arc<ReasoningCipher>>,
    /// Background Responses running on this server.
    background: Arc<Background>,
    /// Where background `store: false` Responses are kept briefly; `None`
    /// when no private temporary store could be created.
    temporary: Option<Arc<Temporary>>,
    /// The opt-in experimental decision service; `None` without
    /// `--experimental-decision-head`.
    decisions: Option<Arc<decisions::Decisions>>,
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
            models_json(&state.model, state.context, state.created),
            &state,
            &parts.headers,
        )
        .await;
    }
    if parts.method == Method::POST && parts.uri.path() == decisions::ROUTE {
        return decisions::handle(&state, &parts.headers, body).await;
    }
    if parts.method == Method::POST && parts.uri.path() == decisions::OPENAI_ROUTE {
        return error_response(
            StatusCode::NOT_FOUND,
            &decisions::openai_refusal(),
            &state,
            &parts.headers,
        )
        .await;
    }
    // Authentication has already passed, so a stored response is never read or
    // deleted for a caller without the key.
    if let Some(rest) = parts.uri.path().strip_prefix("/v1/responses/")
        && (matches!(parts.method, Method::GET | Method::DELETE)
            || (parts.method == Method::POST && rest.ends_with("/cancel")))
    {
        return stored_response(
            &state,
            &parts.method,
            rest,
            parts.uri.query(),
            &parts.headers,
        )
        .await;
    }
    let count_tokens =
        parts.method == Method::POST && parts.uri.path() == "/v1/responses/input_tokens";
    let api = match (parts.method, parts.uri.path()) {
        (Method::POST, "/v1/chat/completions") => Api::Chat,
        (Method::POST, "/v1/completions") => Api::Completion,
        (Method::POST, "/v1/responses" | "/v1/responses/input_tokens") => Api::Responses,
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
    // Off the runtime: resolving `previous_response_id` reads the store.
    let (thinking, store) = (state.thinking, state.responses.clone());
    let (cipher, model) = (state.cipher.clone(), Arc::clone(&state.model));
    let temporary = state.temporary.clone();
    if count_tokens {
        let engine = state.engine.clone();
        let counted = tokio::task::spawn_blocking(move || {
            let request = responses::prepare_input_tokens(
                &body,
                thinking,
                store.as_ref(),
                cipher.as_ref(),
                &model,
            )?;
            engine.count_chat_tokens(&request)
        })
        .await
        .unwrap_or_else(|error| Err(crate::Error::Generation(error.to_string())));
        return match counted {
            Ok(tokens) => {
                json_response(
                    StatusCode::OK,
                    json!({"object":"response.input_tokens","input_tokens":tokens}),
                    &state,
                    &parts.headers,
                )
                .await
            }
            Err(error) => {
                error_response(
                    error_status(&error),
                    &error.to_string(),
                    &state,
                    &parts.headers,
                )
                .await
            }
        };
    }
    let prepared = tokio::task::spawn_blocking(move || {
        prepare_retaining(
            api,
            &body,
            thinking,
            (store.as_ref(), temporary.as_ref()),
            cipher.as_ref(),
            &model,
        )
    })
    .await
    .unwrap_or_else(|error| {
        Err(crate::Error::Generation(format!(
            "request preparation failed: {error}"
        )))
    });
    let prepared = match prepared {
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
    let background = matches!(&protocol, Protocol::Responses(echo) if echo.background());
    if background && state.background.closing() {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "the server is shutting down",
            &state,
            &parts.headers,
        )
        .await;
    }
    let responses = matches!(&protocol, Protocol::Responses(_));
    let reply = Reply::new(protocol, Arc::clone(&state.model));
    // Submission compiles a structured response format synchronously before
    // queueing (about half a second the first time, building the tokenizer's
    // grammar tables), so it runs on the blocking pool, never on a runtime
    // worker. A dropped request drops the stream, which cancels the job.
    let engine = state.engine.clone();
    let events = tokio::task::spawn_blocking(move || match request {
        GenerationRequest::Chat(request) => engine.chat(request),
        GenerationRequest::Completion(request) => engine.complete(request),
    })
    .await
    .unwrap_or_else(|error| {
        Err(crate::Error::Generation(format!(
            "request submission failed: {error}"
        )))
    });
    let events = match events {
        Ok(events) => events,
        Err(error) => {
            if matches!(error, crate::Error::QueueFull) {
                return queue_full_response(&state, &parts.headers).await;
            }
            // The native API names its field response_format; Responses
            // clients need the corresponding field in their own request.
            let error = match error {
                crate::Error::InvalidArgument(message) if responses => {
                    crate::Error::InvalidArgument(message.replacen(
                        "invalid response_format:",
                        "invalid text.format:",
                        1,
                    ))
                }
                other => other,
            };
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
    if background {
        // Not waiting for the first event: a pre-generation failure becomes a
        // `failed` status on the stored response, not this request's status.
        let started = background::start(
            Arc::clone(&state.background),
            events,
            reply,
            admitted,
            stream,
        )
        .await;
        return match started {
            Ok(queued) if stream => {
                // The client follows the journal from its first event; going
                // away only ends this stream, never the job.
                let id = queued["id"].as_str().unwrap_or_default().to_owned();
                stream_events(&state, id, None, &parts.headers).await
            }
            Ok(queued) => json_response(StatusCode::OK, queued, &state, &parts.headers).await,
            Err((status, message)) => {
                error_response(status, &message, &state, &parts.headers).await
            }
        };
    }
    if stream {
        match start_stream(events, reply, admitted, state.stall).await {
            Ok(events) => event_stream(events, &state),
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

/// A `text/event-stream` response whose frames arrive on `frames`.
fn event_stream(
    frames: tokio::sync::mpsc::Receiver<Result<bytes::Bytes, std::io::Error>>,
    state: &AppState,
) -> Response {
    let mut response = Response::new(Body::from_stream(ReceiverStream::new(frames)));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache"),
    );
    add_alt_svc(&mut response, state);
    response
}

/// Stream background response `id`'s events after `after`, or answer with
/// the status that refuses it before any stream starts.
async fn stream_events(
    state: &AppState,
    id: String,
    after: Option<u64>,
    headers: &axum::http::HeaderMap,
) -> Response {
    let not_found = format!("response with id {id:?} not found");
    let background = Arc::clone(&state.background);
    let (temporary, durable) = (state.temporary.clone(), state.responses.clone());
    let subscribed = tokio::task::spawn_blocking(move || {
        let located = locate(temporary.as_ref(), durable.as_ref(), &background, &id)?;
        resume::subscribe(&background, &located.store, &id, after)
    })
    .await
    .unwrap_or_else(|error| {
        Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("response store task failed: {error}"),
        ))
    });
    match subscribed {
        Ok(Some(subscription)) => event_stream(resume::start(subscription, state.stall), state),
        Ok(None) => error_response(StatusCode::NOT_FOUND, &not_found, state, headers).await,
        Err((status, message)) => error_response(status, &message, state, headers).await,
    }
}

#[derive(Clone, Copy)]
enum Api {
    Completion,
    Chat,
    Responses,
}

/// [`prepare_retaining`] on a server with no temporary store.
#[cfg(test)]
fn prepare(
    api: Api,
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<(bool, GenerationRequest, Protocol)> {
    prepare_retaining(api, body, thinking, (store, None), cipher, model)
}

/// Parse a request body for `api` into the engine request and how to answer
/// it, keeping background `store: false` Responses in `temporary`.
fn prepare_retaining(
    api: Api,
    body: &[u8],
    thinking: bool,
    (store, temporary): (Option<&Arc<ResponseStore>>, Option<&Arc<Temporary>>),
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
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
            let prepared =
                prepare_responses_retaining(body, thinking, store, temporary, cipher, model)?;
            Ok((
                prepared.stream,
                GenerationRequest::Chat(prepared.request),
                Protocol::Responses(Arc::new(prepared.echo)),
            ))
        }
    }
}

/// `GET` and `DELETE /v1/responses/{id}`, `GET .../{id}/input_items` and
/// `POST .../{id}/cancel`.
///
/// `rest` is the path after `/v1/responses/`. Unknown, malformed, deleted,
/// expired and never-stored IDs are all the same 404, and so is every ID but
/// a retained temporary one when no store is configured: none of them names a
/// response this server can produce. See [`locate`] for which store answers.
async fn stored_response(
    state: &AppState,
    method: &Method,
    rest: &str,
    query: Option<&str>,
    headers: &axum::http::HeaderMap,
) -> Response {
    let (id, items, cancel) = match rest.split_once('/') {
        None if *method != Method::POST => (rest, false, false),
        Some((id, "input_items")) if *method == Method::GET => (id, true, false),
        Some((id, "cancel")) if *method == Method::POST => (id, false, true),
        _ => {
            return error_response(StatusCode::NOT_FOUND, "route not found", state, headers).await;
        }
    };
    let not_found = format!("response with id {id:?} not found");
    if state.responses.is_none() && state.temporary.is_none() {
        let message = format!("{not_found}: this server was started without --response-store");
        return error_response(StatusCode::NOT_FOUND, &message, state, headers).await;
    }
    let outcome = match stored_request(method, items, query) {
        Ok(Retrieval::Stream(after)) => {
            return stream_events(state, id.to_owned(), after, headers).await;
        }
        Ok(retrieval) => {
            let page = retrieval.into_page();
            let (id, method) = (id.to_owned(), method.clone());
            let background = Arc::clone(&state.background);
            let (temporary, durable) = (state.temporary.clone(), state.responses.clone());
            tokio::task::spawn_blocking(move || {
                let located = locate(temporary.as_ref(), durable.as_ref(), &background, &id)?;
                let store = &located.store;
                if cancel {
                    background.cancel(store, &id)
                } else if method == Method::DELETE {
                    let found = background.delete(store, &id)?;
                    if let Some(temporary) = &located.temporary {
                        temporary.forget(&id);
                    }
                    Ok(found.then(|| deleted(&id)))
                } else {
                    stored_answer(store, &method, &id, page.as_ref())
                }
            })
            .await
            .unwrap_or_else(|error| {
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("response store task failed: {error}"),
                ))
            })
        }
        Err(message) => Err((StatusCode::BAD_REQUEST, message)),
    };
    match outcome {
        Ok(Some(value)) => json_response(StatusCode::OK, value, state, headers).await,
        Ok(None) => error_response(StatusCode::NOT_FOUND, &not_found, state, headers).await,
        Err((status, message)) => error_response(status, &message, state, headers).await,
    }
}

/// Why an ID that is not a temporary response is never found without a store.
const WITHOUT_STORE: &str = "this server was started without --response-store, so it keeps only \
     background store=false responses, and only briefly";

/// The store that answers for response `id`.
#[derive(Debug)]
struct Located {
    store: Arc<ResponseStore>,
    /// Set when that is the temporary store.
    temporary: Option<Arc<Temporary>>,
}

/// Which store answers for response `id`: the temporary one while it keeps
/// `id`, otherwise the configured one. Never both, so a temporary response is
/// never looked up in, or confused with, the durable store.
///
/// A temporary response whose retention has ended is deleted here and is
/// then the same 404 as any unknown ID. Blocks on the store.
fn locate(
    temporary: Option<&Arc<Temporary>>,
    durable: Option<&Arc<ResponseStore>>,
    background: &Background,
    id: &str,
) -> Result<Located, Failure> {
    let not_found = format!("response with id {id:?} not found");
    if let Some(temporary) = temporary {
        match temporary.holds(background, id) {
            Some(true) => {
                return Ok(Located {
                    store: Arc::clone(temporary.store()),
                    temporary: Some(Arc::clone(temporary)),
                });
            }
            Some(false) => return Err((StatusCode::NOT_FOUND, not_found)),
            None => {}
        }
    }
    durable.map_or_else(
        || {
            Err((
                StatusCode::NOT_FOUND,
                format!("{not_found}: {WITHOUT_STORE}"),
            ))
        },
        |store| {
            Ok(Located {
                store: Arc::clone(store),
                temporary: None,
            })
        },
    )
}

/// Validate the query of a stored-response request.
///
/// `GET /v1/responses/{id}` streams with `stream=true`, resuming after
/// `starting_after` (see [`resume::retrieval`]); `include` has nothing to
/// add. Other parameters are ignored.
fn stored_request(method: &Method, items: bool, query: Option<&str>) -> Result<Retrieval, String> {
    if items {
        return ItemPage::parse(query)
            .map(Retrieval::Items)
            .map_err(|error| match error {
                PageError::Invalid(message) | PageError::AfterNotFound(message) => message,
            });
    }
    if *method == Method::GET {
        return resume::retrieval(query);
    }
    Ok(Retrieval::Document)
}

/// The answer to a successful `DELETE /v1/responses/{id}`.
fn deleted(id: &str) -> serde_json::Value {
    json!({"id":id,"object":"response.deleted","deleted":true})
}

/// Read, list or delete stored response `id`; `Ok(None)` is a 404.
fn stored_answer(
    store: &ResponseStore,
    method: &Method,
    id: &str,
    page: Option<&ItemPage>,
) -> Result<Option<serde_json::Value>, (StatusCode, String)> {
    let failed = |error: std::io::Error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("response store failed: {error}"),
        )
    };
    if *method == Method::DELETE {
        return Ok(background::delete_unowned(store, id)?.then(|| deleted(id)));
    }
    let Some(stored) = store.load(id).map_err(failed)? else {
        return Ok(None);
    };
    let Some(page) = page else {
        return Ok(Some(stored.response));
    };
    match page.list(&stored.input_items) {
        Ok(list) => Ok(Some(list)),
        Err(PageError::AfterNotFound(after)) => Err((
            StatusCode::NOT_FOUND,
            format!("input item with id {after:?} not found in response {id}"),
        )),
        Err(PageError::Invalid(message)) => Err((StatusCode::BAD_REQUEST, message)),
    }
}

/// `GET /v1/models`: the one loaded model and the context the engine admitted
/// for it.
///
/// `context_length` (the name `OpenRouter` and LM Studio read) and
/// `max_model_len` (vLLM's) are both the per-request admission limit from
/// `BonsaiInfo::context`, prompt plus output, not the checkpoint's training
/// length: it is what a request can actually use on this machine.
fn models_json(model: &str, context: usize, created: u64) -> serde_json::Value {
    json!({"object":"list","data":[{"id":model,"object":"model","created":created,"owned_by":"local","context_length":context,"max_model_len":context}]})
}

async fn run_async(args: Args) -> crate::Result<()> {
    let cipher = args
        .reasoning_key
        .as_ref()
        .map(|path| {
            ReasoningCipher::open(path)
                .map(Arc::new)
                .map_err(|error| crate::Error::InvalidArgument(error.to_string()))
        })
        .transpose()?;
    // Opened before the model loads, so a bad directory fails in milliseconds.
    let responses = args
        .response_store
        .as_ref()
        .map(|dir| {
            ResponseStore::open(dir).map(Arc::new).map_err(|error| {
                crate::Error::InvalidArgument(format!(
                    "cannot open --response-store {}: {error}",
                    dir.display()
                ))
            })
        })
        .transpose()?;
    // Also before the model loads: a wrong head fails in milliseconds.
    let decisions = decisions::Decisions::start(args.decision_head.as_deref())?;
    // Background `store: false` Responses are kept only in a private store of
    // this process; without one they are refused, and nothing else changes.
    let temporary = Temporary::create()
        .map_err(|error| {
            eprintln!(
                "warning: background requests with store=false are unavailable: cannot create a \
                 private temporary response store: {error}"
            );
        })
        .ok();
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
        created: response::unix_now(),
        context,
        thinking: args.thinking,
        api_key: args.api_key.map(Into::into),
        alt_svc: resources
            .tls
            .as_ref()
            .map(|_| Arc::from(format!("h3=\":{}\"; ma=86400", args.port))),
        depth: QueueDepth::default(),
        stall: args.stall,
        responses,
        cipher,
        background: Arc::default(),
        temporary: temporary.clone(),
        decisions: decisions.clone(),
    };
    let address = SocketAddr::new(args.host, args.port);
    let h3_task = if let Some((cert, key)) = resources.tls {
        Some(tokio::spawn(serve_h3(address, state.clone(), cert, key)))
    } else {
        None
    };
    let listener = tokio::net::TcpListener::bind(address).await?;
    // Started once nothing can fail early, since it holds the store until
    // shutdown closes it.
    if let Some(temporary) = &temporary
        && let Err(error) = temporary.reap(Arc::clone(&state.background))
    {
        // Expiry still happens on access; unread responses wait for shutdown.
        eprintln!("warning: temporary responses expire only when accessed: {error}");
    }
    eprintln!("Bonsai server listening on http://{address}");
    let background = Arc::clone(&state.background);
    let app = Router::new().fallback(any(route)).with_state(state);
    // Background jobs are settled and cancelled as soon as shutdown begins,
    // while axum is still draining foreground connections.
    let closing = Arc::clone(&background);
    let closing_decisions = decisions.clone();
    let served = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown().await;
            // Waiting decisions end at once, cancelling their engine work,
            // so they never hold up the drain.
            decisions::close(closing_decisions.as_deref());
            let _ = tokio::task::spawn_blocking(move || closing.close()).await;
        })
        .await
        .map_err(crate::Error::Io);
    // Again, for a server that stopped without the signal; settling is idempotent.
    decisions::close(decisions.as_deref());
    let closing = Arc::clone(&background);
    let _ = tokio::task::spawn_blocking(move || closing.close()).await;
    // Each cancelled job still holds its engine slot and lease until the worker
    // lets go of it, which is at most one prefill chunk or decode step away.
    background.drained().await;
    // No job writes any more: temporary responses are discarded, never kept
    // across a restart.
    if let Some(temporary) = temporary {
        let _ = tokio::task::spawn_blocking(move || temporary.close()).await;
    }
    if let Some(task) = h3_task {
        task.abort();
    }
    served
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

#[cfg(test)]
#[allow(clippy::expect_used)]
mod store_tests;

#[cfg(test)]
#[allow(clippy::expect_used)]
mod background_tests;

#[cfg(test)]
#[allow(clippy::expect_used)]
mod resume_tests;
