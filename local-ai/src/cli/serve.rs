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
use http_body_util::{BodyExt, LengthLimitError, Limited};
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;

use local_engine::{Engine, EngineHandle};

use crate::resources::Resources;

mod background;
mod backpressure;
mod batches;
mod chunked;
mod compaction;
mod decisions;
mod embeddings;
mod evals;
mod hosted;
mod http3;
mod journal;
mod options;
mod reasoning_crypto;
mod request;
mod response;
mod responses;
mod resume;
mod services;
mod sse;
mod store;
mod tool_loop;
mod vector_stores;

#[cfg(test)]
use self::request::message_text;
#[cfg(test)]
use axum::http::HeaderMap;
#[cfg(test)]
use local_engine::Event;

use self::background::{Background, Failure, Resumed, Temporary};
use self::chunked::start_chunked;
use self::http3::serve_h3;
use self::options::{parse, usage};
use self::reasoning_crypto::ReasoningCipher;
use self::request::{GenerationRequest, prepare_generation};
use self::response::{
    Protocol, Reply, add_alt_svc, error_response, error_status, error_status_message,
    json_response, queue_full_response, wants_zstd,
};
use self::responses::prepare_responses_conversing;
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
    /// Hosted tool permissions and bounds; none by default.
    tools: options::ToolArgs,
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
    /// Native durable services, available automatically in the user's data directory.
    services: Option<Arc<services::Services>>,
    batches: Option<Arc<batches::Batches>>,
    vector_stores: Option<Arc<vector_stores::VectorStores>>,
    evals: Option<Arc<evals::Evals>>,
    /// Hosted tools this server may run for Responses; `None` grants none.
    tools: Option<Arc<tool_loop::Tools>>,
    /// Native text embeddings, when the embedding model is installed.
    embeddings: Option<Arc<embeddings::Embeddings>>,
}

/// Requests the engine has accepted and not finished yet: the queue depth as
/// this server sees it.
///
/// It cannot be read off the engine, because `try_send` reports only that the
/// queue was full, and it is only ever read to explain a rejection, so it is
/// tracked here. Every job reaching the queue comes through this module, so it
/// counts queued and running requests together.
#[derive(Clone, Default)]
struct QueueDepth(Arc<Depth>);

#[derive(Default)]
struct Depth {
    count: AtomicUsize,
    /// Told each time a slot is released, for resumptions the full queue
    /// deferred (see [`resume_interrupted`]).
    freed: tokio::sync::Notify,
}

impl QueueDepth {
    fn load(&self) -> usize {
        self.0.count.load(Ordering::SeqCst)
    }

    /// Take a queue slot. The returned guard releases it on drop, so an early
    /// return, a failed generation and a disconnected client all release it.
    fn admit(&self) -> Admitted {
        self.0.count.fetch_add(1, Ordering::SeqCst);
        Admitted(Arc::clone(&self.0))
    }

    /// Wait until a slot is released. A release with nobody waiting is kept
    /// for the next wait, so none is missed between a full queue and the wait.
    async fn freed(&self) {
        self.0.freed.notified().await;
    }
}

struct Admitted(Arc<Depth>);

impl Drop for Admitted {
    fn drop(&mut self) {
        self.0.count.fetch_sub(1, Ordering::SeqCst);
        self.0.freed.notify_one();
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
    if services::matches(parts.uri.path()) {
        let response = services::handle(&state, &parts, body).await;
        if parts.method == Method::DELETE
            && parts.uri.path().starts_with("/v1/files/")
            && response.status().is_success()
            && let Some(vector_stores) = &state.vector_stores
        {
            // The deleted File left every vector store; drop its documents.
            vector_stores.notify();
        }
        return response;
    }
    if batches::matches(parts.uri.path()) {
        return batches::handle(&state, &parts, body).await;
    }
    if vector_stores::matches(parts.uri.path()) {
        return vector_stores::handle(&state, &parts, body).await;
    }
    if evals::matches(parts.uri.path()) {
        return evals::handle(&state, &parts, body).await;
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
            models_json(
                &state.model,
                state.context,
                state.created,
                state.embeddings.is_some(),
            ),
            &state,
            &parts.headers,
        )
        .await;
    }
    if parts.method == Method::POST && parts.uri.path() == embeddings::ROUTE {
        return embeddings::handle(&state, &parts.headers, body).await;
    }
    if parts.method == Method::POST && parts.uri.path() == compaction::ROUTE {
        return compaction::handle(&state, &parts.headers, body).await;
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
    let body = match Limited::new(body, MAX_REQUEST_BYTES).collect().await {
        Ok(value) => value.to_bytes().to_vec(),
        Err(error) => {
            let status = if error.is::<LengthLimitError>() {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            return error_response(
                status,
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
    let services = state.services.clone();
    if count_tokens {
        let engine = state.engine.clone();
        let counted = tokio::task::spawn_blocking(move || {
            let request = responses::prepare_input_tokens_conversing(
                &body,
                thinking,
                store.as_ref(),
                services.as_deref().map(services::Services::store),
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
    let prepared = if matches!(api, Api::Responses) {
        // Compaction and reasoning summaries run engine work of their own.
        compaction::prepare_responses(&state, body).await
    } else {
        tokio::task::spawn_blocking(move || {
            prepare_retaining(
                api,
                &body,
                thinking,
                (store.as_ref(), temporary.as_ref()),
                services.as_deref().map(services::Services::store),
                cipher.as_ref(),
                &model,
            )
        })
        .await
        .unwrap_or_else(|error| {
            Err(crate::Error::Generation(format!(
                "request preparation failed: {error}"
            )))
        })
    };
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
    let (stream, mut request, protocol) = prepared;
    let tool_loop = match hosted_loop(&state, &protocol, &mut request) {
        Ok(tool_loop) => tool_loop,
        Err(message) => {
            return error_response(StatusCode::BAD_REQUEST, &message, &state, &parts.headers).await;
        }
    };
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
        Ok(events) => match tool_loop {
            Some(tool_loop) => tool_loop::start(tool_loop, events),
            None => events,
        },
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

/// The tool loop for a Response that declared hosted tools, refusing one
/// whose tools this server was not permitted to run.
fn hosted_loop(
    state: &AppState,
    protocol: &Protocol,
    request: &mut GenerationRequest,
) -> Result<Option<tool_loop::Loop>, String> {
    let (Protocol::Responses(echo), GenerationRequest::Chat(request)) = (protocol, request) else {
        return Ok(None);
    };
    if echo.kinds().all_hosted().next().is_none() {
        return Ok(None);
    }
    tool_loop::check(state.tools.as_deref(), echo.kinds())?;
    let Some(tools) = &state.tools else {
        return Ok(None);
    };
    let engine = state.engine.clone();
    // Every round renders the same growing conversation, so one prompt-cache
    // affinity lets each round reuse the previous one's prefix.
    request
        .session
        .get_or_insert_with(|| response::new_id("tools_"));
    Ok(Some(tool_loop::Loop {
        submit: Box::new(move |request| engine.chat(request)),
        request: request.clone(),
        toolbox: tools.toolbox.clone(),
        kinds: echo.kinds().clone(),
        bounds: tools.bounds,
        max_calls: echo.max_tool_calls(),
    }))
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
    let resuming = state.clone();
    let subscribed = tokio::task::spawn_blocking(move || {
        let state = resuming;
        let background = &state.background;
        let located = locate(
            state.temporary.as_ref(),
            state.responses.as_ref(),
            background,
            &id,
        )?;
        if adopt_orphan(&state, &located, &id)? {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "response {id} was interrupted by a restart and is waiting for an engine \
                     queue slot to be resumed; retry shortly"
                ),
            ));
        }
        resume::subscribe(background, &located.store, &id, after)
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
    prepare_retaining(api, body, thinking, (store, None), None, cipher, model)
}

/// Parse a request body for `api` into the engine request and how to answer
/// it, keeping background `store: false` Responses in `temporary`.
fn prepare_retaining(
    api: Api,
    body: &[u8],
    thinking: bool,
    (store, temporary): (Option<&Arc<ResponseStore>>, Option<&Arc<Temporary>>),
    conversations: Option<&local_services::Store>,
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
            let prepared = prepare_responses_conversing(
                body,
                thinking,
                store,
                temporary,
                conversations,
                cipher,
                model,
            )?;
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
            let resuming = state.clone();
            tokio::task::spawn_blocking(move || {
                let located = locate(temporary.as_ref(), durable.as_ref(), &background, &id)?;
                let store = &located.store;
                if method == Method::GET && page.is_none() {
                    // Retrieving an orphaned job resumes it; one waiting for a
                    // queue slot is reported as it stands.
                    adopt_orphan(&resuming, &located, &id)?;
                }
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

/// Resume interrupted background response `id` in `store` here and start its
/// pump: `Ok(true)` when the full engine queue (or shutdown) leaves it
/// pending with no writer. Blocks on the store and on submission.
fn resume_background(
    state: &AppState,
    store: &Arc<ResponseStore>,
    id: &str,
) -> Result<bool, Failure> {
    let (engine, depth) = (state.engine.clone(), state.depth.clone());
    let resumed = state.background.resume(
        store,
        id,
        (state.thinking, state.cipher.as_ref()),
        move |request| {
            let mut events = engine.chat(request)?;
            let cancel = events.cancel_handle();
            // No deadline: the job has no client to stall, and a closed
            // channel is `None`.
            let next = move || events.next_signal(None).ok().flatten();
            Ok((next, move || cancel.cancel(), Some(depth.admit())))
        },
    )?;
    Ok(match resumed {
        Resumed::Started(pump) => {
            tokio::task::spawn_blocking(move || pump.run());
            false
        }
        Resumed::Settled | Resumed::Owned => false,
        Resumed::Deferred => true,
    })
}

/// Resume response `id` if it is a durable background Response left pending
/// by a writer that is gone (another server sharing the store died): `true`
/// when it waits for a queue slot. Nothing for any other response, so
/// reading one never creates a lease sidecar. Blocks on the store.
fn adopt_orphan(state: &AppState, located: &Located, id: &str) -> Result<bool, Failure> {
    if located.temporary.is_some() || state.background.watch(id).is_some() {
        return Ok(false);
    }
    let orphan = matches!(
        located.store.load(id),
        Ok(Some(stored)) if stored.request.is_some() && background::pending(&stored.response)
    );
    if orphan {
        resume_background(state, &located.store, id)
    } else {
        Ok(false)
    }
}

/// Resume the interrupted background Responses `ids` in order, returning
/// those the full engine queue deferred.
async fn resume_each(
    state: &AppState,
    store: &Arc<ResponseStore>,
    mut ids: std::collections::VecDeque<String>,
) -> std::collections::VecDeque<String> {
    while let Some(id) = ids.front().cloned() {
        let (resuming, store) = (state.clone(), Arc::clone(store));
        let resumed =
            tokio::task::spawn_blocking(move || resume_background(&resuming, &store, &id)).await;
        match resumed {
            Ok(Ok(true)) => return ids,
            Ok(Ok(false)) => {}
            Ok(Err((_, message))) => eprintln!("background response not resumed: {message}"),
            Err(error) => eprintln!("background response not resumed: {error}"),
        }
        ids.pop_front();
    }
    ids
}

/// Resume every background Response a crash or restart interrupted in
/// `store`: as many as the engine queue admits now, before the server
/// serves, and each deferred one once a queue slot is released, on a task of
/// its own that waits for those releases and stops at shutdown.
async fn resume_interrupted(state: &AppState, store: &Arc<ResponseStore>) {
    let listed = Arc::clone(store);
    let ids = match tokio::task::spawn_blocking(move || listed.resumable()).await {
        Ok(Ok(ids)) => ids,
        Ok(Err(error)) => {
            eprintln!("warning: interrupted background responses not listed: {error}");
            return;
        }
        Err(error) => {
            eprintln!("warning: interrupted background responses not listed: {error}");
            return;
        }
    };
    let mut deferred = resume_each(state, store, ids.into()).await;
    if deferred.is_empty() {
        return;
    }
    let (state, store) = (state.clone(), Arc::clone(store));
    tokio::spawn(async move {
        while !deferred.is_empty() && !state.background.closing() {
            state.depth.freed().await;
            deferred = resume_each(&state, &store, deferred).await;
        }
    });
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
fn models_json(model: &str, context: usize, created: u64, embeddings: bool) -> serde_json::Value {
    let mut data = vec![
        json!({"id":model,"object":"model","created":created,"owned_by":"local","context_length":context,"max_model_len":context}),
    ];
    if embeddings {
        data.push(embeddings::model_json(created));
    }
    json!({"object":"list","data":data})
}

// Keep service startup and shutdown ownership together.
#[allow(clippy::too_many_lines)]
async fn run_async(args: Args) -> crate::Result<()> {
    let services = services::Services::open_default()?;
    let vector_stores = match vector_stores::VectorStores::start(services.store()) {
        Ok(vector_stores) => Some(vector_stores),
        // Another server on this data directory owns the index: serve
        // everything else.
        Err(local_services::Error::Conflict(message)) => {
            eprintln!("warning: vector stores are unavailable: {message}");
            None
        }
        Err(error) => {
            return Err(crate::Error::InvalidArgument(format!(
                "cannot start vector stores: {error}"
            )));
        }
    };
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
    // Hosted tools are opened before the model loads too, so a bad root or
    // domain fails in milliseconds; nothing is granted without a flag.
    let tools = open_tools(&args.tools)?;
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
    let engine = engine.into_handle();
    // Beside the Bonsai model's own folder: `models/embeddinggemma2-text`.
    let embeddings = match resources.model.parent().and_then(std::path::Path::parent) {
        Some(models_dir) => embeddings::Embeddings::start(models_dir, engine.activity())?,
        None => None,
    };
    let mut state = AppState {
        engine,
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
        services: Some(services),
        batches: None,
        vector_stores: vector_stores.clone(),
        evals: None,
        tools,
        embeddings,
    };
    let address = SocketAddr::new(args.host, args.port);
    let listener = tokio::net::TcpListener::bind(address).await?;
    let batches = batches::Batches::start(&state)?;
    state.batches = Some(Arc::clone(&batches));
    let evals = evals::Evals::start(&state, &batches)?;
    state.evals = Some(Arc::clone(&evals));
    let h3_task = if let Some((cert, key)) = resources.tls {
        Some(tokio::spawn(serve_h3(address, state.clone(), cert, key)))
    } else {
        None
    };
    // Started once nothing can fail early, since it holds the store until
    // shutdown closes it.
    if let Some(temporary) = &temporary
        && let Err(error) = temporary.reap(Arc::clone(&state.background))
    {
        // Expiry still happens on access; unread responses wait for shutdown.
        eprintln!("warning: temporary responses expire only when accessed: {error}");
    }
    // Before serving, so a client finds every resumed job already running.
    if let Some(store) = state.responses.clone() {
        resume_interrupted(&state, &store).await;
    }
    eprintln!("Bonsai server listening on http://{address}");
    let background = Arc::clone(&state.background);
    let app = Router::new().fallback(any(route)).with_state(state);
    // Background jobs are settled and cancelled as soon as shutdown begins,
    // while axum is still draining foreground connections.
    let closing = Arc::clone(&background);
    let closing_decisions = decisions.clone();
    let closing_batches = Arc::clone(&batches);
    let closing_evals = Arc::clone(&evals);
    let served = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown().await;
            closing_evals.close();
            closing_batches.close();
            // Waiting decisions end at once, cancelling their engine work,
            // so they never hold up the drain.
            decisions::close(closing_decisions.as_deref());
            let _ = tokio::task::spawn_blocking(move || closing.close()).await;
        })
        .await
        .map_err(crate::Error::Io);
    // Again, for a server that stopped without the signal; settling is idempotent.
    decisions::close(decisions.as_deref());
    evals.close();
    batches.close();
    evals.drained().await;
    batches.drained().await;
    // The indexer finishes its current file; an unfinished one resumes on start.
    if let Some(vector_stores) = vector_stores {
        let _ = tokio::task::spawn_blocking(move || vector_stores.close()).await;
    }
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

/// The hosted tools `args` grant, or `None` when they grant nothing.
fn open_tools(args: &options::ToolArgs) -> crate::Result<Option<Arc<tool_loop::Tools>>> {
    if !args.any() {
        return Ok(None);
    }
    let toolbox = local_services::tools::Toolbox::new(
        local_services::tools::Permissions {
            root: args.root.clone(),
            write: args.write,
            shell: args.shell,
            fetch_domains: args.fetch_domains.clone(),
        },
        local_services::tools::Limits {
            call_timeout: args.call_timeout,
            ..local_services::tools::Limits::default()
        },
    )
    .map_err(|error| crate::Error::InvalidArgument(format!("hosted tools: {error}")))?;
    eprintln!(
        "hosted tools enabled: root={} write={} shell={} fetch={:?}",
        toolbox
            .root()
            .map_or_else(|| "none".to_owned(), |root| root.display().to_string()),
        toolbox.permissions().write,
        toolbox.permissions().shell,
        toolbox.permissions().fetch_domains
    );
    Ok(Some(Arc::new(tool_loop::Tools {
        toolbox,
        bounds: tool_loop::Bounds {
            max_iterations: args.max_iterations,
            budget: args.budget,
            call_timeout: args.call_timeout,
        },
    })))
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

#[cfg(test)]
#[allow(clippy::expect_used)]
mod recovery_tests;
