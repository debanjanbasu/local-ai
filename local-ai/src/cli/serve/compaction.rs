//! The HTTP adapter for compaction and reasoning summaries: runs the summary
//! generations [`super::responses`] plans on the engine.
//!
//! `POST /v1/responses/compact` answers with a `response.compaction` object;
//! a `POST /v1/responses` with `context_management` is compacted here, before
//! its generation is queued, when its prompt is due; a Response with
//! `reasoning.summary` gets the engine-backed [`Summarizer`]. Each summary is
//! one more generation of the loaded model, queued like any request.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use http_body_util::{BodyExt as _, LengthLimitError, Limited};
use tokio_stream::StreamExt;

use local_engine::{ChatMessage, ChatRequest, EngineHandle, Event, Stats};
use local_services::compaction::should_compact;

use super::response::{Protocol, error_response, error_status, json_response};
use super::responses::{
    Generate, PreparedResponses, Summarizer, compaction_resource, prepare_compact,
    prepare_responses_conversing,
};
use super::{AppState, MAX_REQUEST_BYTES, request::GenerationRequest, services};

/// The route this module answers.
pub(super) const ROUTE: &str = "/v1/responses/compact";

/// The share of a token budget a compaction keeps verbatim, as newest items:
/// a quarter, so a compacted prompt is that plus one summary.
const KEEP_DIVISOR: usize = 4;

/// Run `request` to its end on `engine` and collect its visible content.
///
/// Dropping the future drops the event stream, which cancels the generation,
/// so a client that goes away stops it.
async fn generate(engine: &EngineHandle, request: ChatRequest) -> crate::Result<(String, Stats)> {
    let engine = engine.clone();
    // Submission may compile the no-call tool grammar, so it runs off the
    // runtime, like every other submission.
    let mut events = tokio::task::spawn_blocking(move || engine.chat(request))
        .await
        .map_err(|error| crate::Error::Generation(error.to_string()))??;
    let mut text = String::new();
    while let Some(event) = StreamExt::next(&mut events).await {
        match event {
            Event::Content(piece) => text.push_str(&piece),
            Event::Finished(stats) => return Ok((text, *stats)),
            Event::Error(message) => return Err(crate::Error::Generation(message)),
            Event::Reasoning(_) | Event::ToolCall(_) | Event::TokenIds(_) => {}
        }
    }
    Err(crate::Error::Generation(
        "the engine stopped before the summary finished".into(),
    ))
}

/// The engine-backed [`Generate`]: blocks the calling thread (a pump's
/// blocking-pool thread) until the generation ends.
fn blocking_generate(engine: EngineHandle) -> Generate {
    Arc::new(move |request, piece| {
        // Leaving the loop early drops the stream, which cancels the run.
        for event in engine.chat(request)? {
            match event {
                Event::Content(text) => {
                    if !piece(&text) {
                        return Err(crate::Error::Generation(
                            "the client went away during the summary".into(),
                        ));
                    }
                }
                Event::Finished(stats) => return Ok(*stats),
                Event::Error(message) => return Err(crate::Error::Generation(message)),
                Event::Reasoning(_) | Event::ToolCall(_) | Event::TokenIds(_) => {}
            }
        }
        Err(crate::Error::Generation(
            "the engine stopped before the summary finished".into(),
        ))
    })
}

/// Count `messages` as `template` would be counted with them.
fn counter(
    engine: EngineHandle,
    template: ChatRequest,
) -> impl Fn(Vec<ChatMessage>) -> crate::Result<usize> {
    move |messages| {
        engine.count_chat_tokens(&ChatRequest {
            messages,
            ..template.clone()
        })
    }
}

/// Prepare a `POST /v1/responses` body for generation: parse and resolve it,
/// compact it first when `context_management` says it is due, and attach the
/// summarizer `reasoning.summary` needs.
pub(super) async fn prepare_responses(
    state: &AppState,
    body: Vec<u8>,
) -> crate::Result<(bool, GenerationRequest, Protocol)> {
    let (thinking, store) = (state.thinking, state.responses.clone());
    let (cipher, model) = (state.cipher.clone(), Arc::clone(&state.model));
    let temporary = state.temporary.clone();
    let services = state.services.clone();
    let mut prepared = tokio::task::spawn_blocking(move || {
        prepare_responses_conversing(
            &body,
            thinking,
            store.as_ref(),
            temporary.as_ref(),
            services.as_deref().map(services::Services::store),
            cipher.as_ref(),
            &model,
        )
    })
    .await
    .map_err(|error| crate::Error::Generation(format!("request preparation failed: {error}")))??;
    if let Some(entry) = prepared.context_compaction {
        prepared = compact_if_due(state, prepared, entry.threshold).await?;
    }
    if prepared.echo.summarizes() {
        let generate = blocking_generate(state.engine.clone());
        prepared.echo.set_summarizer(Summarizer(generate));
    }
    Ok((
        prepared.stream,
        GenerationRequest::Chat(prepared.request),
        Protocol::Responses(Arc::new(prepared.echo)),
    ))
}

/// Compact `prepared` when its prompt has reached `threshold` or leaves less
/// than its `max_output_tokens` of the context, before it is generated.
async fn compact_if_due(
    state: &AppState,
    prepared: PreparedResponses,
    threshold: Option<usize>,
) -> crate::Result<PreparedResponses> {
    let Some(cipher) = state.cipher.clone() else {
        return Err(crate::Error::InvalidArgument(
            "context_management needs --reasoning-key".into(),
        ));
    };
    let (engine, context) = (state.engine.clone(), state.context);
    let (mut prepared, pass) = tokio::task::spawn_blocking(move || {
        let count = counter(engine, prepared.request.clone());
        let tokens = count(prepared.request.messages.clone())?;
        let reserve = prepared.request.max_tokens;
        if !should_compact(tokens, threshold, reserve, context) {
            return Ok((prepared, None));
        }
        let budget = threshold.unwrap_or_else(|| context.saturating_sub(reserve));
        let pass = prepared.history.compaction(
            budget / KEEP_DIVISOR,
            prepared.request.tools.clone(),
            prepared.request.session.clone(),
            count,
        )?;
        Ok::<_, crate::Error>((prepared, pass))
    })
    .await
    .map_err(|error| crate::Error::Generation(error.to_string()))??;
    let Some(pass) = pass else {
        // Nothing can be summarized; the prompt is generated as it is, and
        // fails as over-long if it is.
        return Ok(prepared);
    };
    let _admitted = state.depth.admit();
    let (summary, _) = generate(&state.engine, pass.request.clone()).await?;
    let (item, history) = pass.finish(&summary, &prepared.history, &cipher, &state.model)?;
    prepared.compacted(item, history)?;
    Ok(prepared)
}

/// `POST /v1/responses/compact`.
pub(super) async fn handle(state: &AppState, headers: &HeaderMap, body: Body) -> Response {
    let body = match Limited::new(body, MAX_REQUEST_BYTES).collect().await {
        Ok(value) => value.to_bytes().to_vec(),
        Err(error) => {
            let status = if error.is::<LengthLimitError>() {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            let message = format!("invalid HTTP body: {error}");
            return error_response(status, &message, state, headers).await;
        }
    };
    match compact(state, body).await {
        Ok(resource) => json_response(StatusCode::OK, resource, state, headers).await,
        Err(error) => {
            error_response(error_status(&error), &error.to_string(), state, headers).await
        }
    }
}

async fn compact(state: &AppState, body: Vec<u8>) -> crate::Result<serde_json::Value> {
    let (store, cipher) = (state.responses.clone(), state.cipher.clone());
    let (engine, model) = (state.engine.clone(), Arc::clone(&state.model));
    let (prepared, pass) = tokio::task::spawn_blocking(move || {
        let prepared = prepare_compact(&body, store.as_ref(), cipher.as_ref(), &model)?;
        let template = local_engine::compaction::compaction_request(
            Vec::new(),
            Vec::new(),
            prepared.session.clone(),
        );
        let count = counter(engine, template);
        let tokens = count(prepared.history.messages()?)?;
        let pass = prepared.history.compaction(
            tokens / KEEP_DIVISOR,
            Vec::new(),
            prepared.session.clone(),
            count,
        )?;
        Ok::<_, crate::Error>((prepared, pass))
    })
    .await
    .map_err(|error| crate::Error::Generation(error.to_string()))??;
    let Some(pass) = pass else {
        // Nothing to summarize: the window is already as compact as it gets.
        return Ok(compaction_resource(&prepared.items, None));
    };
    let _admitted = state.depth.admit();
    let (summary, measured) = generate(&state.engine, pass.request.clone()).await?;
    let mut output = pass.kept().to_vec();
    let (item, _) = pass.finish(&summary, &prepared.history, &prepared.cipher, &state.model)?;
    output.push(item);
    Ok(compaction_resource(&output, Some(&measured)))
}
