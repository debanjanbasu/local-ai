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

use super::admin::Meter;
use super::response::{Protocol, error_response, error_status, json_response};
use super::responses::{
    Generate, History, PreparedResponses, Summarizer, compaction_resource, prepare_compact,
    prepare_responses_conversing,
};
use super::{AppState, MAX_REQUEST_BYTES, services};

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
            Event::Reasoning(_)
            | Event::ToolCall(_)
            | Event::ToolOutput(_)
            | Event::TokenIds(_) => {}
        }
    }
    Err(crate::Error::Generation(
        "the engine stopped before the summary finished".into(),
    ))
}

/// The engine-backed [`Generate`]: blocks the calling thread (a pump's
/// blocking-pool thread) until the generation ends.
fn blocking_generate(engine: EngineHandle, meter: Meter) -> Generate {
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
                Event::Finished(stats) => {
                    meter.generation(&stats);
                    return Ok(*stats);
                }
                Event::Error(message) => return Err(crate::Error::Generation(message)),
                Event::Reasoning(_)
                | Event::ToolCall(_)
                | Event::ToolOutput(_)
                | Event::TokenIds(_) => {}
            }
        }
        Err(crate::Error::Generation(
            "the engine stopped before the summary finished".into(),
        ))
    })
}

/// The engine-backed [`Summarizer`], for a new or a resumed Response; each
/// summary generation is recorded by `meter`.
pub(super) fn summarizer(engine: EngineHandle, meter: Meter) -> Summarizer {
    Summarizer(blocking_generate(engine, meter))
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

/// Whether compacted `history` counts fewer prompt tokens, as `template`
/// counts them, than the `tokens` the window it replaces counted.
///
/// A summary of a short history can outgrow the items it replaced; such a
/// compaction is discarded rather than handed back as a longer prompt.
async fn shrinks(
    engine: &EngineHandle,
    template: ChatRequest,
    history: &History,
    tokens: usize,
) -> crate::Result<bool> {
    let (engine, messages) = (engine.clone(), history.messages()?);
    tokio::task::spawn_blocking(move || Ok(counter(engine, template)(messages)? < tokens))
        .await
        .map_err(|error| crate::Error::Generation(error.to_string()))?
}

/// Prepare a `POST /v1/responses` body for generation: parse and resolve it,
/// compact it first when `context_management` says it is due, and attach the
/// summarizer `reasoning.summary` needs.
pub(super) async fn prepare_responses(
    state: &AppState,
    body: Vec<u8>,
    meter: &Meter,
) -> crate::Result<(bool, local_engine::ChatRequest, Protocol)> {
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
        prepared = compact_if_due(state, prepared, entry.threshold, meter).await?;
    }
    if prepared.echo.summarizes() {
        prepared
            .echo
            .set_summarizer(summarizer(state.engine.clone(), meter.clone()));
    }
    Ok((
        prepared.stream,
        prepared.request,
        Protocol::Responses(Arc::new(prepared.echo)),
    ))
}

/// Compact `prepared` when its prompt has reached `threshold` or leaves less
/// than its `max_output_tokens` of the context, before it is generated.
async fn compact_if_due(
    state: &AppState,
    prepared: PreparedResponses,
    threshold: Option<usize>,
    meter: &Meter,
) -> crate::Result<PreparedResponses> {
    let Some(cipher) = state.cipher.clone() else {
        return Err(crate::Error::InvalidArgument(
            "context_management needs --reasoning-key".into(),
        ));
    };
    let (engine, context) = (state.engine.clone(), state.context);
    let (mut prepared, pass, tokens) = tokio::task::spawn_blocking(move || {
        let count = counter(engine, prepared.request.clone());
        let tokens = count(prepared.request.messages.clone())?;
        let reserve = prepared.request.max_tokens;
        if !should_compact(tokens, threshold, reserve, context) {
            return Ok((prepared, None, tokens));
        }
        let budget = threshold.unwrap_or_else(|| context.saturating_sub(reserve));
        let pass = prepared.history.compaction(
            budget / KEEP_DIVISOR,
            prepared.request.tools.clone(),
            prepared.request.session.clone(),
            count,
        )?;
        Ok::<_, crate::Error>((prepared, pass, tokens))
    })
    .await
    .map_err(|error| crate::Error::Generation(error.to_string()))??;
    let Some(pass) = pass else {
        // Nothing can be summarized; the prompt is generated as it is, and
        // fails as over-long if it is.
        return Ok(prepared);
    };
    let _admitted = state.depth.admit();
    let (summary, measured) = generate(&state.engine, pass.request.clone()).await?;
    meter.spawn_generation(&measured);
    // It ran for this Response, so its tokens are in its usage even when
    // its item is discarded below.
    prepared.echo.spend(&measured);
    let (item, history) = pass.finish(&summary, &prepared.history, &cipher, &state.model)?;
    // A compaction that does not shorten the prompt is discarded, like one
    // with nothing to summarize.
    if shrinks(&state.engine, prepared.request.clone(), &history, tokens).await? {
        prepared.compacted(item, history)?;
    }
    Ok(prepared)
}

/// `POST /v1/responses/compact`.
pub(super) async fn handle(
    state: &AppState,
    headers: &HeaderMap,
    body: Body,
    meter: &Meter,
) -> Response {
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
    match compact(state, body, meter).await {
        Ok(resource) => json_response(StatusCode::OK, resource, state, headers).await,
        Err(error) => {
            error_response(error_status(&error), &error.to_string(), state, headers).await
        }
    }
}

async fn compact(
    state: &AppState,
    body: Vec<u8>,
    meter: &Meter,
) -> crate::Result<serde_json::Value> {
    let (store, cipher) = (state.responses.clone(), state.cipher.clone());
    let (engine, model) = (state.engine.clone(), Arc::clone(&state.model));
    let (prepared, pass, template, tokens) = tokio::task::spawn_blocking(move || {
        let prepared = prepare_compact(&body, store.as_ref(), cipher.as_ref(), &model)?;
        let template = local_engine::compaction::compaction_request(
            Vec::new(),
            Vec::new(),
            prepared.session.clone(),
        );
        let count = counter(engine, template.clone());
        let tokens = count(prepared.history.messages()?)?;
        let pass = prepared.history.compaction(
            tokens / KEEP_DIVISOR,
            Vec::new(),
            prepared.session.clone(),
            count,
        )?;
        Ok::<_, crate::Error>((prepared, pass, template, tokens))
    })
    .await
    .map_err(|error| crate::Error::Generation(error.to_string()))??;
    let Some(pass) = pass else {
        // Nothing to summarize: the window is already as compact as it gets.
        return Ok(compaction_resource(&prepared.items, None));
    };
    let _admitted = state.depth.admit();
    let (summary, measured) = generate(&state.engine, pass.request.clone()).await?;
    meter.spawn_generation(&measured);
    let mut output = pass.kept().to_vec();
    let (item, history) =
        pass.finish(&summary, &prepared.history, &prepared.cipher, &state.model)?;
    if !shrinks(&state.engine, template, &history, tokens).await? {
        // The summary outgrew what it replaced: the window comes back
        // unchanged, with the usage the summary generation still cost.
        return Ok(compaction_resource(&prepared.items, Some(&measured)));
    }
    output.push(item);
    Ok(compaction_resource(&output, Some(&measured)))
}
