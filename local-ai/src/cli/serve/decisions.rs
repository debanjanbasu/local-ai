//! EXPERIMENTAL: `POST /v1/experimental/decisions`, the opt-in HTTP bridge to
//! the native judgment-head decisions ([`local_engine::EngineHandle::decide`]).
//!
//! Served only when the server was started with
//! `--experimental-decision-head FILE`; the head is never loaded otherwise.
//!
//! # Why this is not `POST /v1/decisions`
//!
//! `OpenAI`'s `createDecision` response (`DecisionResponse` and
//! `AnswerResource` in the pinned `OpenAPI` spec, and the Decisions guide's
//! "Interpret the answers") requires a `confidence` on every `choice` and
//! `score` answer, separate from the probability distribution, and may answer
//! a question with a `refusal`. The judgment head produces neither: it
//! returns its softmax over the options and nothing else, and the loaded
//! head's `calibration_scope` (or [`UNKNOWN_CALIBRATION_SCOPE`]) says what
//! even those probabilities are not. A
//! `confidence` here could only be invented, so `/v1/decisions` is refused
//! with an explanation and this route answers in its own shape instead.
//!
//! # The bridge
//!
//! The request borrows the Decisions request's vocabulary (`input`,
//! `questions` of type `predicate`, `choice` or `score`, optional `name`),
//! restricted to what the native renderer reproduces exactly: a text-string
//! `input` (the rendered `State:`), `instructions` (the `Question:`), and
//! option values or level labels without descriptions. Anything else —
//! messages, images, descriptions, more choices than option letters — is
//! refused rather than approximated.
//!
//! A head declaring a renderer outside [`DECISION_RENDERERS`] is refused
//! at startup. Each response's `experimental` object reports the renderer,
//! whether the head declared it (`renderer_source` `"artifact"`) or a legacy
//! head left it assumed (`"assumed"`), and the head's own calibration scope,
//! all taken from the decisions that answered it.
//!
//! Each question is one native decision over the same input. Questions run
//! one after another on the loaded model (no second model), each taking one
//! slot of the engine's bounded FIFO queue while it waits or runs, so a
//! many-question request never holds more than one engine slot and other
//! requests interleave with it. At most [`MAX_CONCURRENT`] decision requests
//! are admitted at once; more are answered `503` with `Retry-After`.
//!
//! A client that goes away drops the handler, which drops the
//! [`local_engine::PendingDecision`] and so cancels the decision, queued or
//! running. Shutdown ([`Decisions::close`]) refuses new requests and ends
//! every waiting one with a `503`, cancelling its decision the same way. No
//! wait polls: each one is a future woken by the engine or by shutdown.

use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;
use http_body_util::BodyExt;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};

use local_engine::judgment::{
    DECISION_RENDERER, DECISION_RENDERERS, Decision, DecisionKind, DecisionRequest, DecisionValue,
    JudgmentHead, MAX_DECISION_OPTIONS, UNKNOWN_CALIBRATION_SCOPE,
};

use super::response::{error_response, error_status, json_response, queue_full_response};
use super::{AppState, MAX_REQUEST_BYTES};

/// The experimental route.
pub(super) const ROUTE: &str = "/v1/experimental/decisions";

/// `OpenAI`'s route, which this server refuses with an explanation.
pub(super) const OPENAI_ROUTE: &str = "/v1/decisions";

/// The CLI flag that enables [`ROUTE`].
pub(super) const FLAG: &str = "--experimental-decision-head";

/// Decision requests admitted at once.
///
/// Each one holds at most one engine queue slot at a time, so this is the
/// engine's waiting capacity: more could only wait for a slot, or fail with
/// a full queue after earlier questions already ran.
pub(super) const MAX_CONCURRENT: usize = local_engine::resources::SERVE_QUEUE;

/// Questions one request may ask: `OpenAI`'s `questions` `maxItems`.
const MAX_QUESTIONS: usize = 200;

/// Score levels per question: `OpenAI`'s `levels` `minItems` and `maxItems`.
const SCORE_LEVELS: std::ops::RangeInclusive<usize> = 2..=10;

/// Choices per question: `OpenAI`'s `minItems`, and the native renderer's
/// option letters `A`..`Z` (`OpenAI` allows 255).
const CHOICES: std::ops::RangeInclusive<usize> = 2..=MAX_DECISION_OPTIONS;

/// Served with a `503` so a rejected client can retry instead of guessing.
const RETRY_AFTER_SECONDS: u32 = 1;

/// What every answer omits, and why; echoed in each response.
const LIMITATIONS: &str = "experimental judgment-head probabilities only: no confidence and no \
     refusal are produced, so this is not the OpenAI Decisions API";

/// The decision service: the opt-in head, admission, and shutdown.
pub(super) struct Decisions {
    head: Arc<JudgmentHead>,
    slots: Arc<Semaphore>,
    closing: watch::Sender<bool>,
}

impl Decisions {
    /// A service answering with `head`.
    pub(super) fn new(head: JudgmentHead) -> Self {
        Self {
            head: Arc::new(head),
            slots: Arc::new(Semaphore::new(MAX_CONCURRENT)),
            closing: watch::Sender::new(false),
        }
    }

    /// Load and validate the head at `path` for a model of `width`, so a
    /// wrong file (including one declaring a renderer the native decisions
    /// do not reproduce) fails at startup rather than on the first request.
    pub(super) fn open(path: &Path, width: usize) -> crate::Result<Self> {
        let head = JudgmentHead::open(path).map_err(|error| {
            crate::Error::InvalidArgument(format!("cannot open {FLAG} {}: {error}", path.display()))
        })?;
        if head.width() != width {
            return Err(crate::Error::InvalidArgument(format!(
                "{FLAG} {}: head width {} does not match the model width {width}",
                path.display(),
                head.width()
            )));
        }
        if let Some(renderer) = head.renderer()
            && !DECISION_RENDERERS.contains(&renderer)
        {
            return Err(crate::Error::InvalidArgument(format!(
                "{FLAG} {}: head renderer {renderer:?} is not supported natively",
                path.display()
            )));
        }
        Ok(Self::new(head))
    }

    /// What the loaded head says about itself, for the startup notice.
    fn provenance(&self) -> String {
        let renderer = self.head.renderer().map_or_else(
            || format!("renderer {DECISION_RENDERER} (assumed: legacy head declares none)"),
            |renderer| format!("renderer {renderer} (declared by the head)"),
        );
        let scope = self
            .head
            .calibration_scope()
            .unwrap_or(UNKNOWN_CALIBRATION_SCOPE);
        format!("{renderer}; calibration scope: {scope}")
    }

    /// The service `--experimental-decision-head` asks for, if any, with
    /// its experimental status and the head's own provenance announced.
    pub(super) fn start(path: Option<&Path>) -> crate::Result<Option<Arc<Self>>> {
        let Some(path) = path else {
            return Ok(None);
        };
        let decisions = Self::open(path, local_engine::bonsai::WIDTH)?;
        eprintln!(
            "EXPERIMENTAL decisions at POST {ROUTE} (head probabilities only; no confidence or \
             refusal; not the OpenAI Decisions API): {}",
            decisions.provenance()
        );
        Ok(Some(Arc::new(decisions)))
    }

    /// Refuse new requests and end waiting ones; idempotent.
    pub(super) fn close(&self) {
        self.closing.send_replace(true);
    }

    fn closing(&self) -> bool {
        *self.closing.borrow()
    }

    /// One admission slot, or `None` when [`MAX_CONCURRENT`] are taken.
    fn admit(&self) -> Option<OwnedSemaphorePermit> {
        Arc::clone(&self.slots).try_acquire_owned().ok()
    }
}

/// Close `decisions`, if the server has them; see [`Decisions::close`].
pub(super) fn close(decisions: Option<&Decisions>) {
    if let Some(decisions) = decisions {
        decisions.close();
    }
}

/// A Decisions-style request body, restricted to what is supported.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestBody {
    /// Accepted and ignored: the loaded model answers.
    #[allow(dead_code)]
    model: Option<String>,
    input: Value,
    questions: Vec<Question>,
    /// Accepted and ignored: an opaque caller identifier.
    #[allow(dead_code)]
    safety_identifier: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum Question {
    Predicate {
        name: Option<String>,
        instructions: String,
    },
    Choice {
        name: Option<String>,
        instructions: String,
        choices: Vec<ChoiceOption>,
    },
    Score {
        name: Option<String>,
        instructions: String,
        levels: Vec<Level>,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChoiceOption {
    value: ChoiceValue,
    description: Option<String>,
}

/// Choice values are typed: a string and a boolean are distinct.
#[derive(Deserialize)]
#[serde(untagged)]
enum ChoiceValue {
    Bool(bool),
    Text(String),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Level {
    label: String,
    description: Option<String>,
}

/// The answer shape a question asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Predicate,
    Choice,
    Score,
}

/// One validated question.
#[derive(Debug)]
struct Asked {
    name: Option<String>,
    shape: Shape,
    instructions: String,
    kind: DecisionKind,
}

/// A validated request: the input, once, and its questions.
#[derive(Debug)]
pub(super) struct Prepared {
    state: String,
    questions: Vec<Asked>,
}

impl Asked {
    /// The native request over `state`, built only when it is submitted so
    /// a long input is copied for one question at a time.
    fn request(&self, state: &str) -> DecisionRequest {
        DecisionRequest {
            state: state.to_owned(),
            question: self.instructions.clone(),
            kind: self.kind.clone(),
        }
    }

    /// This question's answer from the native `decision`.
    fn answer(&self, decision: &Decision) -> Result<Value, String> {
        if decision
            .options
            .iter()
            .any(|option| !option.probability.is_finite())
            || decision.score.is_some_and(|score| !score.is_finite())
        {
            return Err("the judgment head produced a non-finite probability".into());
        }
        let name = &self.name;
        match self.shape {
            Shape::Predicate => {
                let probability = decision
                    .options
                    .iter()
                    .find(|option| option.option == DecisionValue::Bool(true))
                    .map(|option| option.probability)
                    .ok_or("the predicate decision has no Yes option")?;
                Ok(json!({"type":"predicate","name":name,"probability":probability}))
            }
            Shape::Choice => {
                let probabilities: Vec<Value> = decision
                    .options
                    .iter()
                    .map(|option| {
                        json!({"value":value_json(&option.option),"probability":option.probability})
                    })
                    .collect();
                Ok(json!({
                    "type":"choice",
                    "name":name,
                    "choice":value_json(&decision.value),
                    "probabilities":probabilities,
                }))
            }
            Shape::Score => {
                let score = decision.score.ok_or("the score decision has no score")?;
                let probabilities: Vec<Value> = decision
                    .options
                    .iter()
                    .enumerate()
                    .map(|(index, option)| {
                        json!({
                            "value":index,
                            "label":option.option.rendered(),
                            "probability":option.probability,
                        })
                    })
                    .collect();
                Ok(json!({
                    "type":"score",
                    "name":name,
                    "score":score,
                    "probabilities":probabilities,
                }))
            }
        }
    }
}

fn value_json(value: &DecisionValue) -> Value {
    match value {
        DecisionValue::Text(text) => json!(text),
        DecisionValue::Bool(value) => json!(value),
    }
}

/// Parse and validate a request body, refusing what cannot be answered
/// faithfully. Pure CPU work; no engine is involved.
pub(super) fn prepare(body: &[u8]) -> Result<Prepared, String> {
    let body: RequestBody =
        serde_json::from_slice(body).map_err(|error| format!("invalid JSON body: {error}"))?;
    let state = match body.input {
        Value::String(text) => text,
        Value::Array(_) => {
            return Err(
                "input must be a text string: messages and images are not supported by \
                 the experimental judgment head"
                    .into(),
            );
        }
        _ => return Err("input must be a text string".into()),
    };
    if state.trim().is_empty() {
        return Err("input is empty".into());
    }
    if !(1..=MAX_QUESTIONS).contains(&body.questions.len()) {
        return Err(format!(
            "questions must hold 1..={MAX_QUESTIONS} questions, got {}",
            body.questions.len()
        ));
    }
    let questions = body
        .questions
        .into_iter()
        .enumerate()
        .map(|(index, question)| {
            asked(question).map_err(|error| format!("questions[{index}]: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Prepared { state, questions })
}

/// Validate one question, including everything the native renderer checks
/// about it, so no question runs before a later one is found invalid.
fn asked(question: Question) -> Result<Asked, String> {
    let undescribed = |description: Option<&String>, what: &str| {
        if description.is_some_and(|text| !text.is_empty()) {
            Err(format!(
                "{what} descriptions are not supported: the judgment head's renderer has no \
                 place for them"
            ))
        } else {
            Ok(())
        }
    };
    let (name, shape, instructions, kind) = match question {
        Question::Predicate { name, instructions } => (
            name,
            Shape::Predicate,
            instructions,
            DecisionKind::Predicate,
        ),
        Question::Choice {
            name,
            instructions,
            choices,
        } => {
            if !CHOICES.contains(&choices.len()) {
                return Err(format!(
                    "choices must hold {}..={} values (the renderer letters options A..Z), \
                     got {}",
                    CHOICES.start(),
                    CHOICES.end(),
                    choices.len()
                ));
            }
            let mut values = Vec::with_capacity(choices.len());
            for choice in choices {
                undescribed(choice.description.as_ref(), "choice")?;
                values.push(match choice.value {
                    ChoiceValue::Bool(value) => DecisionValue::Bool(value),
                    ChoiceValue::Text(text) => DecisionValue::Text(text),
                });
            }
            (
                name,
                Shape::Choice,
                instructions,
                DecisionKind::Choice(values),
            )
        }
        Question::Score {
            name,
            instructions,
            levels,
        } => {
            if !SCORE_LEVELS.contains(&levels.len()) {
                return Err(format!(
                    "levels must hold {}..={} levels, got {}",
                    SCORE_LEVELS.start(),
                    SCORE_LEVELS.end(),
                    levels.len()
                ));
            }
            let mut labels = Vec::with_capacity(levels.len());
            for level in levels {
                undescribed(level.description.as_ref(), "level")?;
                labels.push(level.label);
            }
            (
                name,
                Shape::Score,
                instructions,
                DecisionKind::Score(labels),
            )
        }
    };
    let asked = Asked {
        name,
        shape,
        instructions,
        kind,
    };
    // The native checks need a non-empty state; the input's own checks
    // (special tokens included) run at submission.
    asked
        .request("-")
        .render()
        .map_err(|error| error.to_string())?;
    Ok(asked)
}

/// Why a request was not answered.
#[derive(Debug)]
pub(super) enum Failure {
    ShuttingDown,
    QueueFull,
    Cancelled,
    Engine(crate::Error),
    Internal(String),
}

/// The answers to `prepared`, the prompt tokens they read and the head
/// provenance the decisions reported.
#[derive(Debug)]
pub(super) struct Answered {
    answers: Vec<Value>,
    input_tokens: usize,
    provenance: Provenance,
}

/// A decision's renderer and calibration provenance, as the native decision
/// reported it.
#[derive(Debug)]
struct Provenance {
    renderer: &'static str,
    renderer_declared: bool,
    calibration_scope: String,
}

impl Provenance {
    fn of(decision: &Decision) -> Self {
        Self {
            renderer: decision.renderer,
            renderer_declared: decision.renderer_declared,
            calibration_scope: decision.calibration_scope.clone(),
        }
    }
}

/// Answer `prepared`'s questions in order, submitting each with `submit`
/// only once the previous one has resolved.
///
/// `submit` queues one native decision and yields its pending result. When
/// `closing` turns true (or its service is gone) the current decision is
/// dropped, cancelling it, and the request ends; so does dropping this
/// future.
pub(super) async fn answer<S, Submitted, Pending>(
    prepared: &Prepared,
    mut submit: S,
    mut closing: watch::Receiver<bool>,
) -> Result<Answered, Failure>
where
    S: FnMut(DecisionRequest) -> Submitted,
    Submitted: Future<Output = crate::Result<Pending>>,
    Pending: Future<Output = crate::Result<Option<Decision>>>,
{
    let mut answers = Vec::with_capacity(prepared.questions.len());
    let mut input_tokens = 0usize;
    let mut provenance: Option<Provenance> = None;
    for question in &prepared.questions {
        let request = question.request(&prepared.state);
        let decided = async { submit(request).await?.await };
        let decided = tokio::select! {
            biased;
            _ = closing.wait_for(|closing| *closing) => return Err(Failure::ShuttingDown),
            decided = decided => decided,
        };
        let decision = match decided {
            Ok(Some(decision)) => decision,
            Ok(None) => return Err(Failure::Cancelled),
            Err(crate::Error::QueueFull) => return Err(Failure::QueueFull),
            Err(error) => return Err(Failure::Engine(error)),
        };
        input_tokens = input_tokens.saturating_add(decision.prompt_tokens);
        // Every question uses the service's same immutable head.
        provenance.get_or_insert_with(|| Provenance::of(&decision));
        answers.push(question.answer(&decision).map_err(Failure::Internal)?);
    }
    let provenance =
        provenance.ok_or_else(|| Failure::Internal("the request asked no questions".into()))?;
    Ok(Answered {
        answers,
        input_tokens,
        provenance,
    })
}

/// The response body for `answered`; its `experimental` provenance is what
/// the decisions reported for the loaded head.
pub(super) fn response_json(model: &str, answered: Answered) -> Value {
    let tokens = answered.input_tokens;
    let Provenance {
        renderer,
        renderer_declared,
        calibration_scope,
    } = answered.provenance;
    json!({
        "object":"experimental.decision",
        "model":model,
        "answers":Value::Array(answered.answers),
        "usage":{"input_tokens":tokens,"output_tokens":0,"total_tokens":tokens},
        "experimental":{
            "renderer":renderer,
            "renderer_source":if renderer_declared { "artifact" } else { "assumed" },
            "calibration_scope":calibration_scope,
            "limitations":LIMITATIONS,
        },
    })
}

/// Why `POST /v1/decisions` is not served.
pub(super) fn openai_refusal() -> String {
    format!(
        "{OPENAI_ROUTE} is not implemented: OpenAI Decisions answers carry a confidence and may \
         be refusals, which this server's experimental judgment head does not produce. Start the \
         server with {FLAG} FILE and use POST {ROUTE}, which returns the head's probabilities \
         only"
    )
}

/// `POST /v1/experimental/decisions`.
pub(super) async fn handle(state: &AppState, headers: &HeaderMap, body: Body) -> Response {
    let Some(decisions) = state.decisions.clone() else {
        let message = format!("route not found: {ROUTE} is served only with {FLAG} FILE");
        return error_response(StatusCode::NOT_FOUND, &message, state, headers).await;
    };
    if decisions.closing() {
        return shutting_down(state, headers).await;
    }
    let body = match body.collect().await {
        Ok(value) => value.to_bytes(),
        Err(error) => {
            let message = format!("invalid HTTP body: {error}");
            return error_response(StatusCode::BAD_REQUEST, &message, state, headers).await;
        }
    };
    if body.len() > MAX_REQUEST_BYTES {
        let message = "HTTP request is too large";
        return error_response(StatusCode::PAYLOAD_TOO_LARGE, message, state, headers).await;
    }
    let prepared = tokio::task::spawn_blocking(move || prepare(&body))
        .await
        .unwrap_or_else(|error| Err(format!("request preparation failed: {error}")));
    let prepared = match prepared {
        Ok(prepared) => prepared,
        Err(message) => {
            let message = format!("invalid argument: decision: {message}");
            return error_response(StatusCode::BAD_REQUEST, &message, state, headers).await;
        }
    };
    let Some(admitted) = decisions.admit() else {
        return busy(state, headers).await;
    };
    let (engine, head, depth) = (
        state.engine.clone(),
        Arc::clone(&decisions.head),
        state.depth.clone(),
    );
    // Validation, rendering and tokenization of a long input run on the
    // blocking pool; the wait is a future the engine resolves.
    let submit = move |request: DecisionRequest| {
        let (engine, head, depth) = (engine.clone(), Arc::clone(&head), depth.clone());
        async move {
            let pending = tokio::task::spawn_blocking(move || engine.decide(head, &request))
                .await
                .unwrap_or_else(|error| {
                    Err(crate::Error::Generation(format!(
                        "decision submission failed: {error}"
                    )))
                })?;
            let queued = depth.admit();
            Ok(async move {
                let _queued = queued;
                pending.await
            })
        }
    };
    let answered = answer(&prepared, submit, decisions.closing.subscribe()).await;
    drop(admitted);
    match answered {
        Ok(answered) => {
            let value = response_json(&state.model, answered);
            json_response(StatusCode::OK, value, state, headers).await
        }
        Err(Failure::ShuttingDown) => shutting_down(state, headers).await,
        Err(Failure::QueueFull) => queue_full_response(state, headers).await,
        Err(Failure::Cancelled) => {
            let message = "the decision was cancelled";
            error_response(StatusCode::SERVICE_UNAVAILABLE, message, state, headers).await
        }
        Err(Failure::Engine(error)) => {
            error_response(error_status(&error), &error.to_string(), state, headers).await
        }
        Err(Failure::Internal(message)) => {
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &message, state, headers).await
        }
    }
}

async fn shutting_down(state: &AppState, headers: &HeaderMap) -> Response {
    let message = "the server is shutting down";
    error_response(StatusCode::SERVICE_UNAVAILABLE, message, state, headers).await
}

/// Every admission slot is taken.
async fn busy(state: &AppState, headers: &HeaderMap) -> Response {
    let message = format!(
        "experimental decisions are busy: {MAX_CONCURRENT} requests are being answered. Retry \
         after {RETRY_AFTER_SECONDS}s with your own backoff."
    );
    let mut response =
        error_response(StatusCode::SERVICE_UNAVAILABLE, &message, state, headers).await;
    response.headers_mut().insert(
        header::RETRY_AFTER,
        header::HeaderValue::from(RETRY_AFTER_SECONDS),
    );
    response
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
