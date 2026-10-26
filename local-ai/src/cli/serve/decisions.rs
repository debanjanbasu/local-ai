//! `POST /v1/decisions`: `OpenAI`'s `createDecision`, served only with
//! `--decisions-artifact DIR` and only within a pre-registered, evaluated
//! scope.
//!
//! The directory holds a judgment head, its out-of-scope gate and the
//! calibration artifact that passed the pre-registered acceptance criteria
//! (`docs/DECISIONS_EVALUATION.md` part 3). [`CalibratedHead::open`]
//! verifies the artifact's content hash, the head's and gate's file hashes,
//! its acceptance and its declared scope at startup; any mismatch stops the
//! server before the model loads. Without the flag the route is refused with
//! an explanation.
//!
//! # What is answered
//!
//! Each question is answered in order, in the pinned spec's shape:
//!
//! - a `predicate` question in the declared scope ([`DecisionFamily::of`]:
//!   `CodeReviewer` `needs_comment` or `CommitPackFT` `message_match`, exact
//!   question forms, over a text input starting `Diff:` or
//!   `Context before hunk:`) runs one native decision with the artifact's
//!   head and is answered `{"type":"predicate","name","probability"}` with the
//!   **calibrated** `P(Yes)` — unless [`CalibratedHead::judge`] refuses it:
//!   prompt over the declared token limit, the feature gate's distance above
//!   its threshold, a family whose calibration split found no threshold
//!   meeting the 10% target risk (`CodeReviewer`: always refused), or a
//!   top-label `confidence` below the family's threshold;
//! - every other question (choice, score, other predicates, message or image
//!   inputs) is answered `{"type":"refusal","name"}` without GPU work.
//!
//! The spec's predicate answer has no `confidence` field, so the confidence
//! (the calibrated probability of the more likely answer) decides refusal
//! and is not echoed; choice and score answers, which carry one, are never
//! produced because no choice or score family passed evaluation.
//!
//! Admission: at most [`MAX_CONCURRENT`] requests at once (more are a `503`
//! with `Retry-After`), one engine queue slot per running question, and work
//! a dropped or shut-down request no longer needs is cancelled.

use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;
use bytes::Bytes;
use http_body_util::{BodyExt, LengthLimitError, Limited};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};

use local_engine::judgment::{
    CalibratedHead, Decision, DecisionFamily, DecisionKind, DecisionRequest, Judgment,
};

use super::response::{error_response, error_status, json_response, queue_full_response};
use super::{AppState, MAX_REQUEST_BYTES};

/// `OpenAI`'s route.
pub(super) const ROUTE: &str = "/v1/decisions";

/// The CLI flag that enables [`ROUTE`].
pub(super) const FLAG: &str = "--decisions-artifact";

/// Requests admitted at once: one per engine queue slot, so a full house
/// of single-question requests can all be queued.
pub(super) const MAX_CONCURRENT: usize = local_engine::resources::SERVE_QUEUE;

/// `OpenAI`'s `questions` bounds.
const QUESTIONS: std::ops::RangeInclusive<usize> = 1..=200;

const RETRY_AFTER_SECONDS: u32 = 1;

/// The calibrated decision service.
pub(super) struct Decisions {
    calibrated: Arc<CalibratedHead>,
    slots: Arc<Semaphore>,
    closing: watch::Sender<bool>,
}

impl Decisions {
    pub(super) fn new(calibrated: CalibratedHead) -> Self {
        Self {
            calibrated: Arc::new(calibrated),
            slots: Arc::new(Semaphore::new(MAX_CONCURRENT)),
            closing: watch::Sender::new(false),
        }
    }

    /// Open and verify the artifact directory for a model of `width`.
    pub(super) fn open(dir: &Path, width: usize) -> crate::Result<Self> {
        let calibrated = CalibratedHead::open(dir).map_err(|error| {
            crate::Error::InvalidArgument(format!("cannot open {FLAG} {}: {error}", dir.display()))
        })?;
        if calibrated.head().width() != width {
            return Err(crate::Error::InvalidArgument(format!(
                "{FLAG} {}: head width {} does not match the model width {width}",
                dir.display(),
                calibrated.head().width()
            )));
        }
        Ok(Self::new(calibrated))
    }

    /// The service `--decisions-artifact` asks for, if any.
    pub(super) fn start(dir: Option<&Path>) -> crate::Result<Option<Arc<Self>>> {
        let Some(dir) = dir else {
            return Ok(None);
        };
        let service = Self::open(dir, local_engine::bonsai::WIDTH)?;
        eprintln!(
            "decisions at POST {ROUTE}: calibrated and scope-gated (CodeReviewer needs_comment, \
             always refused; CommitPackFT message_match), artifact {}, head {}, gate {}",
            service.calibrated.content_sha256(),
            service.calibrated.head_sha256(),
            service.calibrated.gate_sha256()
        );
        Ok(Some(Arc::new(service)))
    }

    /// Refuse new requests and end waiting ones; idempotent.
    pub(super) fn close(&self) {
        self.closing.send_replace(true);
    }

    fn closing(&self) -> bool {
        *self.closing.borrow()
    }

    fn admit(&self) -> Option<OwnedSemaphorePermit> {
        Arc::clone(&self.slots).try_acquire_owned().ok()
    }
}

/// Close `service`, if the server has one.
pub(super) fn close(service: Option<&Decisions>) {
    if let Some(service) = service {
        service.close();
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestBody {
    /// Required by the spec; checked by [`super::models::check_generation`],
    /// the loaded model answers.
    model: String,
    input: Value,
    questions: Vec<Question>,
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
        #[allow(dead_code)]
        instructions: String,
        #[allow(dead_code)]
        choices: Vec<Value>,
    },
    Score {
        name: Option<String>,
        #[allow(dead_code)]
        instructions: String,
        #[allow(dead_code)]
        levels: Vec<Value>,
    },
}

/// One question: refused up front, or a native decision of a declared family.
#[derive(Debug)]
enum Asked {
    Refused {
        name: Option<String>,
    },
    InScope {
        name: Option<String>,
        family: DecisionFamily,
        request: DecisionRequest,
    },
}

/// A validated request.
#[derive(Debug)]
pub(super) struct Prepared {
    questions: Vec<Asked>,
}

/// Parse and validate a body; decide each question's scope. Pure CPU work.
pub(super) fn prepare(body: &[u8]) -> Result<Prepared, String> {
    let body: RequestBody =
        serde_json::from_slice(body).map_err(|error| format!("invalid JSON body: {error}"))?;
    super::models::check_generation(Some(&body.model)).map_err(|error| error.to_string())?;
    if !QUESTIONS.contains(&body.questions.len()) {
        return Err(format!(
            "questions must hold {}..={} questions, got {}",
            QUESTIONS.start(),
            QUESTIONS.end(),
            body.questions.len()
        ));
    }
    // Message and image inputs are valid requests this server cannot read:
    // every question is refused.
    let state = match body.input {
        Value::String(text) => Some(text),
        Value::Array(_) => None,
        _ => return Err("input must be a string or an array of messages".into()),
    };
    let questions = body
        .questions
        .into_iter()
        .map(|question| match (question, &state) {
            (Question::Predicate { name, instructions }, Some(state)) => {
                let request = DecisionRequest {
                    state: state.clone(),
                    question: instructions,
                    kind: DecisionKind::Predicate,
                };
                match DecisionFamily::of(&request) {
                    Some(family) => Asked::InScope {
                        name,
                        family,
                        request,
                    },
                    None => Asked::Refused { name },
                }
            }
            (
                Question::Predicate { name, .. }
                | Question::Choice { name, .. }
                | Question::Score { name, .. },
                _,
            ) => Asked::Refused { name },
        })
        .collect();
    Ok(Prepared { questions })
}

/// Why a request was not answered.
#[derive(Debug)]
pub(super) enum Failure {
    ShuttingDown,
    QueueFull,
    Cancelled,
    Engine(crate::Error),
}

/// The answers and the prompt tokens their decisions read.
#[derive(Debug)]
pub(super) struct Answered {
    answers: Vec<Value>,
    input_tokens: usize,
}

fn refusal(name: Option<&String>) -> Value {
    json!({"type":"refusal","name":name})
}

/// The spec-shaped answer for `judgment`.
pub(super) fn answer_json(name: Option<&String>, judgment: &Judgment) -> Value {
    if judgment.refusal.is_some() {
        refusal(name)
    } else {
        json!({"type":"predicate","name":name,"probability":judgment.probability_yes})
    }
}

/// Answer `prepared` in order, submitting each in-scope question only once
/// the previous one resolved, so one request holds at most one engine queue
/// slot. Shutdown ends the wait at once; dropping the returned future drops
/// the pending decision, which cancels its engine work.
pub(super) async fn answer<S, Submitted, Pending>(
    prepared: &Prepared,
    calibrated: &CalibratedHead,
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
    for question in &prepared.questions {
        let (name, family, request) = match question {
            Asked::Refused { name } => {
                answers.push(refusal(name.as_ref()));
                continue;
            }
            Asked::InScope {
                name,
                family,
                request,
            } => (name, *family, request.clone()),
        };
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
        let judgment = calibrated
            .judge(family, &decision)
            .map_err(Failure::Engine)?;
        answers.push(answer_json(name.as_ref(), &judgment));
    }
    Ok(Answered {
        answers,
        input_tokens,
    })
}

/// The `DecisionResponse` body.
pub(super) fn response_json(model: &str, answered: Answered) -> Value {
    let tokens = answered.input_tokens;
    json!({
        "model":model,
        "answers":Value::Array(answered.answers),
        "usage":{
            "input_tokens":tokens,
            "input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},
            "output_tokens":0,
            "output_tokens_details":{"reasoning_tokens":0},
            "total_tokens":tokens,
        },
    })
}

/// Why `POST /v1/decisions` is not served on this server.
pub(super) fn refusal_message() -> String {
    format!(
        "{ROUTE} is served only with {FLAG} DIR: a judgment head, its out-of-scope gate and a \
         calibration artifact that passed the pre-registered evaluation in \
         docs/DECISIONS_EVALUATION.md (part 3). Without one, answers would carry no calibrated \
         probability and no refusal."
    )
}

/// `POST /v1/decisions`.
pub(super) async fn handle(state: &AppState, headers: &HeaderMap, body: Body) -> Response {
    let Some(service) = state.decisions.clone() else {
        return error_response(StatusCode::NOT_FOUND, &refusal_message(), state, headers).await;
    };
    if service.closing() {
        return shutting_down(state, headers).await;
    }
    let body = match read_body(body).await {
        Ok(body) => body,
        Err((status, message)) => return error_response(status, &message, state, headers).await,
    };
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
    let Some(admitted) = service.admit() else {
        let message = format!(
            "decisions are busy: {MAX_CONCURRENT} requests are being answered. Retry after \
             {RETRY_AFTER_SECONDS}s with your own backoff."
        );
        let mut response =
            error_response(StatusCode::SERVICE_UNAVAILABLE, &message, state, headers).await;
        response.headers_mut().insert(
            header::RETRY_AFTER,
            header::HeaderValue::from(RETRY_AFTER_SECONDS),
        );
        return response;
    };
    let (engine, head, depth) = (
        state.engine.clone(),
        service.calibrated.head(),
        state.depth.clone(),
    );
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
    let answered = answer(
        &prepared,
        &service.calibrated,
        submit,
        service.closing.subscribe(),
    )
    .await;
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
    }
}

/// The request body, read no further than [`MAX_REQUEST_BYTES`]: a larger one
/// is a `413` as soon as the limit is crossed, not after it has all arrived.
async fn read_body(body: Body) -> Result<Bytes, (StatusCode, String)> {
    match Limited::new(body, MAX_REQUEST_BYTES).collect().await {
        Ok(value) => Ok(value.to_bytes()),
        Err(error) if error.is::<LengthLimitError>() => Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "HTTP request is too large".to_owned(),
        )),
        Err(error) => Err((
            StatusCode::BAD_REQUEST,
            format!("invalid HTTP body: {error}"),
        )),
    }
}

async fn shutting_down(state: &AppState, headers: &HeaderMap) -> Response {
    let message = "the server is shutting down";
    error_response(StatusCode::SERVICE_UNAVAILABLE, message, state, headers).await
}

#[cfg(test)]
mod tests;
