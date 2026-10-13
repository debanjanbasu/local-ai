//! `POST /v1/moderations` on calibrated heads over the native `EmbeddingGemma`
//! 2 text embeddings ([`local_engine::moderation`]).
//!
//! Served only when both the embedding model and the moderation artifact
//! (`models/moderation/moderation-heads.json`, see `docs/MODERATION.md`) are
//! installed; there is no flag. The artifact is checked on startup against
//! its pinned SHA-256 and the encoder payload it was trained on; a bad
//! artifact stops the server rather than serving unvalidated scores.
//!
//! The request follows the pinned `CreateModerationRequest`. `input` is a
//! string or an array of strings (one result each, at most [`MAX_INPUTS`]),
//! or an array of multi-modal objects (one result for the whole array, its
//! text parts joined by newlines). Image parts are refused with `400` until
//! native image embeddings exist (the integration point: embed the image
//! with the vision encoder and score it with heads trained on image
//! embeddings; the text heads must not be reused on image vectors). `model`
//! may be omitted, [`MODERATION_MODEL_ID`], or one of the `OpenAI` moderation
//! model names, which are accepted as aliases; the response always names
//! [`MODERATION_MODEL_ID`].
//!
//! The response has the spec's full shape for all 13 categories. A category
//! the artifact does not support (no data, or it failed its pre-registered
//! evaluation) is answered honestly as not scored: `category_applied_input_types`
//! is `[]` (the score applies to no input), the score is `0`, and the
//! category is `false` (`null` for `illicit` and `illicit/violent`, the only
//! categories whose type allows it). It never contributes to `flagged`. The
//! unscored categories are also listed in the `x-moderation-unscored-categories`
//! response header.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use http_body_util::BodyExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use local_engine::moderation::{
    self, CATEGORIES, MODERATION_MODEL_ID, ModerationHeads, Scores, installed_heads,
};

use super::embeddings::{self, CancelOnDrop, Embeddings};
use super::response::{error_response, json_response, new_id};
use super::{AppState, MAX_REQUEST_BYTES};

pub(super) const ROUTE: &str = "/v1/moderations";
/// Strings scored per request.
const MAX_INPUTS: usize = 256;
/// Tokens over every window of a request.
const MAX_TOTAL_TOKENS: usize = 300_000;
/// `OpenAI` moderation model names accepted as aliases.
const ALIASES: [&str; 4] = [
    "omni-moderation-latest",
    "omni-moderation-2024-09-26",
    "text-moderation-latest",
    "text-moderation-stable",
];
/// Categories whose `categories` value may be `null` in the spec.
const NULLABLE: [&str; 2] = ["illicit", "illicit/violent"];
pub(super) const UNSCORED_HEADER: &str = "x-moderation-unscored-categories";

pub(super) struct Moderations {
    heads: ModerationHeads,
    embeddings: Arc<Embeddings>,
}

impl Moderations {
    /// The service, when the artifact is installed in `models_dir` and the
    /// embedding service runs.
    pub(super) fn start(
        models_dir: &Path,
        embeddings: Option<&Arc<Embeddings>>,
    ) -> crate::Result<Option<Arc<Self>>> {
        let Some(path) = installed_heads(models_dir) else {
            return Ok(None);
        };
        let Some(embeddings) = embeddings else {
            eprintln!(
                "moderation artifact {} is installed but the embedding model is not; \
                 {ROUTE} is not served",
                path.display()
            );
            return Ok(None);
        };
        let heads = ModerationHeads::open(&path, &embeddings.payload_sha256())
            .map_err(|error| crate::Error::InvalidArgument(error.to_string()))?;
        eprintln!(
            "moderations at POST {ROUTE}: {MODERATION_MODEL_ID} from {} (sha256 {}); scored: {}",
            path.display(),
            heads.sha256(),
            heads.supported().collect::<Vec<_>>().join(", ")
        );
        Ok(Some(Arc::new(Self {
            heads,
            embeddings: Arc::clone(embeddings),
        })))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestBody {
    input: Value,
    model: Option<String>,
}

/// Texts to score, one result each.
fn parse(body: &[u8]) -> Result<Vec<String>, String> {
    let request: RequestBody = serde_json::from_slice(body)
        .map_err(|error| format!("invalid moderation request: {error}"))?;
    if let Some(model) = request.model.as_deref()
        && model != MODERATION_MODEL_ID
        && !ALIASES.contains(&model)
    {
        return Err(format!(
            "model {model:?} is not served here; use {MODERATION_MODEL_ID:?} (or omit it)"
        ));
    }
    match request.input {
        Value::String(text) => Ok(vec![text]),
        Value::Array(items) if items.is_empty() => Err("input must not be an empty array".into()),
        Value::Array(items) if items.len() > MAX_INPUTS => Err(format!(
            "input holds {} items; the limit is {MAX_INPUTS}",
            items.len()
        )),
        Value::Array(items) if items.iter().all(Value::is_string) => Ok(items
            .into_iter()
            .map(|item| item.as_str().unwrap_or_default().to_owned())
            .collect()),
        Value::Array(items) if items.iter().all(Value::is_object) => {
            let mut parts = Vec::with_capacity(items.len());
            for item in &items {
                match item.get("type").and_then(Value::as_str) {
                    Some("text") => match (
                        item.get("text").and_then(Value::as_str),
                        item.as_object().map(Map::len),
                    ) {
                        (Some(text), Some(2)) => parts.push(text),
                        _ => {
                            return Err("a text input is {\"type\":\"text\",\"text\":string}".into());
                        }
                    },
                    Some("image_url") => {
                        return Err(
                            "image inputs are not supported: this server has no native image \
                                    embeddings yet, and its moderation heads score text only"
                                .into(),
                        );
                    }
                    _ => return Err("multi-modal inputs must have type text or image_url".into()),
                }
            }
            Ok(vec![parts.join("\n")])
        }
        _ => Err(
            "input must be a string, an array of strings or an array of multi-modal inputs".into(),
        ),
    }
}

/// One spec result object.
fn result_json(heads: &ModerationHeads, scores: &Scores) -> Value {
    let (mut categories, mut values, mut applied) = (Map::new(), Map::new(), Map::new());
    for (index, name) in CATEGORIES.iter().enumerate() {
        let key = (*name).to_owned();
        match (scores.scores[index], scores.flags[index]) {
            (Some(score), Some(flag)) if heads.head(name).is_some() => {
                categories.insert(key.clone(), json!(flag));
                values.insert(key.clone(), json!(score));
                applied.insert(key, json!(["text"]));
            }
            _ => {
                let unscored = if NULLABLE.contains(name) {
                    Value::Null
                } else {
                    json!(false)
                };
                categories.insert(key.clone(), unscored);
                values.insert(key.clone(), json!(0.0));
                applied.insert(key, json!([]));
            }
        }
    }
    json!({
        "flagged":scores.flagged(),
        "categories":categories,
        "category_scores":values,
        "category_applied_input_types":applied,
    })
}

fn response_json(heads: &ModerationHeads, results: &[Scores]) -> Value {
    json!({
        "id":new_id("modr-"),
        "model":MODERATION_MODEL_ID,
        "results":results.iter().map(|scores| result_json(heads, scores)).collect::<Vec<_>>(),
    })
}

fn unscored(heads: &ModerationHeads) -> String {
    CATEGORIES
        .iter()
        .filter(|name| heads.head(name).is_none())
        .copied()
        .collect::<Vec<_>>()
        .join(",")
}

enum Failure {
    Invalid(String),
    Internal(String),
}

fn score(
    service: &Moderations,
    texts: &[String],
    cancelled: &AtomicBool,
) -> Result<Vec<Scores>, Failure> {
    service.embeddings.with_model(cancelled, |model, pace| {
        let mut windows = Vec::with_capacity(texts.len());
        let mut total = 0_usize;
        for text in texts {
            let ids = moderation::windows(model, text)
                .map_err(|error| Failure::Invalid(error.to_string()))?;
            for window in &ids {
                local_engine::embedding::EmbeddingModel::validate_ids(window)
                    .map_err(|error| Failure::Invalid(error.to_string()))?;
                total += window.len();
            }
            if total > MAX_TOTAL_TOKENS {
                return Err(Failure::Invalid(format!(
                    "the inputs hold more than {MAX_TOTAL_TOKENS} tokens in total"
                )));
            }
            windows.push(ids);
        }
        let flat: Vec<Vec<u32>> = windows.iter().flatten().cloned().collect();
        let vectors = model
            .embed_ids(&flat, pace)
            .map_err(|error| Failure::Internal(error.to_string()))?;
        let mut start = 0;
        Ok(windows
            .iter()
            .map(|ids| {
                let scores = service.heads.score(&vectors[start..start + ids.len()]);
                start += ids.len();
                scores
            })
            .collect())
    })
}

pub(super) async fn handle(state: &AppState, headers: &HeaderMap, body: Body) -> Response {
    let Some(service) = state.moderations.clone() else {
        return error_response(StatusCode::NOT_FOUND, "route not found", state, headers).await;
    };
    let body = match http_body_util::Limited::new(body, MAX_REQUEST_BYTES)
        .collect()
        .await
    {
        Ok(collected) => collected.to_bytes(),
        Err(error) => {
            let status = if error.is::<http_body_util::LengthLimitError>() {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            return error_response(
                status,
                &format!("invalid HTTP body: {error}"),
                state,
                headers,
            )
            .await;
        }
    };
    let texts = match parse(&body) {
        Ok(texts) => texts,
        Err(message) => {
            return error_response(StatusCode::BAD_REQUEST, &message, state, headers).await;
        }
    };
    let Some(permit) = service.embeddings.try_admit() else {
        return embeddings::busy(state, headers).await;
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    let _guard = CancelOnDrop(Arc::clone(&cancelled));
    let worker = Arc::clone(&service);
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        score(&worker, &texts, &cancelled)
    })
    .await;
    match result {
        Ok(Ok(results)) => {
            let mut response = json_response(
                StatusCode::OK,
                response_json(&service.heads, &results),
                state,
                headers,
            )
            .await;
            if let Ok(value) = HeaderValue::from_str(&unscored(&service.heads)) {
                response.headers_mut().insert(UNSCORED_HEADER, value);
            }
            response
        }
        Ok(Err(Failure::Invalid(message))) => {
            error_response(StatusCode::BAD_REQUEST, &message, state, headers).await
        }
        Ok(Err(Failure::Internal(message))) => {
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &message, state, headers).await
        }
        Err(error) => {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &error.to_string(),
                state,
                headers,
            )
            .await
        }
    }
}

#[cfg(test)]
mod tests;
