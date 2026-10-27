//! `POST /v1/embeddings` on the native `EmbeddingGemma` 2 text encoder
//! ([`local_engine::embedding`]).
//!
//! The BF16 artifact beside the Bonsai checkpoint
//! (`models/embeddinggemma2-text/`, see `docs/EMBEDDINGS.md`) is required:
//! the server does not start without it, and there is no flag.
//!
//! The request follows the pinned `CreateEmbeddingRequest`: `input` is a
//! string, an array of strings, an array of token ids or an array of token-id
//! arrays (at most 2,048 inputs, no empty strings, at most 300,000 tokens in
//! total); `model` must be [`EMBEDDING_MODEL_ID`]; `encoding_format` is
//! `float` (default) or `base64` (little-endian F32); `dimensions` is one of
//! the Matryoshka widths 768, 512, 256 or 128 (the vector is truncated, then
//! renormalized); `user` is accepted and ignored. Text is embedded exactly as
//! sent: retrieval quality depends on the model card's prompts
//! (`task: search result | query: …`, `title: none | text: …`), which the
//! client adds. Token ids are `EmbeddingGemma` 2 ids, used as given (no
//! `<bos>`/`<eos>` added). One input may hold at most
//! [`MAX_INPUT_TOKENS`] tokens; longer inputs are refused, never truncated.
//!
//! # Scheduling
//!
//! Embedding runs on its own Metal queue, one request at a time, at low
//! priority: it submits a few encoder layers per command buffer and, while
//! Bonsai generations are running, waits for the engine worker to finish a
//! decode step between submissions ([`EngineActivity::yield_to_decode`]), so
//! each decode step is delayed by at most one short embedding submission.
//! At most [`MAX_PENDING`] embedding requests are admitted (running or
//! waiting); more are answered `503` with `Retry-After`. A client that goes
//! away cancels its request at the next submission boundary.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;
use base64::Engine as _;
use http_body_util::BodyExt;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Semaphore;

use local_engine::activity::EngineActivity;
use local_engine::embedding::{
    EMBEDDING_MODEL_ID, EmbeddingModel, MATRYOSHKA_DIMENSIONS, MAX_INPUT_TOKENS, installed_model,
    truncate_normalized,
};

use super::response::{error_response, json_response};
use super::{AppState, MAX_REQUEST_BYTES};

pub(super) const ROUTE: &str = "/v1/embeddings";

/// Embedding requests admitted at once (one runs, the rest wait in order).
pub(super) const MAX_PENDING: usize = 8;
/// `CreateEmbeddingRequest`: at most 2,048 inputs per request.
const MAX_INPUTS: usize = 2048;
/// `CreateEmbeddingRequest`: at most 300,000 tokens summed over the inputs.
const MAX_TOTAL_TOKENS: usize = 300_000;
/// Longest wait for one decode step before the next embedding submission
/// goes ahead anyway, so embeddings are slowed, never starved.
const DECODE_YIELD_LIMIT: Duration = Duration::from_millis(500);
const RETRY_AFTER_SECONDS: u32 = 1;

pub(super) struct Embeddings {
    model: Arc<Mutex<EmbeddingModel>>,
    slots: Arc<Semaphore>,
    activity: Arc<EngineActivity>,
}

impl Embeddings {
    /// The service, from the required install in `models_dir` (the
    /// directory holding the Bonsai model's own folder).
    pub(super) fn start(
        models_dir: &Path,
        activity: Arc<EngineActivity>,
    ) -> crate::Result<Arc<Self>> {
        let dir = installed_model(models_dir)?;
        let model = EmbeddingModel::open(&dir).map_err(|error| {
            super::artifact_error(format!(
                "cannot load the embedding model in {}: {error}",
                dir.display()
            ))
        })?;
        eprintln!(
            "embeddings at POST {ROUTE}: {EMBEDDING_MODEL_ID} text encoder from {} (BF16, payload \
             sha256 {})",
            dir.display(),
            model.payload_sha256()
        );
        Ok(Arc::new(Self {
            model: Arc::new(Mutex::new(model)),
            slots: Arc::new(Semaphore::new(MAX_PENDING)),
            activity,
        }))
    }

    /// SHA-256 of the loaded encoder's payload.
    pub(super) fn payload_sha256(&self) -> String {
        self.model
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .payload_sha256()
            .to_owned()
    }

    /// One admission slot, shared by every route that runs the encoder.
    pub(super) fn try_admit(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        Arc::clone(&self.slots).try_acquire_owned().ok()
    }

    /// Run `work` on the locked encoder with this service's pacing: yield to
    /// Bonsai decode between submissions and stop once `cancelled` is set.
    pub(super) fn with_model<T>(
        &self,
        cancelled: &AtomicBool,
        work: impl FnOnce(&mut EmbeddingModel, &mut dyn FnMut() -> local_engine::Result<()>) -> T,
    ) -> T {
        let mut model = self.model.lock().unwrap_or_else(PoisonError::into_inner);
        let activity = Arc::clone(&self.activity);
        let mut pace = || {
            if cancelled.load(Ordering::SeqCst) {
                return Err(local_engine::Error::Generation(
                    "embedding request cancelled".into(),
                ));
            }
            activity.yield_to_decode(DECODE_YIELD_LIMIT);
            Ok(())
        };
        work(&mut model, &mut pace)
    }
}

impl Embeddings {
    /// The model as the vector stores' chunk embedder (same model and lock,
    /// same yielding to decode).
    pub(super) fn chunk_embedder(&self) -> Arc<dyn local_services::vector_stores::ChunkEmbedder> {
        Arc::new(crate::cli::vector_embedder::ModelChunkEmbedder::new(
            Arc::clone(&self.model),
            Arc::clone(&self.activity),
        ))
    }
}

/// `/v1/models` entry of the embedding model.
pub(super) fn model_json(created: u64) -> Value {
    json!({"id":EMBEDDING_MODEL_ID,"object":"model","created":created,"owned_by":"local",
           "context_length":MAX_INPUT_TOKENS})
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestBody {
    input: Value,
    model: String,
    encoding_format: Option<String>,
    dimensions: Option<u64>,
    /// Accepted and ignored.
    #[expect(dead_code)]
    user: Option<String>,
}

/// One input as sent: text to tokenize, or token ids to use as given.
#[derive(Debug, PartialEq, Eq)]
enum Input {
    Text(String),
    Tokens(Vec<u32>),
}

#[derive(Debug)]
struct Parsed {
    inputs: Vec<Input>,
    base64: bool,
    dimensions: usize,
}

fn token(value: &Value) -> Option<u32> {
    value.as_u64().and_then(|id| u32::try_from(id).ok())
}

fn tokens(values: &[Value]) -> Result<Vec<u32>, String> {
    if values.is_empty() {
        return Err("a token array must not be empty".into());
    }
    values
        .iter()
        .map(|value| token(value).ok_or_else(|| "token ids must be non-negative integers".into()))
        .collect()
}

fn parse(body: &[u8]) -> Result<Parsed, String> {
    let request: RequestBody = serde_json::from_slice(body)
        .map_err(|error| format!("invalid embeddings request: {error}"))?;
    if super::models::resolve(&request.model, super::models::BONSAI_MODEL_ID)
        != Some(super::models::Served::Embeddings)
    {
        return Err(format!(
            "model {:?} is not served here; use {EMBEDDING_MODEL_ID:?}",
            request.model
        ));
    }
    let base64 = match request.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return Err(format!(
                "encoding_format must be float or base64, not {other:?}"
            ));
        }
    };
    let dimensions = match request.dimensions {
        None => MATRYOSHKA_DIMENSIONS[0],
        Some(d) if MATRYOSHKA_DIMENSIONS.iter().any(|m| *m as u64 == d) => d as usize,
        Some(d) => {
            return Err(format!(
                "dimensions must be one of the trained Matryoshka widths {MATRYOSHKA_DIMENSIONS:?}, not {d}"
            ));
        }
    };
    let inputs = match request.input {
        Value::String(text) => vec![Input::Text(text)],
        Value::Array(items) if items.is_empty() => {
            return Err("input must not be an empty array".into());
        }
        Value::Array(items) if items.len() > MAX_INPUTS => {
            return Err(format!(
                "input holds {} items; the limit is {MAX_INPUTS}",
                items.len()
            ));
        }
        Value::Array(items) if items.iter().all(Value::is_string) => items
            .into_iter()
            .map(|item| Input::Text(item.as_str().unwrap_or_default().to_owned()))
            .collect(),
        Value::Array(items) if items.iter().all(Value::is_number) => {
            vec![Input::Tokens(tokens(&items)?)]
        }
        Value::Array(items) if items.iter().all(Value::is_array) => items
            .iter()
            .map(|item| tokens(item.as_array().map_or(&[][..], Vec::as_slice)).map(Input::Tokens))
            .collect::<Result<_, _>>()?,
        _ => {
            return Err(
                "input must be a string, an array of strings, an array of token ids or an array of \
                 token-id arrays"
                    .into(),
            );
        }
    };
    if inputs
        .iter()
        .any(|input| matches!(input, Input::Text(text) if text.is_empty()))
    {
        return Err("input strings must not be empty".into());
    }
    Ok(Parsed {
        inputs,
        base64,
        dimensions,
    })
}

/// Sets the flag when dropped: the handler future went away (client gone).
pub(super) struct CancelOnDrop(pub(super) Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

enum Failure {
    Invalid(String),
    Internal(String),
}

fn embed(
    service: &Embeddings,
    inputs: Vec<Input>,
    cancelled: &AtomicBool,
) -> Result<(Vec<Vec<f32>>, usize), Failure> {
    service.with_model(cancelled, |model, pace| {
        let mut ids = Vec::with_capacity(inputs.len());
        let mut total = 0_usize;
        for input in inputs {
            let row = match input {
                Input::Text(text) => model
                    .tokenize(&text)
                    .map_err(|error| Failure::Invalid(error.to_string()))?,
                Input::Tokens(row) => row,
            };
            EmbeddingModel::validate_ids(&row)
                .map_err(|error| Failure::Invalid(error.to_string()))?;
            total += row.len();
            if total > MAX_TOTAL_TOKENS {
                return Err(Failure::Invalid(format!(
                    "the inputs hold more than {MAX_TOTAL_TOKENS} tokens in total"
                )));
            }
            ids.push(row);
        }
        let vectors = model
            .embed_ids(&ids, pace)
            .map_err(|error| Failure::Internal(error.to_string()))?;
        Ok((vectors, total))
    })
}

fn response_json(vectors: &[Vec<f32>], tokens: usize, base64: bool, dimensions: usize) -> Value {
    let data: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            let vector = truncate_normalized(vector, dimensions);
            let embedding = if base64 {
                let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
                Value::String(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                json!(vector)
            };
            json!({"object":"embedding","index":index,"embedding":embedding})
        })
        .collect();
    json!({
        "object":"list",
        "data":data,
        "model":EMBEDDING_MODEL_ID,
        "usage":{"prompt_tokens":tokens,"total_tokens":tokens},
    })
}

pub(super) async fn busy(state: &AppState, headers: &HeaderMap) -> Response {
    let mut response = json_response(
        StatusCode::SERVICE_UNAVAILABLE,
        json!({"error":{
            "message":format!(
                "{MAX_PENDING} embedding requests are already running or waiting; retry after \
                 {RETRY_AFTER_SECONDS}s with your own backoff"
            ),
            "type":"server_error",
            "code":null,
            "param":null,
        }}),
        state,
        headers,
    )
    .await;
    response.headers_mut().insert(
        header::RETRY_AFTER,
        header::HeaderValue::from(RETRY_AFTER_SECONDS),
    );
    response
}

pub(super) async fn handle(
    state: &AppState,
    headers: &HeaderMap,
    body: Body,
    caller: &super::admin::Caller,
) -> Response {
    let service = Arc::clone(&state.embeddings);
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
    let parsed = match parse(&body) {
        Ok(parsed) => parsed,
        Err(message) => {
            return error_response(StatusCode::BAD_REQUEST, &message, state, headers).await;
        }
    };
    let Some(permit) = service.try_admit() else {
        return busy(state, headers).await;
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    let _guard = CancelOnDrop(Arc::clone(&cancelled));
    let Parsed {
        inputs,
        base64,
        dimensions,
    } = parsed;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        embed(&service, inputs, &cancelled)
    })
    .await;
    match result {
        Ok(Ok((vectors, tokens))) => {
            super::admin::record(
                state,
                super::admin::token_usage(
                    caller,
                    local_services::admin::usage::UsageKind::Embeddings,
                    EMBEDDING_MODEL_ID,
                    tokens,
                ),
            );
            json_response(
                StatusCode::OK,
                response_json(&vectors, tokens, base64, dimensions),
                state,
                headers,
            )
            .await
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
mod tests {
    use super::*;

    fn body(value: &Value) -> Vec<u8> {
        serde_json::to_vec(value).expect("json")
    }

    #[test]
    fn inputs_follow_the_create_embedding_request_shapes() {
        let one = parse(&body(&json!({"input":"hi","model":EMBEDDING_MODEL_ID}))).expect("string");
        assert_eq!(one.inputs, vec![Input::Text("hi".into())]);
        assert_eq!(one.dimensions, 768);
        assert!(!one.base64);
        let many = parse(&body(&json!({"input":["a","b"],"model":EMBEDDING_MODEL_ID,
            "encoding_format":"base64","dimensions":256})))
        .expect("strings");
        assert_eq!(many.inputs.len(), 2);
        assert!(many.base64);
        assert_eq!(many.dimensions, 256);
        let ids = parse(&body(&json!({"input":[2,9,1],"model":EMBEDDING_MODEL_ID}))).expect("ids");
        assert_eq!(ids.inputs, vec![Input::Tokens(vec![2, 9, 1])]);
        let nested = parse(&body(
            &json!({"input":[[2,1],[2,5,1]],"model":EMBEDDING_MODEL_ID}),
        ))
        .expect("nested");
        assert_eq!(nested.inputs.len(), 2);
    }

    #[test]
    fn invalid_requests_are_refused() {
        for request in [
            json!({"input":"","model":EMBEDDING_MODEL_ID}),
            json!({"input":[],"model":EMBEDDING_MODEL_ID}),
            json!({"input":["a",""],"model":EMBEDDING_MODEL_ID}),
            json!({"input":[[]],"model":EMBEDDING_MODEL_ID}),
            json!({"input":[1,-2],"model":EMBEDDING_MODEL_ID}),
            json!({"input":["a",1],"model":EMBEDDING_MODEL_ID}),
            json!({"input":"a","model":"ternary-bonsai-2-27b"}),
            json!({"input":"a"}),
            json!({"input":"a","model":EMBEDDING_MODEL_ID,"dimensions":100}),
            json!({"input":"a","model":EMBEDDING_MODEL_ID,"encoding_format":"int8"}),
            json!({"input":"a","model":EMBEDDING_MODEL_ID,"extra":true}),
        ] {
            assert!(parse(&body(&request)).is_err(), "{request}");
        }
        // OpenAI's embedding model names are aliases of the local model.
        assert!(
            parse(&body(
                &json!({"input":"a","model":"text-embedding-3-small"})
            ))
            .is_ok()
        );
        let too_many: Vec<&str> = vec!["x"; MAX_INPUTS + 1];
        assert!(parse(&body(&json!({"input":too_many,"model":EMBEDDING_MODEL_ID}))).is_err());
    }

    #[test]
    fn responses_truncate_renormalize_and_encode() {
        let vectors = vec![vec![3.0_f32, 4.0, 0.0, 0.0]];
        let value = response_json(&vectors, 7, false, 2);
        assert_eq!(value["object"], "list");
        assert_eq!(value["usage"]["prompt_tokens"], 7);
        assert_eq!(value["data"][0]["object"], "embedding");
        let embedding = value["data"][0]["embedding"].as_array().expect("floats");
        assert_eq!(embedding.len(), 2);
        assert!((embedding[0].as_f64().expect("f") - 0.6).abs() < 1e-6);
        let encoded = response_json(&vectors, 7, true, 2);
        let text = encoded["data"][0]["embedding"].as_str().expect("base64");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(text)
            .expect("decode");
        assert_eq!(bytes.len(), 8);
        assert!((f32::from_le_bytes(bytes[4..8].try_into().expect("4")) - 0.8).abs() < 1e-6);
    }
}
