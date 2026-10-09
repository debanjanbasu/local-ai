use std::hash::{BuildHasher, Hasher, RandomState};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;
use serde_json::{Value, json};

use local_engine::bonsai_native::MAX_BATCH_SEQUENCES;
use local_engine::resources::SERVE_QUEUE;
use local_engine::{Stats, ToolCall};

use super::AppState;

/// zstd level for JSON responses.
///
/// Level 22 is the maximum and was chosen without measurement. Measured on
/// 2 MiB bodies of real English text placed in `choices[0].text`, the shape
/// this server sends, on the target M2 with `zstd::stream::encode_all` over a
/// `serde_json::to_vec` body:
///
/// | level | bytes  | ratio  | ratio vs 22 | median ms |
/// |-------|--------|--------|-------------|-----------|
/// | 22    | 429525 | 5.0185 | baseline    |     638   |
/// | 19    | 429627 | 5.0173 |    0.02%    |     482   |
/// | 17    | 433398 | 4.9736 |    0.89%    |     337   |
/// | 15    | 457918 | 4.7073 |    6.20%    |     233   |
/// | 9     | 474662 | 4.5412 |    9.51%    |      29   |
/// | 6     | 489939 | 4.3996 |   12.33%    |      19   |
/// | 3     | 538030 | 4.0064 |   20.17%    |       6   |
///
/// A second corpus of manual pages gives the same ordering (level 3 is 21.1%
/// worse), so this is a property of the levels and not of one sample. Levels 19
/// and 17 are both within 1% of 22 on both corpora, so 22 is chosen for the
/// best ratio rather than for a cheaper CPU bill. An incompressible body ties at
/// every level because there is nothing to compress.
///
/// What 22 costs over 17 is time, not size: ~638 ms versus ~337 ms for a 2 MiB
/// body. That work runs on the blocking pool and never on a runtime worker, so
/// it delays only this response. Level 3 is the opposite trade — about 20% worse
/// ratio, or ~108 KiB per 2 MiB body, for a 100x smaller CPU bill.
///
/// A zstd dictionary was measured for this path and rejected. A 1 MB dictionary
/// at level 9 dominates this setting on both axes (+5.6% ratio, ~14x less
/// time), but RFC 8878 §6 says `application/zstd` payloads should not use a
/// dictionary and §7.4 confirms none are published for public use. Decoding
/// without the dictionary fails hard with `Dictionary mismatch` and yields zero
/// bytes, so every stock client — `curl`, `httpx`, `requests`, `undici` — would
/// break. The frame header does carry a `Dictionary_ID`, but there is no HTTP
/// mechanism to deliver the dictionary it points at.
pub(super) const ZSTD_LEVEL: i32 = 22;

/// Served with the 503 so a rejected client can retry instead of guessing.
const RETRY_AFTER_SECONDS: u32 = 1;

const JSON_ENCODE_FAILED: &[u8] =
    b"{\"error\":{\"message\":\"JSON encoding failed\",\"type\":\"server_error\",\"code\":null,\"param\":null}}";

/// How long the collector may wait for the next item of a response.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EventWait {
    /// No deadline at all, used until the engine has produced an event.
    Unbounded,
    /// The stall budget, used once the engine has shown it is producing.
    Bounded(Duration),
}

/// The wait to give the next item of a non-streaming response.
///
/// The first *event* is deliberately unbounded, and the stall budget starts only
/// once one has arrived. A prefill boundary does not count as one, which is the
/// whole reason this is a named rule and not a flag someone can flip: nothing is
/// emitted per token during prefill, `PREFILL_CHUNK` is 128 and this M2 prefills
/// at 3.6-4.2 tok/s, so one chunk is about 35 s of working, and the
/// `--stall-timeout` floor is 10 s. A boundary therefore says the engine is alive
/// and is written to the client, but it does not start a clock, because arming
/// that clock on it would cancel every prompt longer than one chunk.
///
/// A stall means the engine was producing and the client stopped consuming, which
/// is only knowable after an event; a slow first token is the model working.
/// Putting a deadline on the wait for the first of those would cancel long
/// prefills that were never stalled, so `Unbounded` is a separate outcome here
/// rather than a large timeout that would quietly become one.
pub(super) const fn event_wait(first_seen: bool, stall: Duration) -> EventWait {
    if first_seen {
        EventWait::Bounded(stall)
    } else {
        EventWait::Unbounded
    }
}

/// Encode and optionally compress a JSON body off the runtime.
///
/// `serde_json::to_vec` and zstd are both synchronous CPU work, and at the 2
/// MiB body cap that is a substantial fraction of a second, so both run on the
/// blocking pool rather than on a runtime worker.
#[allow(clippy::needless_pass_by_value)]
pub(super) async fn json_response(
    status: StatusCode,
    value: Value,
    state: &AppState,
    request_headers: &HeaderMap,
) -> Response {
    let zstd = wants_zstd(request_headers);
    let bytes = tokio::task::spawn_blocking(move || {
        let mut bytes = serde_json::to_vec(&value).unwrap_or_else(|_| JSON_ENCODE_FAILED.to_vec());
        // `bulk` tells zstd the size up front, so it sizes its tables to the
        // body: ~1.5 MB of working memory at level 22, where the stream API
        // without a size allocates and clears ~740 MB per call.
        if zstd && let Ok(compressed) = zstd::bulk::compress(&bytes, ZSTD_LEVEL) {
            bytes = compressed;
        }
        bytes
    })
    .await
    .unwrap_or_else(|_| JSON_ENCODE_FAILED.to_vec());
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    if zstd {
        response.headers_mut().insert(
            header::CONTENT_ENCODING,
            header::HeaderValue::from_static("zstd"),
        );
    }
    add_alt_svc(&mut response, state);
    response
}

pub(super) fn wants_zstd(request_headers: &HeaderMap) -> bool {
    request_headers
        .get(header::ACCEPT_ENCODING)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(',')
                .any(|item| item.trim().split(';').next() == Some("zstd"))
        })
}

pub(super) async fn error_response(
    status: StatusCode,
    message: &str,
    state: &AppState,
    headers: &HeaderMap,
) -> Response {
    let kind = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    json_response(
        status,
        json!({"error":{"message":message,"type":kind,"code":null,"param":null}}),
        state,
        headers,
    )
    .await
}

/// Rejection for a full engine queue.
///
/// Up to `MAX_BATCH_SEQUENCES` requests decode together, one batched pass per
/// token for all of them, and `SERVE_QUEUE` more wait behind them; this is the
/// answer once both are full. What a rejected client lacks is a schedule, so
/// this response carries the queue depth, the capacity that depth is measured
/// against, and a `Retry-After` to retry on.
///
/// The depth is the occupancy at the instant of rejection: requests running
/// and waiting. Waiting requests are admitted in arrival order as running ones
/// finish.
pub(super) async fn queue_full_response(state: &AppState, headers: &HeaderMap) -> Response {
    let depth = state.depth.load();
    let message = format!(
        "engine queue is full: {depth} requests running or waiting; up to \
         {MAX_BATCH_SEQUENCES} decode together and {SERVE_QUEUE} more may wait. Retry after \
         {RETRY_AFTER_SECONDS}s with your own backoff."
    );
    let mut response = json_response(
        StatusCode::SERVICE_UNAVAILABLE,
        json!({"error":{
            "message":message,
            "type":"server_error",
            "code":null,
            "param":null,
            "queue_depth":depth,
            "queue_capacity":SERVE_QUEUE,
            "retry_after_seconds":RETRY_AFTER_SECONDS,
        }}),
        state,
        headers,
    )
    .await;
    if let Ok(value) = header::HeaderValue::from_str(&RETRY_AFTER_SECONDS.to_string()) {
        response.headers_mut().insert(header::RETRY_AFTER, value);
    }
    response
}

pub(super) fn add_alt_svc(response: &mut Response, state: &AppState) {
    if let Some(value) = &state.alt_svc
        && let Ok(value) = header::HeaderValue::from_str(value)
    {
        response.headers_mut().insert(header::ALT_SVC, value);
    }
}

pub(super) const fn finish_reason(reason: crate::bonsai_model::StopReason) -> &'static str {
    match reason {
        crate::bonsai_model::StopReason::Eos => "stop",
        crate::bonsai_model::StopReason::TokenLimit => "length",
        crate::bonsai_model::StopReason::Cancelled => "cancelled",
    }
}

/// Chat finish reason, which reports `tool_calls` when the model ended its turn
/// on one or more complete calls.
///
/// A token limit still wins: a call that was emitted before the budget ran out
/// is complete, but the turn was cut short and the client has to know.
pub(super) const fn chat_finish_reason(
    reason: crate::bonsai_model::StopReason,
    called: bool,
) -> &'static str {
    match reason {
        crate::bonsai_model::StopReason::Eos if called => "tool_calls",
        other => finish_reason(other),
    }
}

/// Seconds since the Unix epoch, for the `created` members.
pub(super) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// A fresh opaque identifier: `prefix` followed by 32 hex digits.
///
/// Unique within the process by a counter and across restarts by the clock and
/// `RandomState`'s per-process keys. It names a response, never a secret.
pub(super) fn new_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    hasher.write_u128(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos()),
    );
    let high = hasher.finish();
    hasher.write_u64(high);
    let low = hasher.finish();
    format!("{prefix}{high:016x}{low:016x}")
}

/// Which API a generation answers, with what its documents need to echo.
#[derive(Clone)]
pub(super) enum Protocol {
    /// `POST /v1/completions`.
    Completion,
    /// `POST /v1/chat/completions`.
    Chat {
        /// `stream_options.include_usage`: a final chunk with `usage`.
        include_usage: bool,
    },
    /// `POST /v1/responses`.
    Responses(Arc<super::responses::Echo>),
}

/// Everything a response document is labelled with, fixed when the request is
/// accepted so every frame of one response carries the same identity.
#[derive(Clone)]
pub(super) struct Reply {
    pub(super) protocol: Protocol,
    pub(super) id: String,
    pub(super) created: u64,
    pub(super) model: Arc<str>,
}

impl Reply {
    pub(super) fn new(protocol: Protocol, model: Arc<str>) -> Self {
        let prefix = match &protocol {
            Protocol::Completion => "cmpl-",
            Protocol::Chat { .. } => "chatcmpl-",
            Protocol::Responses(_) => "resp_",
        };
        Self {
            protocol,
            id: new_id(prefix),
            created: unix_now(),
            model,
        }
    }

    pub(super) const fn chat(&self) -> bool {
        matches!(self.protocol, Protocol::Chat { .. })
    }
}

/// Chat and completion `usage`, with the engine's prompt-cache reuse.
pub(super) fn usage_json(stats: &Stats) -> Value {
    let generation = &stats.generation;
    json!({"prompt_tokens":generation.prompt_tokens,"prompt_tokens_details":{"cached_tokens":generation.reused_prompt_tokens,"cache_source":stats.cache_source},"completion_tokens":generation.generated_tokens,"completion_tokens_details":{"reasoning_tokens":stats.reasoning_tokens},"total_tokens":generation.prompt_tokens+generation.generated_tokens})
}

/// A tool call in the Chat Completions wire shape, at its position in the turn.
///
/// `arguments` is a JSON *string* on the wire. The engine hands over a parsed,
/// validated value, so this re-encodes it; the text is canonical JSON rather
/// than the model's exact spelling, which is the same document.
pub(super) fn chat_tool_call(index: usize, call: &ToolCall) -> Value {
    json!({"index":index,"id":call.id,"type":"function","function":{"name":call.name,"arguments":arguments_text(call)}})
}

pub(super) fn arguments_text(call: &ToolCall) -> String {
    serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_owned())
}

pub(super) const fn error_status(error: &crate::Error) -> StatusCode {
    match error {
        crate::Error::QueueFull => StatusCode::SERVICE_UNAVAILABLE,
        crate::Error::InvalidArgument(_) | crate::Error::ContextOverflow(_) => {
            StatusCode::BAD_REQUEST
        }
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

pub(super) fn error_status_message(error: &str) -> StatusCode {
    if error.starts_with("invalid argument:") || error.starts_with("context overflow:") {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}
