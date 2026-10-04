use std::io::Cursor;
use std::sync::mpsc::RecvTimeoutError;
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;
use serde_json::{Value, json};

use local_engine::resources::SERVE_QUEUE;
use local_engine::{Event, EventStream, Stats};

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
const ZSTD_LEVEL: i32 = 22;

/// Served with the 503 so a rejected client can retry instead of guessing.
const RETRY_AFTER_SECONDS: u32 = 1;

const JSON_ENCODE_FAILED: &[u8] =
    b"{\"error\":{\"message\":\"JSON encoding failed\",\"type\":\"server_error\"}}";

/// How long the collector may wait for the next event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EventWait {
    /// No deadline at all, used for the first event.
    Unbounded,
    /// The stall budget, used once the engine has shown it is producing.
    Bounded(Duration),
}

/// The wait to give the next event of a non-streaming collection.
///
/// The first event is deliberately unbounded, and the stall budget starts only
/// once it has arrived. Nothing is emitted during prefill, `PREFILL_CHUNK` is
/// 128 and this M2 prefills at 3.6-4.2 tok/s, so one prefill chunk is about
/// 35 s of complete silence, and a prompt over 128 tokens is ordinary. A stall
/// means the engine was producing and the client stopped consuming, which is
/// only knowable after the first event; a slow first token is the model
/// working. Putting a deadline on the first wait would cancel long prefills
/// that were never stalled, so `Unbounded` is a separate outcome here rather
/// than a large timeout that would quietly become one.
pub(super) const fn event_wait(first_seen: bool, stall: Duration) -> EventWait {
    if first_seen {
        EventWait::Bounded(stall)
    } else {
        EventWait::Unbounded
    }
}

/// Collect a whole response, bounding the wait for each event after the first.
///
/// This path cannot see a client disconnect: the body is buffered here and
/// written once at the end, so the engine is never backpressured and it keeps
/// generating for a socket nobody is reading. What it can bound is a
/// generation that has stopped producing, which is the same engine-side
/// silence the streaming path treats as a stall.
pub(super) fn collect_response(
    mut events: EventStream,
    chat: bool,
    model: &str,
    stall: Duration,
) -> Result<Value, String> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut stats = None;
    let mut first_seen = false;
    loop {
        let event = match event_wait(first_seen, stall) {
            EventWait::Unbounded => events.next(),
            EventWait::Bounded(budget) => match events.next_timeout(budget) {
                Ok(event) => event,
                Err(RecvTimeoutError::Timeout) => {
                    // Dropping `events` cancels too, but setting the handle
                    // first stops the engine without waiting for the unwind.
                    events.cancel_handle().cancel();
                    eprintln!(
                        "dropping request: no event for {}s after the first token, so the \
                         generation is cancelled and the queue slot released",
                        budget.as_secs()
                    );
                    return Err(format!(
                        "generation stalled: no event for {}s after the first token, so it \
                         was cancelled",
                        budget.as_secs()
                    ));
                }
                Err(RecvTimeoutError::Disconnected) => None,
            },
        };
        let Some(event) = event else {
            break;
        };
        first_seen = true;
        match event {
            Event::Content(piece) => content.push_str(&piece),
            Event::Reasoning(piece) => reasoning.push_str(&piece),
            Event::Finished(value) => stats = Some(*value),
            Event::Error(error) => return Err(error),
            Event::TokenIds(_) => {}
        }
    }
    let stats = stats.ok_or_else(|| "engine worker stopped".to_owned())?;
    let choice = if chat {
        json!({"index":0,"message":{"role":"assistant","content":content,"reasoning_content":reasoning},"finish_reason":finish_reason(stats.stop_reason)})
    } else {
        json!({"index":0,"text":content,"finish_reason":finish_reason(stats.stop_reason)})
    };
    Ok(response_json(&choice, chat, model, &stats))
}

fn response_json(choice: &Value, chat: bool, model: &str, stats: &Stats) -> Value {
    let generation = &stats.generation;
    json!({"id":"local","object":if chat {"chat.completion"} else {"text_completion"},"model":model,"choices":[choice],"usage":{"prompt_tokens":generation.prompt_tokens,"prompt_tokens_details":{"cached_tokens":generation.reused_prompt_tokens,"cache_source":stats.cache_source},"completion_tokens":generation.generated_tokens,"total_tokens":generation.prompt_tokens+generation.generated_tokens},"timings":{"prefill_seconds":generation.prefill.as_secs_f64(),"first_token_seconds":generation.first_token.map(|duration| duration.as_secs_f64()),"elapsed_seconds":generation.elapsed.as_secs_f64()},"speculation":{"mtp":{"rounds":generation.mtp.rounds,"proposed_tokens":generation.mtp.proposed_tokens,"accepted_tokens":generation.mtp.accepted_tokens},"lookup":{"rounds":generation.ngram.rounds,"proposed_tokens":generation.ngram.proposed_tokens,"accepted_tokens":generation.ngram.accepted_tokens,"cpu_seconds":generation.ngram.lookup.as_secs_f64()}}})
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
        if zstd && let Ok(compressed) = zstd::stream::encode_all(Cursor::new(&bytes), ZSTD_LEVEL) {
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
        json!({"error":{"message":message,"type":kind}}),
        state,
        headers,
    )
    .await
}

/// Rejection for a full engine queue.
///
/// This adds information, not throughput. The engine is single-flight by
/// design: one command queue, one position cursor, and `&mut self` held for a
/// whole generation. Measured on this M2, aggregate throughput is 6.12 tok/s
/// at one client and 4.69 tok/s at three, so a second client costs 0.77x
/// instead of adding capacity. What a rejected client lacks is a schedule, so
/// this response carries the queue depth, the capacity that depth is measured
/// against, and a `Retry-After` to retry on.
///
/// The depth is the occupancy at the instant of rejection, which is the number
/// of requests ahead of the rejected one. An admitted client still waits
/// invisibly: the engine drains in arrival order, so a client's position is
/// its admission order, and nothing here shortens that wait.
pub(super) async fn queue_full_response(state: &AppState, headers: &HeaderMap) -> Response {
    let depth = state.depth.load();
    let message = format!(
        "engine queue is full: {depth} of {SERVE_QUEUE} slots in use; this server runs one \
         generation at a time, so adding clients does not add throughput. Retry after \
         {RETRY_AFTER_SECONDS}s with your own backoff."
    );
    let mut response = json_response(
        StatusCode::SERVICE_UNAVAILABLE,
        json!({"error":{
            "message":message,
            "type":"server_error",
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
