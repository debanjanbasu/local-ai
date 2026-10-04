use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde_json::json;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;

use local_engine::{Event, EventStream};

use super::{Admitted, response::finish_reason};

/// Retry interval for a frame the client has not taken yet.
///
/// Only a stalled client ever waits this long: a live one leaves the capacity-1
/// slot free, so the first attempt sends and this constant is never reached.
const STALL_RETRY: Duration = Duration::from_millis(2);

/// Start a streaming response and hand the frame receiver to axum.
///
/// The first event is awaited here, before the caller builds a response, so a
/// pre-generation validation error becomes a real HTTP status code instead of
/// a 200 whose first frame is an error.
pub(super) async fn start_stream(
    mut events: EventStream,
    chat: bool,
    model: Arc<str>,
    admitted: Admitted,
    stall: Duration,
) -> Result<mpsc::Receiver<Result<Bytes, std::io::Error>>, String> {
    // The first event has no deadline, deliberately and not by accident: it is
    // prefill, not a stall. Nothing is emitted while the prompt is prefilled,
    // `PREFILL_CHUNK` is 128 and this M2 prefills at 3.6-4.2 tok/s, so one
    // chunk is about 35 s of complete silence and a prompt over 128 tokens is
    // ordinary. A stall means the engine was producing and the client stopped
    // consuming, which is only knowable once the engine has produced
    // something, so every budget on this path starts after this call returns.
    // `response::event_wait` is the same rule for the non-streaming path.
    let (first, events) = tokio::task::spawn_blocking(move || {
        let first = events.next();
        (first, events)
    })
    .await
    .map_err(|error| format!("generation task failed: {error}"))?;
    if let Some(Event::Error(error)) = &first {
        return Err(error.clone());
    }
    // Capacity 1 is deliberate backpressure, not a buffer: a slow reader stops
    // this pump, and the pump stops the engine once the engine's own
    // `EVENT_BUFFER` fills. It costs nothing here because frames are per token.
    let (sender, receiver) = mpsc::channel(1);
    // Detached, because this function returns the receiver and the pump owns
    // the only `EventStream`; the receiver is what ends the pump.
    tokio::task::spawn_blocking(move || {
        pump(first, events, chat, model, sender, admitted, stall);
    });
    Ok(receiver)
}

/// Push frames into `sender` until the stream ends or the client goes away.
///
/// The client-disconnect chain is why the loop breaks on a closed channel: axum
/// drops the body, `ReceiverStream` drops, the `Receiver` drops, the `Sender`
/// drops, `send_frame` reports `Closed`, the loop breaks, this function
/// returns, `events` drops, and `EventStream`'s `Drop` cancels the generation.
/// Ignoring it would let a disconnected client run a full-length generation.
///
/// A client that stops reading with its socket still open has no such event —
/// TCP cannot tell it from a slow one and axum surfaces no per-response signal —
/// so the only way out is the stall budget in [`send_frame`]. A stall returns
/// from here on the same terms, and the cascade above is what stops the engine
/// and frees the queue slot.
///
/// `_admission` is a drop guard held for the pump's lifetime, so the queue slot
/// is released exactly when this function returns, on every path out of it.
// `model`, `sender` and `stall` are owned because the detached task has to keep
// all three alive for the whole pump; none is consumed by the body.
#[allow(clippy::needless_pass_by_value)]
fn pump(
    first: Option<Event>,
    events: EventStream,
    chat: bool,
    model: Arc<str>,
    sender: mpsc::Sender<Result<Bytes, std::io::Error>>,
    _admission: Admitted,
    stall: Duration,
) {
    if let Some(event) = first
        && !deliver(&sender, &model, chat, event, stall)
    {
        return;
    }
    for event in events {
        if !deliver(&sender, &model, chat, event, stall) {
            return;
        }
    }
}

/// Hand one event to the client, reporting whether the stream should continue.
///
/// The stall is logged rather than returned because it is the one end with no
/// socket event behind it, and a server that silently drops a generation is
/// indistinguishable from one that finished. The frame count is deliberately
/// absent: the budget is a wall-clock wait for the client, not a token count,
/// so it says the same thing for a 10-token and a 4000-token generation.
fn deliver(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    model: &str,
    chat: bool,
    event: Event,
    stall: Duration,
) -> bool {
    match send_stream_event(sender, model, chat, event, stall) {
        FrameOutcome::Sent => true,
        FrameOutcome::Closed => false,
        FrameOutcome::Stalled => {
            eprintln!(
                "dropping stream: the client stopped accepting frames for {}s, so the \
                 generation is cancelled and the queue slot released",
                stall.as_secs()
            );
            false
        }
    }
}

/// What became of a frame offered to the client.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FrameOutcome {
    /// The frame is in the channel, or never needed sending.
    Sent,
    /// The body was dropped, so the client is gone. The normal end.
    Closed,
    /// The client stopped taking frames for the whole stall budget.
    Stalled,
}

/// Offer one frame to `sender`, abandoning it if the client stops reading.
///
/// A bare `blocking_send` is what wedges a server: it parks this thread until
/// the slot frees, and a client that stops reading with its socket open never
/// frees it — no TCP event distinguishes that client from a slow one. Polling
/// with [`mpsc::Sender::try_send`] is bounded because the deadline is checked
/// here; `try_send` hands the frame back on `Full`, so the retry keeps the same
/// allocation instead of cloning a frame per attempt.
///
/// `try_send` rather than `blocking_send` is also what makes the healthy case
/// free: a live reader leaves the capacity-1 slot free, the first `try_send`
/// succeeds, and the frame is on its way without a clock read or a sleep. The
/// deadline is therefore started on the first retry, not before the send.
pub(super) fn send_frame(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    frame: Bytes,
    stall: Duration,
) -> FrameOutcome {
    let mut frame = Ok(frame);
    let mut retrying_since = None;
    loop {
        match sender.try_send(frame) {
            Ok(()) => return FrameOutcome::Sent,
            Err(TrySendError::Closed(_)) => return FrameOutcome::Closed,
            Err(TrySendError::Full(returned)) => frame = returned,
        }
        let started = *retrying_since.get_or_insert_with(Instant::now);
        let waited = started.elapsed();
        if waited >= stall {
            return FrameOutcome::Stalled;
        }
        sleep(STALL_RETRY.min(stall.saturating_sub(waited)));
    }
}

fn send_stream_event(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    model: &str,
    chat: bool,
    event: Event,
    stall: Duration,
) -> FrameOutcome {
    let object = if chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    let data = match event {
        Event::Content(piece) if chat => json!({"id":"local","object":object,"model":model,"choices":[{"index":0,"delta":{"content":piece},"finish_reason":null}]}).to_string(),
        Event::Reasoning(piece) if chat => json!({"id":"local","object":object,"model":model,"choices":[{"index":0,"delta":{"reasoning_content":piece},"finish_reason":null}]}).to_string(),
        Event::Content(piece) | Event::Reasoning(piece) => json!({"id":"local","object":object,"model":model,"choices":[{"index":0,"text":piece,"finish_reason":null}]}).to_string(),
        Event::Finished(stats) => {
            let choice = if chat { json!({"index":0,"delta":{},"finish_reason":finish_reason(stats.stop_reason)}) } else { json!({"index":0,"text":"","finish_reason":finish_reason(stats.stop_reason)}) };
            format!("{}\n\ndata: [DONE]", json!({"id":"local","object":object,"model":model,"choices":[choice]}))
        }
        Event::Error(error) => json!({"error":{"message":error,"type":"server_error"}}).to_string(),
        Event::TokenIds(_) => return FrameOutcome::Sent,
    };
    send_frame(sender, Bytes::from(format!("data: {data}\n\n")), stall)
}
