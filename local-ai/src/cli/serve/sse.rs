use std::sync::Arc;

use bytes::Bytes;
use serde_json::json;
use tokio::sync::mpsc;

use local_engine::{Event, EventStream};

use super::{Admitted, response::finish_reason};

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
) -> Result<mpsc::Receiver<Result<Bytes, std::io::Error>>, String> {
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
    tokio::task::spawn_blocking(move || pump(first, events, chat, model, sender, admitted));
    Ok(receiver)
}

/// Push frames into `sender` until the stream ends or the client goes away.
///
/// The client-disconnect chain is why the loop breaks on send error: axum
/// drops the body, `ReceiverStream` drops, the `Receiver` drops, the `Sender`
/// drops, `blocking_send` fails, the loop breaks, this function returns,
/// `events` drops, and `EventStream`'s `Drop` cancels the generation. Ignoring
/// the error would let a disconnected client run a full-length generation.
///
/// `_admission` is a drop guard held for the pump's lifetime, so the queue slot
/// is released exactly when this function returns, on every path out of it.
// `model` and `sender` are owned because the detached task has to keep both
// alive for the whole pump; neither is consumed by the body.
#[allow(clippy::needless_pass_by_value)]
fn pump(
    first: Option<Event>,
    events: EventStream,
    chat: bool,
    model: Arc<str>,
    sender: mpsc::Sender<Result<Bytes, std::io::Error>>,
    _admission: Admitted,
) {
    if let Some(event) = first
        && send_stream_event(&sender, &model, chat, event).is_err()
    {
        return;
    }
    for event in events {
        if send_stream_event(&sender, &model, chat, event).is_err() {
            return;
        }
    }
}

fn send_stream_event(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    model: &str,
    chat: bool,
    event: Event,
) -> Result<(), ()> {
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
        Event::TokenIds(_) => return Ok(()),
    };
    sender
        .blocking_send(Ok(Bytes::from(format!("data: {data}\n\n"))))
        .map_err(|_| ())
}
