use std::thread::sleep;
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;

use local_engine::{Event, EventStream};

use super::Admitted;
use super::response::{
    Protocol, Reply, chat_finish_reason, chat_tool_call, finish_reason, usage_json,
};
use super::responses::{ResponsesState, sse_frame};

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
    reply: Reply,
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
    let frames = Frames::new(reply);
    tokio::task::spawn_blocking(move || {
        pump(first, events, frames, sender, admitted, stall);
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
// `sender` is owned because the detached task has to keep it alive for the
// whole pump; it is not consumed by the body.
#[allow(clippy::needless_pass_by_value)]
fn pump(
    first: Option<Event>,
    events: EventStream,
    mut frames: Frames,
    sender: mpsc::Sender<Result<Bytes, std::io::Error>>,
    _admission: Admitted,
    stall: Duration,
) {
    if !deliver(&sender, frames.start(), stall) {
        return;
    }
    for event in first.into_iter().chain(events) {
        let (out, terminal) = frames.event(event);
        if !deliver(&sender, out, stall) || terminal {
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
    frames: Vec<String>,
    stall: Duration,
) -> bool {
    for frame in frames {
        match send_frame(sender, Bytes::from(frame), stall) {
            FrameOutcome::Sent => {}
            FrameOutcome::Closed => return false,
            FrameOutcome::Stalled => {
                eprintln!(
                    "dropping stream: the client stopped accepting frames for {}s, so the \
                     generation is cancelled and the queue slot released",
                    stall.as_secs()
                );
                return false;
            }
        }
    }
    true
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

/// Turns engine events into the SSE frames of one API's streaming format.
///
/// Stateful because Chat and Responses frames are: Chat numbers tool calls and
/// announces the role once, and a Responses stream is a lifecycle of numbered
/// item events rather than one frame per token.
pub(super) enum Frames {
    Completion(Reply),
    Chat {
        reply: Reply,
        include_usage: bool,
        /// Whether a chunk has carried `role` yet; `OpenAI` sends it once, first.
        announced: bool,
        /// Complete calls sent so far, which is also the next call's `index`.
        calls: usize,
    },
    Responses(Box<ResponsesState>),
}

impl Frames {
    pub(super) fn new(reply: Reply) -> Self {
        match reply.protocol.clone() {
            Protocol::Completion => Self::Completion(reply),
            Protocol::Chat { include_usage } => Self::Chat {
                reply,
                include_usage,
                announced: false,
                calls: 0,
            },
            Protocol::Responses(echo) => {
                Self::Responses(Box::new(ResponsesState::new(&reply, echo, true)))
            }
        }
    }

    /// Frames sent before the first engine event: the Responses `created` and
    /// `in_progress` pair, and nothing for the older APIs.
    pub(super) fn start(&mut self) -> Vec<String> {
        match self {
            Self::Responses(state) => {
                state.start();
                state.take_events().iter().map(sse_frame).collect()
            }
            Self::Completion(_) | Self::Chat { .. } => Vec::new(),
        }
    }

    /// The frames for one event, and whether it ended the stream.
    pub(super) fn event(&mut self, event: Event) -> (Vec<String>, bool) {
        let terminal = matches!(event, Event::Finished(_) | Event::Error(_));
        let frames = match self {
            Self::Completion(reply) => completion_frames(reply, event),
            Self::Chat {
                reply,
                include_usage,
                announced,
                calls,
            } => chat_frames(reply, *include_usage, announced, calls, event),
            Self::Responses(state) => {
                state.event(event);
                state.take_events().iter().map(sse_frame).collect()
            }
        };
        (frames, terminal)
    }
}

fn data(value: &Value) -> String {
    format!("data: {value}\n\n")
}

fn error_frame(error: &str) -> String {
    data(&json!({"error":{"message":error,"type":"server_error"}}))
}

fn completion_frames(reply: &Reply, event: Event) -> Vec<String> {
    let chunk = |text: &str, finish: Option<&str>| json!({"id":reply.id,"object":"text_completion","created":reply.created,"model":reply.model.as_ref(),"choices":[{"index":0,"text":text,"finish_reason":finish}]});
    match event {
        Event::Content(piece) | Event::Reasoning(piece) => vec![data(&chunk(&piece, None))],
        Event::Finished(stats) => vec![
            data(&chunk("", Some(finish_reason(stats.stop_reason)))),
            "data: [DONE]\n\n".to_owned(),
        ],
        Event::Error(error) => vec![error_frame(&error)],
        // A raw completion renders no tools, so the engine cannot report a call.
        Event::TokenIds(_) | Event::ToolCall(_) => Vec::new(),
    }
}

fn chat_frames(
    reply: &Reply,
    include_usage: bool,
    announced: &mut bool,
    calls: &mut usize,
    event: Event,
) -> Vec<String> {
    let chunk = |delta: Value, finish: Option<&str>| json!({"id":reply.id,"object":"chat.completion.chunk","created":reply.created,"model":reply.model.as_ref(),"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
    let mut delta = match event {
        Event::Content(piece) => json!({"content":piece}),
        Event::Reasoning(piece) => json!({"reasoning_content":piece}),
        // The engine reports a call only once it is complete and validated, so
        // the whole call — ID, name and every argument — goes out in one delta.
        // That is buffering in the engine, not incremental argument streaming,
        // and clients that concatenate `arguments` fragments read it unchanged.
        Event::ToolCall(call) => {
            let index = *calls;
            *calls += 1;
            json!({"tool_calls":[chat_tool_call(index, &call)]})
        }
        Event::Finished(stats) => {
            let finish = chat_finish_reason(stats.stop_reason, *calls > 0);
            let mut frames = vec![data(&chunk(json!({}), Some(finish)))];
            if include_usage {
                frames.push(data(&json!({"id":reply.id,"object":"chat.completion.chunk","created":reply.created,"model":reply.model.as_ref(),"choices":[],"usage":usage_json(&stats)})));
            }
            frames.push("data: [DONE]\n\n".to_owned());
            return frames;
        }
        Event::Error(error) => return vec![error_frame(&error)],
        Event::TokenIds(_) => return Vec::new(),
    };
    if !*announced {
        *announced = true;
        if let Value::Object(map) = &mut delta {
            map.insert("role".into(), json!("assistant"));
        }
    }
    vec![data(&chunk(delta, None))]
}
