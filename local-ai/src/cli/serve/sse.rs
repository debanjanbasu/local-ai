use std::time::Duration;

use bytes::Bytes;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use local_engine::{Event, EventStream};

use super::Admitted;
use super::backpressure::{FrameOutcome, send_frame};
use super::response::{
    Protocol, Reply, arguments_text, chat_finish_reason, chat_tool_call, checked,
    legacy_finish_reason, legacy_function_call, usage_json,
};
use super::responses::{ResponsesState, sse_frame};

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
    let mut frames = Frames::new(reply);
    if let Frames::Responses(state) = &mut frames {
        // A reasoning summary is generated after the main generation's last
        // event; its frames go out as they are made, on the same terms.
        let sender = sender.clone();
        state.set_flush(Box::new(move |events| {
            deliver(&sender, events.iter().map(sse_frame).collect(), stall)
        }));
    }
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

/// Turns engine events into the SSE frames of one API's streaming format.
///
/// Stateful because Chat and Responses frames are: Chat numbers tool calls and
/// announces the role once, and a Responses stream is a lifecycle of numbered
/// item events rather than one frame per token.
pub(super) enum Frames {
    Chat {
        reply: Reply,
        include_usage: bool,
        /// Calls go out as the deprecated `function_call`.
        functions: bool,
        /// Whether a chunk has carried `role` yet; `OpenAI` sends it once, first.
        announced: bool,
        /// Calls begun so far, which is also the next call's `index`.
        calls: usize,
        /// The call whose arguments are streaming: its ID and the bytes of
        /// its arguments sent.
        open: Option<(String, usize)>,
    },
    Responses(Box<ResponsesState>),
}

impl Frames {
    pub(super) fn new(reply: Reply) -> Self {
        match reply.protocol.clone() {
            Protocol::Chat {
                include_usage,
                functions,
            } => Self::Chat {
                reply,
                include_usage,
                functions,
                announced: false,
                calls: 0,
                open: None,
            },
            Protocol::Responses(echo) => {
                Self::Responses(Box::new(ResponsesState::new(&reply, echo, true)))
            }
        }
    }

    /// Frames sent before the first engine event: the Responses `created` and
    /// `in_progress` pair, and nothing for Chat Completions.
    pub(super) fn start(&mut self) -> Vec<String> {
        match self {
            Self::Responses(state) => {
                state.start();
                state.take_events().iter().map(sse_frame).collect()
            }
            Self::Chat { .. } => Vec::new(),
        }
    }

    /// The frames for one event, and whether it ended the stream.
    pub(super) fn event(&mut self, event: Event) -> (Vec<String>, bool) {
        let terminal = matches!(event, Event::Finished(_) | Event::Error(_));
        let frames = match self {
            Self::Chat {
                reply,
                include_usage,
                functions,
                announced,
                calls,
                open,
            } => chat_frames(
                reply,
                (*include_usage, *functions),
                announced,
                (calls, open),
                checked(event),
            ),
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
    data(&json!({"error":{"message":error,"type":"server_error","code":null,"param":null}}))
}

fn chat_frames(
    reply: &Reply,
    (include_usage, functions): (bool, bool),
    announced: &mut bool,
    (calls, open): (&mut usize, &mut Option<(String, usize)>),
    event: Event,
) -> Vec<String> {
    let chunk = |delta: Value, finish: Option<&str>| json!({"id":reply.id,"object":"chat.completion.chunk","created":reply.created,"model":reply.model.as_ref(),"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
    let mut delta = match event {
        Event::Content(piece) => json!({"content":piece}),
        Event::Reasoning(piece) => json!({"reasoning_content":piece}),
        // A call streams as the engine writes it: its first delta carries the
        // ID, type and name, every later one only more `arguments`, and the
        // validated call then sends whatever its arguments still lack
        // (usually the closing brace), so the fragments concatenate to them.
        Event::ToolCallDelta(delta) => {
            if open.as_ref().is_some_and(|(id, _)| *id == delta.id) {
                if let Some((_, sent)) = open.as_mut() {
                    *sent += delta.arguments.len();
                }
                json!({"tool_calls":[{"index":*calls - 1,"function":{"arguments":delta.arguments}}]})
            } else {
                *open = Some((delta.id.clone(), delta.arguments.len()));
                *calls += 1;
                json!({"tool_calls":[{"index":*calls - 1,"id":delta.id,"type":"function","function":{"name":delta.name,"arguments":delta.arguments}}]})
            }
        }
        Event::ToolCall(call) => match open.take() {
            Some((id, sent)) if id == call.id => {
                let arguments = arguments_text(&call);
                debug_assert!(sent <= arguments.len(), "streamed past the arguments");
                let rest = arguments.get(sent..).unwrap_or_default();
                if rest.is_empty() {
                    return Vec::new();
                }
                json!({"tool_calls":[{"index":*calls - 1,"function":{"arguments":rest}}]})
            }
            _ => {
                let index = *calls;
                *calls += 1;
                json!({"tool_calls":[chat_tool_call(index, &call)]})
            }
        },
        Event::Finished(stats) => {
            let finish =
                legacy_finish_reason(chat_finish_reason(stats.stop_reason, *calls > 0), functions);
            let mut frames = vec![data(&chunk(json!({}), Some(finish)))];
            if include_usage {
                frames.push(data(&json!({"id":reply.id,"object":"chat.completion.chunk","created":reply.created,"model":reply.model.as_ref(),"choices":[],"usage":usage_json(&stats)})));
            }
            frames.push("data: [DONE]\n\n".to_owned());
            return frames;
        }
        Event::Error(error) => return vec![error_frame(&error)],
        // Hosted tool results only run in the Responses loop.
        Event::TokenIds(_) | Event::ToolOutput(_) => return Vec::new(),
    };
    if functions
        && let Value::Object(map) = &mut delta
        && let Some(calls) = map.remove("tool_calls")
    {
        map.insert("function_call".into(), legacy_function_call(&calls[0]));
    }
    if !*announced {
        *announced = true;
        if let Value::Object(map) = &mut delta {
            map.insert("role".into(), json!("assistant"));
        }
    }
    vec![data(&chunk(delta, None))]
}
