use std::io::Write as _;
use std::sync::Arc;
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use zstd::stream::write::Encoder;

use local_engine::{Event, EventStream, Signal, Stats};

use super::Admitted;
use super::response::{EventWait, ZSTD_LEVEL, event_wait, finish_reason};
use super::sse::{FrameOutcome, send_frame};

/// Body bytes written per frame when the generation is outrunning the clock.
///
/// One bound is not enough, because the two answer different questions. The byte
/// bound stops a fast generation paying a chunk header and a write syscall per
/// token; the interval bound is the one the fix depends on, because at the
/// 2.6-17 tok/s measured here a byte bound alone is kilobytes of text, which is
/// up to two and a half minutes of silence at the slow end — the whole failure
/// this path exists to remove.
///
/// Measured on this M2 over 1139 tokens of real generated text (4455 bytes of it)
/// at `ZSTD_LEVEL`, one `zstd` flush per frame and 18 bytes of HTTP chunk
/// overhead per frame counted in:
///
/// | frame written every | plain bytes | zstd bytes | zstd ratio |
/// |---------------------|-------------|------------|------------|
/// | 1 piece             | 24957       | 28332      | 0.16x      |
/// | 16 pieces           | 5733        | 4384       | 1.02x      |
/// | 64 pieces           | 4761        | 2743       | 1.62x      |
/// | the whole body      | 4455        | 1903       | 2.34x      |
///
/// A frame per piece is 5.6x the answer in transfer encoding, and it makes `zstd`
/// worse than sending nothing compressed at all. The interval bound holds the
/// frame count near one per second, which on that corpus falls between the
/// 16- and 64-piece rows, and a client that has gone away is noticed in about a
/// second instead of at the end of the generation.
pub(super) const BODY_FLUSH_BYTES: usize = 8 * 1024;

/// Longest a frame may wait for one that is not full.
///
/// Measured time to first byte and generation length on this M2 put the slowest
/// observed decode at 2.6 tok/s, where `BODY_FLUSH_BYTES` is 150 s of output. One
/// second is short enough that a write fails while the abandoned generation can
/// still be cancelled, and long enough that the generation is not spending its
/// time on writes.
pub(super) const BODY_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// The members this body writes piece by piece, in the order it writes them.
///
/// A chat has two because the chat template emits reasoning *before* the answer
/// and this body's order follows arrival: buffering reasoning instead would make
/// a thinking request as silent as the buffered body this path replaces, since
/// reasoning is most of a thinking generation. A completion has one.
///
/// The order is the opposite of what a serialized map gives, and has to be.
/// `serde_json`'s map is a `BTreeMap`, so a buffered body sorts `finish_reason`
/// and `role` ahead of the strings, and neither is known until the end. Writing
/// the strings first is the only order in which they can be written as they are
/// produced. The reassembled document has the same members with the same values;
/// only their order differs, which JSON does not attach meaning to.
const fn members(chat: bool) -> &'static [&'static str] {
    if chat {
        &["reasoning_content", "content"]
    } else {
        &["text"]
    }
}

/// The bytes that open a streamed non-streaming body, first member named and its
/// value left unopened.
///
/// Leaving the value unopened is what buys a heartbeat. Prefill has to be able to
/// write something before there is any text to write, and the only JSON bytes
/// legal between a `:` and the value that follows it are whitespace: a space
/// written once the quotes are open would silently become part of the answer.
const fn stream_head(chat: bool) -> &'static str {
    if chat {
        "{\"choices\":[{\"index\":0,\"message\":{\"reasoning_content\":"
    } else {
        "{\"choices\":[{\"index\":0,\"text\":"
    }
}

/// Window for the streamed body: 2 MiB.
///
/// A streamed body's size is unknown when it opens, so zstd falls back to level
/// 22's full table sizes (window 2^27, hash 2^25, chain 2^27) and allocates and
/// clears ~740 MB per response. These are the logs zstd itself picks for level
/// 22 once it knows the input is at most 2 MiB. Measured with zstd 0.13.3 on a
/// 1.5 MB text body written in 256-byte flushed pieces: the same 555,029
/// compressed bytes, 39 MB peak instead of 744 MB, 206 ms instead of 304 ms.
/// A larger body still compresses correctly; it only loses matches further
/// back than 2 MiB.
const STREAM_WINDOW_LOG: u32 = 21;

fn open_encoder() -> std::io::Result<Encoder<'static, Vec<u8>>> {
    use zstd::stream::raw::CParameter;
    let mut encoder = Encoder::new(Vec::new(), ZSTD_LEVEL)?;
    encoder.set_parameter(CParameter::WindowLog(STREAM_WINDOW_LOG))?;
    encoder.set_parameter(CParameter::HashLog(STREAM_WINDOW_LOG + 1))?;
    encoder.set_parameter(CParameter::ChainLog(STREAM_WINDOW_LOG + 1))?;
    Ok(encoder)
}

/// How body bytes reach the wire.
enum Codec {
    /// Plain bytes: every client that did not ask for zstd, and the same choice
    /// the streaming path already makes for a body it cannot buffer.
    Plain,
    /// One zstd frame, built as the body arrives and taken out when it ends.
    ///
    /// The `Option` is there because `finish` consumes the encoder, and a `Box`
    /// because an encoder is much larger than a `Vec` and the two variants sit
    /// next to each other on every frame.
    Zstd(Option<Box<Encoder<'static, Vec<u8>>>>),
}

/// A non-streaming response body, written as the generation produces it.
pub(super) struct Body {
    chat: bool,
    model: Arc<str>,
    /// How many of [`members`] have been declared. The head declares the first,
    /// so this starts at one.
    declared: usize,
    /// Whether the last declared member's string is open for more pieces.
    open: bool,
    /// Bytes produced but not yet handed to the socket.
    pending: Vec<u8>,
    /// When the last frame was written, for [`Self::due`]'s other bound.
    last_flush: Instant,
    codec: Codec,
}

/// A non-streaming response body, written as the generation produces it.
impl Body {
    /// Open a body, reporting whether it is really being compressed.
    ///
    /// The answer is returned rather than taken from the request because the
    /// response only claims `Content-Encoding: zstd` when it is true. An encoder
    /// this build cannot open falls back to plain bytes rather than failing a
    /// request over its transfer encoding.
    pub(super) fn open(chat: bool, model: Arc<str>, zstd: bool) -> (Self, bool) {
        let codec = if zstd {
            open_encoder().map_or(Codec::Plain, |encoder| Codec::Zstd(Some(Box::new(encoder))))
        } else {
            Codec::Plain
        };
        let compressed = matches!(codec, Codec::Zstd(_));
        let mut body = Self {
            chat,
            model,
            declared: 1,
            open: false,
            pending: Vec::new(),
            last_flush: Instant::now(),
            codec,
        };
        body.pending.extend_from_slice(stream_head(chat).as_bytes());
        (body, compressed)
    }

    /// Write one piece of the member the body is currently on, opening its value
    /// on the first of them.
    fn piece(&mut self, piece: &str) {
        if !self.open {
            self.pending.push(b'"');
            self.open = true;
        }
        self.pending.extend_from_slice(&json_interior(piece));
    }

    /// Write one reasoning piece.
    ///
    /// The chat template emits every reasoning piece before the first content
    /// piece — `EventSplitter` closes its reasoning half at `</think>` and never
    /// reopens it — so this can only ever be the member the head already named.
    fn reasoning(&mut self, piece: &str) {
        debug_assert_eq!(self.declared, 1, "reasoning arrived after content");
        self.piece(piece);
    }

    /// Write one content piece, naming and closing whatever came before it.
    ///
    /// With thinking on, the member before `content` is the empty
    /// `reasoning_content` of a model that did not reason; with thinking off there
    /// is nothing before it and this only opens `text`.
    fn content(&mut self, piece: &str) {
        self.declare_rest();
        self.piece(piece);
    }

    /// Close the value the cursor is on, as the empty string if nothing was ever
    /// written into it.
    ///
    /// An unopened member is not an open one: the head leaves its value unopened
    /// so a heartbeat has a place where JSON whitespace is legal, so closing one
    /// means writing both quotes of an empty string.
    fn close_value(&mut self) {
        self.pending
            .extend_from_slice(if self.open { b"\"" } else { b"\"\"" });
        self.open = false;
    }

    /// Close the member the head left open and name the next, until every
    /// streamed member has been declared.
    fn declare_rest(&mut self) {
        while self.declared < members(self.chat).len() {
            self.close_value();
            self.pending.push(b',');
            self.pending
                .extend_from_slice(json_string(members(self.chat)[self.declared]).as_bytes());
            self.pending.push(b':');
            self.declared += 1;
        }
    }

    /// Write a prefill boundary as JSON whitespace.
    ///
    /// Only legal before the content string is opened, which is exactly when
    /// prefill runs: decode emits an event per token instead, so the pump asks to
    /// write on its own once enough text has accumulated. A space here is
    /// insignificant to every JSON parser and invisible in the answer.
    fn heartbeat(&mut self) {
        self.pending.push(b' ');
    }

    /// Close the last streamed value, the choice and the document behind it.
    fn finish(&mut self, stats: &Stats) {
        // Nothing streamed at all is still a complete document: every member the
        // head declared and every one after it is closed as the empty string.
        self.declare_rest();
        self.close_value();
        self.pending
            .extend_from_slice(&stream_tail(self.chat, &self.model, stats));
    }

    /// Whether enough has accumulated, or enough time has passed, to write.
    ///
    /// Both bounds are needed, and the time one is the load-bearing half: a byte
    /// bound alone would make a slow generation as invisible to a client that has
    /// gone away as the buffered body it replaces. See [`BODY_FLUSH_BYTES`].
    ///
    /// A body that has nothing pending is never due, so a piece that only feeds
    /// the reasoning buffer cannot produce an empty frame.
    fn due(&self) -> bool {
        !self.pending.is_empty()
            && (self.pending.len() >= BODY_FLUSH_BYTES
                || self.last_flush.elapsed() >= BODY_FLUSH_INTERVAL)
    }

    /// Take the frame to write, closing the transfer encoding on the last one.
    ///
    /// `None` means there is nothing to write yet, which is a normal outcome: the
    /// pump asks on every item and only accumulated text or a prefill boundary
    /// produces a frame.
    pub(super) fn take(&mut self, last: bool) -> Option<Bytes> {
        let pending = std::mem::take(&mut self.pending);
        let Codec::Zstd(slot) = &mut self.codec else {
            if pending.is_empty() {
                return None;
            }
            self.last_flush = Instant::now();
            return Some(Bytes::from(pending));
        };
        let encoder = slot.as_mut()?;
        let _ = encoder.write_all(&pending);
        if last {
            // The frame's epilogue is how a decoder learns the body ended, so it
            // goes out with the last flush rather than being left to a drop that
            // may never come.
            self.last_flush = Instant::now();
            return slot
                .take()
                .map(|encoder| Bytes::from(encoder.finish().unwrap_or_default()));
        }
        if pending.is_empty() {
            return None;
        }
        // zstd holds a block back to compress it better, so a write without this
        // emits nothing at all for a body this size. The flush is what puts the
        // bytes on the wire; it does not end the frame.
        let _ = encoder.flush();
        let produced = std::mem::take(encoder.get_mut());
        if produced.is_empty() {
            return None;
        }
        self.last_flush = Instant::now();
        Some(Bytes::from(produced))
    }
}

/// What the pump should do with the body after one signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Flow {
    /// Keep waiting; the body grew but not enough to be worth a write.
    Keep,
    /// Write what has accumulated.
    Flush,
    /// Write what has accumulated and close the transfer encoding.
    Last,
    /// The document cannot be completed, so stop writing.
    Stop,
}

/// Fold one signal into the body and say what to write.
pub(super) fn absorb(body: &mut Body, signal: Signal) -> Flow {
    match signal {
        // A boundary exists to be written, so it always is: reporting it and not
        // flushing it would leave prefill as silent as it was.
        Signal::Progress(_) => {
            body.heartbeat();
            Flow::Flush
        }
        Signal::Event(Event::Content(piece)) => {
            body.content(&piece);
            due(body)
        }
        // Reasoning is streamed like content, not buffered: a thinking generation
        // is mostly reasoning, and a body that withheld it would be as silent as
        // the buffered document this path replaces.
        Signal::Event(Event::Reasoning(piece)) => {
            body.reasoning(&piece);
            due(body)
        }
        // Token IDs are not part of this document.
        Signal::Event(Event::TokenIds(_)) => Flow::Keep,
        Signal::Event(Event::Finished(stats)) => {
            body.finish(&stats);
            Flow::Last
        }
        Signal::Event(Event::Error(error)) => {
            // The status line is long gone by the time the engine fails partway
            // through, so the body is left truncated on purpose: an unparseable
            // document says this failed, where a shorter well-formed one would say
            // this finished and be wrong.
            eprintln!(
                "truncating response: the engine failed after the body had started, so the \
                 client reads an incomplete document rather than a short answer: {error}"
            );
            Flow::Stop
        }
    }
}

fn due(body: &Body) -> Flow {
    if body.due() { Flow::Flush } else { Flow::Keep }
}

/// The bytes that close a streamed choice and the document behind it.
///
/// Every member here is end-dependent, which is the other half of why the
/// streamed strings come first: `role` and `finish_reason` are not known until
/// the generation has stopped.
fn stream_tail(chat: bool, model: &str, stats: &Stats) -> Vec<u8> {
    let mut tail = Vec::new();
    let stop = json_string(finish_reason(stats.stop_reason));
    if chat {
        // `message` is still open, and this brace is what ends it.
        tail.extend_from_slice(b",\"role\":\"assistant\"}");
    }
    tail.extend_from_slice(b",\"finish_reason\":");
    tail.extend_from_slice(stop.as_bytes());
    // The same two bytes close the choice and the array in either shape, which
    // is what `stream_head` opened; the brace at the end closes the document.
    tail.extend_from_slice(b"}]");
    for (key, value) in trailer(chat, model, stats) {
        tail.push(b',');
        tail.extend_from_slice(json_string(key).as_bytes());
        tail.push(b':');
        tail.extend_from_slice(&encode(&value));
    }
    tail.push(b'}');
    tail
}

/// The top-level members that follow `choices`, in map order.
///
/// Written out member by member rather than taken from a serialized map because
/// the streamed body has to carry the members after `choices` by hand, and a map
/// serialized whole would put `choices` back at the front.
fn trailer(chat: bool, model: &str, stats: &Stats) -> Vec<(&'static str, Value)> {
    vec![
        ("id", json!("local")),
        ("model", json!(model)),
        (
            "object",
            json!(if chat {
                "chat.completion"
            } else {
                "text_completion"
            }),
        ),
        ("speculation", speculation_json(stats)),
        ("timings", timings_json(stats)),
        ("usage", usage_json(stats)),
    ]
}

fn usage_json(stats: &Stats) -> Value {
    let generation = &stats.generation;
    json!({"prompt_tokens":generation.prompt_tokens,"prompt_tokens_details":{"cached_tokens":generation.reused_prompt_tokens,"cache_source":stats.cache_source},"completion_tokens":generation.generated_tokens,"total_tokens":generation.prompt_tokens+generation.generated_tokens})
}

fn timings_json(stats: &Stats) -> Value {
    let generation = &stats.generation;
    json!({"prefill_seconds":generation.prefill.as_secs_f64(),"first_token_seconds":generation.first_token.map(|duration| duration.as_secs_f64()),"elapsed_seconds":generation.elapsed.as_secs_f64()})
}

fn speculation_json(stats: &Stats) -> Value {
    let generation = &stats.generation;
    json!({"mtp":{"rounds":generation.mtp.rounds,"proposed_tokens":generation.mtp.proposed_tokens,"accepted_tokens":generation.mtp.accepted_tokens},"lookup":{"rounds":generation.ngram.rounds,"proposed_tokens":generation.ngram.proposed_tokens,"accepted_tokens":generation.ngram.accepted_tokens,"cpu_seconds":generation.ngram.lookup.as_secs_f64()}})
}

/// Serialize a value that cannot fail to serialize.
fn encode(value: &Value) -> Vec<u8> {
    // Every value reaching here is built from strings, integers and floats, none
    // of which `serde_json` can reject. An empty result would be a truncated body
    // rather than a panic, which the workspace denies.
    serde_json::to_vec(value).unwrap_or_default()
}

/// A JSON string literal, including its quotes.
fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// The escaped interior of a JSON string literal, without its quotes.
///
/// Escaping is per character, so the pieces of a streamed string concatenate to
/// exactly the bytes `serde_json` would have written for the whole of it. The
/// quotes are the caller's to write, once each, around every piece.
fn json_interior(value: &str) -> Vec<u8> {
    let quoted = json_string(value);
    quoted
        .strip_prefix('"')
        .and_then(|interior| interior.strip_suffix('"'))
        .unwrap_or(quoted.as_str())
        .as_bytes()
        .to_vec()
}

/// Start a non-streaming response and hand its body frames to axum.
///
/// The first item is awaited here, before the caller builds a response, for two
/// reasons. A pre-generation failure still becomes a real HTTP status code rather
/// than a 200 whose body starts with an error. And a prefill boundary is the
/// first thing a long prompt produces, so waiting for one costs a scheduling
/// quantum instead of the whole prefill — which is what lets the body reach the
/// wire while the prompt is still being consumed.
pub(super) async fn start_chunked(
    mut events: EventStream,
    chat: bool,
    model: Arc<str>,
    admitted: Admitted,
    stall: Duration,
    zstd: bool,
) -> Result<(mpsc::Receiver<Result<Bytes, std::io::Error>>, bool), String> {
    // Unbounded for the same reason the streaming path's first event is: it is
    // prefill, not a stall, and `response::event_wait` is the same rule.
    let (first, events) = tokio::task::spawn_blocking(move || {
        // No deadline, so the wait cannot report a timeout; a closed channel is
        // reported as `Ok(None)`, which is the end this has to handle.
        let first = events.next_signal(None).ok().flatten();
        (first, events)
    })
    .await
    .map_err(|error| format!("generation task failed: {error}"))?;
    let first = first.ok_or_else(|| "engine worker stopped".to_owned())?;
    if let Signal::Event(Event::Error(error)) = &first {
        return Err(error.clone());
    }
    let (body, compressed) = Body::open(chat, model, zstd);
    // Capacity 1 is deliberate backpressure, not a buffer, for the same reason
    // as the streaming path: a slow reader stops this pump, and this pump is what
    // holds the engine.
    let (sender, receiver) = mpsc::channel(1);
    // Detached, because this function returns the receiver and the pump owns the
    // only `EventStream`; the receiver is what ends the pump.
    tokio::task::spawn_blocking(move || pump(first, events, body, sender, admitted, stall));
    Ok((receiver, compressed))
}

/// Write body frames until the generation ends or the client goes away.
///
/// The client-disconnect chain is the one [`super::sse::pump`] documents, and it
/// is the reason this body streams at all: axum drops the body, the receiver
/// drops, the sender drops, `send_frame` reports [`FrameOutcome::Closed`], this
/// function returns, `events` drops, and `EventStream`'s `Drop` cancels the
/// generation. A buffered body has no such write to fail, so a client that had
/// already gone left the engine running for the rest of the generation.
///
/// A client that stops reading with its socket still open has no event either,
/// which is what the stall budget in `send_frame` is for, and the pump logs it as
/// the streaming path does rather than dropping a generation silently.
///
/// `_admission` is a drop guard held for the pump's lifetime, so the queue slot is
/// released exactly when this function returns, on every path out of it.
// `model`, `sender` and `stall` are owned because the detached task has to keep
// all three alive for the whole pump; none is consumed by the body.
#[allow(clippy::needless_pass_by_value)]
fn pump(
    first: Signal,
    mut events: EventStream,
    mut body: Body,
    sender: mpsc::Sender<Result<Bytes, std::io::Error>>,
    _admission: Admitted,
    stall: Duration,
) {
    let mut queued = Some(first);
    let mut first_seen = false;
    // The head goes out before anything is waited for, so the first byte is on
    // the wire as soon as the generation has proved it started, even when prefill
    // was served entirely from cache and never reports a boundary.
    if deliver(&sender, body.take(false), stall).is_none() {
        return;
    }
    loop {
        let Some(signal) = queued
            .take()
            .map_or_else(|| await_signal(&mut events, first_seen, stall), Some)
        else {
            return;
        };
        // Only generated text starts the clock, never a prefill boundary:
        // `PREFILL_CHUNK` is 128 tokens and this M2 prefills at 3.6-4.2 tok/s, so
        // one chunk is about 35 s of work while the `--stall-timeout` floor is
        // 10 s. Arming the budget on a boundary would cancel ordinary long
        // prompts, which is the failure this path exists to remove.
        first_seen |= matches!(signal, Signal::Event(_));
        let flow = absorb(&mut body, signal);
        if flow == Flow::Stop {
            return;
        }
        if flow != Flow::Keep && deliver(&sender, body.take(flow == Flow::Last), stall).is_none() {
            return;
        }
    }
}

/// Write one frame, reporting whether the body may continue.
fn deliver(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    frame: Option<Bytes>,
    stall: Duration,
) -> Option<FrameOutcome> {
    let frame = frame?;
    match send_frame(sender, frame, stall) {
        FrameOutcome::Sent => Some(FrameOutcome::Sent),
        FrameOutcome::Closed => None,
        FrameOutcome::Stalled => {
            eprintln!(
                "dropping request: the client stopped accepting body frames for {}s, so the \
                 generation is cancelled and the queue slot released",
                stall.as_secs()
            );
            None
        }
    }
}

/// Wait for the next signal under the budget that applies to this request.
///
/// `None` ends the pump, for end of stream or for a stall whose message is
/// already logged and whose cancellation has already been asked for.
fn await_signal(events: &mut EventStream, first_seen: bool, stall: Duration) -> Option<Signal> {
    let wait = event_wait(first_seen, stall);
    let outcome = match wait {
        EventWait::Unbounded => events.next_signal(None),
        EventWait::Bounded(budget) => events.next_signal(Some(budget)),
    };
    match outcome {
        Ok(signal) => signal,
        Err(RecvTimeoutError::Timeout) => {
            let EventWait::Bounded(budget) = wait else {
                return None;
            };
            // Dropping `events` cancels too, but setting the handle first stops
            // the engine without waiting for the unwind.
            events.cancel_handle().cancel();
            eprintln!(
                "dropping request: no event for {}s after the first token, so the generation is \
                 cancelled and the queue slot released",
                budget.as_secs()
            );
            None
        }
        Err(RecvTimeoutError::Disconnected) => None,
    }
}
