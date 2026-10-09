use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::mpsc;

use serde_json::{Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::{GenerationStats, PrefillProgress, Signal, Stats};

use super::chunked::{BODY_FLUSH_BYTES, BODY_FLUSH_INTERVAL, Body, Flow, absorb};
use super::response::{EventWait, event_wait};
use super::sse::{FrameOutcome, send_frame};
use super::*;
fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).into()).collect()
}
#[test]
fn defaults_and_bounds() {
    let parsed = parse(&args(&["--no-thinking"])).expect("options");
    assert!(!parsed.thinking);
    for invalid in [
        vec!["--context", "4096"],
        vec!["--prefill-chunk", "128"],
        vec!["--max-queue", "8"],
        vec!["--http3"],
    ] {
        assert!(parse(&args(&invalid)).is_err());
    }
}
#[test]
fn thinking_is_not_a_flag_because_it_already_is_the_default() {
    // Reasoning is on unless the flag turns it off, so a positive spelling
    // could only ever be a no-op. The parser used to accept it anyway, which
    // made it a flag that silently did nothing while the help never listed it.
    let error = parse(&args(&["--thinking"])).expect_err("rejected");
    assert_eq!(error, "unknown option: --thinking", "{error}");
}
#[test]
fn rejects_media_message_parts() {
    assert!(message_text(&json!([{"type":"image_url"}])).is_err());
    assert_eq!(
        message_text(&json!([{"type":"text","text":"a"},{"type":"text","text":"b"}]))
            .expect("text"),
        "ab"
    );
}
#[test]
fn zstd_json_and_sse_framing() {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::ACCEPT_ENCODING,
        header::HeaderValue::from_static("gzip, zstd"),
    );
    assert!(wants_zstd(&headers));
}
#[test]
fn stall_timeout_defaults_to_thirty_seconds_and_rejects_nonsense() {
    assert_eq!(
        parse(&args(&[])).expect("defaults").stall,
        Duration::from_secs(30)
    );
    for accepted in [10, 30, 600, 3600] {
        assert_eq!(
            parse(&args(&["--stall-timeout", &accepted.to_string()]))
                .expect("accepted")
                .stall,
            Duration::from_secs(accepted),
            "{accepted} seconds"
        );
    }
    // 9 is the last rejected value, so a change to the floor has to be
    // deliberate here rather than silently widening what the server accepts.
    for rejected in [
        "0",
        "-1",
        "1",
        "5",
        "9",
        "1.5",
        "abc",
        "",
        "3601",
        "99999999999999999999",
    ] {
        let error = parse(&args(&["--stall-timeout", rejected])).expect_err("rejected");
        assert!(
            error.contains("--stall-timeout"),
            "{rejected:?} produced {error:?}"
        );
    }
    assert!(
        parse(&args(&["--stall-timeout"]))
            .expect_err("missing value")
            .contains("--stall-timeout requires a value")
    );
}
#[test]
fn only_waits_after_the_first_event_are_bounded() {
    // The premise of the exemption, so the reasoning recorded next to
    // `event_wait` cannot rot unnoticed: one prefill chunk of silence.
    assert_eq!(local_engine::resources::PREFILL_CHUNK, 128);
    let configured = parse(&args(&[])).expect("defaults").stall;
    // Time to first token is the model working, not a stalled consumer, so the
    // budget cannot apply to it. `Unbounded` is a separate outcome rather than
    // a large timeout, so it cannot be tightened into a real deadline.
    assert_eq!(event_wait(false, configured), EventWait::Unbounded);
    assert_eq!(
        event_wait(false, Duration::from_secs(1)),
        EventWait::Unbounded
    );
    // Every later wait gets the configured budget, not a fixed one.
    assert_eq!(
        event_wait(true, configured),
        EventWait::Bounded(Duration::from_secs(30))
    );
    assert_eq!(
        event_wait(true, Duration::from_secs(1)),
        EventWait::Bounded(Duration::from_secs(1))
    );
}
#[test]
fn bounded_frame_send_gives_up_only_on_a_client_that_stopped_reading() {
    let (sender, mut receiver) = mpsc::channel(1);
    let stall = Duration::from_millis(200);
    let frame = || Bytes::from_static(b"data: {}\n\n");

    // The capacity-1 slot starts free, so a live client never waits at all.
    assert_eq!(send_frame(&sender, frame(), stall), FrameOutcome::Sent);

    // The slot is now taken and nobody drains it: the frame is abandoned, and
    // only after the whole budget has passed.
    let started = Instant::now();
    assert_eq!(send_frame(&sender, frame(), stall), FrameOutcome::Stalled);
    let waited = started.elapsed();
    assert!(waited >= stall, "gave up after {waited:?}, under {stall:?}");
    assert!(waited < stall * 8, "waited {waited:?} to give up");

    // Draining the channel is what releases the next frame.
    assert!(receiver.try_recv().is_ok());
    assert_eq!(send_frame(&sender, frame(), stall), FrameOutcome::Sent);

    // A dropped receiver is the client-disconnect chain, not a stall, and must
    // not be reported as one.
    drop(receiver);
    assert_eq!(send_frame(&sender, frame(), stall), FrameOutcome::Closed);
}

/// The bytes a streamed body opens with, as the tests see them.
const STREAM_HEAD_CHAT: &str = "{\"choices\":[{\"index\":0,\"message\":{\"reasoning_content\":";

/// The one document shape both the buffered and the streamed body must produce.
///
/// Written out in full rather than assembled from `chunked`'s own helpers, so it
/// is an independent statement of the contract instead of a restatement of the
/// code under test. Every measurement is left at its default, which is why the
/// numbers here are all zero.
fn expected(chat: bool, content: &str, reasoning: &str) -> Value {
    let choice = if chat {
        json!({"index":0,"message":{"content":content,"reasoning_content":reasoning,"role":"assistant"},"finish_reason":"stop"})
    } else {
        json!({"index":0,"text":content,"finish_reason":"stop"})
    };
    json!({"choices":[choice],"created":0,"id":"local","model":"local","object":if chat {"chat.completion"} else {"text_completion"},"speculation":{"batched_tokens":0,"lookup":{"accepted_tokens":0,"cpu_seconds":0.0,"proposed_tokens":0,"rounds":0},"mtp":{"accepted_tokens":0,"proposed_tokens":0,"rounds":0}},"timings":{"elapsed_seconds":0.0,"first_token_seconds":null,"prefill_seconds":0.0},"usage":{"completion_tokens":0,"completion_tokens_details":{"reasoning_tokens":0},"prompt_tokens":7,"prompt_tokens_details":{"cache_source":"none","cached_tokens":0},"total_tokens":7}})
}

/// A response identity fixed for comparison against `expected`.
fn reply(chat: bool) -> Reply {
    Reply {
        protocol: if chat {
            Protocol::Chat {
                include_usage: false,
            }
        } else {
            Protocol::Completion
        },
        id: "local".into(),
        created: 0,
        model: "local".into(),
    }
}

/// A finished request whose only measurement is a prompt length.
fn stats() -> Stats {
    Stats {
        stop_reason: StopReason::Eos,
        cache_source: PromptCacheSource::None,
        reasoning_tokens: 0,
        generation: GenerationStats {
            prompt_tokens: 7,
            ..GenerationStats::default()
        },
    }
}

/// Fold signals through a body exactly as the pump does, and collect the frames.
fn frames(signals: Vec<Signal>, chat: bool, zstd: bool) -> Vec<Vec<u8>> {
    let (mut body, compressed) = Body::open(reply(chat), zstd);
    // The response may only claim a transfer encoding that is really in use.
    assert_eq!(
        compressed, zstd,
        "a body claimed an encoding it did not apply"
    );
    let mut out = Vec::new();
    out.extend(body.take(false).map(|frame| frame.to_vec()));
    for signal in signals {
        let flow = absorb(&mut body, signal);
        if flow == Flow::Stop {
            break;
        }
        if flow != Flow::Keep
            && let Some(frame) = body.take(flow == Flow::Last)
        {
            out.push(frame.to_vec());
        }
    }
    out
}
#[test]
fn only_finished_closes_the_document() {
    let (mut body, _) = Body::open(reply(true), false);
    // Token IDs are a streaming frame, not a member of this document, and
    // reasoning is held back because this body's order puts it after the answer.
    assert_eq!(
        absorb(&mut body, Signal::Event(Event::TokenIds(vec![1, 2, 3]))),
        Flow::Keep
    );
    assert_eq!(
        absorb(&mut body, Signal::Event(Event::Reasoning("why".into()))),
        Flow::Keep
    );
    // A boundary is written the moment it arrives, which is its only purpose.
    assert_eq!(
        absorb(
            &mut body,
            Signal::Progress(PrefillProgress {
                tokens: 0,
                chunks: 1
            })
        ),
        Flow::Flush
    );
    assert_eq!(
        absorb(&mut body, Signal::Event(Event::Finished(Box::new(stats())))),
        Flow::Last
    );
    assert_eq!(
        absorb(&mut body, Signal::Event(Event::Error("boom".into()))),
        Flow::Stop
    );
}

#[test]
fn a_prefill_boundary_is_written_as_whitespace_outside_the_answer() {
    let ticks = (1..=3).map(|chunks| {
        Signal::Progress(PrefillProgress {
            tokens: chunks * 128,
            chunks,
        })
    });
    let frames = frames(ticks.collect(), true, false);
    // The head goes out on its own, then one frame per boundary.
    assert_eq!(frames.len(), 4, "{frames:?}");
    assert_eq!(frames[0], STREAM_HEAD_CHAT.as_bytes());
    for frame in &frames[1..] {
        assert_eq!(frame, b" ", "{frame:?}");
    }
    // No quote has been written, so no whitespace can have landed inside the
    // content string and become part of the answer.
    let body = frames.concat();
    assert_eq!(body, [STREAM_HEAD_CHAT, "   "].concat().into_bytes());
}

#[test]
fn streamed_body_reassembles_to_the_buffered_document() {
    for chat in [true, false] {
        let mut signals = vec![Signal::Progress(PrefillProgress {
            tokens: 0,
            chunks: 1,
        })];
        // Reasoning is a chat-only event: a raw completion is generated with
        // `EventSplitter::new(false)`, so it never emits one.
        if chat {
            signals.push(Signal::Event(Event::Reasoning("Because.".into())));
        }
        signals.extend([
            Signal::Event(Event::Content("Hello".into())),
            Signal::Event(Event::Content(", world".into())),
            Signal::Event(Event::TokenIds(vec![7, 8])),
            Signal::Event(Event::Finished(Box::new(stats()))),
        ]);
        let frames = frames(signals, chat, false);
        let body = frames.concat();
        let streamed: Value = serde_json::from_slice(&body).expect("streamed body is valid JSON");
        assert_eq!(
            streamed,
            expected(chat, "Hello, world", "Because."),
            "{chat}"
        );
    }
}

#[test]
fn streamed_body_escapes_what_json_has_to_escape() {
    // Escaping is per character, so the pieces have to reassemble to exactly the
    // bytes a buffered body would have written. Anything that needs a backslash,
    // a control escape or multi-byte UTF-8 is checked here.
    let pieces = [
        "quote \" and backslash \\ ",
        "newline \n tab \t bell \u{7} ",
        "unicode é \u{2192} \u{1f600} ",
        "end",
    ];
    let signals = pieces
        .iter()
        .map(|piece| Signal::Event(Event::Content((*piece).into())))
        .chain([Signal::Event(Event::Finished(Box::new(stats())))])
        .collect();
    let body = frames(signals, true, false).concat();
    let streamed: Value = serde_json::from_slice(&body).expect("valid JSON");
    assert_eq!(streamed, expected(true, &pieces.concat(), ""));
    // The control character itself must be escaped away, not passed through.
    assert!(!body.contains(&0x07_u8), "raw bell byte in the body");
    // A backslash in the content is doubled on the wire and restored by the parse.
    assert!(
        body.windows(2).any(|pair| pair == b"\\\\"),
        "unescaped backslash"
    );
}

#[test]
fn an_empty_answer_still_produces_a_complete_document() {
    // The quote that opens the content string is written by the first piece, so
    // an answer of zero pieces has to have it written by the tail instead.
    let frames = frames(
        vec![Signal::Event(Event::Finished(Box::new(stats())))],
        true,
        false,
    );
    let streamed: Value = serde_json::from_slice(&frames.concat()).expect("valid JSON");
    assert_eq!(streamed, expected(true, "", ""));
}

#[test]
fn a_full_frame_is_written_without_waiting_for_the_interval() {
    // 96 KiB of answer in 1 KiB pieces. The interval alone would take 96 frames
    // of a second each; the byte bound is what stops a fast generation from
    // paying a chunk header per token.
    let piece = "x".repeat(1024);
    let signals = (0..96)
        .map(|_| Signal::Event(Event::Content(piece.clone())))
        .chain([Signal::Event(Event::Finished(Box::new(stats())))])
        .collect();
    let frames = frames(signals, true, false);
    // The head the pump writes before it waits for anything, twelve body frames
    // of 8 KiB, and the tail that closes the document.
    assert_eq!(frames.len(), 14, "{} frames", frames.len());
    assert_eq!(frames[0], STREAM_HEAD_CHAT.as_bytes());
    for frame in &frames[1..frames.len() - 1] {
        assert!(
            frame.len() >= BODY_FLUSH_BYTES,
            "a mid-answer frame of {} bytes is under {BODY_FLUSH_BYTES}",
            frame.len()
        );
    }
    let streamed: Value = serde_json::from_slice(&frames.concat()).expect("valid JSON");
    assert_eq!(streamed, expected(true, &piece.repeat(96), ""));
}

#[test]
fn the_interval_writes_a_frame_that_fills_no_byte_bound() {
    // The byte bound alone would let a slow generation go as unnoticed to a
    // client that has gone away as the buffered body this replaces: 8 KiB of text
    // is 150 s at the 2.6 tok/s measured here. This is the bound that makes the
    // write fail while the abandoned generation can still be dropped.
    let piece = "y".repeat(16);
    let (mut body, _) = Body::open(reply(true), false);
    let mut written = Vec::new();
    // The head the pump writes before it waits for anything.
    written.extend(body.take(false).map(|frame| frame.to_vec()));
    for _ in 0..2 {
        // A piece that neither fills a frame nor arrives a second after the last
        // one is held back. This is a real second of wall clock, which is the
        // price of proving the bound that matters is the one that fires.
        assert_eq!(
            absorb(&mut body, Signal::Event(Event::Content(piece.clone()))),
            Flow::Keep
        );
        std::thread::sleep(BODY_FLUSH_INTERVAL);
        assert_eq!(
            absorb(&mut body, Signal::Event(Event::Content(piece.clone()))),
            Flow::Flush
        );
        written.extend(body.take(false).map(|frame| frame.to_vec()));
    }
    assert_eq!(
        absorb(&mut body, Signal::Event(Event::Finished(Box::new(stats())))),
        Flow::Last
    );
    written.extend(body.take(true).map(|frame| frame.to_vec()));
    let total: usize = written.iter().map(Vec::len).sum();
    assert!(
        total < BODY_FLUSH_BYTES,
        "{total} bytes never reached the byte bound, so no frame above came from it"
    );
    assert_eq!(written.len(), 4, "{written:?}");
    let streamed: Value = serde_json::from_slice(&written.concat()).expect("valid JSON");
    assert_eq!(streamed, expected(true, &piece.repeat(4), ""));
}

#[test]
fn an_error_after_the_body_started_stops_the_body_instead_of_completing_it() {
    let frames = frames(
        vec![
            // Enough text to be written before the failure, so what a client
            // holds when the engine gives up is visible.
            Signal::Event(Event::Content("partial ".repeat(4096))),
            Signal::Event(Event::Error("metal said no".into())),
        ],
        true,
        false,
    );
    // The head and one body frame, then nothing: the status line is long gone by
    // then, so the only honest thing left to tell the client is that the
    // document is incomplete.
    assert_eq!(frames.len(), 2, "{frames:?}");
    let body = frames.concat();
    assert!(serde_json::from_slice::<Value>(&body).is_err(), "{body:?}");
    assert!(!body.windows(12).any(|pair| pair == b"finish_reason"));
}

#[test]
fn a_zstd_body_streams_frames_and_reassembles_to_the_same_document() {
    let piece = "compress me ".repeat(4096);
    let signals = [
        vec![Signal::Progress(PrefillProgress {
            tokens: 0,
            chunks: 1,
        })],
        (0..8)
            .map(|_| Signal::Event(Event::Content(piece.clone())))
            .collect(),
        vec![Signal::Event(Event::Finished(Box::new(stats())))],
    ]
    .concat();
    let plain = frames(signals.clone(), true, false);
    let compressed = frames(signals, true, true);
    // The head goes out compressed before any text exists, which is the whole
    // reason the encoder is flushed rather than left to buffer a whole block.
    assert!(compressed.len() >= 3, "{compressed:?}");
    for frame in &compressed[..compressed.len() - 1] {
        assert!(!frame.is_empty(), "an empty zstd frame carries nothing");
    }
    let decoded = zstd::decode_all(&compressed.concat()[..]).expect("one frame sequence");
    assert_eq!(
        serde_json::from_slice::<Value>(&decoded).expect("valid JSON"),
        serde_json::from_slice::<Value>(&plain.concat()[..]).expect("valid JSON")
    );
    // Compression still pays for itself, or is at worst a wash: a frame per piece
    // would make it worse than nothing at all, which is what `BODY_FLUSH_BYTES`
    // and `BODY_FLUSH_INTERVAL` exist to prevent. See that constant's table.
    assert!(
        compressed.concat().len() < plain.concat().len() + plain.concat().len() / 2,
        "{} compressed vs {} plain",
        compressed.concat().len(),
        plain.concat().len()
    );
}
#[test]
#[ignore = "requires the pinned model; validates server requests before generation"]
fn serve_real_bonsai_validates_before_streaming_and_preserves_history() {
    // `cargo test` runs this from `local-ai/`, where the shipped relative
    // default does not resolve, so name the same pinned file from the root.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    );
    let handle = Engine::open_model(path).expect("open engine").into_handle();
    for body in [
        json!({"prompt":"", "stream":true, "max_tokens":0}),
        json!({"prompt":"hello", "stream":true, "temperature":-1}),
        json!({"prompt":"hello", "stream":true, "max_tokens":usize::MAX}),
    ] {
        let prepared =
            prepare_generation(body.to_string().as_bytes(), false, true).expect("parse request");
        let GenerationRequest::Completion(request) = prepared.request else {
            unreachable!("completion request")
        };
        assert!(matches!(
            handle.complete(request).expect("queue").next(),
            Some(Event::Error(_))
        ));
    }
    let body = json!({"messages":[
            {"role":"user","content":"First"},
            {"role":"assistant","content":"Answer","reasoning_content":"prior reasoning"},
            {"role":"user","content":"Second"}
        ], "max_tokens":0});
    let prepared =
        prepare_generation(body.to_string().as_bytes(), true, true).expect("prepare history");
    let GenerationRequest::Chat(request) = prepared.request else {
        unreachable!("chat request")
    };
    assert!(
        handle
            .chat(request)
            .expect("queue")
            .all(|event| !matches!(event, Event::Error(_)))
    );
}
