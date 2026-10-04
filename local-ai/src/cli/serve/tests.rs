use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::mpsc;

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

#[test]
#[ignore = "requires BONSAI_GGUF; validates server requests before generation"]
fn serve_real_bonsai_validates_before_streaming_and_preserves_history() {
    let path = std::env::var("BONSAI_GGUF").expect("BONSAI_GGUF");
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
