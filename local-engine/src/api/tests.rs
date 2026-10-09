use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{CompletionRequest, Event, PrefillProgress, Sampling, Signal};
use crate::Engine;

#[test]
fn a_silent_stream_times_out_without_closing_the_generation() {
    let (sender, receiver) = tokio::sync::mpsc::channel(1);
    let mut events = super::EventStream {
        receiver,
        progress: std::collections::VecDeque::new(),
        cancel: crate::bonsai_model::CancelToken::new(),
        not_sync: std::marker::PhantomData,
    };
    let (resume, waiting) = std::sync::mpsc::channel();
    let producer = std::thread::spawn(move || {
        // Close after one second if the wait is broken, so this regression
        // fails rather than hanging the suite indefinitely.
        if waiting.recv_timeout(Duration::from_secs(1)).is_ok() {
            sender
                .blocking_send(super::Delivered::Event(Event::Content("ready".into())))
                .expect("stream remains open after a timeout");
        }
    });
    assert!(matches!(
        events.next_timeout(Duration::from_millis(10)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    resume.send(()).expect("resume producer");
    assert!(matches!(events.next(), Some(Event::Content(text)) if text == "ready"));
    assert!(events.next().is_none());
    producer.join().expect("producer");
}

/// Prefill reports every chunk, in order, before the first event exists.
///
/// This is the claim the whole prefill heartbeat rests on: the model emits
/// nothing per token until decode starts, so a long prompt is minutes of silence
/// unless each chunk says it is still working. The prompt is made unique per run
/// because a prompt the cache already holds is prefilled as zero chunks, which
/// would make this vacuously true.
#[test]
#[ignore = "requires the pinned model; a real prefill is the only thing that reports"]
fn prefill_reports_every_chunk_before_the_first_event() {
    let handle = Engine::open().expect("open engine").into_handle();
    let marker = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after the epoch")
        .as_nanos();
    // 400 words is several chunks at `PREFILL_CHUNK` of 128, and one generated
    // token is enough because the assertion is about what arrives before it.
    let request = CompletionRequest {
        prompt: format!("{} {marker}", "word ".repeat(400)),
        max_tokens: 1,
        sampling: Sampling::default(),
        session: None,
    };
    let mut events = handle.complete(request).expect("queue");
    let mut boundaries: Vec<PrefillProgress> = Vec::new();
    let first_event = loop {
        match events.next_signal(None).expect("signal") {
            Some(Signal::Progress(progress)) => boundaries.push(progress),
            Some(Signal::Event(event)) => break event,
            None => break Event::Error("engine worker stopped".into()),
        }
    };
    assert!(
        boundaries.len() >= 2,
        "a multi-chunk prompt reported {boundaries:?}"
    );
    assert!(
        boundaries
            .windows(2)
            .all(|pair| pair[1].chunks == pair[0].chunks + 1),
        "chunk counts are not consecutive: {boundaries:?}"
    );
    // Tokens already in the K/V cache count, so the numbers climb and never
    // restart at the reusable boundary the way a per-call counter would.
    assert!(
        boundaries
            .windows(2)
            .all(|pair| pair[1].tokens >= pair[0].tokens),
        "token counts went backwards: {boundaries:?}"
    );
    assert!(
        boundaries.iter().all(|boundary| boundary.chunks >= 1),
        "the first chunk must be 1 rather than 0: {boundaries:?}"
    );
    assert!(
        !matches!(first_event, Event::Error(_)),
        "the request failed: {first_event:?}"
    );
}

/// A boundary is kept for `next_signal`, not dropped by `next_timeout`.
///
/// `next_timeout` is a wait on generated text and its deadline measures silence
/// from generated text, so a prefill boundary neither satisfies it nor resets it:
/// the protection for a long prefill is the *unbounded* first wait a caller has to
/// ask for, which is what `EventStream::next_signal(None)` is. What must hold on
/// this side is that the boundaries stay reachable afterwards, because the server
/// reads them from the same stream it read its events from, and a boundary dropped
/// here would silently disable the prefill heartbeat.
#[test]
#[ignore = "requires the pinned model; a real prefill is what reports"]
fn a_prefill_boundary_survives_a_wait_that_only_wanted_events() {
    let handle = Engine::open().expect("open engine").into_handle();
    let marker = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after the epoch")
        .as_nanos();
    let request = CompletionRequest {
        prompt: format!("{} {marker}", "word ".repeat(400)),
        max_tokens: 8,
        sampling: Sampling::default(),
        session: None,
    };
    let mut events = handle.complete(request).expect("queue");
    // Generous on purpose: this wait is allowed to cover the whole prefill, which
    // is the point. Only the *absence* of a deadline is what a caller needs there.
    let budget = Duration::from_secs(3600);
    loop {
        match events.next_timeout(budget) {
            Ok(Some(Event::Finished(_)) | None) => break,
            Ok(Some(_)) => {}
            Err(error) => panic!("an hour-long wait on a healthy generation expired: {error}"),
        }
    }
    // The generation is over and every boundary it reported is still here.
    let mut kept = 0;
    while let Ok(Some(Signal::Progress(_))) = events.next_signal(Some(Duration::from_millis(1))) {
        kept += 1;
    }
    assert!(
        kept >= 2,
        "a multi-chunk prefill left {kept} boundaries for `next_signal`"
    );
}
