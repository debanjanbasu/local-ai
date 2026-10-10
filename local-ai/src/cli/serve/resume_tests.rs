//! CPU-only tests for streamed background Responses: the journal's exact
//! order and cursors, live subscribers that leave or stall while generation
//! continues, cancellation and deletion waking subscribers, write failures,
//! crash recovery of a cut journal, and another server's live lease — all
//! driven by scripted engine signals, so no model or GPU is touched.

use std::sync::atomic::Ordering;
use std::sync::{Arc, mpsc as std_mpsc};
use std::thread;
use std::time::Duration;

use axum::http::StatusCode;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use local_engine::bonsai_model::StopReason;
use local_engine::{Event, PrefillProgress, Signal};

use super::QueueDepth;
use super::background::{Background, Pump};
use super::background_tests::{
    Scratch, Step, drained, event, script, stats, status, stopper, stored, streamed_doc,
};
use super::journal::terminal_events;
use super::responses::sse_frame;
use super::resume::{Subscription, subscribe};
use super::store::ResponseStore;

const STALL: Duration = Duration::from_secs(30);

/// Admit a streamed background job with `steps`.
fn admit(
    background: &Arc<Background>,
    store: &Arc<ResponseStore>,
    depth: &QueueDepth,
    steps: impl FnOnce(&str) -> Vec<Step>,
) -> (String, Pump, Arc<std::sync::atomic::AtomicBool>) {
    let doc = streamed_doc(
        &json!({"input":"hi","background":true,"stream":true}),
        store,
        true,
    );
    let id = doc.id().to_owned();
    let (stopped, stop) = stopper();
    let (queued, pump) = background
        .admit(doc, script(steps(&id)), stop, Some(depth.admit()))
        .expect("admitted");
    assert_eq!(queued["status"], "queued");
    (id, pump, stopped)
}

fn finished_steps() -> Vec<Step> {
    vec![
        Box::new(|| {
            Signal::Progress(PrefillProgress {
                tokens: 0,
                chunks: 1,
            })
        }),
        event(Event::Reasoning("think".into())),
        event(Event::Content("Hi".into())),
        event(Event::Content(" there".into())),
        event(Event::Finished(stats(StopReason::Eos))),
    ]
}

fn open(
    background: &Background,
    store: &Arc<ResponseStore>,
    id: &str,
    after: Option<u64>,
) -> Subscription {
    subscribe(background, store, id, after)
        .expect("subscribable")
        .expect("found")
}

/// Run `subscription` on its own thread, as the server does.
fn spawn(
    subscription: Subscription,
    stall: Duration,
) -> (
    mpsc::Receiver<Result<bytes::Bytes, std::io::Error>>,
    thread::JoinHandle<()>,
) {
    let (sender, receiver) = mpsc::channel(1);
    let thread = thread::spawn(move || subscription.run(&sender, stall));
    (receiver, thread)
}

fn frames_of(mut receiver: mpsc::Receiver<Result<bytes::Bytes, std::io::Error>>) -> Vec<String> {
    let mut frames = Vec::new();
    while let Some(Ok(frame)) = receiver.blocking_recv() {
        frames.push(String::from_utf8(frame.to_vec()).expect("utf-8"));
    }
    frames
}

/// Every frame `subscription` sends until its stream ends.
fn collect(subscription: Subscription) -> Vec<String> {
    let (receiver, thread) = spawn(subscription, STALL);
    let frames = frames_of(receiver);
    thread.join().expect("subscriber");
    frames
}

/// The event in `frame`, checking the frame is exactly the one `sse_frame`
/// makes of it.
fn parse(frame: &str) -> Value {
    let data = frame
        .lines()
        .nth(1)
        .and_then(|line| line.strip_prefix("data: "))
        .expect("data line");
    let event: Value = serde_json::from_str(data).expect("json");
    assert_eq!(sse_frame(&event), frame, "frame shape");
    event
}

fn kinds(frames: &[String]) -> Vec<String> {
    frames
        .iter()
        .map(|frame| parse(frame)["type"].as_str().expect("type").to_owned())
        .collect()
}

fn assert_numbered(frames: &[String], first: u64) {
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(parse(frame)["sequence_number"], first + index as u64);
    }
}

fn journal_lines(store: &ResponseStore, id: &str) -> Vec<String> {
    let path = store.journal_file(id).expect("path");
    std::fs::read_to_string(path)
        .expect("journal")
        .lines()
        .map(str::to_owned)
        .collect()
}

#[test]
fn events_are_numbered_in_order_and_resume_strictly_after_the_cursor() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let (id, pump, stopped) = admit(&background, &store, &depth, |_| finished_steps());
    pump.run();
    drained(&background);
    assert!(!stopped.load(Ordering::SeqCst));
    let frames = collect(open(&background, &store, &id, None));
    assert_eq!(
        kinds(&frames),
        [
            "response.created",
            "response.queued",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_numbered(&frames, 0);
    let events: Vec<Value> = frames.iter().map(|frame| parse(frame)).collect();
    assert_eq!(events[0]["response"]["status"], "queued");
    assert_eq!(events[1]["response"]["status"], "queued");
    assert_eq!(events[2]["response"]["status"], "in_progress");
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "completed");
    assert_eq!(
        events[16]["response"], record,
        "the end is the stored record"
    );
    assert_eq!(
        super::stored_answer(&store, &axum::http::Method::GET, &id, None).expect("get"),
        Some(record),
        "the JSON document is unchanged"
    );

    for after in [0, 1, 5, 15] {
        let resumed = collect(open(&background, &store, &id, Some(after)));
        assert_eq!(resumed, frames[after as usize + 1..], "after {after}");
    }
    for after in [16, 1000] {
        assert!(
            collect(open(&background, &store, &id, Some(after))).is_empty(),
            "a cursor at or past the end is an empty stream"
        );
    }
    // A restarted server replays the identical bytes: no regenerated
    // timestamps or IDs.
    let restarted = scratch.open();
    assert_eq!(
        collect(open(&Background::default(), &restarted, &id, None)),
        frames
    );
}

#[test]
fn a_live_subscriber_follows_and_one_that_leaves_does_not_cancel() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let (gate, opened) = std_mpsc::channel::<()>();
    let (id, pump, stopped) = admit(&background, &store, &depth, move |_| {
        let mut steps: Vec<Step> = vec![Box::new(move || {
            opened.recv().expect("gate");
            Signal::Event(Event::Content("a".into()))
        })];
        steps.extend(finished_steps());
        steps
    });
    let follower = spawn(open(&background, &store, &id, None), STALL);
    let (mut leaver, leaver_thread) = spawn(open(&background, &store, &id, None), STALL);
    let mut seen = Vec::new();
    for _ in 0..2 {
        let frame = leaver.blocking_recv().expect("frame").expect("bytes");
        seen.push(String::from_utf8(frame.to_vec()).expect("utf-8"));
    }
    assert_eq!(kinds(&seen), ["response.created", "response.queued"]);
    // No `in_progress` before the engine signals: the leaver is parked.
    drop(leaver);
    leaver_thread
        .join()
        .expect("a parked subscriber wakes when its client leaves");
    let pump = thread::spawn(move || pump.run());
    gate.send(()).expect("open the gate");
    pump.join().expect("pump");
    let frames = frames_of(follower.0);
    follower.1.join().expect("follower");
    assert!(
        !stopped.load(Ordering::SeqCst),
        "leaving never cancels the job"
    );
    assert_eq!(status(&store, &id), "completed");
    assert_eq!(&frames[..2], &seen[..]);
    assert_eq!(kinds(&frames)[2], "response.in_progress");
    assert_eq!(
        kinds(&frames).last().map(String::as_str),
        Some("response.completed")
    );
    assert_numbered(&frames, 0);
    assert_eq!(collect(open(&background, &store, &id, None)), frames);
    assert_eq!(depth.load(), 0);
}

#[test]
fn a_stalled_subscriber_is_dropped_without_blocking_generation() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let (id, pump, stopped) = admit(&background, &store, &depth, |_| finished_steps());
    // Never read: one frame fills the slot, the next waits out the budget.
    let (receiver, stalled) = spawn(
        open(&background, &store, &id, None),
        Duration::from_millis(20),
    );
    pump.run();
    assert_eq!(status(&store, &id), "completed", "generation ran on");
    stalled.join().expect("the stalled subscriber gave up");
    drop(receiver);
    assert!(!stopped.load(Ordering::SeqCst));
    assert_eq!(depth.load(), 0);
}

#[test]
fn cancellation_wakes_subscribers_and_ends_without_a_false_terminal() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();

    // Queued: no engine signal yet, so no `in_progress`.
    let (id, pump, stopped) = admit(&background, &store, &depth, |_| Vec::new());
    let (receiver, thread) = spawn(open(&background, &store, &id, None), STALL);
    let cancelled = background
        .cancel(&store, &id)
        .expect("cancel")
        .expect("found");
    let frames = frames_of(receiver);
    thread.join().expect("woken by the cancellation");
    assert!(stopped.load(Ordering::SeqCst));
    assert_eq!(
        kinds(&frames),
        ["response.created", "response.queued", "error"]
    );
    let end = parse(&frames[2]);
    assert_eq!(end["code"], "response_cancelled");
    assert_eq!(end["sequence_number"], 2);
    pump.run();
    assert_eq!(stored(&store, &id), Some(cancelled));
    assert_eq!(collect(open(&background, &store, &id, None)), frames);

    // Mid-generation: the open item ends incomplete, and the late finish
    // never reaches the journal.
    let (id, pump, _) = admit(&background, &store, &depth, |id| {
        let (background, store, id) = (Arc::clone(&background), Arc::clone(&store), id.to_owned());
        vec![
            event(Event::Content("par".into())),
            Box::new(move || {
                background.cancel(&store, &id).expect("cancel");
                Signal::Event(Event::Content("tial".into()))
            }),
            event(Event::Finished(stats(StopReason::Eos))),
        ]
    });
    pump.run();
    let frames = collect(open(&background, &store, &id, None));
    assert_eq!(
        kinds(&frames),
        [
            "response.created",
            "response.queued",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "error",
        ]
    );
    assert_numbered(&frames, 0);
    assert_eq!(parse(&frames[8])["item"]["status"], "incomplete");
    assert_eq!(status(&store, &id), "cancelled");
}

#[test]
fn deletion_stops_subscribers_removes_the_journal_and_keeps_the_lock() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let (id, pump, stopped) = admit(&background, &store, &depth, |_| finished_steps());
    let journal = store.journal_file(&id).expect("path");
    assert!(journal.exists());
    let (receiver, thread) = spawn(open(&background, &store, &id, None), STALL);
    assert!(background.delete(&store, &id).expect("delete"));
    let frames = frames_of(receiver);
    thread.join().expect("woken by the deletion");
    assert!(frames.len() <= 2, "nothing after the deletion");
    assert!(stopped.load(Ordering::SeqCst));
    assert!(!journal.exists(), "event data is removed");
    assert!(scratch.0.join(format!(".{id}.lock")).exists(), "lock kept");
    assert!(
        subscribe(&background, &store, &id, None)
            .expect("lookup")
            .is_none()
    );
    pump.run();
    assert_eq!(stored(&store, &id), None, "never resurrected");
    assert!(!journal.exists(), "nor its journal");
    assert!(
        subscribe(&background, &store, &id, None)
            .expect("lookup")
            .is_none()
    );
}

#[test]
fn only_streamed_background_responses_can_be_streamed() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let refused = |id: &str| {
        subscribe(&background, &store, id, None)
            .err()
            .expect("refused")
    };
    let mut foreground = streamed_doc(&json!({"input":"hi","stream":true}), &store, true);
    foreground.event(Event::Finished(stats(StopReason::Eos)));
    let (code, message) = refused(foreground.id());
    assert_eq!(code, StatusCode::BAD_REQUEST);
    assert!(message.contains("background"));

    let doc = streamed_doc(&json!({"input":"hi","background":true}), &store, false);
    let id = doc.id().to_owned();
    let (_, stop) = stopper();
    let (_, pump) = background
        .admit(
            doc,
            script(vec![event(Event::Finished(stats(StopReason::Eos)))]),
            stop,
            Some(depth.admit()),
        )
        .expect("admitted");
    pump.run();
    assert!(store.journal_file(&id).is_some_and(|path| !path.exists()));
    let (code, message) = refused(&id);
    assert_eq!(code, StatusCode::BAD_REQUEST);
    assert!(message.contains("stream=true"));
    assert!(
        subscribe(&background, &store, "resp_missing", None)
            .expect("lookup")
            .is_none()
    );
}

#[test]
fn a_journal_write_failure_ends_the_stream_from_the_stored_failure() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let (id, pump, stopped) = admit(&background, &store, &depth, |id| {
        let (background, store, id) = (Arc::clone(&background), Arc::clone(&store), id.to_owned());
        vec![
            event(Event::Content("a".into())),
            Box::new(move || {
                background.break_journal(&store, &id);
                Signal::Event(Event::Content("b".into()))
            }),
            event(Event::Finished(stats(StopReason::Eos))),
        ]
    });
    let (receiver, thread) = spawn(open(&background, &store, &id, None), STALL);
    pump.run();
    let frames = frames_of(receiver);
    thread.join().expect("subscriber");
    assert!(stopped.load(Ordering::SeqCst), "the job stopped");
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "failed");
    assert!(
        record["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("events could not be stored"))
    );
    assert_eq!(
        kinds(&frames),
        [
            "response.created",
            "response.queued",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "error",
            "response.failed",
        ]
    );
    assert_numbered(&frames, 0);
    assert_eq!(parse(&frames[7])["response"], record);
    assert_eq!(journal_lines(&store, &id).len(), 6, "no partial batch");
    assert_eq!(
        collect(open(&Background::default(), &scratch.open(), &id, None)),
        frames,
        "replay after a restart matches what the live client saw"
    );
}

#[test]
fn a_torn_journal_write_closes_live_streams_without_renumbering_any_event() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let (id, pump, stopped) = admit(&background, &store, &depth, |id| {
        let (background, store, id) = (Arc::clone(&background), Arc::clone(&store), id.to_owned());
        vec![
            event(Event::Reasoning("think".into())),
            Box::new(move || {
                background.tear_journal(&store, &id);
                Signal::Event(Event::Content("Hi".into()))
            }),
            event(Event::Finished(stats(StopReason::Eos))),
        ]
    });
    let (receiver, thread) = spawn(open(&background, &store, &id, None), STALL);
    pump.run();
    let frames = frames_of(receiver);
    thread.join().expect("subscriber");
    drained(&background);
    assert!(stopped.load(Ordering::SeqCst), "the job stopped");
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "failed");
    // The live stream ends on what was published: the lines the failed write
    // left behind already hold the next sequence numbers, so no end is made up.
    assert_eq!(
        kinds(&frames),
        [
            "response.created",
            "response.queued",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
        ]
    );
    assert_numbered(&frames, 0);
    let path = store.journal_file(&id).expect("path");
    assert!(
        !std::fs::read_to_string(path)
            .expect("journal")
            .ends_with('\n'),
        "the failed batch was left in part"
    );

    let replay = collect(open(&Background::default(), &scratch.open(), &id, None));
    assert_eq!(&replay[..frames.len()], &frames[..], "the published prefix");
    assert_eq!(
        kinds(&replay[frames.len()..]),
        [
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "error",
            "response.failed",
        ],
        "the whole lines on disk, then the end after them"
    );
    assert_numbered(&replay, 0);
    assert_eq!(parse(replay.last().expect("end"))["response"], record);
    for after in 0..replay.len() as u64 {
        assert_eq!(
            collect(open(&background, &store, &id, Some(after))),
            replay[after as usize + 1..],
            "after {after}"
        );
    }
}

#[test]
fn a_failed_end_cut_after_its_error_event_is_not_repeated() {
    let scratch = Scratch::new();
    let (id, lines, frames) = completed(&scratch);
    let store = scratch.open();
    let mut record = store.load(&id).expect("readable").expect("stored");
    record.response["status"] = json!("failed");
    record.response["error"] = json!({"code":"server_error","message":"the disk failed"});
    store
        .save(&record.response, &record.input_items)
        .expect("saved");
    // The end's write stopped after its `error` line.
    let kept = lines.len() - 1;
    let mut error = terminal_events(&record.response).remove(0);
    error["sequence_number"] = json!(kept);
    let mut whole = lines[..kept].to_vec();
    whole.push(serde_json::to_string(&error).expect("json"));
    rewrite(&store, &id, &whole, "{\"type\":\"response.fai");
    let replay = collect(open(&Background::default(), &scratch.open(), &id, None));
    assert_eq!(&replay[..kept], &frames[..kept]);
    assert_eq!(kinds(&replay[kept..]), ["error", "response.failed"]);
    assert_numbered(&replay, 0);
    assert_eq!(parse(&replay[kept])["code"], "server_error");
    assert_eq!(parse(&replay[kept + 1])["response"], record.response);
}

#[cfg(unix)]
#[test]
fn a_store_failure_never_advertises_completion_and_recovery_ends_the_stream() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let (id, pump, stopped) = admit(&background, &store, &depth, |_| {
        vec![
            event(Event::Content("Hi".into())),
            event(Event::Finished(stats(StopReason::Eos))),
        ]
    });
    let (receiver, thread) = spawn(open(&background, &store, &id, None), STALL);
    scratch.set_mode(0o500);
    pump.run();
    scratch.set_mode(0o700);
    let frames = frames_of(receiver);
    thread.join().expect("subscriber");
    assert!(stopped.load(Ordering::SeqCst));
    assert_eq!(
        kinds(&frames),
        ["response.created", "response.queued"],
        "neither in_progress nor completed was durable"
    );
    assert_eq!(status(&store, &id), "queued");
    drained(&background);
    let restarted = scratch.open();
    assert_eq!(status(&restarted, &id), "failed");
    let replay = collect(open(&Background::default(), &restarted, &id, None));
    assert_eq!(
        kinds(&replay),
        [
            "response.created",
            "response.queued",
            "error",
            "response.failed"
        ]
    );
    assert_numbered(&replay, 0);
    assert_eq!(&replay[..2], &frames[..]);
    assert_eq!(
        parse(&replay[3])["response"],
        stored(&restarted, &id).expect("stored")
    );
}

/// A completed streamed response: its ID, journal path and frames.
fn completed(scratch: &Scratch) -> (String, Vec<String>, Vec<String>) {
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let (id, pump, _) = admit(&background, &store, &depth, |_| finished_steps());
    pump.run();
    drained(&background);
    let frames = collect(open(&background, &store, &id, None));
    let lines = journal_lines(&store, &id);
    (id, lines, frames)
}

fn rewrite(store: &ResponseStore, id: &str, lines: &[String], tail: &str) {
    let mut text = lines.join("\n");
    if !lines.is_empty() {
        text.push('\n');
    }
    text.push_str(tail);
    std::fs::write(store.journal_file(id).expect("path"), text).expect("rewrite");
}

fn set_status(store: &ResponseStore, id: &str, status: &str) {
    let mut record = store.load(id).expect("readable").expect("stored");
    record.response["status"] = json!(status);
    record.response["completed_at"] = Value::Null;
    store
        .save(&record.response, &record.input_items)
        .expect("saved");
}

#[test]
fn recovery_reconciles_a_cut_journal_with_its_record() {
    const PARTIAL: &str = "{\"type\":\"response.outp";

    // Crashed mid-generation, mid-append: failed, with the end appended right
    // after the last whole event.
    let scratch = Scratch::new();
    let (id, lines, frames) = completed(&scratch);
    let store = scratch.open();
    rewrite(&store, &id, &lines[..lines.len() - 1], PARTIAL);
    set_status(&store, &id, "in_progress");
    let reopened = scratch.open();
    let record = stored(&reopened, &id).expect("stored");
    assert_eq!(record["status"], "failed");
    let replay = collect(open(&Background::default(), &reopened, &id, None));
    assert_eq!(&replay[..frames.len() - 1], &frames[..frames.len() - 1]);
    assert_eq!(
        kinds(&replay[frames.len() - 1..]),
        ["error", "response.failed"]
    );
    assert_numbered(&replay, 0);
    assert_eq!(parse(&replay[frames.len()])["response"], record);
    assert_eq!(
        journal_lines(&reopened, &id).len(),
        replay.len(),
        "tail cut"
    );

    // Crashed after the completed record, before its events: still
    // completed, and the replay is byte-for-byte the original.
    let scratch = Scratch::new();
    let (id, lines, frames) = completed(&scratch);
    let store = scratch.open();
    rewrite(&store, &id, &lines[..lines.len() - 1], PARTIAL);
    let reopened = scratch.open();
    assert_eq!(status(&reopened, &id), "completed", "not falsely failed");
    assert_eq!(
        collect(open(&Background::default(), &reopened, &id, None)),
        frames
    );

    // Crashed after publishing the end but before the record caught up: the
    // published end wins.
    let scratch = Scratch::new();
    let (id, lines, frames) = completed(&scratch);
    let store = scratch.open();
    let record = stored(&store, &id).expect("stored");
    set_status(&store, &id, "in_progress");
    rewrite(&store, &id, &lines, PARTIAL);
    let reopened = scratch.open();
    assert_eq!(stored(&reopened, &id), Some(record));
    assert_eq!(
        collect(open(&Background::default(), &reopened, &id, None)),
        frames
    );
    assert_eq!(
        journal_lines(&reopened, &id),
        lines,
        "tail cut, nothing added"
    );
}

#[test]
fn another_servers_live_stream_is_a_conflict_and_its_finished_one_replays() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let refusal = Arc::new(std::sync::Mutex::new(None));
    let (id, pump, _) = admit(&background, &store, &depth, |id| {
        let (dir, id, refusal) = (scratch.0.clone(), id.to_owned(), Arc::clone(&refusal));
        vec![
            event(Event::Content("a".into())),
            Box::new(move || {
                let other = Arc::new(ResponseStore::open(&dir).expect("second store"));
                *refusal.lock().expect("log") =
                    subscribe(&Background::default(), &other, &id, None)
                        .err()
                        .map(|(code, _)| code);
                Signal::Event(Event::Finished(stats(StopReason::Eos)))
            }),
        ]
    });
    pump.run();
    assert_eq!(*refusal.lock().expect("log"), Some(StatusCode::CONFLICT));
    assert_eq!(
        status(&store, &id),
        "completed",
        "the owner was not disturbed"
    );
    let own = collect(open(&background, &store, &id, None));
    let other = scratch.open();
    assert_eq!(
        collect(open(&Background::default(), &other, &id, None)),
        own
    );
}
