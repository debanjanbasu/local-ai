//! CPU-only tests for resuming background Responses after a crash: a job
//! killed at the queued, prefill, generation and terminal-write boundaries is
//! regenerated from its stored request by a restarted server, its journal
//! continues after the restart boundary without repeating or contradicting an
//! event, and leases, cancellation, deletion and torn journals keep their
//! meaning. Scripted engine signals stand in for the model.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock, RwLockReadGuard};

use serde_json::{Value, json};

use local_engine::bonsai_model::StopReason;
use local_engine::{ChatRequest, Event, PrefillProgress, Signal};

use super::Admitted;
use super::background::{Background, Pump, Resumed};
use super::background_tests::{
    Scratch, Step, event, manual_temporary, script, stats, status, stopper, stored, streamed_doc,
    temporary_doc,
};
use super::response::{Protocol, Reply};
use super::responses::{ResponsesState, prepare_responses_with};
use super::resume::{Subscription, subscribe};
use super::store::ResponseStore;

const STALL: std::time::Duration = std::time::Duration::from_secs(30);

/// Held for writing while a test spawns a process. A child holds a copy of
/// every descriptor until it execs, so a lease another test drops meanwhile
/// would still look held to the resumption that follows.
static SPAWNING: RwLock<()> = RwLock::new(());

pub(super) fn no_spawns() -> RwLockReadGuard<'static, ()> {
    SPAWNING.read().unwrap_or_else(PoisonError::into_inner)
}

fn body(stream: bool) -> Value {
    json!({"input":"hi","instructions":"be brief","background":true,"stream":stream,"max_output_tokens":64})
}

/// Admit a background job for `body(stream)` and kill its process after
/// `signals` of `steps`.
fn crashed(store: &Arc<ResponseStore>, stream: bool, steps: Vec<Step>, signals: usize) -> String {
    let background = Arc::new(Background::default());
    let doc = streamed_doc(&body(stream), store, stream);
    let id = doc.id().to_owned();
    let (_, stop) = stopper();
    let (_, pump) = background
        .admit(doc, script(steps), stop, None)
        .expect("admitted");
    pump.crash_after(signals);
    id
}

fn prefill() -> Step {
    Box::new(|| {
        Signal::Progress(PrefillProgress {
            tokens: 0,
            chunks: 1,
        })
    })
}

fn regenerated() -> Vec<Step> {
    vec![
        prefill(),
        event(Event::Reasoning("again".into())),
        event(Event::Content("Hello".into())),
        event(Event::Finished(stats(StopReason::Eos))),
    ]
}

/// What a restarted server's engine was asked, and how often.
#[derive(Default)]
struct Engine {
    requests: Mutex<Vec<String>>,
    stopped: Arc<AtomicBool>,
}

type Submitted = (
    Box<dyn FnMut() -> Option<Signal> + Send>,
    Box<dyn Fn() + Send + Sync>,
    Option<Admitted>,
);

impl Engine {
    fn submit(
        &self,
        steps: Vec<Step>,
    ) -> impl FnOnce(ChatRequest, &Reply) -> crate::Result<Submitted> + '_ {
        move |request, _| {
            self.requests
                .lock()
                .expect("log")
                .push(format!("{request:?}"));
            let flag = Arc::clone(&self.stopped);
            Ok((
                Box::new(script(steps)),
                Box::new(move || flag.store(true, Ordering::SeqCst)),
                None,
            ))
        }
    }

    fn asked(&self) -> usize {
        self.requests.lock().expect("log").len()
    }
}

/// Resume `id` on a fresh server over `store`, expecting it to start.
fn resume(
    background: &Arc<Background>,
    store: &Arc<ResponseStore>,
    id: &str,
    engine: &Engine,
) -> Pump {
    let Resumed::Started(pump) = background
        .resume(store, id, (true, None, None), engine.submit(regenerated()))
        .expect("resumable")
    else {
        unreachable!("expected the job to start")
    };
    pump
}

fn outcome(
    background: &Arc<Background>,
    store: &Arc<ResponseStore>,
    id: &str,
    engine: &Engine,
) -> &'static str {
    match background
        .resume(store, id, (true, None, None), engine.submit(regenerated()))
        .expect("resume ran")
    {
        Resumed::Started(_) => "started",
        Resumed::Settled => "settled",
        Resumed::Owned => "owned",
        Resumed::Deferred => "deferred",
    }
}

fn journal(store: &ResponseStore, id: &str) -> Vec<Value> {
    std::fs::read_to_string(store.journal_file(id).expect("path"))
        .expect("journal")
        .lines()
        .map(|line| serde_json::from_str(line).expect("event"))
        .collect()
}

fn kinds(events: &[Value]) -> Vec<&str> {
    events
        .iter()
        .map(|event| event["type"].as_str().expect("type"))
        .collect()
}

/// Every frame `subscription` sends, parsed.
fn frames(subscription: Subscription) -> Vec<Value> {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
    let thread = std::thread::spawn(move || subscription.run(&sender, STALL));
    let mut events = Vec::new();
    while let Some(Ok(frame)) = receiver.blocking_recv() {
        let text = String::from_utf8(frame.to_vec()).expect("utf-8");
        let data = text
            .lines()
            .nth(1)
            .and_then(|line| line.strip_prefix("data: "))
            .expect("data line");
        events.push(serde_json::from_str(data).expect("json"));
    }
    thread.join().expect("subscriber");
    events
}

fn stream(
    background: &Background,
    store: &Arc<ResponseStore>,
    id: &str,
    after: Option<u64>,
) -> Vec<Value> {
    frames(
        subscribe(background, store, id, after)
            .expect("subscribable")
            .expect("found"),
    )
}

/// The invariants a stream that crossed a restart must keep: numbered from
/// 0 without gaps, one opening, one end that is the stored record, every
/// output index added once and finished once with a unique item ID, and the
/// final output exactly the finished items in index order.
fn assert_consistent(events: &[Value], record: &Value) {
    for (index, event) in events.iter().enumerate() {
        assert_eq!(event["sequence_number"], index as u64, "{event}");
    }
    let count = |kind: &str| events.iter().filter(|event| event["type"] == kind).count();
    assert_eq!(count("response.created"), 1);
    assert_eq!(count("response.queued"), 1);
    assert_eq!(count("response.in_progress"), 1);
    let last = events.last().expect("events");
    assert_eq!(last["response"], *record, "the end is the record");
    let mut added = HashSet::new();
    let mut done: HashMap<u64, Value> = HashMap::new();
    let mut ids = HashSet::new();
    for event in events {
        let index = event["output_index"].as_u64();
        match event["type"].as_str() {
            Some("response.output_item.added") => {
                assert!(added.insert(index.expect("index")), "index added twice");
                assert!(
                    ids.insert(event["item"]["id"].as_str().expect("id").to_owned()),
                    "item ID reused"
                );
            }
            Some("response.output_item.done") => {
                let index = index.expect("index");
                assert!(added.contains(&index), "done before added");
                assert!(
                    done.insert(index, event["item"].clone()).is_none(),
                    "done twice"
                );
            }
            Some("response.content_part.added" | "response.content_part.done") => {
                assert!(
                    !done.contains_key(&index.expect("index")),
                    "part after done"
                );
            }
            _ => {}
        }
    }
    assert_eq!(added.len(), done.len(), "every item finished");
    let output: Vec<Value> = (0..done.len() as u64)
        .map(|index| done[&index].clone())
        .collect();
    assert_eq!(
        record["output"],
        json!(output),
        "output is what was streamed"
    );
}

#[test]
fn a_job_killed_while_queued_is_regenerated_from_its_request() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let id = crashed(&scratch.open(), true, Vec::new(), 0);
    // The restarted server's store leaves it pending, with its request.
    let store = scratch.open();
    let pending = store.load(&id).expect("readable").expect("stored");
    assert_eq!(pending.response["status"], "queued");
    assert!(pending.request.is_some());
    let background = Arc::new(Background::default());
    let engine = Engine::default();
    let pump = resume(&background, &store, &id, &engine);
    // The engine is asked exactly what the original request asked.
    let original = prepare_responses_with(
        body(true).to_string().as_bytes(),
        true,
        Some(&store),
        None,
        "m",
    )
    .expect("prepared");
    assert_eq!(
        *engine.requests.lock().expect("log"),
        [format!("{:?}", original.request)]
    );
    // A client following live sees the rest; a replay sees all of it.
    let live = std::thread::spawn({
        let (background, store, id) = (Arc::clone(&background), Arc::clone(&store), id.clone());
        move || stream(&background, &store, &id, Some(1))
    });
    pump.run();
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "completed");
    assert_eq!(record["id"], id.as_str());
    let all = stream(&background, &store, &id, None);
    assert_consistent(&all, &record);
    assert_eq!(live.join().expect("live"), all[2..]);
    assert_eq!(
        kinds(&all),
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
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let ended = store.load(&id).expect("readable").expect("stored");
    assert!(
        ended.request.is_none(),
        "a terminal record drops its request"
    );
}

#[test]
fn a_job_killed_during_prefill_resumes_in_progress_without_a_second_announcement() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let id = crashed(&scratch.open(), true, vec![prefill()], 1);
    let store = scratch.open();
    assert_eq!(status(&store, &id), "in_progress");
    assert_eq!(
        kinds(&journal(&store, &id)),
        [
            "response.created",
            "response.queued",
            "response.in_progress"
        ]
    );
    let background = Arc::new(Background::default());
    resume(&background, &store, &id, &Engine::default()).run();
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "completed");
    assert_consistent(&stream(&background, &store, &id, None), &record);
}

#[test]
fn a_job_killed_mid_generation_finishes_the_lost_items_at_the_restart_boundary() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let steps = vec![
        prefill(),
        event(Event::Reasoning("think".into())),
        event(Event::Content("Hi".into())),
        event(Event::Content(" there".into())),
    ];
    let id = crashed(&scratch.open(), true, steps, 3);
    let store = scratch.open();
    let before = journal(&store, &id);
    let seen = before.len() as u64 - 1;
    assert_eq!(before.last().expect("event")["delta"], "Hi");
    let background = Arc::new(Background::default());
    let pump = resume(&background, &store, &id, &Engine::default());
    // A client that saw everything before the crash resumes after it.
    let resumed = std::thread::spawn({
        let (background, store, id) = (Arc::clone(&background), Arc::clone(&store), id.clone());
        move || stream(&background, &store, &id, Some(seen))
    });
    pump.run();
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "completed");
    let all = stream(&background, &store, &id, None);
    assert_consistent(&all, &record);
    assert_eq!(all[..before.len()], before[..], "nothing seen is rewritten");
    assert_eq!(resumed.join().expect("resumed"), all[before.len()..]);
    // The boundary finishes the open message with the text it had, then the
    // regenerated items follow at the next indices.
    let boundary = &all[before.len()..before.len() + 3];
    assert_eq!(
        kinds(boundary),
        [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done"
        ]
    );
    assert_eq!(boundary[0]["text"], "Hi");
    let output = record["output"].as_array().expect("output");
    let summary: Vec<(&str, &str)> = output
        .iter()
        .map(|item| {
            (
                item["type"].as_str().expect("type"),
                item["status"].as_str().expect("status"),
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            ("reasoning", "completed"),
            ("message", "incomplete"),
            ("reasoning", "completed"),
            ("message", "completed")
        ]
    );
    assert_eq!(output[1]["content"][0]["text"], "Hi");
    assert_eq!(output[3]["content"][0]["text"], "Hello");
}

#[test]
fn a_job_killed_twice_keeps_one_consistent_stream_and_its_request() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let steps = vec![prefill(), event(Event::Content("one".into()))];
    let id = crashed(&scratch.open(), true, steps, 2);
    // The first restart dies too, part way through its regeneration.
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let engine = Engine::default();
    let Resumed::Started(second) = background
        .resume(
            &store,
            &id,
            (true, None, None),
            engine.submit(vec![
                event(Event::Content("two".into())),
                event(Event::Content("!".into())),
            ]),
        )
        .expect("resumed")
    else {
        unreachable!("started")
    };
    second.crash_after(1);
    let store = scratch.open();
    assert!(
        store
            .load(&id)
            .expect("readable")
            .expect("stored")
            .request
            .is_some(),
        "still resumable"
    );
    let background = Arc::new(Background::default());
    resume(&background, &store, &id, &Engine::default()).run();
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "completed");
    assert_consistent(&stream(&background, &store, &id, None), &record);
    let texts: Vec<&str> = record["output"]
        .as_array()
        .expect("output")
        .iter()
        .filter_map(|item| item["content"][0]["text"].as_str())
        .collect();
    assert_eq!(texts, ["one", "two", "again", "Hello"]);
}

#[cfg(unix)]
#[test]
fn a_terminal_write_that_never_landed_is_regenerated() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let doc = streamed_doc(&body(true), &store, true);
    let id = doc.id().to_owned();
    let (_, stop) = stopper();
    let dir = scratch.0.clone();
    let steps: Vec<Step> = vec![
        prefill(),
        event(Event::Content("Hi".into())),
        // Every record write fails from here: the output reaches the
        // journal, but neither the completed record nor the fallback failure
        // does, as if the process died at the terminal write.
        Box::new(move || {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o500)).expect("chmod");
            Signal::Event(Event::Finished(stats(StopReason::Eos)))
        }),
    ];
    let (_, pump) = background
        .admit(doc, script(steps), stop, None)
        .expect("admitted");
    pump.run();
    scratch.set_mode(0o700);
    let store = scratch.open();
    assert_eq!(status(&store, &id), "in_progress");
    let before = journal(&store, &id);
    assert_eq!(
        before.last().expect("event")["type"],
        "response.output_item.done"
    );
    let background = Arc::new(Background::default());
    resume(&background, &store, &id, &Engine::default()).run();
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "completed");
    let all = stream(&background, &store, &id, None);
    assert_consistent(&all, &record);
    assert_eq!(all[..before.len()], before[..]);
    assert_eq!(record["output"].as_array().expect("output").len(), 3);
}

#[test]
fn a_job_killed_after_its_terminal_record_is_never_regenerated() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let store = scratch.open();
    let doc = streamed_doc(&body(true), &store, true);
    let id = doc.id().to_owned();
    let (_, stop) = stopper();
    let (_, pump) = Arc::new(Background::default())
        .admit(doc, script(regenerated()), stop, None)
        .expect("admitted");
    pump.run();
    // The process died after the completed record was durable but before
    // the end reached the journal.
    let path = store.journal_file(&id).expect("path");
    let mut lines = journal(&store, &id);
    assert_eq!(lines.pop().expect("end")["type"], "response.completed");
    let raw = std::fs::read_to_string(&path).expect("journal");
    let kept: Vec<&str> = raw.lines().take(lines.len()).collect();
    std::fs::write(&path, kept.join("\n") + "\n").expect("rewrite");
    let store = scratch.open();
    assert_eq!(store.resumable().expect("listed"), Vec::<String>::new());
    let background = Arc::new(Background::default());
    let engine = Engine::default();
    assert_eq!(outcome(&background, &store, &id, &engine), "settled");
    assert_eq!(engine.asked(), 0, "a durable end is never generated again");
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "completed");
    let all = stream(&background, &store, &id, None);
    assert_consistent(&all, &record);
    assert_eq!(all[..lines.len()], lines[..]);
}

#[test]
fn a_non_streamed_job_restarts_with_only_the_regenerated_output() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let steps = vec![prefill(), event(Event::Content("lost".into()))];
    let id = crashed(&scratch.open(), false, steps, 2);
    let store = scratch.open();
    assert_eq!(status(&store, &id), "in_progress");
    assert!(store.open_journal(&id).expect("readable").is_none());
    let background = Arc::new(Background::default());
    resume(&background, &store, &id, &Engine::default()).run();
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "completed");
    let texts: Vec<&str> = record["output"]
        .as_array()
        .expect("output")
        .iter()
        .filter_map(|item| item["content"][0]["text"].as_str())
        .collect();
    assert_eq!(texts, ["again", "Hello"]);
}

#[test]
fn a_held_lease_is_never_resumed_and_exactly_one_server_resumes_after_it() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let id = crashed(&scratch.open(), true, vec![prefill()], 1);
    let store = scratch.open();
    let before = journal(&store, &id);
    let engine = Engine::default();
    let lease = scratch.open().lease(&id).expect("lease").expect("free");
    assert_eq!(
        outcome(&Arc::new(Background::default()), &store, &id, &engine),
        "owned"
    );
    assert_eq!(engine.asked(), 0);
    assert_eq!(
        journal(&store, &id),
        before,
        "nothing written under a live lease"
    );
    drop(lease);
    // Two servers race; the first wins and the other sees its live lease.
    let (first, second) = (
        Arc::new(Background::default()),
        Arc::new(Background::default()),
    );
    let pump = resume(&first, &store, &id, &engine);
    let other = scratch.open();
    assert_eq!(outcome(&second, &other, &id, &engine), "owned");
    pump.run();
    assert_eq!(outcome(&second, &other, &id, &engine), "settled");
    assert_eq!(engine.asked(), 1, "generated once");
    assert_eq!(status(&store, &id), "completed");
}

/// A lease held by another process, through the system `lockf`, which
/// takes the same `flock` lock.
#[cfg(target_os = "macos")]
#[test]
fn another_process_holding_the_lease_is_never_resumed() {
    use std::io::{BufRead as _, BufReader, Write as _};
    use std::process::{Command, Stdio};

    let scratch = Scratch::new();
    let id = crashed(&scratch.open(), true, vec![prefill()], 1);
    let store = scratch.open();
    let sidecar = scratch.0.join(format!(".{id}.lock"));
    // `spawn` returns once the child has exec'd, dropping its copies.
    let spawning = SPAWNING.write().unwrap_or_else(PoisonError::into_inner);
    let mut holder = Command::new("/usr/bin/lockf")
        .arg("-k")
        .arg(&sidecar)
        .args(["/bin/sh", "-c", "echo locked; read _"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("lockf runs");
    drop(spawning);
    let mut line = String::new();
    BufReader::new(holder.stdout.take().expect("stdout"))
        .read_line(&mut line)
        .expect("holder reports");
    assert_eq!(line.trim(), "locked");
    let engine = Engine::default();
    let background = Arc::new(Background::default());
    assert_eq!(outcome(&background, &store, &id, &engine), "owned");
    assert_eq!(engine.asked(), 0);
    holder
        .stdin
        .take()
        .expect("stdin")
        .write_all(b"\n")
        .expect("release");
    assert!(holder.wait().expect("holder exits").success());
    resume(&background, &store, &id, &engine).run();
    assert_eq!(status(&store, &id), "completed");
}

#[test]
fn a_torn_journal_is_cut_to_its_last_whole_event_and_continued() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let steps = vec![prefill(), event(Event::Content("Hi".into()))];
    let id = crashed(&scratch.open(), true, steps, 2);
    let store = scratch.open();
    let path = store.journal_file(&id).expect("path");
    let mut lines: Vec<String> = std::fs::read_to_string(&path)
        .expect("journal")
        .lines()
        .map(str::to_owned)
        .collect();
    // The last batch landed in part: the item and its part, but not the
    // delta, then a torn fragment and garbage after it.
    lines.truncate(lines.len() - 2);
    let mut text = lines.join("\n");
    text.push_str("\n{\"type\":\"response.content_part.added\",\"sequ");
    std::fs::write(&path, text).expect("rewrite");
    let background = Arc::new(Background::default());
    resume(&background, &store, &id, &Engine::default()).run();
    let record = stored(&store, &id).expect("stored");
    let all = stream(&background, &store, &id, None);
    assert_consistent(&all, &record);
    let output = record["output"].as_array().expect("output");
    assert_eq!(output[0]["status"], "incomplete");
    assert_eq!(output[0]["content"][0]["text"], "");
    // The missing part is added before it is finished.
    let at = lines.len();
    assert_eq!(
        kinds(&all[at..at + 4]),
        [
            "response.content_part.added",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done"
        ]
    );

    // A whole but corrupt line ends the readable journal just as a torn one
    // does: it and everything after it are cut, and the stream continues.
    let steps = vec![prefill(), event(Event::Content("Hi".into()))];
    let id = crashed(&store, true, steps, 2);
    let store = scratch.open();
    let path = store.journal_file(&id).expect("path");
    let kept = journal(&store, &id);
    let mut text = std::fs::read_to_string(&path).expect("journal");
    text.push_str("not json\n{\"type\":\"response.output_text.delta\"}\n");
    std::fs::write(&path, text).expect("rewrite");
    let background = Arc::new(Background::default());
    resume(&background, &store, &id, &Engine::default()).run();
    let record = stored(&store, &id).expect("stored");
    let all = stream(&background, &store, &id, None);
    assert_consistent(&all, &record);
    assert_eq!(all[..kept.len()], kept[..]);
    assert_eq!(record["output"][0]["content"][0]["text"], "Hi");

    // A corrupt record is skipped by the listing and refused, never resumed.
    std::fs::write(scratch.0.join("resp_corrupt.json"), b"{\"trunc").expect("planted");
    let listed = scratch.open().resumable().expect("listed");
    assert_eq!(listed, Vec::<String>::new());
}

#[test]
fn cancellation_before_or_after_a_restart_wins_over_resumption() {
    let _spawns = no_spawns();
    // Cancelled while orphaned: nothing is generated.
    let scratch = Scratch::new();
    let id = crashed(&scratch.open(), true, vec![prefill()], 1);
    let store = scratch.open();
    let cancelled = Background::default()
        .cancel(&store, &id)
        .expect("cancelled")
        .expect("found");
    assert_eq!(cancelled["status"], "cancelled");
    let engine = Engine::default();
    assert_eq!(
        outcome(&Arc::new(Background::default()), &store, &id, &engine),
        "settled"
    );
    assert_eq!(engine.asked(), 0);
    assert_eq!(status(&store, &id), "cancelled");
    let events = journal(&store, &id);
    assert_eq!(events.last().expect("end")["code"], "response_cancelled");

    // Cancelled just after resuming: the regeneration stops and is drained.
    let id = crashed(
        &store,
        true,
        vec![prefill(), event(Event::Content("Hi".into()))],
        2,
    );
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let engine = Engine::default();
    let pump = resume(&background, &store, &id, &engine);
    let cancelled = background
        .cancel(&store, &id)
        .expect("cancelled")
        .expect("found");
    assert!(
        engine.stopped.load(Ordering::SeqCst),
        "native work cancelled"
    );
    pump.run();
    assert_eq!(status(&store, &id), "cancelled");
    assert_eq!(stored(&store, &id), Some(cancelled));
    let all = stream(&background, &store, &id, None);
    for (index, event) in all.iter().enumerate() {
        assert_eq!(event["sequence_number"], index as u64);
    }
    assert_eq!(all.last().expect("end")["code"], "response_cancelled");
    assert_eq!(
        all.iter()
            .filter(|event| event["type"] == "response.output_item.done")
            .count(),
        1,
        "only the lost item is finished, at the boundary"
    );

    // Deleted while orphaned: nothing is left to resume.
    let id = crashed(&store, true, vec![prefill()], 1);
    let store = scratch.open();
    assert!(super::background::delete_unowned(&store, &id).expect("deleted"));
    let engine = Engine::default();
    assert_eq!(
        outcome(&Arc::new(Background::default()), &store, &id, &engine),
        "settled"
    );
    assert_eq!(engine.asked(), 0);
    assert_eq!(stored(&store, &id), None);
    assert!(store.open_journal(&id).expect("readable").is_none());
}

#[test]
fn a_full_queue_defers_without_writing_twice_and_unresumable_jobs_fail() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let steps = vec![prefill(), event(Event::Content("Hi".into()))];
    let id = crashed(&scratch.open(), true, steps, 2);
    let store = scratch.open();
    let full =
        |_: ChatRequest, _: &Reply| -> crate::Result<Submitted> { Err(crate::Error::QueueFull) };
    for _ in 0..2 {
        let background = Arc::new(Background::default());
        assert!(matches!(
            background
                .resume(&store, &id, (true, None, None), full)
                .expect("ran"),
            Resumed::Deferred
        ));
    }
    let events = journal(&store, &id);
    assert_eq!(
        events
            .iter()
            .filter(|event| event["type"] == "response.output_item.done")
            .count(),
        1,
        "the boundary is appended once"
    );
    assert_eq!(status(&store, &id), "in_progress", "still pending");
    let background = Arc::new(Background::default());
    resume(&background, &store, &id, &Engine::default()).run();
    let record = stored(&store, &id).expect("stored");
    assert_consistent(&stream(&background, &store, &id, None), &record);

    // A record without its request fails, saying why.
    let mut requestless = stored(&store, &id).expect("stored");
    let requestless_id = "resp_requestless";
    requestless["id"] = json!(requestless_id);
    requestless["status"] = json!("in_progress");
    store.save(&requestless, &[]).expect("saved");
    let engine = Engine::default();
    assert_eq!(
        outcome(
            &Arc::new(Background::default()),
            &store,
            requestless_id,
            &engine
        ),
        "settled"
    );
    assert_eq!(engine.asked(), 0);
    let failed = stored(&store, requestless_id).expect("stored");
    assert_eq!(failed["status"], "failed");
    assert!(
        failed["error"]["message"]
            .as_str()
            .expect("message")
            .ends_with("its record keeps no request to regenerate it from")
    );

    // An engine refusal fails it too, rather than leave it pending.
    let id = crashed(&store, false, Vec::new(), 0);
    let refused = |_: ChatRequest, _: &Reply| -> crate::Result<Submitted> {
        Err(crate::Error::InvalidArgument("refused".into()))
    };
    let background = Arc::new(Background::default());
    assert!(matches!(
        background
            .resume(&store, &id, (true, None, None), refused)
            .expect("ran"),
        Resumed::Settled
    ));
    assert_eq!(status(&store, &id), "failed");
}

#[test]
fn shutdown_defers_resumption_and_temporary_jobs_are_never_resumable() {
    let _spawns = no_spawns();
    // A temporary job keeps no request: nothing about it is durable.
    let temporary_dir = Scratch::new();
    let (temporary, _) = manual_temporary(&temporary_dir);
    let doc = temporary_doc(
        &json!({"input":"hi","background":true,"store":false,"stream":true}),
        None,
        &temporary,
        true,
    );
    let id = doc.id().to_owned();
    let (_, stop) = stopper();
    let (_, pump) = Arc::new(Background::default())
        .admit(doc, script(Vec::new()), stop, None)
        .expect("admitted");
    pump.crash_after(0);
    let kept = temporary
        .store()
        .load(&id)
        .expect("readable")
        .expect("stored");
    assert!(kept.request.is_none());
    assert_eq!(
        temporary.store().resumable().expect("listed"),
        Vec::<String>::new()
    );

    let scratch = Scratch::new();
    let id = crashed(&scratch.open(), true, Vec::new(), 0);
    let store = scratch.open();
    let background = Arc::new(Background::default());
    background.close();
    let engine = Engine::default();
    assert_eq!(outcome(&background, &store, &id, &engine), "deferred");
    assert_eq!(engine.asked(), 0);
    assert_eq!(status(&store, &id), "queued", "left for the next server");
    assert_eq!(store.resumable().expect("listed"), [id]);
}

#[test]
fn a_continuation_resumes_from_its_stored_history_even_after_its_parent_is_deleted() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let store = scratch.open();
    let parent = json!({
        "id":"resp_parent","object":"response","created_at":1,"status":"completed",
        "background":false,"model":"m","error":null,
        "output":[{"id":"msg_answer","type":"message","status":"completed","role":"assistant",
            "content":[{"type":"output_text","text":"earlier","annotations":[]}]}],
    });
    let question = json!({"id":"msg_question","type":"message","role":"user",
        "content":[{"type":"input_text","text":"first"}]});
    store.save(&parent, &[question]).expect("parent stored");
    let request = json!({"input":"next","previous_response_id":"resp_parent","background":true});
    let original = prepare_responses_with(
        request.to_string().as_bytes(),
        true,
        Some(&store),
        None,
        "m",
    )
    .expect("prepared");
    let doc = streamed_doc(&request, &store, false);
    let id = doc.id().to_owned();
    let (_, stop) = stopper();
    let (_, pump) = Arc::new(Background::default())
        .admit(doc, script(Vec::new()), stop, None)
        .expect("admitted");
    pump.crash_after(0);
    assert!(store.delete("resp_parent").expect("deleted"));
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let engine = Engine::default();
    resume(&background, &store, &id, &engine).run();
    assert_eq!(
        *engine.requests.lock().expect("log"),
        [format!("{:?}", original.request)]
    );
    let record = store.load(&id).expect("readable").expect("stored");
    assert_eq!(record.response["status"], "completed");
    assert_eq!(record.response["previous_response_id"], "resp_parent");
    assert_eq!(record.input_items.len(), 3, "history kept");
}

#[test]
fn a_hosted_tool_job_resumes_only_if_no_hosted_call_had_started() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let store = scratch.open();
    let mut request = body(true);
    request["tools"] = json!([{"type":"function","name":"local_fs_read"}]);
    let crash = |store: &Arc<ResponseStore>| {
        let doc = streamed_doc(&request, store, true);
        let id = doc.id().to_owned();
        let (_, stop) = stopper();
        let (_, pump) = Arc::new(Background::default())
            .admit(doc, script(vec![prefill()]), stop, None)
            .expect("admitted");
        pump.crash_after(1);
        id
    };
    // No marker: no hosted call ever started, so it is regenerated.
    let id = crash(&store);
    let store = scratch.open();
    assert!(!store.tools_started(&id).expect("readable"));
    let engine = Engine::default();
    let background = Arc::new(Background::default());
    resume(&background, &store, &id, &engine).run();
    assert_eq!(engine.asked(), 1);
    assert_eq!(status(&store, &id), "completed");

    // Marked before a call started executing: it may have run, so the job
    // fails, saying exactly why, and nothing is regenerated.
    let id = crash(&store);
    store.mark_tools_started(&id).expect("marked");
    let store = scratch.open();
    let engine = Engine::default();
    assert_eq!(
        outcome(&Arc::new(Background::default()), &store, &id, &engine),
        "settled"
    );
    assert_eq!(engine.asked(), 0, "nothing is regenerated");
    let failed = stored(&store, &id).expect("stored");
    assert_eq!(failed["status"], "failed");
    assert!(
        failed["error"]["message"]
            .as_str()
            .expect("message")
            .contains("a hosted tool call of this response had started executing")
    );
    // The marker goes with the response.
    assert!(store.delete(&id).expect("deleted"));
    assert!(!store.tools_started(&id).expect("readable"));
}

#[test]
fn a_resumed_job_keeps_its_creator_until_it_ends() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let store = scratch.open();
    let creator = super::admin::Caller::Project {
        key_id: "key_a".into(),
        project_id: "proj_a".into(),
    };
    let mut prepared = prepare_responses_with(
        body(false).to_string().as_bytes(),
        true,
        Some(&store),
        None,
        "m",
    )
    .expect("prepared");
    prepared.echo.set_creator(creator.creator());
    let reply = Reply::new(Protocol::Responses(Arc::new(prepared.echo)), "m".into());
    let Protocol::Responses(echo) = &reply.protocol else {
        unreachable!("responses reply")
    };
    let doc = ResponsesState::new(&reply, Arc::clone(echo), false);
    let id = doc.id().to_owned();
    let (_, stop) = stopper();
    let (_, pump) = Arc::new(Background::default())
        .admit(doc, script(Vec::new()), stop, None)
        .expect("admitted");
    pump.crash_after(0);
    // The restarted server finds who created it beside its request.
    let store = scratch.open();
    let pending = store.load(&id).expect("readable").expect("stored");
    assert!(pending.request.is_some());
    assert_eq!(
        super::admin::Caller::from_creator(pending.creator.as_ref()),
        creator
    );
    // Resumed and killed again, the record still keeps it.
    resume(
        &Arc::new(Background::default()),
        &store,
        &id,
        &Engine::default(),
    )
    .crash_after(1);
    let store = scratch.open();
    let pending = store.load(&id).expect("readable").expect("stored");
    assert_eq!(pending.response["status"], "in_progress");
    assert_eq!(pending.creator, creator.creator());
    // Once it ends the record drops both.
    resume(
        &Arc::new(Background::default()),
        &store,
        &id,
        &Engine::default(),
    )
    .run();
    let ended = store.load(&id).expect("readable").expect("stored");
    assert_eq!(ended.response["status"], "completed");
    assert_eq!((ended.request, ended.creator), (None, None));
}

#[test]
fn a_record_without_a_creator_resumes_unattributed() {
    assert_eq!(
        super::admin::Caller::from_creator(None),
        super::admin::Caller::Anonymous
    );
    for malformed in [
        json!({}),
        json!({"project_id":"p"}),
        json!({"project_id":1,"api_key_id":"k"}),
    ] {
        assert_eq!(
            super::admin::Caller::from_creator(Some(&malformed)),
            super::admin::Caller::Anonymous
        );
    }
    for caller in [
        super::admin::Caller::Anonymous,
        super::admin::Caller::Static,
    ] {
        assert_eq!(caller.creator(), None);
    }
}
