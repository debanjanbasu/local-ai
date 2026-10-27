//! CPU-only tests for background Responses combined with reasoning
//! summaries, `context_management` compaction and hosted tools: the summary
//! journaled as it is generated and replayed exactly, cancellation during a
//! summary, restarts before, during and after a summary or a compaction, the
//! extra generations' tokens in `usage`, and the hosted-tool marker that
//! decides whether a restarted job may be regenerated. Scripted engine
//! signals and fixed summaries stand in for the model.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use serde_json::{Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::{
    ChatMessage, ChatRequest, Event, EventStream, GenerationStats, PrefillProgress, Signal, Stats,
    ToolCall,
};
use local_services::tools::{Limits, Permissions, Toolbox};

use super::background::{Background, Pump, Resumed};
use super::background_tests::{Scratch, Step, drained, event, script, stats, stopper, stored};
use super::reasoning_crypto::ReasoningCipher;
use super::recovery_tests::no_spawns;
use super::response::{Protocol, Reply};
use super::responses::{
    Generate, PreparedResponses, ResponsesState, Summarizer, prepare_responses,
    prepare_responses_with,
};
use super::resume::subscribe;
use super::store::ResponseStore;
use super::tool_loop::{Bounds, Loop, start};

const STALL: std::time::Duration = std::time::Duration::from_secs(30);

fn measured(prompt: usize, reused: usize, generated: usize) -> Stats {
    Stats {
        stop_reason: StopReason::Eos,
        cache_source: PromptCacheSource::None,
        reasoning_tokens: 0,
        generation: GenerationStats {
            prompt_tokens: prompt,
            reused_prompt_tokens: reused,
            generated_tokens: generated,
            ..GenerationStats::default()
        },
    }
}

/// What every fixed summary generation here costs.
fn summary_cost() -> Stats {
    measured(20, 5, 4)
}

/// A summary generation answering with `pieces`, stopping, as the engine's
/// does, once told nobody listens.
fn summarizer(pieces: &'static [&'static str]) -> Summarizer {
    let generate: Generate = Arc::new(move |_, piece| {
        for text in pieces {
            if !piece(text) {
                return Err(crate::Error::Generation("stopped".into()));
            }
        }
        Ok(summary_cost())
    });
    Summarizer(generate)
}

fn prepared(
    body: &Value,
    store: &Arc<ResponseStore>,
    cipher: Option<&Arc<ReasoningCipher>>,
) -> PreparedResponses {
    prepare_responses_with(body.to_string().as_bytes(), Some(store), cipher, "m")
        .expect("request accepted")
}

/// The Response state machine for `prepared`, summarizing with `summarizer`.
fn doc(prepared: PreparedResponses, summarizer: Option<Summarizer>) -> ResponsesState {
    let mut echo = prepared.echo;
    if let Some(summarizer) = summarizer {
        echo.set_summarizer(summarizer);
    }
    let reply = Reply::new(Protocol::Responses(Arc::new(echo)), "m".into());
    let Protocol::Responses(echo) = &reply.protocol else {
        unreachable!("responses reply")
    };
    ResponsesState::new(&reply, Arc::clone(echo), true)
}

fn admit(background: &Arc<Background>, doc: ResponsesState, steps: Vec<Step>) -> (String, Pump) {
    let id = doc.id().to_owned();
    let (_, stop) = stopper();
    let (queued, pump) = background
        .admit(doc, script(steps), stop, None)
        .expect("admitted");
    assert_eq!(queued["status"], "queued");
    (id, pump)
}

fn prefill() -> Step {
    Box::new(|| {
        Signal::Progress(PrefillProgress {
            tokens: 0,
            chunks: 1,
        })
    })
}

fn thought() -> Vec<Step> {
    vec![
        prefill(),
        event(Event::Reasoning("think".into())),
        event(Event::Content("Hi".into())),
        event(Event::Finished(stats(StopReason::Eos))),
    ]
}

fn regenerated() -> Vec<Step> {
    vec![
        prefill(),
        event(Event::Reasoning("again".into())),
        event(Event::Content("Hello".into())),
        event(Event::Finished(stats(StopReason::Eos))),
    ]
}

/// Resume `id` on a fresh server over `store`, regenerating `steps` and
/// recording the request it was asked to generate.
fn resume(
    store: &Arc<ResponseStore>,
    id: &str,
    cipher: Option<&Arc<ReasoningCipher>>,
    summary: Option<&Summarizer>,
    asked: &Mutex<Vec<ChatRequest>>,
) -> (Arc<Background>, Pump) {
    let background = Arc::new(Background::default());
    let resumed = background
        .resume(store, id, (cipher, summary), |request, _: &Reply| {
            asked.lock().expect("log").push(request);
            Ok((script(regenerated()), || {}, None))
        })
        .expect("resume ran");
    let Resumed::Started(pump) = resumed else {
        unreachable!("expected the job to start")
    };
    (background, pump)
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

fn count(events: &[Value], kind: &str) -> usize {
    events.iter().filter(|event| event["type"] == kind).count()
}

/// Every event a client streaming `id` after `after` receives.
fn stream(
    background: &Background,
    store: &Arc<ResponseStore>,
    id: &str,
    after: Option<u64>,
) -> Vec<Value> {
    let subscription = subscribe(background, store, id, after)
        .expect("subscribable")
        .expect("found");
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

/// What a stream must keep across a restart, summaries included: numbered
/// without gaps, one opening, every output index added and finished once
/// with a unique item ID, every summary part opened before its deltas and
/// closed once, with their text, before its item finishes, and the final
/// output exactly the finished items.
fn assert_consistent(events: &[Value], record: &Value) {
    for (index, event) in events.iter().enumerate() {
        assert_eq!(event["sequence_number"], index as u64, "{event}");
    }
    assert_eq!(count(events, "response.created"), 1);
    assert_eq!(count(events, "response.queued"), 1);
    assert_eq!(count(events, "response.in_progress"), 1);
    let mut added = HashSet::new();
    let mut ids = HashSet::new();
    let mut done: HashMap<u64, Value> = HashMap::new();
    // Per output index: the summary text streamed, and whether it closed.
    let mut summaries: HashMap<u64, (String, bool)> = HashMap::new();
    for event in events {
        let index = event["output_index"].as_u64();
        match event["type"].as_str().expect("type") {
            "response.output_item.added" => {
                assert!(added.insert(index.expect("index")), "index added twice");
                assert!(
                    ids.insert(event["item"]["id"].as_str().expect("id").to_owned()),
                    "item ID reused"
                );
            }
            "response.output_item.done" => {
                let index = index.expect("index");
                assert!(added.contains(&index), "done before added");
                if let Some((text, closed)) = summaries.get(&index) {
                    assert!(closed, "item finished with its summary part open");
                    assert_eq!(
                        event["item"]["summary"],
                        json!([{"type":"summary_text","text":text}])
                    );
                }
                assert!(
                    done.insert(index, event["item"].clone()).is_none(),
                    "done twice"
                );
            }
            "response.reasoning_summary_part.added" => {
                let index = index.expect("index");
                assert!(!done.contains_key(&index), "summary after done");
                assert!(
                    summaries.insert(index, (String::new(), false)).is_none(),
                    "summary part opened twice"
                );
            }
            "response.reasoning_summary_text.delta" => {
                let (text, closed) = summaries
                    .get_mut(&index.expect("index"))
                    .expect("delta of an open part");
                assert!(!*closed, "delta after the part closed");
                text.push_str(event["delta"].as_str().expect("delta"));
            }
            "response.reasoning_summary_text.done" => {
                let (text, _) = &summaries[&index.expect("index")];
                assert_eq!(event["text"], *text);
            }
            "response.reasoning_summary_part.done" => {
                let (text, closed) = summaries
                    .get_mut(&index.expect("index"))
                    .expect("an open part");
                assert!(!*closed, "summary part closed twice");
                *closed = true;
                assert_eq!(event["part"]["text"], *text);
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
    if record["status"] != "cancelled" {
        assert_eq!(events.last().expect("events")["response"], *record);
    }
}

/// `(type, status, summary text)` of each output item.
fn shape(record: &Value) -> Vec<(String, String, String)> {
    record["output"]
        .as_array()
        .expect("output")
        .iter()
        .map(|item| {
            let text = |value: &Value| value.as_str().unwrap_or_default().to_owned();
            (
                text(&item["type"]),
                text(&item["status"]),
                text(&item["summary"][0]["text"]),
            )
        })
        .collect()
}

fn usage(record: &Value) -> [u64; 5] {
    let usage = &record["usage"];
    let number = |pointer: &str| {
        usage
            .pointer(pointer)
            .and_then(Value::as_u64)
            .expect(pointer)
    };
    let usage = [
        number("/input_tokens"),
        number("/input_tokens_details/cached_tokens"),
        number("/output_tokens"),
        number("/output_tokens_details/reasoning_tokens"),
        number("/total_tokens"),
    ];
    assert_eq!(usage[4], usage[0] + usage[2], "total is input plus output");
    assert!(usage[1] <= usage[0] && usage[3] <= usage[2]);
    usage
}

fn summarized_body() -> Value {
    json!({"input":"hi","background":true,"stream":true,"reasoning":{"summary":"auto"}})
}

#[test]
fn a_background_summary_is_journaled_as_it_is_generated_and_replays_exactly() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let store = scratch.open();
    // The summary checks, between its pieces, that the first is on disk.
    let path: Arc<Mutex<Option<PathBuf>>> = Arc::default();
    let live = Arc::new(AtomicBool::new(false));
    let generate: Generate = {
        let (path, live) = (Arc::clone(&path), Arc::clone(&live));
        Arc::new(move |_, piece| {
            assert!(piece("Summed"));
            let journal = path.lock().expect("path").clone().expect("set");
            let text = std::fs::read_to_string(journal).expect("journal");
            live.store(
                text.contains("response.reasoning_summary_text.delta"),
                Ordering::SeqCst,
            );
            assert!(piece(" up."));
            Ok(summary_cost())
        })
    };
    let doc = doc(
        prepared(&summarized_body(), &store, None),
        Some(Summarizer(generate)),
    );
    *path.lock().expect("path") = store.journal_file(doc.id());
    let background = Arc::new(Background::default());
    let (id, pump) = admit(&background, doc, thought());
    pump.run();
    drained(&background);
    assert!(
        live.load(Ordering::SeqCst),
        "the first piece was journaled live"
    );
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "completed", "{record}");
    assert_eq!(record["reasoning"]["summary"], "auto");
    let events = stream(&background, &store, &id, None);
    assert_consistent(&events, &record);
    let order = kinds(&events);
    let at = |kind: &str| order.iter().rposition(|seen| *seen == kind).expect(kind);
    assert!(at("response.output_text.done") < at("response.reasoning_summary_part.added"));
    assert!(at("response.reasoning_summary_part.done") < at("response.output_item.done"));
    assert_eq!(*order.last().expect("end"), "response.completed");
    assert_eq!(
        shape(&record),
        [
            ("reasoning".into(), "completed".into(), "Summed up.".into()),
            ("message".into(), "completed".into(), String::new()),
        ]
    );
    // The response's own 7 + 3 and the summary's 20 (5 reused) + 4.
    assert_eq!(usage(&record), [27, 5, 7, 1, 34]);
    for after in 0..events.len() as u64 {
        assert_eq!(
            stream(&background, &store, &id, Some(after)),
            events[after as usize + 1..],
            "after {after}"
        );
    }
}

#[test]
fn cancelling_during_a_background_summary_settles_cancelled_at_once() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let store = scratch.open();
    let (started, began) = mpsc::channel();
    let generate: Generate = Arc::new(move |_, piece| {
        // Bounded: a summary that is never stopped ends, and the test fails.
        for round in 0..100_000 {
            if !piece("x") {
                return Err(crate::Error::Generation("stopped".into()));
            }
            if round == 0 {
                let _ = started.send(());
            }
            std::thread::yield_now();
        }
        Ok(summary_cost())
    });
    let doc = doc(
        prepared(&summarized_body(), &store, None),
        Some(Summarizer(generate)),
    );
    let background = Arc::new(Background::default());
    let (id, pump) = admit(&background, doc, thought());
    let pumping = std::thread::spawn(move || pump.run());
    began
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the summary started");
    let cancelled = background
        .cancel(&store, &id)
        .expect("cancelled")
        .expect("found");
    pumping.join().expect("pump");
    assert_eq!(cancelled["status"], "cancelled", "{cancelled}");
    assert_eq!(cancelled["usage"], Value::Null);
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record, cancelled);
    let events = stream(&background, &store, &id, None);
    assert_consistent(&events, &record);
    assert_eq!(events.last().expect("end")["type"], "error");
    let summary = record["output"][0]["summary"][0]["text"]
        .as_str()
        .expect("partial summary");
    assert!(summary.len() < 100_000, "it stopped part way");
    assert_eq!(record["output"][0]["status"], "incomplete");
}

#[test]
fn a_restart_during_a_background_summary_closes_its_part_and_summarizes_again() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    // A process killed part way through the summary: the pump stops dead,
    // writing nothing more, and its lease is released.
    let crashing: Generate = Arc::new(|_, piece| {
        assert!(piece("Sum"));
        assert!(piece("med"));
        std::panic::resume_unwind(Box::new("simulated crash during the summary"))
    });
    let id = {
        let store = scratch.open();
        let doc = doc(
            prepared(&summarized_body(), &store, None),
            Some(Summarizer(crashing)),
        );
        let (id, pump) = admit(&Arc::new(Background::default()), doc, thought());
        assert!(std::thread::spawn(move || pump.run()).join().is_err());
        id
    };
    let store = scratch.open();
    let before = journal(&store, &id);
    assert_eq!(before.last().expect("event")["delta"], "med");
    let asked = Mutex::default();
    let again = summarizer(&["Again."]);
    let (background, pump) = resume(&store, &id, None, Some(&again), &asked);
    pump.run();
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "completed", "{record}");
    let events = stream(&background, &store, &id, None);
    assert_consistent(&events, &record);
    assert_eq!(
        events[..before.len()],
        before[..],
        "nothing seen is rewritten"
    );
    assert_eq!(
        kinds(&events[before.len()..before.len() + 3]),
        [
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",
        ],
        "the boundary closes the cut summary"
    );
    assert_eq!(
        shape(&record),
        [
            ("reasoning".into(), "incomplete".into(), "Summed".into()),
            ("message".into(), "completed".into(), String::new()),
            ("reasoning".into(), "completed".into(), "Again.".into()),
            ("message".into(), "completed".into(), String::new()),
        ]
    );
    assert_eq!(record["output"][3]["content"][0]["text"], "Hello");
    // Only the resuming server's generations: its answer and its summary.
    assert_eq!(usage(&record), [27, 5, 7, 1, 34]);
    assert_eq!(asked.lock().expect("log").len(), 1);
}

#[test]
fn a_restart_before_the_summary_leaves_that_reasoning_incomplete_and_unsummarized() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let id = {
        let store = scratch.open();
        let doc = doc(
            prepared(&summarized_body(), &store, None),
            Some(summarizer(&["never"])),
        );
        let (id, pump) = admit(&Arc::new(Background::default()), doc, thought());
        // The reasoning is finished and waits for the summary; the answer is
        // part way.
        pump.crash_after(3);
        id
    };
    let store = scratch.open();
    let asked = Mutex::default();
    let again = summarizer(&["Again."]);
    let (background, pump) = resume(&store, &id, None, Some(&again), &asked);
    pump.run();
    let record = stored(&store, &id).expect("stored");
    assert_consistent(&stream(&background, &store, &id, None), &record);
    assert_eq!(
        shape(&record),
        [
            ("reasoning".into(), "incomplete".into(), String::new()),
            ("message".into(), "incomplete".into(), String::new()),
            ("reasoning".into(), "completed".into(), "Again.".into()),
            ("message".into(), "completed".into(), String::new()),
        ]
    );
    assert_eq!(record["output"][0]["summary"], json!([]));
}

// ---- compaction ------------------------------------------------------------

fn cipher(scratch: &Scratch) -> Arc<ReasoningCipher> {
    let dir = scratch.0.with_extension("key");
    std::fs::create_dir_all(&dir).expect("key directory");
    Arc::new(ReasoningCipher::open(dir.join("reasoning.key")).expect("cipher opens"))
}

/// One token per rendered message: enough to plan a compaction.
#[expect(clippy::needless_pass_by_value)]
fn by_messages(messages: Vec<ChatMessage>) -> crate::Result<usize> {
    local_engine::Engine::render_chat_with_tools(&messages, false, &[])?;
    Ok(messages.len())
}

/// A background request compacted before admission, as the server does
/// when its prompt is due, with the compaction generation's cost counted.
fn compacted(
    store: &Arc<ResponseStore>,
    cipher: &Arc<ReasoningCipher>,
) -> (PreparedResponses, Value) {
    let mut input = Vec::new();
    for turn in 0..4 {
        input.push(json!({"type":"message","role":"user","content":format!("question {turn}")}));
        input.push(json!({"type":"message","role":"assistant","content":format!("answer {turn}")}));
    }
    input.push(json!({"type":"message","role":"user","content":"and now?"}));
    let body = json!({"input":input,"instructions":"be brief","background":true,"stream":true,
        "context_management":[{"type":"compaction","compact_threshold":1000}]});
    let mut prepared = prepared(&body, store, Some(cipher));
    let pass = prepared
        .history
        .compaction(2, &prepared.request, by_messages)
        .expect("plans")
        .expect("something to summarize");
    let (item, history) = pass
        .finish(
            "Four questions were answered.",
            &prepared.history,
            cipher,
            "m",
        )
        .expect("seals");
    prepared.echo.spend(&measured(30, 0, 6));
    prepared.compacted(item.clone(), history).expect("renders");
    (prepared, item)
}

#[test]
fn a_background_compaction_is_streamed_once_and_counted_in_usage() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let store = scratch.open();
    let cipher = cipher(&scratch);
    let (prepared, item) = compacted(&store, &cipher);
    let background = Arc::new(Background::default());
    let (id, pump) = admit(&background, doc(prepared, None), thought());
    pump.run();
    drained(&background);
    let record = stored(&store, &id).expect("stored");
    assert_eq!(record["status"], "completed", "{record}");
    assert_eq!(record["output"][0], item);
    let events = stream(&background, &store, &id, None);
    assert_consistent(&events, &record);
    assert_eq!(
        kinds(&events[2..5]),
        [
            "response.in_progress",
            "response.output_item.added",
            "response.output_item.done"
        ]
    );
    assert_eq!(events[3]["item"], item);
    // The response's own 7 + 3 and the compaction's 30 + 6.
    assert_eq!(usage(&record), [37, 0, 9, 1, 46]);
}

#[test]
fn a_restarted_background_compaction_is_reused_not_run_again() {
    let _spawns = no_spawns();
    let scratch = Scratch::new();
    let cipher = cipher(&scratch);
    // Killed while queued (nothing announced) and mid-answer (the
    // compaction item already streamed).
    for signals in [0, 2] {
        let (id, item, messages) = {
            let store = scratch.open();
            let (prepared, item) = compacted(&store, &cipher);
            let messages = format!("{:?}", prepared.request.messages);
            let steps = vec![prefill(), event(Event::Content("Hi".into()))];
            let (id, pump) = admit(&Arc::new(Background::default()), doc(prepared, None), steps);
            pump.crash_after(signals);
            (id, item, messages)
        };
        let store = scratch.open();
        let asked = Mutex::default();
        let (background, pump) = resume(&store, &id, Some(&cipher), None, &asked);
        pump.run();
        let asked = asked.into_inner().expect("log");
        assert_eq!(asked.len(), 1);
        assert_eq!(
            format!("{:?}", asked[0].messages),
            messages,
            "the same compacted prompt"
        );
        let record = stored(&store, &id).expect("stored");
        assert_eq!(record["status"], "completed", "{record}");
        assert_eq!(record["output"][0], item);
        let events = stream(&background, &store, &id, None);
        assert_consistent(&events, &record);
        let announced = events
            .iter()
            .filter(|event| {
                event["type"] == "response.output_item.added"
                    && event["item"]["type"] == "compaction"
            })
            .count();
        assert_eq!(announced, 1, "after {signals} signals");
        // The compaction ran before the restart; only the regeneration is
        // counted.
        assert_eq!(usage(&record), [7, 0, 3, 1, 10]);
    }
}

// ---- hosted tools ----------------------------------------------------------

/// A finished synthetic generation of `events`.
fn generation(events: Vec<Event>) -> EventStream {
    let (relay, stream) = EventStream::relay();
    std::thread::spawn(move || {
        for event in events {
            assert!(relay.blocking_send(Signal::Event(event)));
        }
    })
    .join()
    .expect("queued");
    stream
}

fn write_call(id: &str) -> Event {
    Event::ToolCall(ToolCall {
        id: id.into(),
        name: "local_fs_write".into(),
        arguments: json!({"path":format!("{id}.txt"),"content":"hi"}),
    })
}

#[tokio::test]
async fn the_hosted_tool_marker_is_recorded_once_before_any_call_runs() {
    let root = tempfile::tempdir().expect("root");
    let toolbox = Toolbox::new(
        Permissions {
            root: Some(root.path().to_owned()),
            write: true,
            shell: false,
            fetch_domains: Vec::new(),
        },
        Limits::default(),
    )
    .expect("toolbox");
    let body = json!({"input":"hi","tools":[{"type":"function","name":"local_fs_write"}]});
    let make = |before: super::tool_loop::BeforeTools| {
        let prepared = prepare_responses(body.to_string().as_bytes()).expect("prepared");
        let first = generation(vec![
            write_call("a"),
            write_call("b"),
            Event::Finished(stats(StopReason::Eos)),
        ]);
        let tool_loop = Loop {
            submit: Box::new(|_| {
                Ok(generation(vec![
                    Event::Content("done".into()),
                    Event::Finished(stats(StopReason::Eos)),
                ]))
            }),
            request: prepared.request,
            toolbox: toolbox.clone(),
            vector_stores: None,
            kinds: prepared.echo.kinds().clone(),
            bounds: Bounds {
                max_iterations: 2,
                budget: std::time::Duration::from_secs(60),
                call_timeout: std::time::Duration::from_secs(20),
            },
            max_calls: None,
            before_tools: Some(before),
        };
        start(tool_loop, first, super::admin::Meter::default())
    };
    let drain =
        |stream: EventStream| tokio::task::spawn_blocking(move || stream.collect::<Vec<_>>());

    // A marker that cannot be recorded runs nothing and fails the Response.
    let events = drain(make(Box::new(|| Err("cannot record".into()))))
        .await
        .expect("drained");
    assert!(
        matches!(events.last(), Some(Event::Error(message)) if message == "cannot record"),
        "{events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Event::ToolOutput(_)))
    );
    assert!(!root.path().join("a.txt").exists(), "no call ran");

    // Recorded once, before the first call of the turn runs.
    let ran = Arc::new(AtomicUsize::new(0));
    let before = {
        let (ran, path) = (Arc::clone(&ran), root.path().join("a.txt"));
        Box::new(move || {
            assert!(!path.exists(), "recorded before any call runs");
            ran.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    };
    let events = drain(make(before)).await.expect("drained");
    assert_eq!(ran.load(Ordering::SeqCst), 1);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::ToolOutput(_)))
            .count(),
        2
    );
    assert!(root.path().join("a.txt").exists() && root.path().join("b.txt").exists());
}

#[test]
fn only_a_durable_background_response_records_hosted_calls() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let tools = json!([{"type":"function","name":"local_fs_read"}]);
    let resumable = |body: Value| {
        prepared(&body, &store, None)
            .echo
            .resumable_store()
            .is_some()
    };
    assert!(resumable(
        json!({"input":"hi","background":true,"tools":tools})
    ));
    assert!(!resumable(json!({"input":"hi","tools":tools})));
}
