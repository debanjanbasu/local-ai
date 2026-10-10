//! CPU-only tests for background Responses: admission, durable status, and
//! the settle-once races between the pump, cancellation, deletion and
//! shutdown, all driven by scripted engine signals so no model or GPU is
//! touched.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use axum::http::StatusCode;
use serde_json::{Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::{Event, GenerationStats, PrefillProgress, Signal, Stats};

use super::QueueDepth;
use super::background::{Background, Pump};
use super::response::{Protocol, Reply, new_id};
use super::responses::{ResponsesState, prepare_responses_with};
use super::store::ResponseStore;

/// A store directory under the system temp dir, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        Self(std::env::temp_dir().join(new_id("local-ai-background-test-")))
    }

    fn open(&self) -> Arc<ResponseStore> {
        Arc::new(ResponseStore::open(&self.0).expect("store opens"))
    }

    #[cfg(unix)]
    fn set_mode(&self, mode: u32) {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(mode))
            .expect("chmod scratch");
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        #[cfg(unix)]
        self.set_mode(0o700);
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn stats(stop_reason: StopReason) -> Box<Stats> {
    Box::new(Stats {
        stop_reason,
        cache_source: PromptCacheSource::None,
        reasoning_tokens: 1,
        generation: GenerationStats {
            prompt_tokens: 7,
            generated_tokens: 3,
            ..GenerationStats::default()
        },
    })
}

fn prepare_error(body: &Value, store: Option<&Arc<ResponseStore>>) -> String {
    prepare_responses_with(body.to_string().as_bytes(), true, store, None, "m")
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default()
}

/// The Response state machine for `body` against `store`.
fn doc(body: &Value, store: &Arc<ResponseStore>) -> ResponsesState {
    let prepared =
        prepare_responses_with(body.to_string().as_bytes(), true, Some(store), None, "m")
            .expect("request accepted");
    let reply = Reply::new(Protocol::Responses(Arc::new(prepared.echo)), "m".into());
    let Protocol::Responses(echo) = &reply.protocol else {
        unreachable!("responses reply")
    };
    ResponsesState::new(&reply, Arc::clone(echo), false)
}

fn background_doc(store: &Arc<ResponseStore>) -> ResponsesState {
    doc(&json!({"input":"hi","background":true}), store)
}

/// One scripted engine step: runs when the pump asks for its next signal.
type Step = Box<dyn FnOnce() -> Signal + Send>;

fn event(event: Event) -> Step {
    Box::new(move || Signal::Event(event))
}

/// The engine side of a job: yields `steps` in order, then end-of-stream.
fn script(steps: Vec<Step>) -> impl FnMut() -> Option<Signal> + Send + 'static {
    let mut steps = steps.into_iter();
    move || steps.next().map(|step| step())
}

/// A native cancel that only records it was asked for.
fn stopper() -> (Arc<AtomicBool>, impl Fn() + Send + Sync + 'static) {
    let stopped = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stopped);
    (stopped, move || flag.store(true, Ordering::SeqCst))
}

fn stored(store: &ResponseStore, id: &str) -> Option<Value> {
    store
        .load(id)
        .expect("store readable")
        .map(|stored| stored.response)
}

fn status(store: &ResponseStore, id: &str) -> String {
    stored(store, id)
        .and_then(|response| response["status"].as_str().map(str::to_owned))
        .unwrap_or_else(|| "absent".into())
}

/// Admit a background job with `steps`, returning its ID, pump and stop flag.
fn admit(
    background: &Arc<Background>,
    store: &Arc<ResponseStore>,
    depth: &QueueDepth,
    steps: impl FnOnce(&str) -> Vec<Step>,
) -> (String, Value, Pump, Arc<AtomicBool>) {
    let doc = background_doc(store);
    let id = doc.id().to_owned();
    let (stopped, stop) = stopper();
    let (queued, pump) = background
        .admit(doc, script(steps(&id)), stop, Some(depth.admit()))
        .expect("admitted");
    (id, queued, pump, stopped)
}

fn drained(background: &Background) {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime")
        .block_on(background.drained());
}

#[test]
fn background_needs_a_store_and_refuses_streaming_and_temporary_retention() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let body = json!({"input":"hi","background":true});
    assert!(prepare_error(&body, None).contains("--response-store"));
    assert!(
        prepare_error(
            &json!({"input":"hi","background":true,"store":false}),
            Some(&store)
        )
        .contains("store=false")
    );
    assert!(
        prepare_error(
            &json!({"input":"hi","background":true,"stream":true}),
            Some(&store)
        )
        .contains("stream=true")
    );
    let prepared =
        prepare_responses_with(body.to_string().as_bytes(), true, Some(&store), None, "m")
            .expect("background with a store");
    assert!(prepared.echo.background() && !prepared.stream);
    for foreground in [
        json!({"input":"hi"}),
        json!({"input":"hi","background":false,"stream":true,"store":false}),
    ] {
        let prepared = prepare_responses_with(
            foreground.to_string().as_bytes(),
            true,
            Some(&store),
            None,
            "m",
        )
        .expect("foreground unchanged");
        assert!(!prepared.echo.background());
    }
}

#[test]
fn queued_then_in_progress_then_terminal_are_each_durable() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (id, queued, pump, stopped) = admit(&background, &store, &depth, |id| {
        // Each step records the durable status the pump left before asking.
        let observe = |signal: Signal| -> Step {
            let (store, id, seen) = (Arc::clone(&store), id.to_owned(), Arc::clone(&seen));
            Box::new(move || {
                seen.lock().expect("log").push(status(&store, &id));
                signal
            })
        };
        vec![
            observe(Signal::Progress(PrefillProgress {
                tokens: 0,
                chunks: 1,
            })),
            observe(Signal::Event(Event::Reasoning("think".into()))),
            observe(Signal::Event(Event::Content("Hi".into()))),
            observe(Signal::Event(Event::Finished(stats(StopReason::Eos)))),
        ]
    });
    assert_eq!(queued["status"], "queued");
    assert_eq!(queued["background"], true);
    assert_eq!(queued["store"], true);
    assert_eq!(queued["usage"], Value::Null);
    assert_eq!(stored(&store, &id), Some(queued), "queued is on disk first");
    assert_eq!(depth.load(), 1, "the engine slot is held by the job");
    let items = store
        .load(&id)
        .expect("readable")
        .expect("stored")
        .input_items;
    assert_eq!(items.len(), 1, "input_items are available while queued");

    pump.run();
    assert_eq!(
        *seen.lock().expect("log"),
        ["queued", "in_progress", "in_progress", "in_progress"],
        "a prefill boundary is the first progress"
    );
    let done = stored(&store, &id).expect("stored");
    assert_eq!(done["status"], "completed");
    assert_eq!(done["background"], true);
    assert_eq!(done["usage"]["output_tokens"], 3);
    assert_eq!(done["output"][0]["content"][0]["text"], "think");
    assert_eq!(done["output"][1]["content"][0]["text"], "Hi");
    assert!(
        !stopped.load(Ordering::SeqCst),
        "a finished job is not cancelled"
    );
    assert_eq!(depth.load(), 0, "the slot is released when the stream ends");
    drained(&background);
    assert!(
        store.lease(&id).expect("lease").is_some(),
        "the lease is free"
    );
    // Terminal: cancelling returns it as it ended.
    assert_eq!(
        background.cancel(&store, &id).expect("cancel"),
        Some(done.clone())
    );
    assert_eq!(
        stored(&scratch.open(), &id),
        Some(done),
        "survives reopening"
    );
}

#[test]
fn failures_and_a_stream_that_ends_without_a_result_are_stored_as_failed() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let (id, _, pump, _) = admit(&background, &store, &depth, |_| {
        vec![
            event(Event::Content("par".into())),
            event(Event::Error("metal said no".into())),
        ]
    });
    pump.run();
    let failed = stored(&store, &id).expect("stored");
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["store"], true);
    assert_eq!(failed["error"]["message"], "metal said no");
    assert_eq!(failed["output"][0]["status"], "incomplete");

    let (id, _, pump, _) = admit(&background, &store, &depth, |_| {
        vec![event(Event::Content("par".into()))]
    });
    pump.run();
    let stopped = stored(&store, &id).expect("stored");
    assert_eq!(stopped["status"], "failed");
    assert!(
        stopped["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("stopped"))
    );
    let (id, _, pump, _) = admit(&background, &store, &depth, |_| {
        vec![event(Event::Finished(stats(StopReason::TokenLimit)))]
    });
    pump.run();
    assert_eq!(status(&store, &id), "incomplete");
    assert_eq!(depth.load(), 0);
}

#[test]
fn cancel_settles_once_stops_native_work_and_is_idempotent() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let answers = Arc::new(Mutex::new(Vec::new()));
    let (id, _, pump, stopped) = admit(&background, &store, &depth, |id| {
        let (background, store, id, answers) = (
            Arc::clone(&background),
            Arc::clone(&store),
            id.to_owned(),
            Arc::clone(&answers),
        );
        vec![
            event(Event::Content("par".into())),
            Box::new(move || {
                // Cancelled mid-generation; the engine still delivers a late
                // piece and even a normal finish, which must not win.
                let first = background.cancel(&store, &id).expect("cancel");
                let second = background.cancel(&store, &id).expect("again");
                answers.lock().expect("log").extend([first, second]);
                Signal::Event(Event::Content("tial".into()))
            }),
            event(Event::Finished(stats(StopReason::Eos))),
        ]
    });
    pump.run();
    assert!(stopped.load(Ordering::SeqCst), "native work was cancelled");
    let answers = answers.lock().expect("log").clone();
    let cancelled = answers[0].clone().expect("found");
    assert_eq!(cancelled["status"], "cancelled");
    assert_eq!(cancelled["output"][0]["content"][0]["text"], "par");
    assert_eq!(cancelled["output"][0]["status"], "incomplete");
    assert_eq!(answers[1], answers[0], "a second cancel returns the same");
    assert_eq!(stored(&store, &id), Some(cancelled.clone()), "finish lost");
    assert_eq!(depth.load(), 0);
    drained(&background);
    assert_eq!(
        background.cancel(&store, &id).expect("after"),
        Some(cancelled),
        "still the same once the job has gone"
    );

    // A queued job cancelled before any signal, whose stream then closes
    // without `Finished`, as a cancelled native stream does.
    let (id, _, pump, stopped) = admit(&background, &store, &depth, |_| Vec::new());
    let cancelled = background
        .cancel(&store, &id)
        .expect("cancel")
        .expect("found");
    assert_eq!(cancelled["status"], "cancelled");
    assert!(stopped.load(Ordering::SeqCst));
    pump.run();
    assert_eq!(status(&store, &id), "cancelled");
}

#[test]
fn only_background_responses_can_be_cancelled() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Background::default();
    let mut foreground = doc(&json!({"input":"hi"}), &store);
    let id = foreground.id().to_owned();
    let done = foreground
        .event(Event::Finished(stats(StopReason::Eos)))
        .expect("terminal");
    assert_eq!(done["background"], false);
    assert_eq!(
        status(&store, &id),
        "completed",
        "foreground still stores itself"
    );
    let (code, message) = background
        .cancel(&store, &id)
        .expect_err("foreground refused");
    assert_eq!(code, StatusCode::BAD_REQUEST);
    assert!(message.contains("background"));
    assert_eq!(status(&store, &id), "completed", "left untouched");
    assert_eq!(
        background.cancel(&store, "resp_missing").expect("lookup"),
        None
    );
}

#[test]
fn deletion_during_generation_is_never_written_back() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let deleted = Arc::new(AtomicBool::new(false));
    let (id, _, pump, stopped) = admit(&background, &store, &depth, |id| {
        let (background, store, id, deleted) = (
            Arc::clone(&background),
            Arc::clone(&store),
            id.to_owned(),
            Arc::clone(&deleted),
        );
        vec![
            event(Event::Content("par".into())),
            Box::new(move || {
                deleted.store(
                    background.delete(&store, &id).expect("delete"),
                    Ordering::SeqCst,
                );
                Signal::Event(Event::Finished(stats(StopReason::Eos)))
            }),
        ]
    });
    pump.run();
    assert!(deleted.load(Ordering::SeqCst));
    assert!(
        stopped.load(Ordering::SeqCst),
        "deleting a pending job stops it"
    );
    assert_eq!(stored(&store, &id), None, "the finish did not resurrect it");
    assert_eq!(background.cancel(&store, &id).expect("cancel"), None);
    assert!(!background.delete(&store, &id).expect("again"));

    // Finish first, then delete: gone as well.
    let (id, _, pump, _) = admit(&background, &store, &depth, |_| {
        vec![event(Event::Finished(stats(StopReason::Eos)))]
    });
    pump.run();
    assert!(background.delete(&store, &id).expect("delete"));
    assert_eq!(stored(&store, &id), None);
}

#[test]
fn another_servers_pending_response_is_a_conflict_not_overwritten() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let refusals = Arc::new(Mutex::new(Vec::new()));
    let (id, _, pump, stopped) = admit(&background, &store, &depth, |id| {
        let (scratch_dir, id, refusals) = (scratch.0.clone(), id.to_owned(), Arc::clone(&refusals));
        vec![
            event(Event::Content("par".into())),
            Box::new(move || {
                // A second server on the same directory: opening it must not
                // recover the live job, and it must refuse to touch it.
                let other_store = ResponseStore::open(&scratch_dir).expect("second store");
                let other = Background::default();
                assert_eq!(status(&other_store, &id), "in_progress", "GET sees it");
                let attempts = [
                    other.cancel(&other_store, &id).map(|_| ()),
                    other.delete(&other_store, &id).map(|_| ()),
                    super::stored_answer(&other_store, &axum::http::Method::DELETE, &id, None)
                        .map(|_| ()),
                ];
                refusals.lock().expect("log").extend(attempts);
                Signal::Event(Event::Finished(stats(StopReason::Eos)))
            }),
        ]
    });
    pump.run();
    for refusal in refusals.lock().expect("log").iter() {
        let (code, message) = refusal.clone().expect_err("conflict");
        assert_eq!(code, StatusCode::CONFLICT);
        assert!(message.contains("another server"));
    }
    assert!(!stopped.load(Ordering::SeqCst));
    assert_eq!(status(&store, &id), "completed", "the owner finished it");
    // Terminal records are retrievable, cancellable and deletable from any
    // server sharing the directory.
    let other_store = scratch.open();
    let other = Background::default();
    assert_eq!(
        other
            .cancel(&other_store, &id)
            .expect("terminal")
            .map(|response| response["status"].clone()),
        Some(json!("completed"))
    );
    assert!(other.delete(&other_store, &id).expect("delete"));
}

#[test]
fn shutdown_fails_pending_jobs_refuses_new_ones_and_drains() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let (id, _, pump, stopped) = admit(&background, &store, &depth, |_| {
        vec![
            event(Event::Content("late".into())),
            event(Event::Finished(stats(StopReason::Eos))),
        ]
    });
    assert!(!background.closing());
    background.close();
    assert!(background.closing());
    assert!(stopped.load(Ordering::SeqCst), "native work was cancelled");
    let failed = stored(&store, &id).expect("stored");
    assert_eq!(failed["status"], "failed");
    assert!(
        failed["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("shut down"))
    );
    // A job admitted during shutdown is refused and leaves no pending record.
    let late = background_doc(&store);
    let late_id = late.id().to_owned();
    let (late_stopped, stop) = stopper();
    let (code, _) = background
        .admit(late, script(Vec::new()), stop, Some(depth.admit()))
        .err()
        .expect("refused");
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    assert!(late_stopped.load(Ordering::SeqCst));
    assert_eq!(status(&store, &late_id), "failed");
    assert_eq!(depth.load(), 1, "only the running job still holds a slot");
    pump.run();
    drained(&background);
    assert_eq!(status(&store, &id), "failed", "the late finish did not win");
    assert_eq!(depth.load(), 0);
}

#[cfg(unix)]
#[test]
fn write_failures_never_claim_durable_success() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let (id, _, pump, stopped) = admit(&background, &store, &depth, |_| {
        vec![
            event(Event::Content("Hi".into())),
            event(Event::Finished(stats(StopReason::Eos))),
        ]
    });
    // Nothing more can be written, renamed or removed in the store.
    scratch.set_mode(0o500);
    pump.run();
    scratch.set_mode(0o700);
    assert!(stopped.load(Ordering::SeqCst), "the job stopped instead");
    assert_eq!(status(&store, &id), "queued", "never reported completed");
    // Once the lease is released, a restart fails the stranded record.
    assert_eq!(status(&scratch.open(), &id), "failed");

    // Retrying cancellation must not return an in-memory result as durable
    // when both the cancellation write and the fallback failure write failed.
    let (id, _, pump, stopped) = admit(&background, &store, &depth, |_| Vec::new());
    scratch.set_mode(0o500);
    for _ in 0..2 {
        let (code, _) = background.cancel(&store, &id).expect_err("not durable");
        assert_eq!(code, StatusCode::INTERNAL_SERVER_ERROR);
    }
    scratch.set_mode(0o700);
    assert!(stopped.load(Ordering::SeqCst));
    pump.run();
    assert_eq!(status(&scratch.open(), &id), "failed");

    // The initial write failing refuses the request and cancels the work.
    scratch.set_mode(0o500);
    let doc = background_doc(&store);
    let (_, stop) = stopper();
    let refused = background.admit(doc, script(Vec::new()), stop, Some(depth.admit()));
    scratch.set_mode(0o700);
    let (code, _) = refused.err().expect("refused");
    assert_eq!(code, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(depth.load(), 0);
}
