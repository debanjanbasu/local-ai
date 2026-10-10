//! CPU-only tests for background Responses: admission, durable status, and
//! the settle-once races between the pump, cancellation, deletion and
//! shutdown, all driven by scripted engine signals so no model or GPU is
//! touched.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use serde_json::{Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::{Event, GenerationStats, PrefillProgress, Signal, Stats};

use super::QueueDepth;
use super::background::{Background, Pump, Temporary};
use super::response::{Protocol, Reply, new_id};
use super::responses::{ResponsesState, prepare_responses_retaining, prepare_responses_with};
use super::store::ResponseStore;

/// A store directory under the system temp dir, removed when dropped.
pub(super) struct Scratch(pub(super) PathBuf);

impl Scratch {
    pub(super) fn new() -> Self {
        Self(std::env::temp_dir().join(new_id("local-ai-background-test-")))
    }

    pub(super) fn open(&self) -> Arc<ResponseStore> {
        Arc::new(ResponseStore::open(&self.0).expect("store opens"))
    }

    #[cfg(unix)]
    pub(super) fn set_mode(&self, mode: u32) {
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

pub(super) fn stats(stop_reason: StopReason) -> Box<Stats> {
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
pub(super) fn doc(body: &Value, store: &Arc<ResponseStore>) -> ResponsesState {
    streamed_doc(body, store, false)
}

/// [`doc`], materialising stream events when `streaming`.
pub(super) fn streamed_doc(
    body: &Value,
    store: &Arc<ResponseStore>,
    streaming: bool,
) -> ResponsesState {
    let prepared =
        prepare_responses_with(body.to_string().as_bytes(), true, Some(store), None, "m")
            .expect("request accepted");
    let reply = Reply::new(Protocol::Responses(Arc::new(prepared.echo)), "m".into());
    let Protocol::Responses(echo) = &reply.protocol else {
        unreachable!("responses reply")
    };
    ResponsesState::new(&reply, Arc::clone(echo), streaming)
}

fn background_doc(store: &Arc<ResponseStore>) -> ResponsesState {
    doc(&json!({"input":"hi","background":true}), store)
}

/// One scripted engine step: runs when the pump asks for its next signal.
pub(super) type Step = Box<dyn FnOnce() -> Signal + Send>;

pub(super) fn event(event: Event) -> Step {
    Box::new(move || Signal::Event(event))
}

/// The engine side of a job: yields `steps` in order, then end-of-stream.
pub(super) fn script(steps: Vec<Step>) -> impl FnMut() -> Option<Signal> + Send + 'static {
    let mut steps = steps.into_iter();
    move || steps.next().map(|step| step())
}

/// A native cancel that only records it was asked for.
pub(super) fn stopper() -> (Arc<AtomicBool>, impl Fn() + Send + Sync + 'static) {
    let stopped = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stopped);
    (stopped, move || flag.store(true, Ordering::SeqCst))
}

pub(super) fn stored(store: &ResponseStore, id: &str) -> Option<Value> {
    store
        .load(id)
        .expect("store readable")
        .map(|stored| stored.response)
}

pub(super) fn status(store: &ResponseStore, id: &str) -> String {
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

pub(super) fn drained(background: &Background) {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime")
        .block_on(background.drained());
}

/// How long the tests' temporary responses are retained.
pub(super) const TTL: Duration = Duration::from_secs(600);

/// A temporary store in `scratch` whose clock only moves when the returned
/// offset is advanced.
pub(super) fn manual_temporary(scratch: &Scratch) -> (Arc<Temporary>, Arc<Mutex<Duration>>) {
    let base = Instant::now();
    let offset = Arc::new(Mutex::new(Duration::ZERO));
    let clock = Arc::clone(&offset);
    let temporary = Temporary::with(
        scratch.open(),
        TTL,
        Arc::new(move || base + *clock.lock().expect("clock")),
    );
    (Arc::new(temporary), offset)
}

pub(super) fn advance(offset: &Mutex<Duration>, by: Duration) {
    *offset.lock().expect("clock") += by;
}

/// The Response state machine for `body` on a server with `durable` and
/// `temporary` stores.
pub(super) fn temporary_doc(
    body: &Value,
    durable: Option<&Arc<ResponseStore>>,
    temporary: &Arc<Temporary>,
    streaming: bool,
) -> ResponsesState {
    let prepared = prepare_responses_retaining(
        body.to_string().as_bytes(),
        true,
        durable,
        Some(temporary),
        None,
        "m",
    )
    .expect("request accepted");
    let reply = Reply::new(Protocol::Responses(Arc::new(prepared.echo)), "m".into());
    let Protocol::Responses(echo) = &reply.protocol else {
        unreachable!("responses reply")
    };
    ResponsesState::new(&reply, Arc::clone(echo), streaming)
}

/// Every file name in `dir` mentioning `id`.
fn files_naming(dir: &std::path::Path, id: &str) -> Vec<String> {
    std::fs::read_dir(dir)
        .expect("listable")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.contains(id))
        .collect()
}

#[test]
fn background_store_false_is_temporary_and_refused_without_a_temporary_store() {
    let (durable_dir, temporary_dir) = (Scratch::new(), Scratch::new());
    let durable = durable_dir.open();
    let (temporary, _) = manual_temporary(&temporary_dir);
    let body = json!({"input":"hi","background":true});
    let temporary_body = json!({"input":"hi","background":true,"store":false});
    // Neither store: refused, naming both ways it could be kept.
    let error = prepare_error(&body, None);
    assert!(
        error.contains("background") && error.contains("--response-store"),
        "{error}"
    );
    // A durable store but no temporary one: store=false is still refused.
    assert!(prepare_error(&temporary_body, Some(&durable)).contains("store=false"));

    // store=false goes to the temporary store only, and says store=false.
    for (body, durable) in [
        (&temporary_body, Some(&durable)),
        (&temporary_body, None),
        // Without a durable store, an omitted store means false.
        (&body, None),
    ] {
        let doc = temporary_doc(body, durable, &temporary, false);
        assert!(doc.temporary().is_some(), "{body}");
        assert!(Arc::ptr_eq(doc.store().expect("kept"), temporary.store()));
        let queued = doc.snapshot("queued");
        assert_eq!(queued["store"], false, "{body}");
        assert_eq!(queued["background"], true, "{body}");
    }
    // With a durable store, an omitted store keeps OpenAI's default of true.
    let doc = temporary_doc(&body, Some(&durable), &temporary, false);
    assert!(doc.temporary().is_none());
    assert!(Arc::ptr_eq(doc.store().expect("kept"), &durable));
    assert_eq!(doc.snapshot("queued")["store"], true);
    // store=true still needs a durable store.
    let error = prepare_responses_retaining(
        json!({"input":"hi","background":true,"store":true})
            .to_string()
            .as_bytes(),
        true,
        None,
        Some(&temporary),
        None,
        "m",
    )
    .err()
    .map(|error| error.to_string())
    .unwrap_or_default();
    assert!(error.contains("store=true"), "{error}");
    // Foreground store=false is unchanged: stateless, never temporary.
    let prepared = prepare_responses_retaining(
        json!({"input":"hi","store":false,"stream":true})
            .to_string()
            .as_bytes(),
        true,
        Some(&durable),
        Some(&temporary),
        None,
        "m",
    )
    .expect("foreground");
    assert!(!prepared.echo.background());
    let streamed = prepare_responses_with(
        json!({"input":"hi","background":true,"stream":true})
            .to_string()
            .as_bytes(),
        true,
        Some(&durable),
        None,
        "m",
    )
    .expect("background streaming is accepted");
    assert!(streamed.echo.background() && streamed.stream);
}

#[test]
fn a_temporary_response_expires_only_after_it_ends_and_never_reaches_the_durable_store() {
    let (durable_dir, temporary_dir) = (Scratch::new(), Scratch::new());
    let durable = durable_dir.open();
    let (temporary, clock) = manual_temporary(&temporary_dir);
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let doc = temporary_doc(
        &json!({"input":"hi","background":true,"store":false}),
        Some(&durable),
        &temporary,
        false,
    );
    let id = doc.id().to_owned();
    let (stopped, stop) = stopper();
    let steps = vec![
        event(Event::Content("Hi".into())),
        event(Event::Finished(stats(StopReason::Eos))),
    ];
    let (queued, pump) = background
        .admit(doc, script(steps), stop, Some(depth.admit()))
        .expect("admitted");
    assert_eq!(queued["store"], false);
    assert_eq!(stored(temporary.store(), &id), Some(queued));
    assert!(
        files_naming(&durable_dir.0, &id).is_empty(),
        "nothing durable"
    );
    let located =
        super::locate(Some(&temporary), Some(&durable), &background, &id).expect("located");
    assert!(Arc::ptr_eq(&located.store, temporary.store()));
    assert!(located.temporary.is_some());

    // A job that runs for longer than the retention is never expired.
    advance(&clock, TTL * 6);
    assert_eq!(temporary.expire_due(&background), 0);
    assert_eq!(
        temporary.next_deadline(),
        None,
        "retention starts at the end"
    );
    assert_eq!(status(temporary.store(), &id), "queued");

    pump.run();
    drained(&background);
    assert!(!stopped.load(Ordering::SeqCst));
    let done = stored(temporary.store(), &id).expect("kept");
    assert_eq!(done["status"], "completed");
    assert_eq!(done["store"], false, "reported truthfully");
    assert!(
        files_naming(&durable_dir.0, &id).is_empty(),
        "still nothing durable"
    );
    assert!(temporary.next_deadline().is_some());
    // Retrievable, and cancelling is idempotent, until the retention ends.
    advance(&clock, TTL.saturating_sub(Duration::from_secs(1)));
    assert_eq!(temporary.expire_due(&background), 0);
    let located =
        super::locate(Some(&temporary), Some(&durable), &background, &id).expect("still retained");
    assert_eq!(
        super::stored_answer(&located.store, &axum::http::Method::GET, &id, None).expect("get"),
        Some(done.clone())
    );
    assert_eq!(
        background.cancel(&located.store, &id).expect("cancel"),
        Some(done)
    );
    // Lazily on access: the first lookup past the deadline deletes it.
    advance(&clock, Duration::from_secs(1));
    let (status_code, message) =
        super::locate(Some(&temporary), Some(&durable), &background, &id).expect_err("expired");
    assert_eq!(status_code, StatusCode::NOT_FOUND);
    assert!(message.contains("not found"), "{message}");
    assert!(
        files_naming(&temporary_dir.0, &id)
            .iter()
            .all(|name| name.starts_with('.') && name.contains(".lock"))
    );
    assert!(!temporary.knows(&id) && temporary.next_deadline().is_none());
    // From then on the ID is an ordinary unknown one, wherever it is looked up.
    let located = super::locate(Some(&temporary), Some(&durable), &background, &id)
        .expect("the durable store answers");
    assert!(located.temporary.is_none());
    assert_eq!(
        super::stored_answer(&located.store, &axum::http::Method::GET, &id, None).expect("get"),
        None
    );
    let (_, message) =
        super::locate(Some(&temporary), None, &background, &id).expect_err("no durable store");
    assert!(message.contains("--response-store"), "{message}");
}

#[test]
fn durable_responses_are_never_answered_from_the_temporary_store() {
    let (durable_dir, temporary_dir) = (Scratch::new(), Scratch::new());
    let durable = durable_dir.open();
    let (temporary, clock) = manual_temporary(&temporary_dir);
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    let doc = temporary_doc(
        &json!({"input":"hi","background":true}),
        Some(&durable),
        &temporary,
        false,
    );
    let id = doc.id().to_owned();
    let (_, stop) = stopper();
    let steps = vec![event(Event::Finished(stats(StopReason::Eos)))];
    let (_, pump) = background
        .admit(doc, script(steps), stop, Some(depth.admit()))
        .expect("admitted");
    pump.run();
    assert!(!temporary.knows(&id));
    assert_eq!(files_naming(&temporary_dir.0, &id), Vec::<String>::new());
    advance(&clock, TTL * 10);
    assert_eq!(temporary.expire_due(&background), 0);
    let located =
        super::locate(Some(&temporary), Some(&durable), &background, &id).expect("located");
    assert!(Arc::ptr_eq(&located.store, &durable));
    assert_eq!(status(&durable, &id), "completed", "never expired");
    assert_eq!(stored(&durable, &id).expect("stored")["store"], true);
    // A temporary response cannot be continued, and says why.
    let doc = temporary_doc(
        &json!({"input":"hi","background":true,"store":false}),
        Some(&durable),
        &temporary,
        false,
    );
    let temporary_id = doc.id().to_owned();
    let (_, stop) = stopper();
    let (_, pump) = background
        .admit(doc, script(Vec::new()), stop, None)
        .expect("admitted");
    pump.run();
    let error = prepare_responses_retaining(
        json!({"input":"next","previous_response_id":temporary_id})
            .to_string()
            .as_bytes(),
        true,
        Some(&durable),
        Some(&temporary),
        None,
        "m",
    )
    .err()
    .map(|error| error.to_string())
    .unwrap_or_default();
    assert!(error.contains("store=false"), "{error}");
}

#[test]
fn cancellation_starts_retention_and_expiry_never_cancels_a_draining_job() {
    let temporary_dir = Scratch::new();
    let (temporary, clock) = manual_temporary(&temporary_dir);
    let background = Arc::new(Background::default());
    let body = json!({"input":"hi","background":true,"store":false});

    // Cancelled while the native work still runs: retention counts from the
    // cancellation, and the pump only drains afterwards.
    let doc = temporary_doc(&body, None, &temporary, false);
    let id = doc.id().to_owned();
    let (stopped, stop) = stopper();
    let (_, pump) = background
        .admit(doc, script(Vec::new()), stop, None)
        .expect("admitted");
    let cancelled = background
        .cancel(temporary.store(), &id)
        .expect("cancel")
        .expect("found");
    assert_eq!(cancelled["status"], "cancelled");
    assert_eq!(cancelled["store"], false);
    assert!(stopped.load(Ordering::SeqCst), "native work is cancelled");
    let deadline = temporary
        .next_deadline()
        .expect("retained from the cancellation");
    assert_eq!(
        background.cancel(temporary.store(), &id).expect("again"),
        Some(cancelled),
        "cancelling twice returns the final Response"
    );
    assert_eq!(temporary.next_deadline(), Some(deadline), "not extended");
    pump.run();

    // Ended but still draining: expiry deletes it without cancelling it.
    let doc = temporary_doc(&body, None, &temporary, false);
    let draining = doc.id().to_owned();
    let (stopped, stop) = stopper();
    let (gate, wait) = std::sync::mpsc::channel::<()>();
    let steps: Vec<Step> = vec![
        event(Event::Finished(stats(StopReason::Eos))),
        Box::new(move || {
            let _ = wait.recv();
            Signal::Progress(PrefillProgress {
                tokens: 0,
                chunks: 1,
            })
        }),
    ];
    let (_, pump) = background
        .admit(doc, script(steps), stop, None)
        .expect("admitted");
    let pump = std::thread::spawn(move || pump.run());
    while status(temporary.store(), &draining) != "completed" {
        std::thread::yield_now();
    }
    advance(&clock, TTL);
    assert_eq!(temporary.expire_due(&background), 2, "both are due");
    assert_eq!(stored(temporary.store(), &id), None);
    assert_eq!(stored(temporary.store(), &draining), None);
    assert!(background.watch(&draining).is_some(), "still draining");
    gate.send(()).expect("open the gate");
    pump.join().expect("pump");
    assert!(!stopped.load(Ordering::SeqCst), "expiry never cancels");
    assert_eq!(
        stored(temporary.store(), &draining),
        None,
        "never written back"
    );
    drained(&background);
}

#[test]
fn the_reaper_deletes_due_responses_without_access_and_close_removes_everything() {
    let temporary = Temporary::create().expect("private temporary store");
    let dir = temporary.dir().expect("owned").to_owned();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = std::fs::metadata(&dir).expect("dir").permissions().mode();
        assert_eq!(mode & 0o777, 0o700, "owner-only");
    }
    let temporary_dir = Scratch::new();
    let reaped = Arc::new(Temporary::with(
        temporary_dir.open(),
        Duration::from_millis(20),
        Arc::new(Instant::now),
    ));
    let background = Arc::new(Background::default());
    let expiry = reaped.reap(Arc::clone(&background)).expect("reaper");
    let doc = temporary_doc(
        &json!({"input":"hi","background":true,"store":false}),
        None,
        &reaped,
        false,
    );
    let id = doc.id().to_owned();
    let (_, stop) = stopper();
    let steps = vec![event(Event::Finished(stats(StopReason::Eos)))];
    let (_, pump) = background
        .admit(doc, script(steps), stop, None)
        .expect("admitted");
    pump.run();
    let start = Instant::now();
    while reaped.knows(&id) {
        assert!(start.elapsed() < Duration::from_secs(10), "reaped in time");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(stored(reaped.store(), &id), None, "deleted unread");
    reaped.close();
    expiry.join().expect("the reaper stops on close");

    // Streamed records and journals are owner-only, and close removes them.
    let doc = temporary_doc(
        &json!({"input":"hi","background":true,"store":false,"stream":true}),
        None,
        &temporary,
        true,
    );
    let id = doc.id().to_owned();
    let (_, stop) = stopper();
    let (_, pump) = background
        .admit(doc, script(Vec::new()), stop, None)
        .expect("admitted");
    pump.run();
    #[cfg(unix)]
    for name in files_naming(&dir, &id) {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = std::fs::metadata(dir.join(&name))
            .expect("file")
            .permissions()
            .mode();
        assert_eq!(mode & 0o077, 0, "{name} is owner-only");
    }
    assert!(files_naming(&dir, &id).len() >= 2, "record and journal");
    temporary.close();
    assert!(!dir.exists(), "discarded, never kept across a restart");
    assert!(!temporary.knows(&id));
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
