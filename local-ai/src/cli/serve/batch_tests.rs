//! CPU-only tests for `/v1/batches`: real requests against a real store in a
//! private temporary directory, with the worker driven by scripted engine
//! signals instead of a model.

use std::io::Read as _;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::{Event, GenerationStats, Signal, Stats};
use local_services::Store;
use local_services::batches::{CreateBatch, LineOutcome};
use local_services::files::{CreateFile, FilePurpose};

use super::super::QueueDepth;
use super::super::request::GenerationRequest;
use super::{Batches, Config, Generation, Submit, matches};

/// A store in a temporary directory, removed when dropped.
struct Scratch {
    _dir: tempfile::TempDir,
    store: Store,
}

impl Scratch {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("temporary directory");
        // SQLite refuses a path through a symlink, and the macOS temporary
        // directory is under one (`/var` -> `/private/var`).
        let root: PathBuf = dir
            .path()
            .canonicalize()
            .expect("canonical temporary directory")
            .join("services");
        let store = Store::open(&root).expect("store opens");
        Self { _dir: dir, store }
    }

    fn input(&self, lines: &[Value]) -> String {
        let content = lines.iter().fold(String::new(), |mut content, line| {
            content.push_str(&line.to_string());
            content.push('\n');
            content
        });
        self.store
            .create_file(
                &CreateFile {
                    filename: "input.jsonl".to_owned(),
                    purpose: FilePurpose::Batch,
                    expires_after: None,
                },
                content.as_bytes(),
            )
            .expect("input stored")
            .id
    }
}

fn stats() -> Stats {
    Stats {
        stop_reason: StopReason::Eos,
        cache_source: PromptCacheSource::None,
        reasoning_tokens: 0,
        generation: GenerationStats {
            prompt_tokens: 7,
            generated_tokens: 2,
            ..GenerationStats::default()
        },
    }
}

fn chat(custom_id: &str, text: &str) -> Value {
    json!({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "local", "messages": [{"role": "user", "content": text}]},
    })
}

/// The text the fake engine keys its behaviour on.
fn prompt(request: &GenerationRequest) -> String {
    match request {
        GenerationRequest::Chat(chat) => chat
            .messages
            .last()
            .map(|message| message.content.clone())
            .unwrap_or_default(),
        GenerationRequest::Completion(completion) => completion.prompt.clone(),
    }
}

fn scripted(signals: Vec<Signal>) -> Generation {
    let mut signals = signals.into_iter();
    Generation {
        next: Box::new(move |_: Option<Duration>| Ok(signals.next())),
        cancel: Arc::new(|| {}),
    }
}

/// A generation that produces nothing until it is cancelled, then ends.
fn blocked() -> Generation {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let opener = Arc::clone(&gate);
    Generation {
        next: Box::new(move |_: Option<Duration>| {
            let (lock, signal) = &*gate;
            drop(
                signal
                    .wait_while(lock.lock().expect("gate"), |cancelled| !*cancelled)
                    .expect("gate"),
            );
            Ok(None)
        }),
        cancel: Arc::new(move || {
            let (lock, signal) = &*opener;
            *lock.lock().expect("gate") = true;
            signal.notify_all();
        }),
    }
}

/// What the fake engine was asked, in order.
type Seen = Arc<Mutex<Vec<String>>>;

/// A fake engine: `boom` fails, `block` (when `blocking`) runs until
/// cancelled and reports that it started, anything else answers `echo:`.
fn fake(blocking: bool) -> (Submit, Seen, mpsc::Receiver<()>) {
    let seen: Seen = Arc::default();
    let (started, receiver) = mpsc::channel();
    let started = Mutex::new(started);
    let log = Arc::clone(&seen);
    let submit: Submit = Arc::new(
        move |request: GenerationRequest| -> crate::Result<Generation> {
            let text = prompt(&request);
            log.lock().expect("log").push(text.clone());
            Ok(match text.as_str() {
                "boom" => scripted(vec![Signal::Event(Event::Error(
                    "generation error: boom".into(),
                ))]),
                "block" if blocking => {
                    let _ = started.lock().expect("started").send(());
                    blocked()
                }
                _ => scripted(vec![
                    Signal::Event(Event::Content(format!("echo:{text}"))),
                    Signal::Event(Event::Finished(Box::new(stats()))),
                ]),
            })
        },
    );
    (submit, seen, receiver)
}

fn start(store: &Store, submit: Submit) -> Arc<Batches> {
    Batches::spawn(
        store.clone(),
        submit,
        Config {
            model: "local".into(),
            thinking: false,
            cipher: None,
            depth: QueueDepth::default(),
        },
    )
    .expect("worker starts")
}

async fn stop(batches: &Batches) {
    batches.close();
    batches.drained().await;
}

async fn send(batches: &Batches, method: Method, uri: &str, body: &Value) -> (StatusCode, Value) {
    let body = if body.is_null() {
        Body::empty()
    } else {
        Body::from(body.to_string())
    };
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .body(body)
        .expect("request");
    let (parts, body) = request.into_parts();
    match batches.dispatch(&parts, body).await {
        Ok(value) => (StatusCode::OK, value),
        Err((status, message)) => (status, json!({"error":{"message":message}})),
    }
}

async fn create(batches: &Batches, file: &str, endpoint: &str) -> (StatusCode, Value) {
    let body = json!({"input_file_id":file,"endpoint":endpoint,"completion_window":"24h"});
    send(batches, Method::POST, "/v1/batches", &body).await
}

/// The batch once it is terminal (or after ten seconds, for the assertion
/// to fail on).
async fn settled(store: &Store, id: &str) -> Value {
    let (store, id) = (store.clone(), id.to_owned());
    tokio::task::spawn_blocking(move || {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let batch = store.get_batch(&id).expect("batch readable");
            if batch.status.is_terminal() || Instant::now() > deadline {
                return serde_json::to_value(&batch).expect("batch json");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    })
    .await
    .expect("wait task")
}

async fn started(receiver: mpsc::Receiver<()>) {
    tokio::task::spawn_blocking(move || receiver.recv_timeout(Duration::from_secs(10)))
        .await
        .expect("wait task")
        .expect("the blocking line started");
}

fn records(store: &Store, file: &Value) -> Vec<Value> {
    let Some(file) = file.as_str() else {
        return Vec::new();
    };
    let mut text = String::new();
    store
        .open_file_content(file)
        .expect("result file")
        .content
        .read_to_string(&mut text)
        .expect("result text");
    text.lines()
        .map(|line| serde_json::from_str(line).expect("record json"))
        .collect()
}

fn custom_ids(records: &[Value]) -> Vec<&str> {
    records
        .iter()
        .map(|record| record["custom_id"].as_str().unwrap_or_default())
        .collect()
}

#[test]
fn only_batch_paths_match() {
    assert!(matches("/v1/batches"));
    assert!(matches("/v1/batches/batch_1/cancel"));
    assert!(!matches("/v1/batchesx"));
    assert!(!matches("/v1/responses"));
}

#[tokio::test]
async fn chat_lines_run_in_order_and_keep_their_custom_ids() {
    let scratch = Scratch::new();
    let (submit, seen, _) = fake(false);
    let batches = start(&scratch.store, submit);
    let mut unsupported = chat("c", "never");
    unsupported["body"]["n"] = json!(2);
    let file = scratch.input(&[
        chat("a", "hello"),
        chat("b", "boom"),
        unsupported,
        chat("d", "world"),
    ]);
    let body = json!({
        "input_file_id": file,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"team": "eval"},
    });
    let (status, created) = send(&batches, Method::POST, "/v1/batches", &body).await;
    assert_eq!(status, StatusCode::OK, "{created}");
    assert_eq!(created["object"], "batch");
    assert_eq!(created["status"], "in_progress");
    assert_eq!(created["metadata"], json!({"team":"eval"}));
    let id = created["id"].as_str().expect("id").to_owned();

    let done = settled(&scratch.store, &id).await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(
        done["request_counts"],
        json!({"total":4,"completed":2,"failed":2})
    );
    let output = records(&scratch.store, &done["output_file_id"]);
    assert_eq!(custom_ids(&output), ["a", "d"]);
    let answer = &output[0]["response"];
    assert_eq!(answer["status_code"], 200);
    assert_eq!(
        answer["body"]["choices"][0]["message"]["content"],
        "echo:hello"
    );
    assert_eq!(answer["body"]["model"], "local");
    let errors = records(&scratch.store, &done["error_file_id"]);
    assert_eq!(custom_ids(&errors), ["b", "c"]);
    assert_eq!(errors[0]["response"]["status_code"], 500);
    assert_eq!(errors[1]["response"]["status_code"], 400);
    assert!(
        errors[1]["response"]["body"]["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains('n'))
    );
    // The refused line never reached the engine.
    assert_eq!(*seen.lock().expect("seen"), ["hello", "boom", "world"]);

    let (status, fetched) = send(
        &batches,
        Method::GET,
        &format!("/v1/batches/{id}"),
        &Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(fetched, done);
    stop(&batches).await;
}

#[tokio::test]
async fn responses_and_completions_answer_in_their_own_shapes() {
    let scratch = Scratch::new();
    let (submit, _, _) = fake(false);
    let batches = start(&scratch.store, submit);
    let responses = scratch.input(&[json!({
        "custom_id": "r",
        "method": "POST",
        "url": "/v1/responses",
        "body": {"model": "local", "input": "hi"},
    })]);
    let completions = scratch.input(&[json!({
        "custom_id": "c",
        "method": "POST",
        "url": "/v1/completions",
        "body": {"model": "local", "prompt": "hey"},
    })]);
    let (_, first) = create(&batches, &responses, "/v1/responses").await;
    let (_, second) = create(&batches, &completions, "/v1/completions").await;

    let done = settled(&scratch.store, first["id"].as_str().expect("id")).await;
    assert_eq!(done["status"], "completed", "{done}");
    let output = records(&scratch.store, &done["output_file_id"]);
    assert_eq!(output[0]["response"]["body"]["object"], "response");
    assert_eq!(output[0]["response"]["body"]["status"], "completed");

    let done = settled(&scratch.store, second["id"].as_str().expect("id")).await;
    assert_eq!(done["status"], "completed", "{done}");
    let output = records(&scratch.store, &done["output_file_id"]);
    assert_eq!(
        output[0]["response"]["body"]["choices"][0]["text"],
        "echo:hey"
    );
    stop(&batches).await;
}

#[tokio::test]
async fn refused_requests_queue_nothing() {
    let scratch = Scratch::new();
    let (submit, seen, _) = fake(false);
    let batches = start(&scratch.store, submit);
    let file = scratch.input(&[chat("a", "hi")]);
    let (status, _) = create(&batches, &file, "/v1/embeddings").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let mut streaming = chat("a", "hi");
    streaming["body"]["stream"] = json!(true);
    let streaming = scratch.input(&[streaming]);
    let (status, _) = create(&batches, &streaming, "/v1/chat/completions").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let extra = json!({
        "input_file_id": file,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "output_expires_after": {"anchor": "created_at", "seconds": 3600},
    });
    let (status, _) = send(&batches, Method::POST, "/v1/batches", &extra).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = create(&batches, "file-missing", "/v1/chat/completions").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, list) = send(&batches, Method::GET, "/v1/batches", &Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list["object"], "list");
    assert_eq!(list["data"], json!([]));
    for (method, uri, expected) in [
        (Method::GET, "/v1/batches?limit=0", StatusCode::BAD_REQUEST),
        (
            Method::GET,
            "/v1/batches?order=asc",
            StatusCode::BAD_REQUEST,
        ),
        (
            Method::GET,
            "/v1/batches/batch_x?x=1",
            StatusCode::BAD_REQUEST,
        ),
        (
            Method::GET,
            "/v1/batches/batch_missing",
            StatusCode::NOT_FOUND,
        ),
        (
            Method::POST,
            "/v1/batches/batch_missing/cancel",
            StatusCode::NOT_FOUND,
        ),
        (Method::DELETE, "/v1/batches/batch_x", StatusCode::NOT_FOUND),
        (Method::GET, "/v1/batches/a/b/c", StatusCode::NOT_FOUND),
    ] {
        let (status, _) = send(&batches, method, uri, &Value::Null).await;
        assert_eq!(status, expected, "{uri}");
    }
    assert!(seen.lock().expect("seen").is_empty());
    stop(&batches).await;
}

#[tokio::test]
async fn cancel_interrupts_the_running_line_and_keeps_earlier_results() {
    let scratch = Scratch::new();
    let (submit, seen, receiver) = fake(true);
    let batches = start(&scratch.store, submit);
    let file = scratch.input(&[chat("a", "hello"), chat("b", "block"), chat("c", "later")]);
    let (_, created) = create(&batches, &file, "/v1/chat/completions").await;
    let id = created["id"].as_str().expect("id").to_owned();
    started(receiver).await;

    let uri = format!("/v1/batches/{id}/cancel");
    let (status, cancelling) = send(&batches, Method::POST, &uri, &Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        matches!(
            cancelling["status"].as_str(),
            Some("cancelling" | "cancelled")
        ),
        "{cancelling}"
    );
    let done = settled(&scratch.store, &id).await;
    assert_eq!(done["status"], "cancelled", "{done}");
    assert_eq!(
        done["request_counts"],
        json!({"total":3,"completed":1,"failed":0})
    );
    assert_eq!(
        custom_ids(&records(&scratch.store, &done["output_file_id"])),
        ["a"]
    );
    assert_eq!(done["error_file_id"], Value::Null);
    assert_eq!(*seen.lock().expect("seen"), ["hello", "block"]);
    stop(&batches).await;
}

#[tokio::test]
async fn shutdown_leaves_the_running_line_for_the_next_worker() {
    let scratch = Scratch::new();
    let (submit, _, receiver) = fake(true);
    let batches = start(&scratch.store, submit);
    let file = scratch.input(&[chat("a", "hello"), chat("b", "block")]);
    let (_, created) = create(&batches, &file, "/v1/chat/completions").await;
    let id = created["id"].as_str().expect("id").to_owned();
    started(receiver).await;
    stop(&batches).await;
    let interrupted = scratch.store.get_batch(&id).expect("batch");
    assert_eq!(interrupted.request_counts.completed, 1);
    assert!(!interrupted.status.is_terminal());

    // The next worker generates the interrupted line again, and only it.
    let (submit, seen, _) = fake(false);
    let batches = start(&scratch.store, submit);
    let done = settled(&scratch.store, &id).await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(
        custom_ids(&records(&scratch.store, &done["output_file_id"])),
        ["a", "b"]
    );
    assert_eq!(*seen.lock().expect("seen"), ["block"]);
    stop(&batches).await;
}

#[tokio::test]
async fn a_restarted_worker_skips_settled_lines_and_leased_batches() {
    let scratch = Scratch::new();
    let store = &scratch.store;
    let create_native = |lines: &[Value]| {
        store
            .create_batch(
                &CreateBatch {
                    input_file_id: scratch.input(lines),
                    endpoint: "/v1/chat/completions".to_owned(),
                    completion_window: "24h".to_owned(),
                    metadata: local_services::Metadata::new(),
                },
                "local",
            )
            .expect("batch created")
    };
    let resumed = create_native(&[chat("a", "one"), chat("b", "two")]);
    let held = create_native(&[chat("h", "held")]);
    {
        let lease = store
            .lease_batch(&resumed.id)
            .expect("lease")
            .expect("free");
        let outcome = LineOutcome::Response {
            status_code: 200,
            body: json!({"before": "restart"}),
        };
        assert!(
            store
                .settle_batch_line(&lease, 0, &outcome)
                .expect("settled")
        );
    }
    // Another server is running this one.
    let lease = store.lease_batch(&held.id).expect("lease").expect("free");

    let (submit, seen, _) = fake(false);
    let batches = start(store, submit);
    let done = settled(store, &resumed.id).await;
    assert_eq!(done["status"], "completed", "{done}");
    let output = records(store, &done["output_file_id"]);
    assert_eq!(custom_ids(&output), ["a", "b"]);
    assert_eq!(output[0]["response"]["body"], json!({"before":"restart"}));
    assert_eq!(*seen.lock().expect("seen"), ["two"]);
    let untouched = store.get_batch(&held.id).expect("batch");
    assert_eq!(untouched.request_counts.completed, 0);
    assert!(!untouched.status.is_terminal());

    // Once that server lets go, the next scan reclaims it.
    drop(lease);
    batches.worker.notify();
    let done = settled(store, &held.id).await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(*seen.lock().expect("seen"), ["two", "held"]);
    stop(&batches).await;
}

#[tokio::test]
async fn listing_pages_newest_first() {
    let scratch = Scratch::new();
    let (submit, _, _) = fake(false);
    let batches = start(&scratch.store, submit);
    let mut ids = Vec::new();
    for _ in 0..3 {
        let file = scratch.input(&[chat("a", "hi")]);
        let (_, created) = create(&batches, &file, "/v1/chat/completions").await;
        ids.push(created["id"].as_str().expect("id").to_owned());
    }
    let (status, page) = send(&batches, Method::GET, "/v1/batches?limit=2", &Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(page["data"][0]["id"], ids[2].as_str());
    assert_eq!(page["data"][1]["id"], ids[1].as_str());
    assert_eq!(page["has_more"], true);
    let after = page["last_id"].as_str().expect("last id");
    let (_, rest) = send(
        &batches,
        Method::GET,
        &format!("/v1/batches?after={after}"),
        &Value::Null,
    )
    .await;
    assert_eq!(rest["data"].as_array().map(Vec::len), Some(1));
    assert_eq!(rest["data"][0]["id"], ids[0].as_str());
    assert_eq!(rest["has_more"], false);
    stop(&batches).await;
}

#[tokio::test]
async fn a_batch_for_another_model_is_refused_without_a_new_batch() {
    let scratch = Scratch::new();
    let (submit, seen, _) = fake(false);
    let batches = start(&scratch.store, submit);
    let mut other = chat("a", "hello");
    other["body"]["model"] = json!("other");
    let file = scratch.input(&[other]);
    let (status, refused) = create(&batches, &file, "/v1/chat/completions").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert!(
        refused["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("installed model")),
        "{refused}"
    );
    let (_, page) = send(&batches, Method::GET, "/v1/batches", &Value::Null).await;
    assert_eq!(page["data"], json!([]));
    stop(&batches).await;
    assert!(seen.lock().expect("seen").is_empty());
}

#[tokio::test]
async fn a_resumed_line_for_another_model_fails_without_generating() {
    let scratch = Scratch::new();
    let mut other = chat("a", "hello");
    other["body"]["model"] = json!("other");
    // Stored by a server whose installed model was `other`.
    let stored = scratch
        .store
        .create_batch(
            &CreateBatch {
                input_file_id: scratch.input(&[other]),
                endpoint: "/v1/chat/completions".to_owned(),
                completion_window: "24h".to_owned(),
                metadata: local_services::Metadata::new(),
            },
            "other",
        )
        .expect("batch created");
    let (submit, seen, _) = fake(false);
    let batches = start(&scratch.store, submit);
    let done = settled(&scratch.store, &stored.id).await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(
        done["request_counts"],
        json!({"total":1,"completed":0,"failed":1})
    );
    let errors = records(&scratch.store, &done["error_file_id"]);
    assert_eq!(errors[0]["response"]["status_code"], 400);
    assert!(
        errors[0]["response"]["body"]["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("loaded model")),
        "{errors:?}"
    );
    stop(&batches).await;
    assert!(seen.lock().expect("seen").is_empty());
}
