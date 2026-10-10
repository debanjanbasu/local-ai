#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Batches persistence and the worker protocol, driven without a model.

use std::fs;
use std::io::Read as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use local_services::batches::{
    Batch, BatchStatus, CreateBatch, LineOutcome, ListBatches, MAX_LINES, NextLine, PendingLine,
};
use local_services::files::{
    BATCH_OUTPUT_EXPIRES_AFTER_SECONDS, CreateFile, FileObject, FilePurpose, ListFiles,
};
use local_services::{Error, Metadata, Store};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "local-services-batches-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn open(&self) -> Store {
        Store::open(self.0.join("store")).unwrap()
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The installed model every fixture line names.
const MODEL: &str = "local";

fn line(custom_id: &str, text: &str) -> Value {
    json!({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": MODEL, "messages": [{"role": "user", "content": text}]},
    })
}

fn jsonl(lines: &[Value]) -> String {
    lines.iter().fold(String::new(), |mut content, line| {
        content.push_str(&line.to_string());
        content.push('\n');
        content
    })
}

fn upload(store: &Store, content: &str, purpose: FilePurpose) -> String {
    let filename = if purpose == FilePurpose::Batch {
        "input.jsonl"
    } else {
        "input.txt"
    };
    store
        .create_file(
            &CreateFile {
                filename: filename.to_owned(),
                purpose,
                expires_after: None,
            },
            content.as_bytes(),
        )
        .unwrap()
        .id
}

fn request(file: &str) -> CreateBatch {
    CreateBatch {
        input_file_id: file.to_owned(),
        endpoint: "/v1/chat/completions".to_owned(),
        completion_window: "24h".to_owned(),
        metadata: Metadata::new(),
    }
}

fn create(store: &Store, ids: &[&str]) -> Batch {
    let lines: Vec<Value> = ids.iter().map(|id| line(id, id)).collect();
    let file = upload(store, &jsonl(&lines), FilePurpose::Batch);
    store.create_batch(&request(&file), MODEL).unwrap()
}

fn ok(text: &str) -> LineOutcome {
    LineOutcome::Response {
        status_code: 200,
        body: json!({"text": text}),
    }
}

fn next(store: &Store, lease: &local_services::batches::BatchLease) -> PendingLine {
    match store.next_batch_line(lease).unwrap() {
        NextLine::Line(line) => line,
        other => panic!("expected a pending line, got {other:?}"),
    }
}

fn records(store: &Store, file: Option<&str>) -> Vec<Value> {
    let Some(file) = file else {
        return Vec::new();
    };
    let mut text = String::new();
    store
        .open_file_content(file)
        .unwrap()
        .content
        .read_to_string(&mut text)
        .unwrap();
    text.lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn batch_outputs(store: &Store) -> Vec<FileObject> {
    store
        .list_files(&ListFiles {
            purpose: Some(FilePurpose::BatchOutput),
            ..ListFiles::default()
        })
        .unwrap()
        .data
}

fn custom_ids(records: &[Value]) -> Vec<&str> {
    records
        .iter()
        .map(|record| record["custom_id"].as_str().unwrap())
        .collect()
}

fn database(store: &Store) -> rusqlite::Connection {
    rusqlite::Connection::open(store.root().join("services.sqlite3")).unwrap()
}

fn stored_lines(store: &Store) -> i64 {
    database(store)
        .query_row("SELECT COUNT(*) FROM batch_lines", [], |row| row.get(0))
        .unwrap()
}

#[test]
fn refused_input_stores_nothing() {
    let root = TempRoot::new();
    let store = root.open();
    let good = line("a", "hi");
    let mut stream = line("b", "hi");
    stream["body"]["stream"] = json!(true);
    let mut background = line("b", "hi");
    background["body"]["background"] = json!(true);
    let mut wrong_url = line("b", "hi");
    wrong_url["url"] = json!("/v1/completions");
    let mut other_model = line("b", "hi");
    other_model["body"]["model"] = json!("another");
    let mut extra = line("b", "hi");
    extra["extra"] = json!(1);
    let mut get = line("b", "hi");
    get["method"] = json!("GET");
    let cases = [
        jsonl(&[good.clone(), line("x", "1"), line("a", "dup")]),
        jsonl(&[good.clone(), stream]),
        jsonl(&[good.clone(), background]),
        jsonl(&[good.clone(), wrong_url]),
        jsonl(&[good.clone(), other_model]),
        jsonl(&[good.clone(), extra]),
        jsonl(&[good.clone(), get]),
        format!("{good}\n\n{}\n", line("b", "hi")),
        format!("{good}\nnot json\n"),
    ];
    for content in &cases {
        let file = upload(&store, content, FilePurpose::Batch);
        let error = store.create_batch(&request(&file), MODEL).unwrap_err();
        assert!(
            matches!(error, Error::InvalidArgument(_)),
            "{content}: {error}"
        );
    }

    let file = upload(
        &store,
        &jsonl(std::slice::from_ref(&good)),
        FilePurpose::Batch,
    );
    let mut embeddings = request(&file);
    embeddings.endpoint = "/v1/embeddings".to_owned();
    let mut window = request(&file);
    window.completion_window = "48h".to_owned();
    let mut metadata = request(&file);
    metadata.metadata = (0..17).map(|n| (format!("k{n}"), "v".to_owned())).collect();
    let user_data = upload(&store, &jsonl(&[good]), FilePurpose::UserData);
    for bad in [
        embeddings,
        window,
        metadata,
        request(&user_data),
        request("file-missing"),
    ] {
        let error = store.create_batch(&bad, MODEL).unwrap_err();
        assert!(matches!(error, Error::InvalidArgument(_)), "{error}");
    }

    let listed = store.list_batches(&ListBatches::default()).unwrap();
    assert_eq!(listed.data, [] as [local_services::batches::Batch; 0]);
    assert_eq!(stored_lines(&store), 0);
}

#[test]
fn a_batch_for_another_model_is_refused_and_stores_nothing() {
    let root = TempRoot::new();
    let store = root.open();
    let file = upload(&store, &jsonl(&[line("a", "1")]), FilePurpose::Batch);
    let error = store.create_batch(&request(&file), "other").unwrap_err();
    assert!(matches!(error, Error::InvalidArgument(_)), "{error}");
    let listed = store.list_batches(&ListBatches::default()).unwrap();
    assert_eq!(listed.data, [] as [Batch; 0]);
    assert_eq!(stored_lines(&store), 0);

    // The same file is accepted for the model it names.
    let batch = store.create_batch(&request(&file), MODEL).unwrap();
    assert_eq!(batch.model, MODEL);
}

#[test]
fn more_than_the_line_limit_is_refused() {
    let root = TempRoot::new();
    let store = root.open();
    let lines: Vec<Value> = (0..=MAX_LINES).map(|n| line(&n.to_string(), "x")).collect();
    let file = upload(&store, &jsonl(&lines), FilePurpose::Batch);
    assert!(matches!(
        store.create_batch(&request(&file), MODEL),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(stored_lines(&store), 0);
}

#[test]
fn lines_settle_once_and_results_keep_their_custom_ids() {
    let root = TempRoot::new();
    let store = root.open();
    let mut metadata = Metadata::new();
    metadata.insert("team".to_owned(), "eval".to_owned());
    let file = upload(
        &store,
        &jsonl(&[line("a", "1"), line("b", "2"), line("c", "3")]),
        FilePurpose::Batch,
    );
    let mut create = request(&file);
    create.metadata = metadata.clone();
    let batch = store.create_batch(&create, MODEL).unwrap();
    assert_eq!(batch.status, BatchStatus::InProgress);
    assert_eq!(batch.model, "local");
    assert_eq!(batch.metadata, metadata);
    assert_eq!(batch.request_counts.total, 3);
    assert_eq!(batch.expires_at, batch.created_at + 86_400);
    assert_eq!(store.active_batches().unwrap(), vec![batch.id.clone()]);

    let lease = store.lease_batch(&batch.id).unwrap().unwrap();
    let first = next(&store, &lease);
    assert_eq!((first.line, first.custom_id.as_str()), (0, "a"));
    assert_eq!(first.body["messages"][0]["content"], "1");
    assert!(store.settle_batch_line(&lease, 0, &ok("one")).unwrap());
    assert!(!store.settle_batch_line(&lease, 0, &ok("again")).unwrap());
    let second = next(&store, &lease);
    assert_eq!(second.custom_id, "b");
    let refused = LineOutcome::Response {
        status_code: 400,
        body: json!({"error": {"message": "bad"}}),
    };
    assert!(store.settle_batch_line(&lease, 1, &refused).unwrap());
    let third = next(&store, &lease);
    let failed = LineOutcome::Error {
        code: "engine".to_owned(),
        message: "down".to_owned(),
    };
    assert!(store.settle_batch_line(&lease, 2, &failed).unwrap());
    assert_eq!(store.next_batch_line(&lease).unwrap(), NextLine::Finish);

    let done = store.finish_batch(&lease).unwrap();
    assert_eq!(done.status, BatchStatus::Completed);
    assert!(done.completed_at.is_some() && done.finalizing_at.is_some());
    assert_eq!(
        (done.request_counts.completed, done.request_counts.failed),
        (1, 2)
    );
    assert_eq!(
        done.output_file_id,
        Some(format!("file_{}_output", batch.id))
    );
    assert_eq!(done.error_file_id, Some(format!("file_{}_error", batch.id)));
    for file in batch_outputs(&store) {
        assert_eq!(file.purpose, FilePurpose::BatchOutput);
        assert_eq!(
            file.expires_at,
            Some(file.created_at + BATCH_OUTPUT_EXPIRES_AFTER_SECONDS)
        );
    }
    assert_eq!(batch_outputs(&store).len(), 2);
    let output = records(&store, done.output_file_id.as_deref());
    assert_eq!(custom_ids(&output), ["a"]);
    assert_eq!(output[0]["id"], first.record_id.as_str());
    assert_eq!(output[0]["response"]["status_code"], 200);
    assert_eq!(output[0]["response"]["body"]["text"], "one");
    let errors = records(&store, done.error_file_id.as_deref());
    assert_eq!(custom_ids(&errors), ["b", "c"]);
    assert_eq!(errors[0]["response"]["status_code"], 400);
    assert_eq!(errors[1]["id"], third.record_id.as_str());
    assert_eq!(errors[1]["error"]["code"], "engine");

    assert_eq!(store.next_batch_line(&lease).unwrap(), NextLine::Done);
    assert_eq!(store.finish_batch(&lease).unwrap(), done);
    assert_eq!(
        store.active_batches().unwrap(),
        [] as [std::string::String; 0]
    );
    assert!(matches!(
        store.cancel_batch(&batch.id),
        Err(Error::Conflict(_))
    ));
}

#[test]
fn cancel_keeps_settled_lines_and_drops_the_rest() {
    let root = TempRoot::new();
    let store = root.open();
    let batch = create(&store, &["a", "b", "c"]);
    let lease = store.lease_batch(&batch.id).unwrap().unwrap();
    next(&store, &lease);
    assert!(store.settle_batch_line(&lease, 0, &ok("a")).unwrap());

    let cancelling = store.cancel_batch(&batch.id).unwrap();
    assert_eq!(cancelling.status, BatchStatus::Cancelling);
    assert!(cancelling.cancelling_at.is_some());
    assert_eq!(store.cancel_batch(&batch.id).unwrap(), cancelling);
    assert_eq!(store.active_batches().unwrap(), vec![batch.id.clone()]);
    assert_eq!(store.next_batch_line(&lease).unwrap(), NextLine::Finish);
    // A line that was already running when the cancel arrived still counts.
    assert!(store.settle_batch_line(&lease, 1, &ok("b")).unwrap());

    let cancelled = store.finish_batch(&lease).unwrap();
    assert_eq!(cancelled.status, BatchStatus::Cancelled);
    assert!(cancelled.cancelled_at.is_some());
    assert_eq!(cancelled.request_counts.total, 3);
    assert_eq!(
        (
            cancelled.request_counts.completed,
            cancelled.request_counts.failed
        ),
        (2, 0)
    );
    let output = records(&store, cancelled.output_file_id.as_deref());
    assert_eq!(custom_ids(&output), ["a", "b"]);
    assert_eq!(cancelled.error_file_id, None);
    assert_eq!(store.cancel_batch(&batch.id).unwrap(), cancelled);
    assert!(matches!(
        store.settle_batch_line(&lease, 2, &ok("c")),
        Err(Error::Conflict(_))
    ));

    // A crash while writing the results leaves the batch cancelling, but its
    // lines stay frozen: the resumed finish writes exactly what was settled.
    database(&store)
        .execute(
            "UPDATE batches SET status = 'cancelling', cancelled_at = NULL WHERE id = ?1",
            [&batch.id],
        )
        .unwrap();
    assert!(matches!(
        store.settle_batch_line(&lease, 2, &ok("c")),
        Err(Error::Conflict(_))
    ));
    let resumed = store.finish_batch(&lease).unwrap();
    assert_eq!(resumed.status, BatchStatus::Cancelled);
    assert_eq!(resumed.output_file_id, cancelled.output_file_id);
    assert_eq!(
        custom_ids(&records(&store, resumed.output_file_id.as_deref())),
        ["a", "b"]
    );
}

#[test]
fn a_finish_interrupted_after_committing_results_reuses_them() {
    let root = TempRoot::new();
    let (id, done) = {
        let store = root.open();
        let batch = create(&store, &["a", "b"]);
        let lease = store.lease_batch(&batch.id).unwrap().unwrap();
        assert!(store.settle_batch_line(&lease, 0, &ok("a")).unwrap());
        let failed = LineOutcome::Error {
            code: "engine".to_owned(),
            message: "down".to_owned(),
        };
        assert!(store.settle_batch_line(&lease, 1, &failed).unwrap());
        (batch.id, store.finish_batch(&lease).unwrap())
    };
    let output_id = done.output_file_id.clone().unwrap();
    let error_id = done.error_file_id.unwrap();
    // Both result files committed, then a crash before the terminal status.
    let interrupt = |store: &Store| {
        database(store)
            .execute(
                "UPDATE batches SET status = 'finalizing', completed_at = NULL,
                                    output_file_id = NULL, error_file_id = NULL
                 WHERE id = ?1",
                [&id],
            )
            .unwrap();
    };

    let store = root.open();
    let committed = batch_outputs(&store);
    interrupt(&store);
    assert_eq!(store.active_batches().unwrap(), vec![id.clone()]);
    let lease = store.lease_batch(&id).unwrap().unwrap();
    assert_eq!(store.next_batch_line(&lease).unwrap(), NextLine::Finish);
    let resumed = store.finish_batch(&lease).unwrap();
    assert_eq!(resumed.status, BatchStatus::Completed);
    assert_eq!(resumed.output_file_id.as_deref(), Some(output_id.as_str()));
    assert_eq!(resumed.error_file_id.as_deref(), Some(error_id.as_str()));
    // One file per kind, the very ones committed before (same lifetime), and
    // no staged copy left behind: the input file plus two results.
    assert_eq!(batch_outputs(&store), committed);
    assert_eq!(committed.len(), 2);
    let blobs = fs::read_dir(store.root().join("file-store").join("blobs")).unwrap();
    assert_eq!(blobs.count(), 3);
    assert_eq!(custom_ids(&records(&store, Some(&output_id))), ["a"]);
    assert_eq!(custom_ids(&records(&store, Some(&error_id))), ["b"]);

    // A result file deleted while finalizing is not written again with a
    // fresh lifetime; the batch still names it.
    interrupt(&store);
    store.delete_file(&output_id).unwrap();
    let again = store.finish_batch(&lease).unwrap();
    assert_eq!(again.status, BatchStatus::Completed);
    assert_eq!(again.output_file_id.as_deref(), Some(output_id.as_str()));
    assert!(matches!(
        store.get_file(&output_id),
        Err(Error::NotFound(_))
    ));
    let left: Vec<String> = batch_outputs(&store)
        .into_iter()
        .map(|file| file.id)
        .collect();
    assert_eq!(left, [error_id]);
}

#[test]
fn a_lease_works_only_on_the_store_that_granted_it() {
    let first = TempRoot::new();
    let second = TempRoot::new();
    let granting = first.open();
    let other = second.open();
    let batch = create(&granting, &["a"]);
    create(&other, &["a"]);
    let lease = granting.lease_batch(&batch.id).unwrap().unwrap();
    assert!(matches!(
        other.next_batch_line(&lease),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        other.settle_batch_line(&lease, 0, &ok("a")),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        other.finish_batch(&lease),
        Err(Error::InvalidArgument(_))
    ));
    // The same directory opened again is the same store.
    assert!(matches!(
        first.open().next_batch_line(&lease).unwrap(),
        NextLine::Line(_)
    ));
}

#[test]
fn a_reopened_store_resumes_after_the_settled_lines() {
    let root = TempRoot::new();
    let (id, first) = {
        let store = root.open();
        let batch = create(&store, &["a", "b", "c"]);
        let lease = store.lease_batch(&batch.id).unwrap().unwrap();
        let first = next(&store, &lease);
        assert!(
            store
                .settle_batch_line(&lease, first.line, &ok("a"))
                .unwrap()
        );
        // The next line was taken but never settled: a crash mid-generation.
        assert_eq!(next(&store, &lease).line, 1);
        (batch.id, first)
    };

    let store = root.open();
    assert_eq!(store.active_batches().unwrap(), vec![id.clone()]);
    let lease = store.lease_batch(&id).unwrap().unwrap();
    let resumed = next(&store, &lease);
    assert_eq!((resumed.line, resumed.custom_id.as_str()), (1, "b"));
    assert!(store.settle_batch_line(&lease, 1, &ok("b")).unwrap());
    assert!(store.settle_batch_line(&lease, 2, &ok("c")).unwrap());
    let done = store.finish_batch(&lease).unwrap();
    assert_eq!(done.request_counts.completed, 3);
    let output = records(&store, done.output_file_id.as_deref());
    assert_eq!(custom_ids(&output), ["a", "b", "c"]);
    assert_eq!(output[0]["id"], first.record_id.as_str());
}

#[test]
fn only_one_holder_owns_a_batch() {
    let root = TempRoot::new();
    let store = root.open();
    let batch = create(&store, &["a"]);
    let lease = store.lease_batch(&batch.id).unwrap().unwrap();
    assert!(store.lease_batch(&batch.id).unwrap().is_none());
    assert!(root.open().lease_batch(&batch.id).unwrap().is_none());
    drop(lease);
    let again = root.open().lease_batch(&batch.id).unwrap();
    assert_eq!(
        again
            .as_ref()
            .map(local_services::batches::BatchLease::batch_id),
        Some(batch.id.as_str())
    );
    assert!(matches!(
        store.lease_batch("batch_missing"),
        Err(Error::NotFound(_))
    ));
    assert!(matches!(
        store.lease_batch("../batch_x"),
        Err(Error::NotFound(_))
    ));
}

#[test]
fn expiry_fails_the_unfinished_lines() {
    let root = TempRoot::new();
    let store = root.open();
    let batch = create(&store, &["a", "b", "c"]);
    let lease = store.lease_batch(&batch.id).unwrap().unwrap();
    next(&store, &lease);
    assert!(store.settle_batch_line(&lease, 0, &ok("a")).unwrap());
    database(&store)
        .execute(
            "UPDATE batches SET expires_at = 1 WHERE id = ?1",
            [&batch.id],
        )
        .unwrap();

    assert_eq!(store.next_batch_line(&lease).unwrap(), NextLine::Finish);
    let expired = store.finish_batch(&lease).unwrap();
    assert_eq!(expired.status, BatchStatus::Expired);
    assert!(expired.expired_at.is_some());
    assert_eq!(
        (
            expired.request_counts.completed,
            expired.request_counts.failed
        ),
        (1, 2)
    );
    assert_eq!(
        custom_ids(&records(&store, expired.output_file_id.as_deref())),
        ["a"]
    );
    let errors = records(&store, expired.error_file_id.as_deref());
    assert_eq!(custom_ids(&errors), ["b", "c"]);
    assert!(
        errors.iter().all(
            |record| record["error"]["code"] == "batch_expired" && record["response"].is_null()
        )
    );
    assert!(matches!(
        store.settle_batch_line(&lease, 1, &ok("late")),
        Err(Error::Conflict(_))
    ));
}

#[test]
fn listing_pages_newest_first() {
    let root = TempRoot::new();
    let store = root.open();
    let ids: Vec<String> = (0..3).map(|_| create(&store, &["a"]).id).collect();
    let page = store
        .list_batches(&ListBatches {
            after: None,
            limit: Some(2),
        })
        .unwrap();
    let listed: Vec<&str> = page.data.iter().map(|batch| batch.id.as_str()).collect();
    assert_eq!(listed, [ids[2].as_str(), ids[1].as_str()]);
    assert!(page.has_more);
    let rest = store
        .list_batches(&ListBatches {
            after: page.last_id.clone(),
            limit: None,
        })
        .unwrap();
    assert_eq!(rest.data.len(), 1);
    assert_eq!(rest.data[0].id, ids[0]);
    assert!(!rest.has_more);
    for limit in [0, 101] {
        assert!(matches!(
            store.list_batches(&ListBatches {
                after: None,
                limit: Some(limit),
            }),
            Err(Error::InvalidArgument(_))
        ));
    }
    assert!(matches!(
        store.list_batches(&ListBatches {
            after: Some("batch_missing".to_owned()),
            limit: None,
        }),
        Err(Error::NotFound(_))
    ));
    let json = serde_json::to_value(&page.data[0]).unwrap();
    assert_eq!(json["object"], "batch");
    assert_eq!(json["endpoint"], "/v1/chat/completions");
    assert_eq!(json["status"], "in_progress");
    assert_eq!(
        json["request_counts"],
        json!({"total":1,"completed":0,"failed":0})
    );
}
