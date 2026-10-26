#![expect(clippy::unwrap_used, clippy::too_many_lines, clippy::assert_is_empty)]
//! Per-project isolation: a store scoped to a project sees only the objects
//! that project created, as if no other existed; an unscoped store sees
//! everything; objects from before projects were recorded are unowned.

use std::fmt::Debug;
use std::fs;
use std::io::Read as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use local_services::batches::{CreateBatch, LineOutcome, ListBatches, NextLine};
use local_services::files::{CompleteUpload, CreateFile, CreateUpload, FilePurpose, ListFiles};
use local_services::vector_stores::{
    CreateFileBatch, CreateVectorStore, CreateVectorStoreFile, ListQuery, UpdateVectorStore,
};
use local_services::{Append, Error, ListItems, Metadata, Store};

const MODEL: &str = "local";

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "local-services-projects-{}-{}",
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

/// `id` with its last character changed: an ID of the same shape that names
/// nothing.
fn absent(id: &str) -> String {
    let mut fake = id.to_owned();
    let last = fake.pop().unwrap();
    fake.push(if last == '0' { '1' } else { '0' });
    fake
}

/// `call` on `id` fails exactly as on an ID that names nothing.
fn as_missing<T: Debug>(id: &str, call: impl Fn(&str) -> Result<T, Error>) {
    let fake = absent(id);
    let (hidden, missing) = (call(id).unwrap_err(), call(&fake).unwrap_err());
    let same = match (&hidden, &missing) {
        (Error::NotFound(hidden), Error::NotFound(missing))
        | (Error::InvalidArgument(hidden), Error::InvalidArgument(missing)) => {
            hidden.replace(id, &fake) == *missing
        }
        _ => false,
    };
    assert!(
        same,
        "{id}: {hidden:?} differs from a missing ID's {missing:?}"
    );
}

fn message(text: &str) -> Value {
    json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
}

fn text_file(store: &Store, content: &str) -> String {
    store
        .create_file(
            &CreateFile {
                filename: "notes.txt".to_owned(),
                purpose: FilePurpose::Assistants,
                expires_after: None,
            },
            content.as_bytes(),
        )
        .unwrap()
        .id
}

fn batch_file(store: &Store) -> String {
    let line = json!({
        "custom_id": "a",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    });
    store
        .create_file(
            &CreateFile {
                filename: "input.jsonl".to_owned(),
                purpose: FilePurpose::Batch,
                expires_after: None,
            },
            format!("{line}\n").as_bytes(),
        )
        .unwrap()
        .id
}

fn batch_request(file: &str) -> CreateBatch {
    CreateBatch {
        input_file_id: file.to_owned(),
        endpoint: "/v1/chat/completions".to_owned(),
        completion_window: "24h".to_owned(),
        metadata: Metadata::new(),
    }
}

fn file_ids(store: &Store) -> Vec<String> {
    let list = store.list_files(&ListFiles::default()).unwrap();
    list.data.into_iter().map(|file| file.id).collect()
}

#[test]
fn conversations_are_private_to_the_project_that_created_them() {
    let root = TempRoot::new();
    let store = root.open();
    let (a, b) = (store.scoped(Some("proj_a")), store.scoped(Some("proj_b")));
    let conversation = a
        .create_conversation(&Metadata::new(), vec![message("hello")])
        .unwrap();
    let id = conversation.id.as_str();
    let item = a.list_items(id, &ListItems::default()).unwrap().data[0]
        .id
        .clone();

    as_missing(id, |id| b.get_conversation(id));
    as_missing(id, |id| b.update_conversation(id, &Metadata::new()));
    as_missing(id, |id| b.add_items(id, vec![message("intrude")]));
    as_missing(id, |id| b.list_items(id, &ListItems::default()));
    as_missing(id, |id| b.get_item(id, &item));
    as_missing(id, |id| b.delete_item(id, &item));
    as_missing(id, |id| b.conversation_history(id));
    as_missing(id, |id| {
        b.append_items(
            id,
            Append {
                request_id: "req".to_owned(),
                expected_version: conversation.version,
                items: vec![message("intrude")],
            },
        )
    });
    as_missing(id, |id| b.delete_conversation(id));

    // Nothing B tried changed it; its owner and a caller without a project
    // still reach it.
    assert_eq!(a.conversation_history(id).unwrap().items.len(), 1);
    assert_eq!(store.get_conversation(id).unwrap(), conversation);

    // One created without a project is unowned: no project key sees it.
    let unowned = store
        .create_conversation(&Metadata::new(), Vec::new())
        .unwrap();
    as_missing(&unowned.id, |id| a.get_conversation(id));
    assert_eq!(store.get_conversation(&unowned.id).unwrap(), unowned);
}

#[test]
fn files_and_uploads_are_private_to_the_project_that_created_them() {
    let root = TempRoot::new();
    let store = root.open();
    let (a, b) = (store.scoped(Some("proj_a")), store.scoped(Some("proj_b")));
    let owned = text_file(&a, "alpha");
    let theirs = text_file(&b, "beta");
    let unowned = text_file(&store, "legacy");

    as_missing(&owned, |id| b.get_file(id));
    as_missing(&owned, |id| {
        b.open_file_content(id).map(|content| content.file)
    });
    as_missing(&owned, |id| b.delete_file(id));
    // Listing shows each project its own files, and a cursor naming another
    // project's file is as unknown as any other.
    assert_eq!(file_ids(&a), std::slice::from_ref(&owned));
    assert_eq!(file_ids(&b), std::slice::from_ref(&theirs));
    as_missing(&owned, |id| {
        b.list_files(&ListFiles {
            after: Some(id.to_owned()),
            ..ListFiles::default()
        })
    });
    // Unowned files are invisible to projects, visible without one.
    as_missing(&unowned, |id| a.get_file(id));
    let mut all = file_ids(&store);
    all.sort();
    let mut expected = vec![owned.clone(), theirs, unowned];
    expected.sort();
    assert_eq!(all, expected);

    // Uploads, and the file a completed upload makes.
    let upload = a
        .create_upload(&CreateUpload {
            filename: "up.txt".to_owned(),
            purpose: FilePurpose::Assistants,
            bytes: 5,
            mime_type: "text/plain".to_owned(),
            expires_after: None,
        })
        .unwrap();
    let id = upload.id.as_str();
    as_missing(id, |id| b.get_upload(id));
    as_missing(id, |id| b.add_upload_part(id, &b"hello"[..]));
    as_missing(id, |id| b.cancel_upload(id));
    let part = a.add_upload_part(id, &b"hello"[..]).unwrap();
    let complete = CompleteUpload {
        part_ids: vec![part.id],
        md5: None,
    };
    as_missing(id, |id| b.complete_upload(id, &complete));
    let made = a.complete_upload(id, &complete).unwrap().file.unwrap();
    as_missing(&made.id, |id| b.get_file(id));
    assert_eq!(a.get_file(&made.id).unwrap(), made);
    assert_eq!(store.get_file(&made.id).unwrap(), made);
    // Still A's: B's attempts deleted nothing.
    let mut text = String::new();
    a.open_file_content(&owned)
        .unwrap()
        .content
        .read_to_string(&mut text)
        .unwrap();
    assert_eq!(text, "alpha");
}

#[test]
fn batches_and_their_files_are_private_to_the_project_that_created_them() {
    let root = TempRoot::new();
    let store = root.open();
    let (a, b) = (store.scoped(Some("proj_a")), store.scoped(Some("proj_b")));
    let input = batch_file(&a);
    // Another project's input file is refused as an unknown one.
    as_missing(&input, |id| {
        b.create_batch(&batch_request(id), MODEL)
            .map(|batch| batch.id)
    });
    let batch = a.create_batch(&batch_request(&input), MODEL).unwrap();
    let id = batch.id.as_str();
    as_missing(id, |id| b.get_batch(id));
    as_missing(id, |id| b.cancel_batch(id));
    as_missing(id, |id| {
        b.list_batches(&ListBatches {
            after: Some(id.to_owned()),
            ..ListBatches::default()
        })
    });
    assert!(
        b.list_batches(&ListBatches::default())
            .unwrap()
            .data
            .is_empty()
    );
    assert_eq!(
        a.list_batches(&ListBatches::default()).unwrap().data,
        std::slice::from_ref(&batch)
    );

    // The worker (no project) runs it; its output belongs to the batch's
    // project.
    let lease = store.lease_batch(id).unwrap().unwrap();
    let NextLine::Line(line) = store.next_batch_line(&lease).unwrap() else {
        panic!("expected a line");
    };
    let outcome = LineOutcome::Response {
        status_code: 200,
        body: json!({"ok": true}),
    };
    store
        .settle_batch_line(&lease, line.line, &outcome)
        .unwrap();
    let finished = store.finish_batch(&lease).unwrap();
    let output = finished.output_file_id.clone().unwrap();
    as_missing(&output, |id| b.get_file(id));
    assert_eq!(a.get_file(&output).unwrap().id, output);
    assert_eq!(a.get_batch(id).unwrap(), finished);

    // A batch created without a project is unowned.
    let unowned = store
        .create_batch(&batch_request(&batch_file(&store)), MODEL)
        .unwrap();
    as_missing(&unowned.id, |id| a.get_batch(id));
    assert_eq!(
        store
            .list_batches(&ListBatches::default())
            .unwrap()
            .data
            .len(),
        2
    );
}

#[test]
fn vector_stores_are_private_to_the_project_that_created_them() {
    let root = TempRoot::new();
    let store = root.open();
    let (a, b) = (store.scoped(Some("proj_a")), store.scoped(Some("proj_b")));
    let file = text_file(&a, "alpha beta");
    let theirs = text_file(&b, "gamma");
    // Another project's file is refused as an unknown one, at creation and
    // when attaching.
    as_missing(&file, |id| {
        b.create_vector_store(&CreateVectorStore {
            file_ids: vec![id.to_owned()],
            ..CreateVectorStore::default()
        })
    });
    let vector_store = a
        .create_vector_store(&CreateVectorStore {
            file_ids: vec![file.clone()],
            ..CreateVectorStore::default()
        })
        .unwrap();
    let id = vector_store.id.as_str();
    as_missing(&theirs, |theirs| {
        a.create_vector_store_file(
            id,
            &CreateVectorStoreFile {
                file_id: theirs.to_owned(),
                ..CreateVectorStoreFile::default()
            },
        )
    });
    let batch = a
        .create_vector_store_file_batch(
            id,
            &CreateFileBatch {
                file_ids: Some(vec![file.clone()]),
                ..CreateFileBatch::default()
            },
        )
        .unwrap();

    as_missing(id, |id| b.get_vector_store(id));
    as_missing(id, |id| {
        b.update_vector_store(id, &UpdateVectorStore::default())
    });
    as_missing(id, |id| {
        b.list_vector_store_files(id, &ListQuery::default())
    });
    as_missing(id, |id| b.get_vector_store_file(id, &file));
    as_missing(id, |id| b.vector_store_file_content(id, &file));
    as_missing(id, |id| {
        b.update_vector_store_file(id, &file, &serde_json::Map::new())
    });
    as_missing(id, |id| b.delete_vector_store_file(id, &file));
    as_missing(id, |id| {
        b.create_vector_store_file(
            id,
            &CreateVectorStoreFile {
                file_id: theirs.clone(),
                ..CreateVectorStoreFile::default()
            },
        )
    });
    as_missing(id, |id| {
        b.create_vector_store_file_batch(
            id,
            &CreateFileBatch {
                file_ids: Some(vec![theirs.clone()]),
                ..CreateFileBatch::default()
            },
        )
    });
    as_missing(id, |id| b.get_vector_store_file_batch(id, &batch.id));
    as_missing(id, |id| b.cancel_vector_store_file_batch(id, &batch.id));
    as_missing(id, |id| {
        b.list_vector_store_file_batch_files(id, &batch.id, &ListQuery::default())
    });
    as_missing(id, |id| {
        b.list_vector_stores(&ListQuery {
            after: Some(id.to_owned()),
            ..ListQuery::default()
        })
    });
    as_missing(id, |id| b.delete_vector_store(id));
    assert!(
        b.list_vector_stores(&ListQuery::default())
            .unwrap()
            .data
            .is_empty()
    );

    // Its owner and a caller without a project still see it whole.
    let listed = a.list_vector_stores(&ListQuery::default()).unwrap().data;
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, id);
    assert_eq!(a.get_vector_store(id).unwrap().file_counts.total, 1);
    assert_eq!(store.get_vector_store(id).unwrap().id, id);

    // One created without a project is unowned.
    let unowned = store
        .create_vector_store(&CreateVectorStore::default())
        .unwrap();
    as_missing(&unowned.id, |id| a.get_vector_store(id));
    assert_eq!(
        store
            .list_vector_stores(&ListQuery::default())
            .unwrap()
            .data
            .len(),
        2
    );
}

#[test]
fn objects_from_before_projects_were_recorded_migrate_unowned() {
    let root = TempRoot::new();
    let (conversation, file, upload, batch, vector_store) = {
        let store = root.open();
        let conversation = store
            .create_conversation(&Metadata::new(), vec![message("old")])
            .unwrap()
            .id;
        let file = text_file(&store, "old text");
        let upload = store
            .create_upload(&CreateUpload {
                filename: "old.txt".to_owned(),
                purpose: FilePurpose::Assistants,
                bytes: 3,
                mime_type: "text/plain".to_owned(),
                expires_after: None,
            })
            .unwrap()
            .id;
        let batch = store
            .create_batch(&batch_request(&batch_file(&store)), MODEL)
            .unwrap()
            .id;
        let vector_store = store
            .create_vector_store(&CreateVectorStore {
                file_ids: vec![file.clone()],
                ..CreateVectorStore::default()
            })
            .unwrap()
            .id;
        // Undo the schema versions that record projects, as a database
        // written before them.
        rusqlite::Connection::open(store.root().join("services.sqlite3"))
            .unwrap()
            .execute_batch(
                "BEGIN;
                 ALTER TABLE conversations DROP COLUMN project_id;
                 UPDATE schema_versions SET version = 2 WHERE component = 'conversations';
                 DROP INDEX files_project_seq;
                 ALTER TABLE files DROP COLUMN project_id;
                 ALTER TABLE uploads DROP COLUMN project_id;
                 UPDATE schema_versions SET version = 2 WHERE component = 'files';
                 DROP INDEX batches_project_seq;
                 UPDATE schema_versions SET version = 2 WHERE component = 'batches';
                 DROP INDEX vector_stores_project_seq;
                 ALTER TABLE vector_stores DROP COLUMN project_id;
                 UPDATE schema_versions SET version = 2 WHERE component = 'vector_stores';
                 COMMIT;",
            )
            .unwrap();
        (conversation, file, upload, batch, vector_store)
    };
    let store = root.open();
    let a = store.scoped(Some("proj_a"));
    // Every old object is still there, whole, for a caller without a project...
    assert_eq!(
        store
            .conversation_history(&conversation)
            .unwrap()
            .items
            .len(),
        1
    );
    let mut text = String::new();
    store
        .open_file_content(&file)
        .unwrap()
        .content
        .read_to_string(&mut text)
        .unwrap();
    assert_eq!(text, "old text");
    assert_eq!(store.get_upload(&upload).unwrap().bytes, 3);
    assert_eq!(store.get_batch(&batch).unwrap().request_counts.total, 1);
    assert_eq!(
        store
            .get_vector_store(&vector_store)
            .unwrap()
            .file_counts
            .total,
        1
    );
    // ...and unowned, so no project sees it.
    as_missing(&conversation, |id| a.get_conversation(id));
    as_missing(&file, |id| a.get_file(id));
    as_missing(&upload, |id| a.get_upload(id));
    as_missing(&batch, |id| a.get_batch(id));
    as_missing(&vector_store, |id| a.get_vector_store(id));
    assert!(file_ids(&a).is_empty());
    // The migrated tables take owned objects.
    let owned = text_file(&a, "new");
    assert_eq!(file_ids(&a), [owned]);
}
