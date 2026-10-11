#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::assert_is_empty,
    clippy::format_collect
)]
//! Vector stores: persistence, the indexing lifecycle, index consistency,
//! bounds, pagination and lexical search, driven without a model.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Map, Value, json};

use local_services::files::{CreateFile, FilePurpose};
use local_services::vector_stores::{
    Attributes, BatchStatus, ChunkingStrategy, CreateFileBatch, CreateVectorStore,
    CreateVectorStoreFile, ErrorCode, ExpirationAnchor, ExpirationPolicy, FileStatus, Filter,
    ListOrder, ListQuery, MAX_FILE_CHUNKS, MAX_INDEXED_FILE_BYTES, Query, Ranker, RankingOptions,
    SearchRequest, StaticChunking, StoreStatus, UpdateVectorStore, VectorIndex, VectorStoreIndexer,
};
use local_services::{Error, Store};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "local-services-vector-stores-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn open(&self) -> (Store, VectorIndex) {
        let store = Store::open(self.0.join("store")).unwrap();
        let index = VectorIndex::open(&store).unwrap();
        (store, index)
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn file(store: &Store, name: &str, content: impl AsRef<[u8]>) -> String {
    store
        .create_file(
            &CreateFile {
                filename: name.to_owned(),
                purpose: FilePurpose::Assistants,
                expires_after: None,
            },
            content.as_ref(),
        )
        .unwrap()
        .id
}

fn vector_store(store: &Store, file_ids: &[&str]) -> String {
    store
        .create_vector_store(&CreateVectorStore {
            name: Some("docs".to_owned()),
            file_ids: file_ids.iter().map(|id| (*id).to_owned()).collect(),
            ..CreateVectorStore::default()
        })
        .unwrap()
        .id
}

fn index_all(index: &VectorIndex) -> usize {
    let mut indexed = 0;
    loop {
        match index.index_next().unwrap() {
            0 => return indexed,
            files => indexed += files,
        }
    }
}

fn search(index: &VectorIndex, id: &str, query: &str) -> Vec<(String, f64, String)> {
    index
        .search(id, &SearchRequest::new(query))
        .unwrap()
        .data
        .into_iter()
        .map(|hit| (hit.file_id, hit.score, hit.content[0].text.clone()))
        .collect()
}

/// Attachments waiting in the garbage outbox.
fn garbage(store: &Store) -> i64 {
    raw(store)
        .query_row(
            "SELECT COUNT(*) FROM vector_store_index_garbage",
            [],
            |row| row.get(0),
        )
        .unwrap()
}

fn raw(store: &Store) -> rusqlite::Connection {
    rusqlite::Connection::open(store.root().join("services.sqlite3")).unwrap()
}

fn attributes(value: Value) -> Attributes {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

fn words(count: usize) -> String {
    (0..count).map(|i| format!("w{i} ")).collect()
}

#[test]
fn files_are_indexed_searched_and_read_back() {
    let root = TempRoot::new();
    let (store, index) = root.open();
    let cats = file(&store, "cats.txt", "Cats are running through the garden.");
    let dogs = file(
        &store,
        "dogs.md",
        "\u{feff}Dogs bark at the mailman every morning.",
    );
    let id = vector_store(&store, &[&cats, &dogs]);
    let created = store.get_vector_store(&id).unwrap();
    assert_eq!(created.status, StoreStatus::InProgress);
    assert_eq!(created.file_counts.in_progress, 2);
    assert_eq!(created.file_counts.total, 2);
    assert_eq!(created.usage_bytes, 0);
    assert_eq!(created.name, "docs");
    let pending = store.get_vector_store_file(&id, &cats).unwrap();
    assert_eq!(pending.status, FileStatus::InProgress);
    assert_eq!(
        pending.chunking_strategy.config,
        StaticChunking {
            max_chunk_size_tokens: 800,
            chunk_overlap_tokens: 400
        }
    );
    // Nothing is searchable before indexing.
    assert!(search(&index, &id, "cats").is_empty());

    assert_eq!(index_all(&index), 2);
    let done = store.get_vector_store(&id).unwrap();
    assert_eq!(done.status, StoreStatus::Completed);
    assert_eq!(done.file_counts.completed, 2);
    assert!(done.usage_bytes > 0);

    // Porter stemming: "run" matches "running".
    let hits = search(&index, &id, "who is run?");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, cats);
    assert!(hits[0].1 > 0.0 && hits[0].1 < 1.0);
    assert_eq!(hits[0].2, "Cats are running through the garden.");
    // No shared word, no hit: lexical, not semantic.
    assert!(search(&index, &id, "feline").is_empty());
    // Punctuation and query syntax in a query are plain words.
    assert!(search(&index, &id, "\"NEAR( OR * AND:").is_empty());

    let content = store.vector_store_file_content(&id, &dogs).unwrap();
    assert_eq!(content.data.len(), 1);
    assert_eq!(
        content.data[0].text,
        "Dogs bark at the mailman every morning."
    );
    assert!(!content.has_more);
    let check = index.check().unwrap();
    assert!(!check.rebuilt);
    assert_eq!(check.chunks, 2);
}

#[test]
fn stores_files_and_the_index_survive_a_restart() {
    let root = TempRoot::new();
    let id;
    let alpha;
    {
        let (store, index) = root.open();
        alpha = file(&store, "a.txt", "alpha bravo charlie");
        id = vector_store(&store, &[&alpha]);
        index_all(&index);
    }
    let (store, index) = root.open();
    assert_eq!(
        store.get_vector_store(&id).unwrap().status,
        StoreStatus::Completed
    );
    assert_eq!(search(&index, &id, "bravo")[0].0, alpha);
    assert!(!index.check().unwrap().rebuilt);
}

#[test]
fn files_left_in_progress_resume_on_the_next_start() {
    let root = TempRoot::new();
    let id;
    let alpha;
    {
        let (store, _index) = root.open();
        alpha = file(&store, "a.txt", "resumable indexing works");
        id = vector_store(&store, &[&alpha]);
        // The process stops before indexing anything.
    }
    let (store, index) = root.open();
    let index = Arc::new(index);
    let indexer = VectorStoreIndexer::start(Arc::clone(&index)).unwrap();
    indexer.wait_idle();
    assert_eq!(
        store.get_vector_store_file(&id, &alpha).unwrap().status,
        FileStatus::Completed
    );
    // A notification after another attach runs again.
    let beta = file(&store, "b.txt", "second file arrives");
    store
        .create_vector_store_file(
            &id,
            &CreateVectorStoreFile {
                file_id: beta.clone(),
                ..CreateVectorStoreFile::default()
            },
        )
        .unwrap();
    indexer.notify();
    indexer.wait_idle();
    assert_eq!(search(&index, &id, "arrives")[0].0, beta);
    // A notification after a deletion removes its documents.
    store.delete_vector_store_file(&id, &alpha).unwrap();
    assert_eq!(garbage(&store), 1);
    indexer.notify();
    indexer.wait_idle();
    assert_eq!(garbage(&store), 0);
    indexer.close();
    let check = index.check().unwrap();
    assert_eq!(
        (check.chunks, check.collected, check.rebuilt),
        (1, 0, false)
    );
}

#[test]
fn only_one_process_owns_the_index() {
    let root = TempRoot::new();
    let (store, index) = root.open();
    assert!(matches!(VectorIndex::open(&store), Err(Error::Conflict(_))));
    drop(index);
    assert!(VectorIndex::open(&store).is_ok());
}

/// The state a crash leaves after an indexing attempt committed the index
/// but not the database: the file back in progress, without text or
/// chunks, listed in the outbox, its documents still in the index.
fn crash_after_index_commit(store: &Store, file_id: &str) {
    let connection = raw(store);
    let attach_id: String = connection
        .query_row(
            "SELECT attach_id FROM vector_store_files WHERE file_id = ?1",
            [file_id],
            |row| row.get(0),
        )
        .unwrap();
    connection
        .execute_batch(&format!(
            "BEGIN;
             UPDATE vector_store_files SET status = 'in_progress', usage_bytes = 0, text = NULL
                 WHERE attach_id = '{attach_id}';
             DELETE FROM vector_store_chunks WHERE attach_id = '{attach_id}';
             INSERT INTO vector_store_index_garbage (attach_id) VALUES ('{attach_id}');
             COMMIT;"
        ))
        .unwrap();
}

#[test]
fn a_crash_between_index_and_database_commits_is_recovered() {
    let root = TempRoot::new();
    let id;
    let alpha;
    {
        let (store, index) = root.open();
        alpha = file(&store, "a.txt", "consistency matters greatly");
        id = vector_store(&store, &[&alpha]);
        index_all(&index);
        crash_after_index_commit(&store, &alpha);
        // The documents are in the index, but the file is not completed:
        // search never shows them.
        assert!(search(&index, &id, "consistency").is_empty());
    }
    // The next start removes them and indexes the file again, once.
    let (store, index) = root.open();
    assert_eq!(garbage(&store), 0);
    assert_eq!(
        store.get_vector_store_file(&id, &alpha).unwrap().status,
        FileStatus::InProgress
    );
    assert_eq!(index_all(&index), 1);
    assert_eq!(search(&index, &id, "consistency").len(), 1);
    let check = index.check().unwrap();
    assert_eq!((check.chunks, check.rebuilt), (1, false));

    // Without a restart, the retry replaces the documents (no duplicates).
    crash_after_index_commit(&store, &alpha);
    assert_eq!(index_all(&index), 1);
    assert_eq!(search(&index, &id, "consistency").len(), 1);
    let check = index.check().unwrap();
    assert_eq!((check.chunks, check.rebuilt), (1, false));
}

#[test]
fn a_crash_between_a_deletion_and_its_collection_is_recovered() {
    let root = TempRoot::new();
    let id;
    let beta;
    {
        let (store, index) = root.open();
        let alpha = file(&store, "a.txt", "shared doomed words");
        beta = file(&store, "b.txt", "shared surviving words");
        id = vector_store(&store, &[&alpha, &beta]);
        index_all(&index);
        // Deleted in the database; the process dies before collecting.
        store.delete_vector_store_file(&id, &alpha).unwrap();
        assert_eq!(garbage(&store), 1);
        let hits = search(&index, &id, "shared");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, beta);
    }
    let (store, index) = root.open();
    assert_eq!(garbage(&store), 0);
    let check = index.check().unwrap();
    assert_eq!((check.chunks, check.rebuilt), (1, false));
    assert!(search(&index, &id, "doomed").is_empty());
    assert_eq!(search(&index, &id, "surviving")[0].0, beta);
}

/// Add a document the database knows nothing about, with Tantivy directly.
fn plant_ghost_document(root: &TempRoot) {
    let index = tantivy::Index::open_in_dir(root.0.join("store/vector_index")).unwrap();
    index.tokenizers().register(
        "vector_store_text",
        tantivy::tokenizer::TextAnalyzer::from(tantivy::tokenizer::SimpleTokenizer::default()),
    );
    let schema = index.schema();
    let field = |name: &str| schema.get_field(name).unwrap();
    let mut writer: tantivy::IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();
    writer
        .add_document(tantivy::doc!(
            field("attach") => "vsfa_ghost",
            field("ordinal") => 0_u64,
            field("text") => "postings"
        ))
        .unwrap();
    writer.commit().unwrap();
    // Two chunks of the database plus the ghost.
    assert_eq!(index.reader().unwrap().searcher().num_docs(), 3);
}

#[test]
fn an_index_damaged_lost_or_unreadable_is_rebuilt() {
    let root = TempRoot::new();
    let id;
    let alpha;
    {
        let (store, index) = root.open();
        alpha = file(&store, "a.txt", "rebuild the lost postings");
        let beta = file(&store, "b.txt", "unrelated words here");
        id = vector_store(&store, &[&alpha, &beta]);
        index_all(&index);
    }
    let directory = root.0.join("store/vector_index");
    // A document outside the protocol (no outbox entry): rebuilt away.
    plant_ghost_document(&root);
    {
        let (_store, index) = root.open();
        let hits = search(&index, &id, "postings");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, alpha);
        let check = index.check().unwrap();
        assert_eq!((check.chunks, check.rebuilt), (2, false));
    }
    // The whole index lost.
    fs::remove_dir_all(&directory).unwrap();
    {
        let (_store, index) = root.open();
        assert_eq!(search(&index, &id, "postings")[0].0, alpha);
        assert!(!index.check().unwrap().rebuilt);
    }
    // Unreadable metadata.
    fs::write(directory.join("meta.json"), b"{ not json").unwrap();
    let (_store, index) = root.open();
    assert_eq!(search(&index, &id, "postings")[0].0, alpha);
    assert_eq!(index.check().unwrap().chunks, 2);
}

#[test]
fn deletions_reach_the_index_through_the_outbox() {
    let root = TempRoot::new();
    let (store, index) = root.open();
    let alpha = file(&store, "a.txt", "shared term alpha");
    let beta = file(&store, "b.txt", "shared term beta");
    let gamma = file(&store, "c.txt", "shared term gamma");
    let id = vector_store(&store, &[&alpha, &beta, &gamma]);
    let other = vector_store(&store, &[&alpha]);
    index_all(&index);
    assert_eq!(index.check().unwrap().chunks, 4);

    // Removing a file from one store keeps the File and the other store.
    let deleted = store.delete_vector_store_file(&id, &beta).unwrap();
    assert!(deleted.deleted);
    assert!(store.get_file(&beta).is_ok());
    assert!(matches!(
        store.get_vector_store_file(&id, &beta),
        Err(Error::NotFound(_))
    ));
    assert!(matches!(
        store.delete_vector_store_file(&id, &beta),
        Err(Error::NotFound(_))
    ));
    let check = index.check().unwrap();
    assert!(!check.rebuilt);
    assert_eq!(check.chunks, 3);

    // Deleting the File removes it from every store.
    store.delete_file(&alpha).unwrap();
    assert!(matches!(
        store.get_vector_store_file(&id, &alpha),
        Err(Error::NotFound(_))
    ));
    assert_eq!(store.get_vector_store(&other).unwrap().file_counts.total, 0);
    assert!(search(&index, &other, "shared").is_empty());
    let check = index.check().unwrap();
    assert!(!check.rebuilt);
    assert_eq!(check.chunks, 1);
    assert_eq!(search(&index, &id, "shared")[0].0, gamma);

    // Deleting a store removes the rest; it is gone after a restart too.
    assert!(store.delete_vector_store(&id).unwrap().deleted);
    assert!(matches!(
        store.get_vector_store(&id),
        Err(Error::NotFound(_))
    ));
    assert!(matches!(
        index.search(&id, &SearchRequest::new("shared")),
        Err(Error::NotFound(_))
    ));
    let check = index.check().unwrap();
    assert!(!check.rebuilt);
    assert_eq!(check.chunks, 0);
    drop(index);
    let (store, _index) = root.open();
    assert!(matches!(
        store.get_vector_store(&id),
        Err(Error::NotFound(_))
    ));
}

#[test]
fn work_on_a_file_removed_mid_indexing_is_discarded() {
    let root = TempRoot::new();
    let (store, index) = root.open();
    let alpha = file(&store, "a.txt", "racing deletion");
    let id = vector_store(&store, &[&alpha]);
    // Re-attaching replaces the attachment the indexer might hold.
    store
        .create_vector_store_file(
            &id,
            &CreateVectorStoreFile {
                file_id: alpha.clone(),
                chunking_strategy: None,
                attributes: Some(attributes(json!({"v": 2}))),
            },
        )
        .unwrap();
    assert_eq!(store.get_vector_store(&id).unwrap().file_counts.total, 1);
    assert_eq!(index_all(&index), 1);
    let file = store.get_vector_store_file(&id, &alpha).unwrap();
    assert_eq!(file.attributes, attributes(json!({"v": 2})));
    assert_eq!(index.check().unwrap().chunks, 1);
}

#[test]
fn unreadable_files_fail_with_the_documented_codes() {
    let root = TempRoot::new();
    let (store, index) = root.open();
    let binary = file(&store, "a.bin", b"\x00\x01\x02text");
    let latin1 = file(&store, "b.txt", b"caf\xe9");
    let blank = file(&store, "c.txt", " \n\t ");
    let large = file(
        &store,
        "d.txt",
        vec![b'a'; usize::try_from(MAX_INDEXED_FILE_BYTES).unwrap() + 1],
    );
    let id = vector_store(&store, &[&binary, &latin1, &blank, &large]);
    assert_eq!(index_all(&index), 4);
    let code = |file_id: &str| {
        let file = store.get_vector_store_file(&id, file_id).unwrap();
        assert_eq!(file.status, FileStatus::Failed);
        assert_eq!(file.usage_bytes, 0);
        file.last_error.unwrap().code
    };
    assert_eq!(code(&binary), ErrorCode::UnsupportedFile);
    assert_eq!(code(&latin1), ErrorCode::UnsupportedFile);
    assert_eq!(code(&blank), ErrorCode::InvalidFile);
    assert_eq!(code(&large), ErrorCode::InvalidFile);
    let counts = store.get_vector_store(&id).unwrap().file_counts;
    assert_eq!(counts.failed, 4);
    assert_eq!(
        store.get_vector_store(&id).unwrap().status,
        StoreStatus::Completed
    );
    assert!(
        store
            .vector_store_file_content(&id, &binary)
            .unwrap()
            .data
            .is_empty()
    );
}

#[test]
fn chunk_counts_are_bounded_and_follow_the_strategy() {
    let root = TempRoot::new();
    let (store, index) = root.open();
    let small = StaticChunking {
        max_chunk_size_tokens: 100,
        chunk_overlap_tokens: 50,
    };
    let id = vector_store(&store, &[]);
    let attach = |file_id: &str| {
        store
            .create_vector_store_file(
                &id,
                &CreateVectorStoreFile {
                    file_id: file_id.to_owned(),
                    chunking_strategy: Some(ChunkingStrategy::Static { config: small }),
                    attributes: None,
                },
            )
            .unwrap()
    };
    // 250 words, windows of 100 every 50: 0-99, 50-149, 100-199, 150-249.
    let text = words(250);
    let four = file(&store, "four.txt", &text);
    attach(&four);
    // Just past the chunk limit, still under the byte limit.
    let words_needed = 100 + MAX_FILE_CHUNKS * 50 + 1;
    let many = file(&store, "many.txt", "a ".repeat(words_needed));
    attach(&many);
    index_all(&index);
    assert_eq!(index.check().unwrap().chunks, 4);
    assert_eq!(
        store.get_vector_store_file(&id, &four).unwrap().status,
        FileStatus::Completed
    );
    let refused = store.get_vector_store_file(&id, &many).unwrap();
    assert_eq!(refused.status, FileStatus::Failed);
    assert_eq!(refused.last_error.unwrap().code, ErrorCode::InvalidFile);
    let hit = search(&index, &id, "w249");
    assert_eq!(hit.len(), 1);
    assert!(hit[0].2.starts_with("w150 ") && hit[0].2.ends_with("w249"));
    // The whole text is returned once, without overlap.
    assert_eq!(
        store.vector_store_file_content(&id, &four).unwrap().data[0].text,
        text
    );
    for config in [
        StaticChunking {
            max_chunk_size_tokens: 99,
            chunk_overlap_tokens: 0,
        },
        StaticChunking {
            max_chunk_size_tokens: 4097,
            chunk_overlap_tokens: 0,
        },
        StaticChunking {
            max_chunk_size_tokens: 200,
            chunk_overlap_tokens: 101,
        },
    ] {
        let error = store
            .create_vector_store_file(
                &id,
                &CreateVectorStoreFile {
                    file_id: four.clone(),
                    chunking_strategy: Some(ChunkingStrategy::Static { config }),
                    attributes: None,
                },
            )
            .unwrap_err();
        assert!(matches!(error, Error::InvalidArgument(_)), "{config:?}");
    }
}

#[test]
fn file_batches_run_cancel_and_list() {
    let root = TempRoot::new();
    let (store, index) = root.open();
    let id = vector_store(&store, &[]);
    let files: Vec<String> = (0..3)
        .map(|i| file(&store, &format!("{i}.txt"), format!("batch file number{i}")))
        .collect();
    let batch = store
        .create_vector_store_file_batch(
            &id,
            &CreateFileBatch {
                file_ids: Some(files.clone()),
                attributes: Some(attributes(json!({"team": "core"}))),
                ..CreateFileBatch::default()
            },
        )
        .unwrap();
    assert_eq!(batch.status, BatchStatus::InProgress);
    assert_eq!(batch.file_counts.in_progress, 3);
    let cancelled = store
        .cancel_vector_store_file_batch(&id, &batch.id)
        .unwrap();
    assert_eq!(cancelled.status, BatchStatus::Cancelled);
    assert_eq!(cancelled.file_counts.cancelled, 3);
    assert_eq!(index.index_next().unwrap(), 0);
    assert_eq!(
        store
            .cancel_vector_store_file_batch(&id, &batch.id)
            .unwrap(),
        cancelled
    );
    let listed = store
        .list_vector_store_file_batch_files(
            &id,
            &batch.id,
            &ListQuery {
                filter: Some(FileStatus::Cancelled),
                ..ListQuery::default()
            },
        )
        .unwrap();
    assert_eq!(listed.data.len(), 3);
    assert_eq!(
        listed.data[0].attributes,
        attributes(json!({"team": "core"}))
    );

    // Per-file entries; a finished batch cannot be cancelled.
    let second = store
        .create_vector_store_file_batch(
            &id,
            &CreateFileBatch {
                files: Some(vec![CreateVectorStoreFile {
                    file_id: files[0].clone(),
                    chunking_strategy: None,
                    attributes: Some(attributes(json!({"n": 0}))),
                }]),
                ..CreateFileBatch::default()
            },
        )
        .unwrap();
    index_all(&index);
    let second = store.get_vector_store_file_batch(&id, &second.id).unwrap();
    assert_eq!(second.status, BatchStatus::Completed);
    assert_eq!(second.file_counts.completed, 1);
    assert!(matches!(
        store.cancel_vector_store_file_batch(&id, &second.id),
        Err(Error::Conflict(_))
    ));
    // The re-attached file left the first batch.
    assert_eq!(
        store
            .get_vector_store_file_batch(&id, &batch.id)
            .unwrap()
            .file_counts
            .total,
        2
    );
    for request in [
        CreateFileBatch::default(),
        CreateFileBatch {
            file_ids: Some(vec![]),
            ..CreateFileBatch::default()
        },
        CreateFileBatch {
            file_ids: Some(vec![files[0].clone(), files[0].clone()]),
            ..CreateFileBatch::default()
        },
        CreateFileBatch {
            file_ids: Some(vec![files[0].clone()]),
            files: Some(vec![]),
            ..CreateFileBatch::default()
        },
        CreateFileBatch {
            file_ids: Some(vec!["x".to_owned(); 2001]),
            ..CreateFileBatch::default()
        },
    ] {
        assert!(matches!(
            store.create_vector_store_file_batch(&id, &request),
            Err(Error::InvalidArgument(_))
        ));
    }
    // A missing File refuses the whole batch.
    assert!(matches!(
        store.create_vector_store_file_batch(
            &id,
            &CreateFileBatch {
                file_ids: Some(vec![files[1].clone(), "file-missing".to_owned()]),
                ..CreateFileBatch::default()
            }
        ),
        Err(Error::NotFound(_))
    ));
    assert!(matches!(
        store.get_vector_store_file_batch(&id, "vsfb_missing"),
        Err(Error::NotFound(_))
    ));
}

#[test]
fn lists_page_in_both_orders_with_both_cursors() {
    let root = TempRoot::new();
    let (store, index) = root.open();
    let ids: Vec<String> = (0..5).map(|_| vector_store(&store, &[])).collect();
    let list = |query: ListQuery| {
        let page = store.list_vector_stores(&query).unwrap();
        (
            page.data.into_iter().map(|s| s.id).collect::<Vec<_>>(),
            page.has_more,
        )
    };
    let newest: Vec<String> = ids.iter().rev().cloned().collect();
    assert_eq!(list(ListQuery::default()), (newest.clone(), false));
    let first = list(ListQuery {
        limit: Some(2),
        ..ListQuery::default()
    });
    assert_eq!(first, (newest[..2].to_vec(), true));
    let second = list(ListQuery {
        limit: Some(2),
        after: Some(newest[1].clone()),
        ..ListQuery::default()
    });
    assert_eq!(second, (newest[2..4].to_vec(), true));
    let back = list(ListQuery {
        limit: Some(2),
        before: Some(newest[2].clone()),
        ..ListQuery::default()
    });
    assert_eq!(back, (newest[..2].to_vec(), false));
    let asc = list(ListQuery {
        limit: Some(3),
        order: ListOrder::Asc,
        after: Some(ids[0].clone()),
        ..ListQuery::default()
    });
    assert_eq!(asc, (ids[1..4].to_vec(), true));
    let between = list(ListQuery {
        order: ListOrder::Asc,
        after: Some(ids[0].clone()),
        before: Some(ids[3].clone()),
        ..ListQuery::default()
    });
    assert_eq!(between, (ids[1..3].to_vec(), false));
    let page = store
        .list_vector_stores(&ListQuery {
            limit: Some(2),
            ..ListQuery::default()
        })
        .unwrap();
    assert_eq!(page.first_id.as_deref(), Some(newest[0].as_str()));
    assert_eq!(page.last_id.as_deref(), Some(newest[1].as_str()));
    for limit in [0, 101] {
        assert!(matches!(
            store.list_vector_stores(&ListQuery {
                limit: Some(limit),
                ..ListQuery::default()
            }),
            Err(Error::InvalidArgument(_))
        ));
    }
    assert!(matches!(
        store.list_vector_stores(&ListQuery {
            after: Some("vs_missing".to_owned()),
            ..ListQuery::default()
        }),
        Err(Error::NotFound(_))
    ));

    // Files page the same way, with a status filter.
    let target = &ids[0];
    let files: Vec<String> = (0..3)
        .map(|i| file(&store, &format!("{i}.txt"), format!("page {i}")))
        .collect();
    for (position, file_id) in files.iter().enumerate() {
        store
            .create_vector_store_file(
                target,
                &CreateVectorStoreFile {
                    file_id: file_id.clone(),
                    ..CreateVectorStoreFile::default()
                },
            )
            .unwrap();
        // Only the first is indexed.
        if position == 0 {
            assert_eq!(index.index_next().unwrap(), 1);
        }
    }
    let page = store
        .list_vector_store_files(
            target,
            &ListQuery {
                limit: Some(1),
                filter: Some(FileStatus::InProgress),
                ..ListQuery::default()
            },
        )
        .unwrap();
    assert_eq!(page.data[0].id, files[2]);
    assert!(page.has_more);
    let completed = store
        .list_vector_store_files(
            target,
            &ListQuery {
                filter: Some(FileStatus::Completed),
                ..ListQuery::default()
            },
        )
        .unwrap();
    assert_eq!(completed.data.len(), 1);
    assert_eq!(completed.data[0].id, files[0]);
}

#[test]
fn attribute_filters_select_files() {
    let root = TempRoot::new();
    let (store, index) = root.open();
    let id = vector_store(&store, &[]);
    let specs = [
        ("a.txt", json!({"lang": "en", "year": 2020, "draft": false})),
        ("b.txt", json!({"lang": "de", "year": 2023, "draft": true})),
        ("c.txt", json!({"year": 2024.5})),
    ];
    let mut names = Vec::new();
    for (name, attrs) in &specs {
        let file_id = file(&store, name, "common words everywhere");
        store
            .create_vector_store_file(
                &id,
                &CreateVectorStoreFile {
                    file_id: file_id.clone(),
                    chunking_strategy: None,
                    attributes: Some(attributes(attrs.clone())),
                },
            )
            .unwrap();
        names.push(file_id);
    }
    index_all(&index);
    let matching = |filter: Value| {
        let filter: Filter = serde_json::from_value(filter).unwrap();
        let mut request = SearchRequest::new("common");
        request.filters = Some(filter);
        let mut found: Vec<usize> = index
            .search(&id, &request)
            .unwrap()
            .data
            .iter()
            .map(|hit| names.iter().position(|n| *n == hit.file_id).unwrap())
            .collect();
        found.sort_unstable();
        found
    };
    assert_eq!(
        matching(json!({"type":"eq","key":"lang","value":"en"})),
        [0]
    );
    assert_eq!(
        matching(json!({"type":"ne","key":"lang","value":"en"})),
        [1, 2]
    );
    assert_eq!(
        matching(json!({"type":"gt","key":"year","value":2020})),
        [1, 2]
    );
    assert_eq!(
        matching(json!({"type":"lte","key":"year","value":2023})),
        [0, 1]
    );
    assert_eq!(
        matching(json!({"type":"eq","key":"draft","value":true})),
        [1]
    );
    assert_eq!(
        matching(json!({"type":"in","key":"lang","value":["de","fr"]})),
        [1]
    );
    assert_eq!(
        matching(json!({"type":"nin","key":"lang","value":["de"]})),
        [0, 2]
    );
    // A string never equals a number.
    assert!(matching(json!({"type":"eq","key":"year","value":"2020"})).is_empty());
    assert_eq!(
        matching(json!({"type":"and","filters":[
            {"type":"gte","key":"year","value":2020},
            {"type":"or","filters":[
                {"type":"eq","key":"lang","value":"de"},
                {"type":"eq","key":"draft","value":false}
            ]}
        ]})),
        [0, 1]
    );
    for bad in [
        json!({"type":"in","key":"lang","value":"en"}),
        json!({"type":"eq","key":"lang","value":["en"]}),
        json!({"type":"and","filters":[]}),
    ] {
        let mut request = SearchRequest::new("common");
        request.filters = Some(serde_json::from_value(bad).unwrap());
        assert!(matches!(
            index.search(&id, &request),
            Err(Error::InvalidArgument(_))
        ));
    }
    let mut deep = json!({"type":"eq","key":"k","value":1});
    for _ in 0..8 {
        deep = json!({"type":"and","filters":[deep]});
    }
    let mut request = SearchRequest::new("common");
    request.filters = Some(serde_json::from_value(deep).unwrap());
    assert!(matches!(
        index.search(&id, &request),
        Err(Error::InvalidArgument(_))
    ));
    // Updating attributes changes what filters see.
    store
        .update_vector_store_file(&id, &names[2], &attributes(json!({"lang": "en"})))
        .unwrap();
    assert_eq!(
        matching(json!({"type":"eq","key":"lang","value":"en"})),
        [0, 2]
    );
    let too_many: Attributes = (0..17).map(|i| (format!("k{i}"), json!(i))).collect();
    for bad in [
        too_many,
        attributes(json!({"k": null})),
        attributes(json!({"k": [1]})),
        attributes(json!({"k": "x".repeat(513)})),
        attributes(json!({"k".repeat(65): 1})),
    ] {
        assert!(matches!(
            store.update_vector_store_file(&id, &names[0], &bad),
            Err(Error::InvalidArgument(_))
        ));
    }
}

#[test]
fn search_requests_are_bounded_and_honest() {
    let root = TempRoot::new();
    let (store, index) = root.open();
    let alpha = file(&store, "a.txt", "apple banana");
    let beta = file(&store, "b.txt", "banana cherry cherry");
    let id = vector_store(&store, &[&alpha, &beta]);
    index_all(&index);
    let run = |request: &SearchRequest| index.search(&id, request);
    let mut request = SearchRequest::new("x");
    for limit in [0, 51] {
        request.max_num_results = Some(limit);
        assert!(matches!(run(&request), Err(Error::InvalidArgument(_))));
    }
    let mut request = SearchRequest::new("banana");
    request.rewrite_query = true;
    assert!(matches!(run(&request), Err(Error::InvalidArgument(_))));
    let mut request = SearchRequest::new("banana");
    request.ranking_options = Some(RankingOptions {
        ranker: Ranker::Default20241115,
        score_threshold: 0.0,
    });
    assert!(matches!(run(&request), Err(Error::InvalidArgument(_))));
    request.ranking_options = Some(RankingOptions {
        ranker: Ranker::None,
        score_threshold: 1.5,
    });
    assert!(matches!(run(&request), Err(Error::InvalidArgument(_))));
    assert!(matches!(
        run(&SearchRequest::new("q".repeat(4097))),
        Err(Error::InvalidArgument(_))
    ));
    let mut request = SearchRequest::new("");
    request.query = Query::Many(vec!["a".to_owned(); 17]);
    assert!(matches!(run(&request), Err(Error::InvalidArgument(_))));
    request.query = Query::Many(vec![]);
    assert!(matches!(run(&request), Err(Error::InvalidArgument(_))));

    // Several queries: each chunk keeps its best score, best first.
    request.query = Query::Many(vec!["apple".to_owned(), "cherry".to_owned()]);
    let results = run(&request).unwrap();
    assert_eq!(results.search_query, ["apple", "cherry"]);
    assert_eq!(results.data.len(), 2);
    assert!(results.data[0].score >= results.data[1].score);
    assert!(!results.has_more);
    assert!(results.next_page.is_none());
    // The threshold drops weaker hits; max_num_results caps the rest.
    let mut request = SearchRequest::new("banana cherry");
    request.max_num_results = Some(1);
    let top = run(&request).unwrap();
    assert_eq!(top.data.len(), 1);
    assert_eq!(top.data[0].file_id, beta);
    request.max_num_results = None;
    request.ranking_options = Some(RankingOptions {
        ranker: Ranker::Auto,
        score_threshold: 0.999_999,
    });
    assert!(run(&request).unwrap().data.is_empty());
    // An empty or punctuation-only query matches nothing.
    assert!(run(&SearchRequest::new("?!")).unwrap().data.is_empty());
}

#[test]
fn stores_validate_update_and_expire() {
    let root = TempRoot::new();
    let (store, index) = root.open();
    let policy = ExpirationPolicy {
        anchor: ExpirationAnchor::LastActiveAt,
        days: 1,
    };
    let created = store
        .create_vector_store(&CreateVectorStore {
            expires_after: Some(policy),
            metadata: Some([("k".to_owned(), "v".to_owned())].into()),
            ..CreateVectorStore::default()
        })
        .unwrap();
    assert_eq!(created.name, "");
    assert_eq!(created.status, StoreStatus::Completed);
    assert_eq!(
        created.expires_at,
        Some(created.last_active_at.unwrap() + 86_400)
    );
    let updated = store
        .update_vector_store(
            &created.id,
            &serde_json::from_value::<UpdateVectorStore>(
                json!({"name": "renamed", "metadata": null}),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(updated.name, "renamed");
    assert!(updated.metadata.is_empty());
    assert_eq!(updated.expires_after, Some(policy));
    // Time passes beyond the policy.
    raw(&store)
        .execute(
            "UPDATE vector_stores SET last_active_at = last_active_at - 2 * 86400 WHERE id = ?1",
            [&created.id],
        )
        .unwrap();
    let expired = store.get_vector_store(&created.id).unwrap();
    assert_eq!(expired.status, StoreStatus::Expired);
    assert!(matches!(
        index.search(&created.id, &SearchRequest::new("x")),
        Err(Error::Conflict(_))
    ));
    let alpha = file(&store, "a.txt", "late");
    assert!(matches!(
        store.create_vector_store_file(
            &created.id,
            &CreateVectorStoreFile {
                file_id: alpha.clone(),
                ..CreateVectorStoreFile::default()
            }
        ),
        Err(Error::Conflict(_))
    ));
    // Clearing the policy revives it.
    let revived = store
        .update_vector_store(
            &created.id,
            &serde_json::from_value::<UpdateVectorStore>(json!({"expires_after": null})).unwrap(),
        )
        .unwrap();
    assert_eq!(revived.status, StoreStatus::Completed);
    assert!(revived.expires_at.is_none());

    for request in [
        CreateVectorStore {
            expires_after: Some(ExpirationPolicy {
                anchor: ExpirationAnchor::LastActiveAt,
                days: 0,
            }),
            ..CreateVectorStore::default()
        },
        CreateVectorStore {
            expires_after: Some(ExpirationPolicy {
                anchor: ExpirationAnchor::LastActiveAt,
                days: 366,
            }),
            ..CreateVectorStore::default()
        },
        CreateVectorStore {
            metadata: Some((0..17).map(|i| (format!("k{i}"), String::new())).collect()),
            ..CreateVectorStore::default()
        },
        CreateVectorStore {
            file_ids: vec![alpha; 501],
            ..CreateVectorStore::default()
        },
        CreateVectorStore {
            name: Some("n".repeat(1025)),
            ..CreateVectorStore::default()
        },
    ] {
        assert!(matches!(
            store.create_vector_store(&request),
            Err(Error::InvalidArgument(_))
        ));
    }
    assert!(matches!(
        store.create_vector_store(&CreateVectorStore {
            file_ids: vec!["file-missing".to_owned()],
            ..CreateVectorStore::default()
        }),
        Err(Error::NotFound(_))
    ));
    // A refused create stores nothing.
    assert_eq!(
        store
            .list_vector_stores(&ListQuery::default())
            .unwrap()
            .data
            .len(),
        1
    );
}
