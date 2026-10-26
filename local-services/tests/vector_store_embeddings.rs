#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::assert_is_empty
)]
//! Vector store embeddings: vectors written with the index, kept consistent
//! with the database across crashes, restarts and deletions, backfilled
//! for stores indexed without them, and the hybrid ranking options; driven
//! by a deterministic fake model.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use local_services::files::{CreateFile, FilePurpose};
use local_services::vector_stores::{
    ChunkEmbedder, CreateVectorStore, FileStatus, Ranker, Ranking, RankingOptions, SearchRequest,
    VectorIndex, VectorStoreIndexer,
};
use local_services::{Error, Store};

const WIDTH: usize = 16;

/// Bag of words hashed into [`WIDTH`] buckets, with "car" and "automobile"
/// sharing one, so the dense ranking finds what words alone cannot.
struct Fake {
    space: String,
    fail: AtomicBool,
    documents: AtomicUsize,
}

impl Fake {
    fn new(space: &str) -> Arc<Self> {
        Arc::new(Self {
            space: space.to_owned(),
            fail: AtomicBool::new(false),
            documents: AtomicUsize::new(0),
        })
    }

    fn vector(text: &str) -> Vec<f32> {
        let mut vector = [0.0_f32; WIDTH];
        for word in text.split_whitespace() {
            let word = word.to_lowercase();
            let word = if word == "automobile" { "car" } else { &word };
            let bucket = word.bytes().fold(7_usize, |hash, byte| {
                hash.wrapping_mul(31).wrapping_add(usize::from(byte))
            }) % WIDTH;
            vector[bucket] += 1.0;
        }
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        vector.iter().map(|v| v / norm).collect()
    }
}

impl ChunkEmbedder for Fake {
    fn space(&self) -> &str {
        &self.space
    }

    fn dimensions(&self) -> usize {
        WIDTH
    }

    fn embed_documents(
        &self,
        texts: &[&str],
        stop: &dyn Fn() -> bool,
    ) -> local_services::Result<Vec<Vec<f32>>> {
        // A document containing "poison" always fails, as one file a model
        // cannot embed would.
        if self.fail.load(Ordering::SeqCst)
            || stop()
            || texts.iter().any(|text| text.contains("poison"))
        {
            return Err(Error::Io(std::io::Error::other("fake embedding failure")));
        }
        self.documents.fetch_add(texts.len(), Ordering::SeqCst);
        Ok(texts.iter().map(|text| Self::vector(text)).collect())
    }

    fn embed_queries(&self, queries: &[&str]) -> local_services::Result<Vec<Vec<f32>>> {
        Ok(queries.iter().map(|text| Self::vector(text)).collect())
    }
}

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "local-services-vector-embeddings-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn open(&self, embedder: Option<&Arc<Fake>>) -> (Store, VectorIndex) {
        let store = Store::open(self.0.join("store")).unwrap();
        let index = VectorIndex::open(&store).unwrap();
        index.set_embedder(embedder.map(|fake| Arc::clone(fake) as Arc<dyn ChunkEmbedder>));
        (store, index)
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn file(store: &Store, name: &str, content: &str) -> String {
    store
        .create_file(
            &CreateFile {
                filename: name.to_owned(),
                purpose: FilePurpose::Assistants,
                expires_after: None,
            },
            content.as_bytes(),
        )
        .unwrap()
        .id
}

fn vector_store(store: &Store, file_ids: &[&str]) -> String {
    store
        .create_vector_store(&CreateVectorStore {
            file_ids: file_ids.iter().map(|id| (*id).to_owned()).collect(),
            ..CreateVectorStore::default()
        })
        .unwrap()
        .id
}

fn index_all(index: &VectorIndex) {
    while index.index_next().unwrap() > 0 {}
}

fn raw(store: &Store) -> rusqlite::Connection {
    rusqlite::Connection::open(store.root().join("services.sqlite3")).unwrap()
}

fn scalar(store: &Store, sql: &str) -> i64 {
    raw(store).query_row(sql, [], |row| row.get(0)).unwrap()
}

fn vectors(store: &Store) -> i64 {
    scalar(store, "SELECT COUNT(*) FROM vector_store_embeddings")
}

/// Completed files and their `embedding_space`, by file ID.
fn spaces(store: &Store) -> BTreeMap<String, Option<String>> {
    let connection = raw(store);
    let mut statement = connection
        .prepare(
            "SELECT file_id, embedding_space FROM vector_store_files WHERE status = 'completed'",
        )
        .unwrap();
    statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

/// Every chunk of a completed, embedded file has exactly one vector, and
/// no other vector exists.
fn assert_vectors_match_chunks(store: &Store) {
    let embedded_chunks = scalar(
        store,
        "SELECT COUNT(*) FROM vector_store_chunks c JOIN vector_store_files f
           ON f.attach_id = c.attach_id
         WHERE f.status = 'completed' AND f.embedding_space IS NOT NULL",
    );
    let matched = scalar(
        store,
        "SELECT COUNT(*) FROM vector_store_embeddings e JOIN vector_store_chunks c
           ON c.attach_id = e.attach_id AND c.ordinal = e.ordinal",
    );
    assert_eq!(vectors(store), embedded_chunks);
    assert_eq!(matched, embedded_chunks);
}

fn cosine(query: &str, document: &str) -> f64 {
    let (left, right) = (Fake::vector(query), Fake::vector(document));
    f64::from(left.iter().zip(&right).map(|(a, b)| a * b).sum::<f32>())
}

fn search(index: &VectorIndex, id: &str, query: &str, ranker: Ranker) -> Vec<(String, f64)> {
    let mut request = SearchRequest::new(query);
    request.ranking_options = Some(RankingOptions {
        ranker,
        score_threshold: 0.0,
    });
    index
        .search(id, &request)
        .unwrap()
        .data
        .into_iter()
        .map(|hit| (hit.file_id, hit.score))
        .collect()
}

#[test]
fn without_a_model_search_is_plain_bm25() {
    let root = TempRoot::new();
    let (store, index) = root.open(None);
    let alpha = file(&store, "a.txt", "car repair manual");
    let beta = file(&store, "b.txt", "repair of bicycles and repair of boats");
    let id = vector_store(&store, &[&alpha, &beta]);
    index_all(&index);
    assert_eq!(index.embed_next().unwrap(), 0);
    assert_eq!(vectors(&store), 0);
    assert!(spaces(&store).values().all(Option::is_none));
    let auto = search(&index, &id, "repair", Ranker::Auto);
    assert_eq!(auto, search(&index, &id, "repair", Ranker::None));
    let lexical = index
        .search_with(&id, &SearchRequest::new("repair"), Ranking::Lexical)
        .unwrap();
    assert_eq!(
        auto,
        lexical
            .data
            .iter()
            .map(|hit| (hit.file_id.clone(), hit.score))
            .collect::<Vec<_>>()
    );
    // BM25's reported score s / (1 + s) is below 1.
    assert!(auto.iter().all(|(_, score)| *score > 0.0 && *score < 1.0));
    // No meaning without a model.
    assert!(search(&index, &id, "automobile", Ranker::Auto).is_empty());
    for ranking in [Ranking::Dense, Ranking::HYBRID] {
        assert!(matches!(
            index.search_with(&id, &SearchRequest::new("repair"), ranking),
            Err(Error::Conflict(_))
        ));
    }
}

#[test]
fn files_are_embedded_with_their_index_and_ranked_by_meaning() {
    let root = TempRoot::new();
    let fake = Fake::new("fake/1");
    let (store, index) = root.open(Some(&fake));
    let alpha = file(&store, "a.txt", "car repair manual");
    let beta = file(&store, "b.txt", "garden tools and seeds");
    let id = vector_store(&store, &[&alpha, &beta]);
    index_all(&index);
    assert_eq!(fake.documents.load(Ordering::SeqCst), 2);
    assert_vectors_match_chunks(&store);
    assert_eq!(vectors(&store), 2);
    assert!(
        spaces(&store)
            .values()
            .all(|space| space.as_deref() == Some("fake/1"))
    );
    // Nothing left to backfill.
    assert_eq!(index.embed_next().unwrap(), 0);

    // "automobile" shares no word with any chunk: BM25 (ranker none) finds
    // nothing; the embedding ranking (auto) finds the car manual first,
    // scored by its cosine similarity.
    assert!(search(&index, &id, "automobile", Ranker::None).is_empty());
    let dense = search(&index, &id, "automobile", Ranker::Auto);
    assert_eq!(dense[0].0, alpha);
    assert!(
        (dense[0].1 - cosine("automobile", "car repair manual")).abs() < 1e-2,
        "{dense:?}"
    );
    assert!(dense.iter().all(|(_, score)| (0.0..=1.0).contains(score)));
    let exact = search(&index, &id, "car repair manual", Ranker::Auto);
    assert!((exact[0].1 - 1.0).abs() < 1e-3, "{exact:?}");
    // The threshold applies to the reported (cosine) score.
    let mut request = SearchRequest::new("automobile");
    request.ranking_options = Some(RankingOptions {
        ranker: Ranker::Auto,
        score_threshold: (dense[0].1 + 0.01).min(1.0),
    });
    assert!(index.search(&id, &request).unwrap().data.is_empty());
    // The reference fusion: half the best score when first in one ranking
    // only, exactly 1 when first in both.
    let fused = index
        .search_with(&id, &SearchRequest::new("automobile"), Ranking::HYBRID)
        .unwrap();
    assert_eq!(fused.data[0].file_id, alpha);
    assert!((fused.data[0].score - 0.5).abs() < 1e-9);
    let both = index
        .search_with(&id, &SearchRequest::new("car repair"), Ranking::HYBRID)
        .unwrap();
    assert_eq!(both.data[0].file_id, alpha);
    assert!((both.data[0].score - 1.0).abs() < 1e-9);
    // Filters and the store boundary hold for the dense ranking too.
    let other = vector_store(&store, &[&beta]);
    index_all(&index);
    let found = search(&index, &other, "automobile", Ranker::Auto);
    assert!(found.iter().all(|(file, _)| *file == beta));
}

#[test]
fn several_stores_are_searched_as_one_ranking() {
    let root = TempRoot::new();
    let fake = Fake::new("fake/1");
    let (store, index) = root.open(Some(&fake));
    let alpha = file(&store, "a.txt", "car repair manual");
    let beta = file(&store, "b.txt", "garden tools and seeds");
    let gamma = file(&store, "c.txt", "automobile engine oil");
    let first = vector_store(&store, &[&alpha, &beta]);
    let second = vector_store(&store, &[&gamma]);
    index_all(&index);
    let found = |stores: &[&str], ranking: Option<Ranking>| {
        index
            .search_stores(stores, &SearchRequest::new("automobile"), ranking)
            .unwrap()
            .data
            .into_iter()
            .map(|hit| (hit.file_id, hit.score))
            .collect::<Vec<_>>()
    };
    // One dense ranking over both stores, best first, each chunk once even
    // when a store is named twice; scores are those of a single-store search.
    let both = found(&[&first, &second, &first], None);
    let files: Vec<&str> = both.iter().map(|(file, _)| file.as_str()).collect();
    assert_eq!(files.len(), 3, "{both:?}");
    assert!(files.contains(&gamma.as_str()) && files.contains(&alpha.as_str()));
    assert!(both.windows(2).all(|pair| pair[0].1 >= pair[1].1));
    let alone = search(&index, &second, "automobile", Ranker::Auto);
    let gamma_score = both.iter().find(|(file, _)| *file == gamma).unwrap().1;
    assert!((gamma_score - alone[0].1).abs() < 1e-9);
    // An explicit ranking (the tool's hybrid_search weights) applies to all.
    let fused = found(
        &[&first, &second],
        Some(Ranking::Fused {
            k: 60.0,
            text_weight: 1.0,
            embedding_weight: 0.0,
        }),
    );
    // Text weight only: just the chunk holding the word ranks, scored 1.
    assert_eq!(fused[0], (gamma, 1.0));
    assert!(matches!(
        index.search_stores(&[], &SearchRequest::new("x"), None),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        index.search_stores(&[&first, "vs_missing"], &SearchRequest::new("x"), None),
        Err(Error::NotFound(_))
    ));
}

#[test]
fn ranking_options_are_validated_and_refused_honestly() {
    let root = TempRoot::new();
    let fake = Fake::new("fake/1");
    let (store, index) = root.open(Some(&fake));
    let alpha = file(&store, "a.txt", "car repair manual");
    let id = vector_store(&store, &[&alpha]);
    index_all(&index);
    let mut request = SearchRequest::new("car");
    request.ranking_options = Some(RankingOptions {
        ranker: Ranker::Default20241115,
        score_threshold: 0.0,
    });
    assert!(matches!(
        index.search(&id, &request),
        Err(Error::InvalidArgument(_))
    ));
    for threshold in [-0.1, 1.1] {
        request.ranking_options = Some(RankingOptions {
            ranker: Ranker::Auto,
            score_threshold: threshold,
        });
        assert!(matches!(
            index.search(&id, &request),
            Err(Error::InvalidArgument(_))
        ));
    }
    request.ranking_options = None;
    request.rewrite_query = true;
    assert!(matches!(
        index.search(&id, &request),
        Err(Error::InvalidArgument(_))
    ));
    // The pinned VectorStoreSearchRequest has no fusion weights: they are
    // unknown fields, refused rather than ignored.
    for body in [
        r#"{"query":"car","ranking_options":{"hybrid_search":{"embedding_weight":1,"text_weight":1}}}"#,
        r#"{"query":"car","ranking_options":{"ranker":"auto","weights":[1,1]}}"#,
    ] {
        assert!(
            serde_json::from_str::<SearchRequest>(body).is_err(),
            "{body}"
        );
    }
    let accepted: SearchRequest = serde_json::from_str(
        r#"{"query":"car","ranking_options":{"ranker":"none","score_threshold":0.25}}"#,
    )
    .unwrap();
    assert_eq!(accepted.ranking_options.unwrap().ranker, Ranker::None);
}

/// What a crash after the index commit but before the confirming
/// transaction leaves: the file in progress without text, chunks or
/// vectors, listed in the outbox, its documents in the index.
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
             UPDATE vector_store_files SET status = 'in_progress', usage_bytes = 0, text = NULL,
                                           embedding_space = NULL
                 WHERE attach_id = '{attach_id}';
             DELETE FROM vector_store_chunks WHERE attach_id = '{attach_id}';
             DELETE FROM vector_store_embeddings WHERE attach_id = '{attach_id}';
             INSERT INTO vector_store_index_garbage (attach_id) VALUES ('{attach_id}');
             COMMIT;"
        ))
        .unwrap();
}

#[test]
fn crashes_between_database_index_and_vector_writes_are_recovered() {
    let root = TempRoot::new();
    let fake = Fake::new("fake/1");
    let id;
    let alpha;
    let beta;
    {
        let (store, index) = root.open(Some(&fake));
        alpha = file(&store, "a.txt", "car repair manual");
        beta = file(&store, "b.txt", "garden tools");
        id = vector_store(&store, &[&alpha, &beta]);
        index_all(&index);
        crash_after_index_commit(&store, &alpha);
        // Dense ranking returns the nearest chunks whatever their distance,
        // but never one of a file that is not completed.
        let found = search(&index, &id, "automobile", Ranker::Auto);
        assert!(found.iter().all(|(file, _)| *file == beta), "{found:?}");
    }
    // Restart: the outbox is collected and the file indexed and embedded
    // again, once.
    let (store, index) = root.open(Some(&fake));
    assert_eq!(
        scalar(&store, "SELECT COUNT(*) FROM vector_store_index_garbage"),
        0
    );
    index_all(&index);
    assert_vectors_match_chunks(&store);
    assert_eq!(vectors(&store), 2);
    assert_eq!(search(&index, &id, "automobile", Ranker::Auto)[0].0, alpha);
    let check = index.check().unwrap();
    assert_eq!(
        (check.chunks, check.rebuilt, check.vectors_repaired),
        (2, false, 0)
    );
    drop(index);

    // Damage the vectors directly: a stray row of an unknown attachment and
    // a missing row of an embedded file. The next start drops both files'
    // bad state and the backfill embeds the file again.
    raw(&store)
        .execute_batch(
            "INSERT INTO vector_store_embeddings (attach_id, ordinal, vector)
                 VALUES ('vsfa_ghost', 0, x'0000');
             DELETE FROM vector_store_embeddings WHERE attach_id =
                 (SELECT attach_id FROM vector_store_files WHERE filename = 'b.txt');",
        )
        .unwrap();
    let (store, index) = root.open(Some(&fake));
    assert_eq!(
        spaces(&store)[&beta],
        None,
        "the damaged file awaits embedding"
    );
    assert_vectors_match_chunks(&store);
    assert_eq!(index.embed_next().unwrap(), 1);
    assert_eq!(index.embed_next().unwrap(), 0);
    assert_vectors_match_chunks(&store);
    assert_eq!(vectors(&store), 2);
    assert_eq!(index.check().unwrap().vectors_repaired, 0);
}

#[test]
fn deletions_remove_vectors_through_the_outbox() {
    let root = TempRoot::new();
    let fake = Fake::new("fake/1");
    let id;
    {
        let (store, index) = root.open(Some(&fake));
        let alpha = file(&store, "a.txt", "car repair manual");
        let beta = file(&store, "b.txt", "garden tools");
        let gamma = file(&store, "c.txt", "automobile engines");
        id = vector_store(&store, &[&alpha, &beta, &gamma]);
        let other = vector_store(&store, &[&beta]);
        index_all(&index);
        assert_eq!(vectors(&store), 4);
        // A vector store file: rows stay until collected, never found.
        store.delete_vector_store_file(&id, &alpha).unwrap();
        assert_eq!(vectors(&store), 4);
        let found = search(&index, &id, "car", Ranker::Auto);
        assert!(found.iter().all(|(file, _)| *file != alpha), "{found:?}");
        index.collect_garbage().unwrap();
        assert_eq!(vectors(&store), 3);
        // A source File, from every store.
        store.delete_file(&beta).unwrap();
        index.collect_garbage().unwrap();
        assert_eq!(vectors(&store), 1);
        // A whole store; the process dies before collecting.
        store.delete_vector_store(&other).unwrap();
        store.delete_vector_store(&id).unwrap();
        assert_eq!(vectors(&store), 1);
    }
    let (store, index) = root.open(Some(&fake));
    assert_eq!(vectors(&store), 0);
    assert_vectors_match_chunks(&store);
    assert!(matches!(
        index.search(&id, &SearchRequest::new("car")),
        Err(Error::NotFound(_))
    ));
}

/// Undo versions 3 and 2 of the schema, as a database written before
/// embeddings.
fn downgrade_to_version_1(store: &Store) {
    raw(store)
        .execute_batch(
            "BEGIN;
             DROP INDEX vector_stores_project_seq;
             ALTER TABLE vector_stores DROP COLUMN project_id;
             DROP INDEX vector_store_files_embedding;
             ALTER TABLE vector_store_files DROP COLUMN embedding_space;
             DROP TABLE vector_store_embeddings;
             UPDATE schema_versions SET version = 1 WHERE component = 'vector_stores';
             COMMIT;",
        )
        .unwrap();
}

#[test]
fn stores_from_before_embeddings_are_migrated_and_backfilled_in_the_background() {
    let root = TempRoot::new();
    let id;
    let alpha;
    {
        let (store, index) = root.open(None);
        alpha = file(&store, "a.txt", "car repair manual");
        let beta = file(&store, "b.txt", &"garden tools ".repeat(700));
        id = vector_store(&store, &[&alpha, &beta]);
        index_all(&index);
        drop(index);
        downgrade_to_version_1(&store);
    }
    // The next start migrates the schema; files stay completed and
    // searchable by keywords; the indexer embeds them in the background.
    let fake = Fake::new("fake/1");
    let (store, index) = root.open(Some(&fake));
    assert_eq!(
        scalar(
            &store,
            "SELECT version FROM schema_versions WHERE component = 'vector_stores'"
        ),
        3
    );
    assert!(spaces(&store).values().all(Option::is_none));
    assert_eq!(search(&index, &id, "car", Ranker::None)[0].0, alpha);
    let index = Arc::new(index);
    let indexer = VectorStoreIndexer::start(Arc::clone(&index)).unwrap();
    indexer.wait_idle();
    assert!(
        spaces(&store)
            .values()
            .all(|space| space.as_deref() == Some("fake/1"))
    );
    assert_vectors_match_chunks(&store);
    assert!(vectors(&store) >= 3, "the long file has several chunks");
    assert_eq!(search(&index, &id, "automobile", Ranker::Auto)[0].0, alpha);

    // Another model (or prompts) names another space: everything is
    // embedded again, replacing the old vectors.
    let rows = vectors(&store);
    let other = Fake::new("fake/2");
    index.set_embedder(Some(Arc::clone(&other) as Arc<dyn ChunkEmbedder>));
    indexer.notify();
    indexer.wait_idle();
    indexer.close();
    assert!(
        spaces(&store)
            .values()
            .all(|space| space.as_deref() == Some("fake/2"))
    );
    assert_eq!(vectors(&store), rows);
    assert_vectors_match_chunks(&store);
}

#[test]
fn embedding_failures_leave_files_searchable_and_retry_later() {
    let root = TempRoot::new();
    let fake = Fake::new("fake/1");
    let (store, index) = root.open(Some(&fake));
    let alpha = file(&store, "a.txt", "car repair manual");
    let id = vector_store(&store, &[&alpha]);
    fake.fail.store(true, Ordering::SeqCst);
    index_all(&index);
    assert_eq!(
        store.get_vector_store_file(&id, &alpha).unwrap().status,
        FileStatus::Completed
    );
    assert_eq!(vectors(&store), 0);
    // Not embedded: auto falls back to the fusion, which finds it through
    // BM25 (first there, absent from the dense ranking: score 0.5).
    let found = search(&index, &id, "car", Ranker::Auto);
    assert_eq!(found[0].0, alpha);
    assert!((found[0].1 - 0.5).abs() < 1e-9, "{found:?}");
    assert!(index.embed_next().is_err());
    fake.fail.store(false, Ordering::SeqCst);
    assert_eq!(index.embed_next().unwrap(), 1);
    assert_vectors_match_chunks(&store);
    assert_eq!(search(&index, &id, "automobile", Ranker::Auto)[0].0, alpha);
}

#[test]
fn a_file_that_cannot_be_embedded_never_holds_up_the_backfill() {
    let root = TempRoot::new();
    let fake = Fake::new("fake/1");
    let (store, index) = root.open(Some(&fake));
    // Indexed while embedding failed: both wait for the backfill, the
    // file that can never be embedded first.
    let poisoned = file(&store, "a.txt", "poison car manual");
    let healthy = file(&store, "b.txt", "automobile repair notes");
    let id = vector_store(&store, &[&poisoned, &healthy]);
    fake.fail.store(true, Ordering::SeqCst);
    index_all(&index);
    fake.fail.store(false, Ordering::SeqCst);
    assert_eq!(vectors(&store), 0);
    // The healthy file is embedded and committed despite the other one.
    assert_eq!(index.embed_next().unwrap(), 1);
    let spaces = spaces(&store);
    assert_eq!(spaces[&healthy].as_deref(), Some("fake/1"));
    assert_eq!(spaces[&poisoned], None);
    assert_vectors_match_chunks(&store);
    // Only the failing file is left: it is retried, and fails alone.
    assert!(index.embed_next().is_err());
    assert_eq!(
        search(&index, &id, "automobile", Ranker::Auto)[0].0,
        healthy
    );
}
