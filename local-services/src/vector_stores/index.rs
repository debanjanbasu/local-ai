//! The ranked index of vector store chunks: one persistent Tantivy index
//! beside the database, derived from it and reconciled with it.
//!
//! See the parent module docs ("Index and consistency") for the protocol.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};

use half::f16;
use rusqlite::{OptionalExtension as _, TransactionBehavior, params};
use tantivy::collector::{Count, TopDocs};
use tantivy::directory::MmapDirectory;
use tantivy::directory::error::LockError;
use tantivy::indexer::IndexWriterOptions;
use tantivy::query::{BooleanQuery, ConstScoreQuery, Occur, TermQuery, TermSetQuery};
use tantivy::schema::{
    Field, IndexRecordOption, STORED, STRING, Schema, TextFieldIndexing, TextOptions, Value as _,
};
use tantivy::tokenizer::{
    AsciiFoldingFilter, Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer,
    TextAnalyzer, TokenStream as _,
};
use tantivy::{
    DocAddress, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, TantivyError, Term,
    doc,
};

use super::{
    ChunkEmbedder, ContentPart, DEFAULT_NUM_RESULTS, IndexCheck, MAX_NUM_RESULTS, MAX_QUERIES,
    MAX_QUERY_CHARS, Query, Ranker, SearchRequest, SearchResult, SearchResults, check_chars,
    corrupt, count, invalid, next_pending, searchable_files, signed, touch, unsigned,
    validate_filter, writable_store,
};
use crate::{Error, Result, Store};

/// The index directory inside the store directory.
const DIRECTORY: &str = "vector_index";
/// The lock file beside it that the owning process holds (see
/// [`VectorIndex`]); taken before the index is touched, so a damaged index is
/// only ever removed by its owner.
const OWNER_LOCK: &str = "vector_index.lock";
/// The analyzer name registered on every open (Tantivy does not persist
/// analyzers, only their names).
const ANALYZER: &str = "vector_store_text";
/// Memory arena of the single indexing thread: Tantivy's documented minimum
/// (15 MB). Larger files flush more segments instead of growing it.
const WRITER_MEMORY_BYTES: usize = 15_000_000;
/// Most files indexed per index commit.
pub const INDEX_BATCH_FILES: usize = 64;
/// Text bytes after which a batch stops taking files (one file may exceed it).
pub const INDEX_BATCH_BYTES: usize = 8 * 1024 * 1024;
/// Searches record activity (`last_active_at`) at most this often per store.
pub const ACTIVITY_GRANULARITY_SECONDS: u64 = 60;
/// Garbage entries removed per index commit.
const GARBAGE_BATCH: usize = 1_000;
/// With an embedder, text bytes after which an indexing or backfill batch
/// stops taking files (one file may exceed it): about 250,000 tokens, under
/// a minute of embedding on an idle GPU.
pub const EMBED_BATCH_BYTES: usize = 1024 * 1024;
/// Chunks handed to the embedder per call; their vectors are stored as F16
/// at once, so a file's vectors cost 2 bytes per dimension in memory until
/// its confirming transaction (at most [`super::MAX_FILE_CHUNKS`] × 1.5 KB
/// at 768 dimensions, 77 MB).
const EMBED_GROUP_CHUNKS: usize = 64;
/// Reciprocal rank fusion constant `k` (Cormack et al. 2009's 60).
pub const RRF_K: f64 = 60.0;
/// Candidates taken from each ranking (per query) before fusion.
pub const FUSION_DEPTH: usize = 100;

/// The persistent ranked index of every store's chunks, and the native
/// search API (`file_search` calls [`Self::search`]).
///
/// One process at a time owns it: opening takes Tantivy's writer lock on
/// the directory, and a second opener (another server on the same data
/// directory) gets [`Error::Conflict`]. All index writes (indexing a file,
/// collecting garbage, checking and rebuilding) run under one mutex, one at
/// a time; searches never wait for them.
pub struct VectorIndex {
    store: Store,
    /// Held for the index's lifetime: the process-wide ownership lock.
    _owner: fs::File,
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    fields: Fields,
    /// The embedding model, when installed (see [`Self::set_embedder`]).
    embedder: RwLock<Option<Arc<dyn ChunkEmbedder>>>,
}

/// How [`VectorIndex::search_with`] ranks chunks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ranking {
    /// BM25 only, scores `s / (1 + s)` (the ranker `none`, and `auto`
    /// without an embedding model).
    Lexical,
    /// Cosine similarity of the query and chunk embeddings only, scores
    /// `max(cos, 0)`.
    Dense,
    /// Weighted reciprocal rank fusion of the BM25 and cosine rankings
    /// ([`FUSION_DEPTH`] candidates each), scores normalized to `(0, 1]`.
    Fused {
        k: f64,
        text_weight: f64,
        embedding_weight: f64,
    },
}

impl Ranking {
    /// What `auto` means when an embedding model is installed (chosen by
    /// measurement; see `docs/EMBEDDINGS.md`).
    pub const HYBRID: Self = Self::Fused {
        k: RRF_K,
        text_weight: 1.0,
        embedding_weight: 1.0,
    };
}

impl std::fmt::Debug for VectorIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VectorIndex")
            .field("root", &self.store.root())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
struct Fields {
    /// The attachment ID: untokenized, for deletion and the store filter.
    attach: Field,
    /// The chunk's position in its file.
    ordinal: Field,
    /// The chunk text, analyzed; frequencies but no positions (BM25 only),
    /// not stored (the database holds the text).
    text: Field,
}

fn schema() -> (Schema, Fields) {
    let mut builder = Schema::builder();
    let attach = builder.add_text_field("attach", STRING | STORED);
    let ordinal = builder.add_u64_field("ordinal", STORED);
    let text = builder.add_text_field(
        "text",
        TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(ANALYZER)
                .set_index_option(IndexRecordOption::WithFreqs),
        ),
    );
    (
        builder.build(),
        Fields {
            attach,
            ordinal,
            text,
        },
    )
}

/// Unicode words, at most 40 bytes, lowercased, folded to ASCII where a
/// folding exists, then reduced by the English (Porter 2) stemmer.
fn analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(40))
        .filter(LowerCaser)
        .filter(AsciiFoldingFilter)
        .filter(Stemmer::new(Language::English))
        .build()
}

impl VectorIndex {
    /// Open (or create) the index of `store`, take its writer lock, and
    /// reconcile it with the database ([`Self::check`]): queued garbage is
    /// removed and an index that still disagrees, cannot be opened, or has
    /// another schema is rebuilt from the stored text before this returns.
    pub fn open(store: &Store) -> Result<Self> {
        let directory = store.root().join(DIRECTORY);
        let owner = lock_owner(&store.root().join(OWNER_LOCK))?;
        let (schema, fields) = schema();
        let index = match open_index(&directory, &schema) {
            Ok(index) => index,
            Err(error) => {
                // Derived data only: start over and rebuild from the database.
                eprintln!(
                    "warning: the vector store index at {} cannot be opened ({error}); \
                     rebuilding it",
                    directory.display()
                );
                remove_index_directory(&directory)?;
                open_index(&directory, &schema)?
            }
        };
        index.tokenizers().register(ANALYZER, analyzer());
        let options = IndexWriterOptions::builder()
            .memory_budget_per_thread(WRITER_MEMORY_BYTES)
            .num_worker_threads(1)
            .num_merge_threads(1)
            .build();
        let writer = index
            .writer_with_options(options)
            .map_err(|error| match error {
                TantivyError::LockFailure(LockError::LockBusy, _) => Error::Conflict(format!(
                    "the vector store index at {} is in use by another process",
                    directory.display()
                )),
                other => index_error(other),
            })?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(index_error)?;
        let opened = Self {
            store: store.clone(),
            _owner: owner,
            index,
            reader,
            writer: Mutex::new(writer),
            fields,
            embedder: RwLock::new(None),
        };
        let check = opened.check()?;
        if check.rebuilt {
            eprintln!(
                "warning: the vector store index disagreed with the database and was rebuilt \
                 ({} chunks)",
                check.chunks
            );
        }
        if check.vectors_repaired > 0 {
            eprintln!(
                "warning: {} vector store files or vector rows disagreed with their chunks; their \
                 vectors were dropped and are embedded again",
                check.vectors_repaired
            );
        }
        Ok(opened)
    }

    /// Install (or remove) the embedding model. From then on files are
    /// embedded as they are indexed, completed files without vectors in its
    /// space are embedded by [`Self::embed_next`] (the indexer's backfill;
    /// notify it after this call), and `auto` searches are hybrid.
    pub fn set_embedder(&self, embedder: Option<Arc<dyn ChunkEmbedder>>) {
        *self
            .embedder
            .write()
            .unwrap_or_else(PoisonError::into_inner) = embedder;
    }

    fn embedder(&self) -> Option<Arc<dyn ChunkEmbedder>> {
        self.embedder
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The store this index belongs to.
    #[must_use]
    pub const fn store(&self) -> &Store {
        &self.store
    }

    /// Rank the completed files' chunks of store `store_id` against the
    /// request (see the parent module docs) and mark the store active.
    ///
    /// This is the native search a `file_search` tool calls.
    pub fn search(&self, store_id: &str, request: &SearchRequest) -> Result<SearchResults> {
        let plan = SearchPlan::new(request)?;
        let ranking = match plan.ranker {
            Ranker::Auto if self.embedder().is_some() => Ranking::HYBRID,
            _ => Ranking::Lexical,
        };
        self.search_planned(store_id, request, &plan, ranking)
    }

    /// [`Self::search`] with an explicit ranking, whatever the request's
    /// `ranker` (validated all the same). For measuring the rankings
    /// against each other through the one search path.
    ///
    /// # Errors
    ///
    /// As [`Self::search`]; [`Error::Conflict`] for [`Ranking::Dense`] or
    /// [`Ranking::Fused`] without an embedding model.
    #[doc(hidden)]
    pub fn search_with(
        &self,
        store_id: &str,
        request: &SearchRequest,
        ranking: Ranking,
    ) -> Result<SearchResults> {
        let plan = SearchPlan::new(request)?;
        self.search_planned(store_id, request, &plan, ranking)
    }

    fn search_planned(
        &self,
        store_id: &str,
        request: &SearchRequest,
        plan: &SearchPlan,
        ranking: Ranking,
    ) -> Result<SearchResults> {
        // Queries are embedded first, outside the database snapshot.
        let dense = match ranking {
            Ranking::Lexical => None,
            Ranking::Dense | Ranking::Fused { .. } => {
                let embedder = self.embedder().ok_or_else(|| {
                    Error::Conflict("no embedding model is installed on this server".to_owned())
                })?;
                let queries: Vec<&str> = plan.queries.iter().map(String::as_str).collect();
                let vectors = embedder.embed_queries(&queries)?;
                check_vectors(&vectors, queries.len(), embedder.dimensions())?;
                Some((embedder.space().to_owned(), vectors))
            }
        };
        let mut connection = self.store.connection()?;
        // One database snapshot for the store, its searchable files and
        // their text.
        let snapshot = connection.transaction()?;
        let target = writable_store(&snapshot, store_id)?;
        let now = crate::now()?;
        if target
            .last_active_at
            .is_none_or(|at| now.saturating_sub(at) >= ACTIVITY_GRANULARITY_SECONDS)
        {
            // A write (and its fsync) at most once per granularity period.
            let writer = self.store.connection()?;
            touch(&writer, store_id)?;
        }
        let files = searchable_files(&snapshot, store_id, request.filters.as_ref())?;
        let mut hits = match (ranking, &dense) {
            (Ranking::Dense, Some((space, vectors))) => {
                let lists = rank_dense(&snapshot, store_id, &files, space, vectors, plan.limit)?;
                best_per_chunk(
                    lists
                        .into_iter()
                        .flatten()
                        .map(|(key, cosine)| (key, f64::from(cosine).max(0.0))),
                )
            }
            (
                Ranking::Fused {
                    k,
                    text_weight,
                    embedding_weight,
                },
                Some((space, vectors)),
            ) => {
                let dense_lists =
                    rank_dense(&snapshot, store_id, &files, space, vectors, FUSION_DEPTH)?;
                let mut fused = Vec::new();
                for (query, dense_list) in plan.queries.iter().zip(dense_lists) {
                    let lexical = self.lexical_hits(query, &files, FUSION_DEPTH)?;
                    fused.extend(fuse(
                        &lexical,
                        &dense_list,
                        k,
                        text_weight,
                        embedding_weight,
                    ));
                }
                best_per_chunk(fused)
            }
            _ => self.rank_lexical(plan, &files)?,
        };
        hits.retain(|(_, score)| *score >= plan.threshold);
        hits.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        hits.truncate(plan.limit);
        let mut data = Vec::with_capacity(hits.len());
        for ((attach_id, ordinal), score) in hits {
            let row: Option<(String, String, String, Vec<u8>)> = snapshot
                .query_row(
                    "SELECT f.file_id, f.filename, f.attributes,
                            substr(f.text, c.start + 1, c.end - c.start)
                     FROM vector_store_chunks c JOIN vector_store_files f
                       ON f.attach_id = c.attach_id
                     WHERE c.attach_id = ?1 AND c.ordinal = ?2 AND f.status = 'completed'",
                    params![attach_id, signed(ordinal)?],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .optional()?;
            // Every admitted document has its chunk row in this snapshot; a
            // missing one would be skipped, never shown.
            let Some((file_id, filename, attributes, text)) = row else {
                continue;
            };
            data.push(SearchResult {
                file_id,
                filename,
                score,
                attributes: serde_json::from_str(&attributes)?,
                content: vec![ContentPart {
                    kind: "text".to_owned(),
                    text: String::from_utf8(text).map_err(|_| corrupt("chunk text"))?,
                }],
            });
        }
        Ok(SearchResults {
            search_query: plan.queries.clone(),
            data,
            has_more: false,
            next_page: None,
        })
    }

    /// The BM25 hits among the chunks of the admitted attachments `files`:
    /// at most `plan.limit` per query, each chunk once with its best
    /// reported score, unordered.
    ///
    /// The rankings (this one, the dense one and their fusion) are the only
    /// step of [`Self::search`] that differs between them: validation, the
    /// store and filter snapshot, the threshold, ordering and reading chunk
    /// text work on chunk keys (attachment, ordinal) and scores, whatever
    /// produced them.
    fn rank_lexical(&self, plan: &SearchPlan, files: &[String]) -> Result<Vec<(ChunkKey, f64)>> {
        let mut hits = Vec::new();
        for query in &plan.queries {
            let top = self.lexical_hits(query, files, plan.limit)?;
            hits.extend(top.into_iter().map(|(key, score)| (key, normalize(score))));
        }
        Ok(best_per_chunk(hits))
    }

    /// The best `limit` BM25 hits of one query among the chunks of `files`,
    /// best first, with raw BM25 scores.
    fn lexical_hits(
        &self,
        query: &str,
        files: &[String],
        limit: usize,
    ) -> Result<Vec<(ChunkKey, f32)>> {
        if files.is_empty() {
            return Ok(Vec::new());
        }
        let words = self.words(query)?;
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let searcher = self.reader.searcher();
        let allowed: Box<dyn tantivy::query::Query> = Box::new(ConstScoreQuery::new(
            Box::new(TermSetQuery::new(
                files
                    .iter()
                    .map(|file| Term::from_field_text(self.fields.attach, file)),
            )),
            0.0,
        ));
        let any_word = BooleanQuery::new(
            words
                .iter()
                .map(|word| {
                    let term = Term::from_field_text(self.fields.text, word);
                    let query: Box<dyn tantivy::query::Query> =
                        Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs));
                    (Occur::Should, query)
                })
                .collect(),
        );
        let query = BooleanQuery::new(vec![
            (Occur::Must, allowed),
            (Occur::Must, Box::new(any_word)),
        ]);
        let top = TopDocs::with_limit(limit).order_by_score();
        searcher
            .search(&query, &top)
            .map_err(index_error)?
            .into_iter()
            .map(|(score, address)| Ok((self.chunk_key(&searcher, address)?, score)))
            .collect()
    }

    /// Index the oldest files still `in_progress` (at most
    /// [`INDEX_BATCH_FILES`], stopping once their text reaches
    /// [`INDEX_BATCH_BYTES`], or [`EMBED_BATCH_BYTES`] with an embedding
    /// model); how many were processed, 0 when none was waiting.
    ///
    /// Order: parse the files and, with an embedding model, embed their
    /// chunks (no lock held: this is the slow part); then, under the writer
    /// lock, keep the files still `in_progress`, record them in the garbage
    /// outbox, write and commit their index documents, then, in one
    /// database transaction, mark each `completed`, insert its chunk rows
    /// and vectors and remove it from the outbox. A crash between the steps
    /// leaves documents the outbox removes; a file deleted, cancelled or
    /// re-attached meanwhile keeps its outbox entry.
    pub fn index_next(&self) -> Result<usize> {
        self.index_batch(&|| false)
    }

    /// [`Self::index_next`], abandoning the batch (an error, files left
    /// `in_progress`) when `stop()` turns true while embedding.
    // The writer lock spans the whole write protocol: the outbox entries,
    // the index commit and the confirming transaction must not interleave
    // with another indexing run or a collection.
    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    pub(crate) fn index_batch(&self, stop: &dyn Fn() -> bool) -> Result<usize> {
        let embedder = self.embedder();
        let budget = if embedder.is_some() {
            EMBED_BATCH_BYTES
        } else {
            INDEX_BATCH_BYTES
        };
        let pending = next_pending(&self.store.connection()?, INDEX_BATCH_FILES)?;
        let mut parsed = Vec::new();
        let mut failed = Vec::new();
        let mut bytes = 0;
        for file in pending {
            if bytes >= budget {
                break;
            }
            match self.store.parse_source(&file) {
                Ok((text, ranges)) => {
                    bytes += text.len();
                    parsed.push((file, text, ranges, None));
                }
                Err(error) => failed.push((file, error)),
            }
        }
        let processed = parsed.len() + failed.len();
        if let Some(embedder) = &embedder {
            for (file, text, ranges, vectors) in &mut parsed {
                match embed_chunks(embedder.as_ref(), text, ranges, stop) {
                    Ok(encoded) => *vectors = Some((embedder.space().to_owned(), encoded)),
                    Err(error) if stop() => return Err(error),
                    // Indexed for BM25 now; the backfill retries the vectors.
                    Err(error) => eprintln!(
                        "warning: embedding vector store file {} failed ({error}); it is \
                         searchable by keywords until its vectors are computed",
                        file.file_id
                    ),
                }
            }
        }
        let mut connection = self.store.connection()?;
        if !failed.is_empty() {
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            for (file, (code, message)) in &failed {
                transaction.execute(
                    "UPDATE vector_store_files SET status = 'failed', error_code = ?2,
                                                   error_message = ?3
                     WHERE attach_id = ?1 AND status = 'in_progress'",
                    params![file.attach_id, code.as_str(), message],
                )?;
            }
            transaction.commit()?;
        }
        if parsed.is_empty() {
            return Ok(processed);
        }
        let mut writer = self.lock();
        // Files only leave `in_progress` for `completed` under this lock, so
        // a file still in progress now cannot be completed by anyone else
        // before the confirming transaction below.
        {
            let mut still = connection.prepare(
                "SELECT 1 FROM vector_store_files WHERE attach_id = ?1 AND status = 'in_progress'",
            )?;
            let mut kept = Vec::with_capacity(parsed.len());
            for entry in parsed {
                if still.exists([&entry.0.attach_id])? {
                    kept.push(entry);
                }
            }
            parsed = kept;
        }
        if parsed.is_empty() {
            return Ok(processed);
        }
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for (file, _, _, _) in &parsed {
            transaction.execute(
                "INSERT OR IGNORE INTO vector_store_index_garbage (attach_id) VALUES (?1)",
                [&file.attach_id],
            )?;
        }
        transaction.commit()?;
        for (file, text, ranges, _) in &parsed {
            writer.delete_term(Term::from_field_text(self.fields.attach, &file.attach_id));
            for (ordinal, &(start, end)) in ranges.iter().enumerate() {
                let part = text
                    .get(start..end)
                    .ok_or_else(|| corrupt("chunk boundary"))?;
                writer
                    .add_document(doc!(
                        self.fields.attach => file.attach_id.as_str(),
                        self.fields.ordinal => count(ordinal),
                        self.fields.text => part
                    ))
                    .map_err(index_error)?;
            }
        }
        self.commit(&mut writer)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        {
            let mut complete = transaction.prepare(
                "UPDATE vector_store_files SET status = 'completed', usage_bytes = ?2, text = ?3,
                                               embedding_space = ?4
                 WHERE attach_id = ?1 AND status = 'in_progress'",
            )?;
            let mut insert = transaction.prepare(
                "INSERT INTO vector_store_chunks (attach_id, ordinal, start, end)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;
            let mut confirm = transaction
                .prepare("DELETE FROM vector_store_index_garbage WHERE attach_id = ?1")?;
            for (file, text, ranges, vectors) in &parsed {
                let changed = complete.execute(params![
                    file.attach_id,
                    signed(count(text.len()))?,
                    text.as_bytes(),
                    vectors.as_ref().map(|(space, _)| space.as_str()),
                ])?;
                if changed == 0 {
                    continue;
                }
                for (ordinal, &(start, end)) in ranges.iter().enumerate() {
                    insert.execute(params![
                        file.attach_id,
                        signed(count(ordinal))?,
                        signed(count(start))?,
                        signed(count(end))?
                    ])?;
                }
                if let Some((_, encoded)) = vectors {
                    store_vectors(&transaction, &file.attach_id, encoded)?;
                }
                confirm.execute([&file.attach_id])?;
            }
        }
        transaction.commit()?;
        Ok(processed)
    }

    /// Embed the oldest completed files whose vectors are missing or in
    /// another space than the installed embedding model's (at most
    /// [`INDEX_BATCH_FILES`], stopping once their text reaches
    /// [`EMBED_BATCH_BYTES`]); how many were embedded, 0 when none was
    /// waiting or no model is installed.
    ///
    /// This is the backfill of stores indexed before embeddings (or while
    /// the model was absent). Each file's vectors replace its old ones and
    /// its `embedding_space` is set in one database transaction, only if
    /// it is still the same completed attachment; nothing else changes, so
    /// a crash at any point leaves the file as it was (embedded again on
    /// the next run).
    pub fn embed_next(&self) -> Result<usize> {
        self.embed_batch(&|| false)
    }

    /// [`Self::embed_next`], abandoning the batch (an error, nothing
    /// written) when `stop()` turns true.
    pub(crate) fn embed_batch(&self, stop: &dyn Fn() -> bool) -> Result<usize> {
        let Some(embedder) = self.embedder() else {
            return Ok(0);
        };
        let space = embedder.space();
        let mut connection = self.store.connection()?;
        let candidates: Vec<(String, i64)> = {
            let mut statement = connection.prepare(
                "SELECT attach_id, length(text) FROM vector_store_files
                 WHERE status = 'completed' AND (embedding_space IS NULL OR embedding_space != ?1)
                 ORDER BY seq LIMIT ?2",
            )?;
            statement
                .query_map(params![space, signed(count(INDEX_BATCH_FILES))?], |row| {
                    Ok((row.get(0)?, row.get::<_, Option<i64>>(1)?.unwrap_or(0)))
                })?
                .collect::<rusqlite::Result<_>>()?
        };
        let mut done = Vec::new();
        let mut bytes = 0_usize;
        for (attach_id, length) in candidates {
            if bytes >= EMBED_BATCH_BYTES {
                break;
            }
            bytes += usize::try_from(length).unwrap_or(0);
            let snapshot = connection.transaction()?;
            let text: Option<Vec<u8>> = snapshot
                .query_row(
                    "SELECT text FROM vector_store_files WHERE attach_id = ?1",
                    [&attach_id],
                    |row| row.get(0),
                )
                .optional()?
                .flatten();
            let ranges = chunk_ranges(&snapshot, &attach_id)?;
            drop(snapshot);
            let Some(text) = text else { continue };
            let text = String::from_utf8(text).map_err(|_| corrupt("vector store file text"))?;
            let encoded = embed_chunks(embedder.as_ref(), &text, &ranges, stop)?;
            done.push((attach_id, ranges.len(), encoded));
        }
        let processed = done.len();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for (attach_id, chunks, encoded) in &done {
            // Still the same completed attachment, with the chunks embedded.
            let current: Option<i64> = transaction
                .query_row(
                    "SELECT (SELECT COUNT(*) FROM vector_store_chunks c
                             WHERE c.attach_id = f.attach_id)
                     FROM vector_store_files f
                     WHERE f.attach_id = ?1 AND f.status = 'completed'
                       AND (f.embedding_space IS NULL OR f.embedding_space != ?2)",
                    params![attach_id, space],
                    |row| row.get(0),
                )
                .optional()?;
            if current != Some(signed(count(*chunks))?) {
                continue;
            }
            transaction.execute(
                "DELETE FROM vector_store_embeddings WHERE attach_id = ?1",
                [attach_id],
            )?;
            store_vectors(&transaction, attach_id, encoded)?;
            transaction.execute(
                "UPDATE vector_store_files SET embedding_space = ?2 WHERE attach_id = ?1",
                params![attach_id, space],
            )?;
        }
        transaction.commit()?;
        Ok(processed)
    }

    /// Remove the index documents of every attachment in the garbage outbox
    /// (deleted files, abandoned indexing attempts); how many attachments
    /// were collected.
    pub fn collect_garbage(&self) -> Result<u64> {
        let mut writer = self.lock();
        self.collect(&mut writer)
    }

    /// Reconcile the index with the database: collect garbage, then compare
    /// the live index documents with the chunk rows of completed files, and
    /// rebuild the whole index from the stored text if they differ.
    ///
    /// Index writes happen only under the writer lock held here, so the
    /// comparison is exact: after collection, documents outside the outbox
    /// belong to completed files only.
    // The writer lock keeps the index fixed while it is compared.
    #[allow(clippy::significant_drop_tightening)]
    pub fn check(&self) -> Result<IndexCheck> {
        let mut writer = self.lock();
        let collected = self.collect(&mut writer)?;
        let mut connection = self.store.connection()?;
        let snapshot = connection.transaction()?;
        let chunks: i64 =
            snapshot.query_row("SELECT COUNT(*) FROM vector_store_chunks", [], |row| {
                row.get(0)
            })?;
        let chunks = unsigned(chunks)?;
        let garbage = garbage(&snapshot, usize::MAX)?;
        let searcher = self.reader.searcher();
        let mut unconfirmed = 0;
        for attach_id in &garbage {
            let term = Term::from_field_text(self.fields.attach, attach_id);
            let query = TermQuery::new(term, IndexRecordOption::Basic);
            unconfirmed += count(searcher.search(&query, &Count).map_err(index_error)?);
        }
        let consistent = searcher.num_docs().checked_sub(unconfirmed) == Some(chunks);
        if !consistent {
            self.rebuild(&mut writer, &snapshot)?;
        }
        drop(snapshot);
        let vectors_repaired = repair_vectors(&mut connection)?;
        Ok(IndexCheck {
            chunks,
            collected,
            rebuilt: !consistent,
            vectors_repaired,
        })
    }

    fn lock(&self) -> MutexGuard<'_, IndexWriter> {
        self.writer.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Commit and make the commit visible to searches.
    fn commit(&self, writer: &mut IndexWriter) -> Result<()> {
        writer.commit().map_err(index_error)?;
        self.reader.reload().map_err(index_error)
    }

    fn collect(&self, writer: &mut IndexWriter) -> Result<u64> {
        let mut collected = 0;
        loop {
            let batch = garbage(&self.store.connection()?, GARBAGE_BATCH)?;
            if batch.is_empty() {
                return Ok(collected);
            }
            for attach_id in &batch {
                writer.delete_term(Term::from_field_text(self.fields.attach, attach_id));
            }
            self.commit(writer)?;
            let mut connection = self.store.connection()?;
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            for attach_id in &batch {
                // Vectors go with the outbox entry, atomically: an outbox
                // entry covers every piece of an attachment's derived data.
                transaction.execute(
                    "DELETE FROM vector_store_embeddings WHERE attach_id = ?1",
                    [attach_id],
                )?;
                transaction.execute(
                    "DELETE FROM vector_store_index_garbage WHERE attach_id = ?1",
                    [attach_id],
                )?;
            }
            transaction.commit()?;
            collected += count(batch.len());
        }
    }

    /// Replace every document by the chunks of the completed files in
    /// `snapshot`.
    fn rebuild(&self, writer: &mut IndexWriter, snapshot: &rusqlite::Connection) -> Result<()> {
        writer.delete_all_documents().map_err(index_error)?;
        let mut files = snapshot.prepare(
            "SELECT attach_id, text FROM vector_store_files
             WHERE status = 'completed' ORDER BY seq",
        )?;
        let mut chunks = snapshot.prepare(
            "SELECT ordinal, start, end FROM vector_store_chunks
             WHERE attach_id = ?1 ORDER BY ordinal",
        )?;
        let mut rows = files.query([])?;
        while let Some(row) = rows.next()? {
            let attach_id: String = row.get(0)?;
            let bytes: Vec<u8> = row.get::<_, Option<Vec<u8>>>(1)?.unwrap_or_default();
            let text = String::from_utf8(bytes).map_err(|_| corrupt("vector store file text"))?;
            let ranges = chunks
                .query_map([&attach_id], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            for (ordinal, start, end) in ranges {
                let range = usize::try_from(start).map_err(|_| corrupt("chunk boundary"))?
                    ..usize::try_from(end).map_err(|_| corrupt("chunk boundary"))?;
                let part = text.get(range).ok_or_else(|| corrupt("chunk boundary"))?;
                writer
                    .add_document(doc!(
                        self.fields.attach => attach_id.as_str(),
                        self.fields.ordinal => unsigned(ordinal)?,
                        self.fields.text => part,
                    ))
                    .map_err(index_error)?;
            }
        }
        self.commit(writer)
    }

    /// The distinct analyzed words of `text`.
    fn words(&self, text: &str) -> Result<BTreeSet<String>> {
        let mut analyzer = self
            .index
            .tokenizer_for_field(self.fields.text)
            .map_err(index_error)?;
        let mut stream = analyzer.token_stream(text);
        let mut words = BTreeSet::new();
        while stream.advance() {
            words.insert(stream.token().text.clone());
        }
        Ok(words)
    }

    fn chunk_key(&self, searcher: &tantivy::Searcher, address: DocAddress) -> Result<ChunkKey> {
        let document: TantivyDocument = searcher.doc(address).map_err(index_error)?;
        let attach = document
            .get_first(self.fields.attach)
            .and_then(|value| value.as_str())
            .ok_or_else(|| corrupt("index document"))?;
        let ordinal = document
            .get_first(self.fields.ordinal)
            .and_then(|value| value.as_u64())
            .ok_or_else(|| corrupt("index document"))?;
        Ok((attach.to_owned(), ordinal))
    }
}

/// A chunk's identity in the index and the database: its attachment and
/// its ordinal in the attachment's file.
type ChunkKey = (String, u64);

/// A validated search request.
struct SearchPlan {
    queries: Vec<String>,
    limit: usize,
    threshold: f64,
    ranker: Ranker,
}

impl SearchPlan {
    fn new(request: &SearchRequest) -> Result<Self> {
        let queries = match &request.query {
            Query::One(query) => vec![query.clone()],
            Query::Many(queries) => queries.clone(),
        };
        if queries.is_empty() || queries.len() > MAX_QUERIES {
            return Err(invalid(format!("query holds 1 to {MAX_QUERIES} strings")));
        }
        for query in &queries {
            check_chars("query", query, MAX_QUERY_CHARS)?;
        }
        if request.rewrite_query {
            return Err(invalid(
                "rewrite_query is not supported: this server has no query rewriter",
            ));
        }
        let limit = request.max_num_results.unwrap_or(DEFAULT_NUM_RESULTS);
        if !(1..=MAX_NUM_RESULTS).contains(&limit) {
            return Err(invalid(format!(
                "max_num_results must be between 1 and {MAX_NUM_RESULTS}"
            )));
        }
        let options = request.ranking_options.unwrap_or_default();
        if options.ranker == Ranker::Default20241115 {
            return Err(invalid(
                "ranker default-2024-11-15 is not available; use auto (hybrid BM25 and \
                 embedding search when an embedding model is installed, BM25 otherwise) or none \
                 (BM25 only)",
            ));
        }
        if !(0.0..=1.0).contains(&options.score_threshold) {
            return Err(invalid(
                "ranking_options.score_threshold must be between 0 and 1",
            ));
        }
        if let Some(filter) = &request.filters {
            validate_filter(filter)?;
        }
        Ok(Self {
            queries,
            limit: usize::try_from(limit).unwrap_or(usize::MAX),
            threshold: options.score_threshold,
            ranker: options.ranker,
        })
    }
}

/// The reported score of a BM25 score `s`: `s / (1 + s)`.
fn normalize(score: f32) -> f64 {
    let score = f64::from(score).max(0.0);
    score / (1.0 + score)
}

/// Each chunk once, with its best score.
fn best_per_chunk(hits: impl IntoIterator<Item = (ChunkKey, f64)>) -> Vec<(ChunkKey, f64)> {
    let mut best: BTreeMap<ChunkKey, f64> = BTreeMap::new();
    for (key, score) in hits {
        let entry = best.entry(key).or_insert(score);
        *entry = entry.max(score);
    }
    best.into_iter().collect()
}

/// Weighted reciprocal rank fusion of one query's BM25 and cosine rankings
/// (each best first): `Σ wᵢ / (k + rankᵢ)` with 1-based ranks, divided by
/// its largest possible value `(w_text + w_embedding) / (k + 1)`, so a chunk
/// ranked first by both scores 1.
fn fuse(
    lexical: &[(ChunkKey, f32)],
    dense: &[(ChunkKey, f32)],
    k: f64,
    text_weight: f64,
    embedding_weight: f64,
) -> Vec<(ChunkKey, f64)> {
    let mut fused: BTreeMap<ChunkKey, f64> = BTreeMap::new();
    for (list, weight) in [(lexical, text_weight), (dense, embedding_weight)] {
        for (rank, (key, _)) in list.iter().enumerate() {
            let rank = f64::from(u32::try_from(rank).unwrap_or(u32::MAX)) + 1.0;
            *fused.entry(key.clone()).or_insert(0.0) += weight / (k + rank);
        }
    }
    let best = (text_weight + embedding_weight) / (k + 1.0);
    fused
        .into_iter()
        .map(|(key, score)| (key, if best > 0.0 { score / best } else { 0.0 }))
        .collect()
}

/// For each query vector, the best `depth` chunks of `files` by cosine
/// similarity (best first), among the files whose vectors are in `space`.
///
/// Brute force: every stored vector of those files is read once (streamed
/// from the database, never all in memory) and compared with every query;
/// memory is `depth` hits per query.
fn rank_dense(
    snapshot: &rusqlite::Connection,
    store_id: &str,
    files: &[String],
    space: &str,
    queries: &[Vec<f32>],
    depth: usize,
) -> Result<Vec<Vec<(ChunkKey, f32)>>> {
    let mut tops: Vec<Vec<(ChunkKey, f32)>> = vec![Vec::new(); queries.len()];
    if files.is_empty() || queries.is_empty() {
        return Ok(tops);
    }
    let embedded: HashSet<String> = {
        let mut statement = snapshot.prepare(
            "SELECT attach_id FROM vector_store_files
             WHERE store_id = ?1 AND status = 'completed' AND embedding_space = ?2",
        )?;
        statement
            .query_map(params![store_id, space], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?
    };
    let width = queries[0].len();
    let mut vector = vec![0.0_f32; width];
    let mut statement = snapshot
        .prepare("SELECT ordinal, vector FROM vector_store_embeddings WHERE attach_id = ?1")?;
    for attach_id in files.iter().filter(|file| embedded.contains(*file)) {
        let mut rows = statement.query([attach_id])?;
        while let Some(row) = rows.next()? {
            let ordinal = unsigned(row.get(0)?)?;
            let bytes = row
                .get_ref(1)?
                .as_blob()
                .map_err(|_| corrupt("chunk vector"))?;
            if !decode(bytes, &mut vector) {
                return Err(corrupt("chunk vector"));
            }
            for (query, top) in queries.iter().zip(&mut tops) {
                let cosine: f32 = query.iter().zip(&vector).map(|(a, b)| a * b).sum();
                top.push(((attach_id.clone(), ordinal), cosine));
                if top.len() >= depth.saturating_mul(4).max(64) {
                    keep_best(top, depth);
                }
            }
        }
    }
    for top in &mut tops {
        keep_best(top, depth);
    }
    Ok(tops)
}

/// Sort hits best first (ties by chunk key) and keep `depth`.
fn keep_best(hits: &mut Vec<(ChunkKey, f32)>, depth: usize) {
    let order = |left: &(ChunkKey, f32), right: &(ChunkKey, f32)| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    };
    if hits.len() > depth && depth > 0 {
        hits.select_nth_unstable_by(depth - 1, order);
    }
    hits.truncate(depth);
    hits.sort_by(order);
}

/// Refuse embedder output of the wrong shape.
fn check_vectors(vectors: &[Vec<f32>], expected: usize, width: usize) -> Result<()> {
    if vectors.len() != expected || vectors.iter().any(|vector| vector.len() != width) {
        return Err(Error::Io(std::io::Error::other(format!(
            "the embedding model returned {} vectors for {expected} inputs, or a width other \
             than {width}",
            vectors.len()
        ))));
    }
    Ok(())
}

/// Embed the chunks `ranges` of `text` with the document prompt, a group at
/// a time, as stored vectors (F16, little-endian).
fn embed_chunks(
    embedder: &dyn ChunkEmbedder,
    text: &str,
    ranges: &[(usize, usize)],
    stop: &dyn Fn() -> bool,
) -> Result<Vec<Vec<u8>>> {
    let mut encoded = Vec::with_capacity(ranges.len());
    for group in ranges.chunks(EMBED_GROUP_CHUNKS) {
        let parts = group
            .iter()
            .map(|&(start, end)| {
                text.get(start..end)
                    .ok_or_else(|| corrupt("chunk boundary"))
            })
            .collect::<Result<Vec<&str>>>()?;
        let vectors = embedder.embed_documents(&parts, stop)?;
        check_vectors(&vectors, parts.len(), embedder.dimensions())?;
        encoded.extend(vectors.iter().map(|vector| encode(vector)));
    }
    Ok(encoded)
}

/// The chunk byte ranges of an attachment, by ordinal.
fn chunk_ranges(connection: &rusqlite::Connection, attach_id: &str) -> Result<Vec<(usize, usize)>> {
    let mut statement = connection.prepare(
        "SELECT start, end FROM vector_store_chunks WHERE attach_id = ?1 ORDER BY ordinal",
    )?;
    let rows = statement
        .query_map([attach_id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    rows.into_iter()
        .map(|(start, end)| {
            Ok((
                usize::try_from(start).map_err(|_| corrupt("chunk boundary"))?,
                usize::try_from(end).map_err(|_| corrupt("chunk boundary"))?,
            ))
        })
        .collect()
}

/// Insert an attachment's vectors, ordinal by ordinal.
fn store_vectors(
    transaction: &rusqlite::Transaction<'_>,
    attach_id: &str,
    encoded: &[Vec<u8>],
) -> Result<()> {
    let mut insert = transaction.prepare(
        "INSERT INTO vector_store_embeddings (attach_id, ordinal, vector) VALUES (?1, ?2, ?3)",
    )?;
    for (ordinal, vector) in encoded.iter().enumerate() {
        insert.execute(params![attach_id, signed(count(ordinal))?, vector])?;
    }
    Ok(())
}

/// Restore the vector invariant (see the parent module docs): drop vector
/// rows that belong neither to a completed, embedded file nor to an
/// attachment awaiting collection, and the vectors of embedded files whose
/// rows do not match their chunks one to one (those are embedded again).
/// How many files and stray rows were repaired.
fn repair_vectors(connection: &mut rusqlite::Connection) -> Result<u64> {
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let stray = transaction.execute(
        "DELETE FROM vector_store_embeddings
         WHERE attach_id NOT IN (SELECT attach_id FROM vector_store_index_garbage)
           AND attach_id NOT IN (SELECT attach_id FROM vector_store_files
                                 WHERE status = 'completed' AND embedding_space IS NOT NULL)",
        [],
    )?;
    let mismatched: Vec<String> = {
        let mut statement = transaction.prepare(
            "SELECT f.attach_id FROM vector_store_files f
             WHERE f.status = 'completed' AND f.embedding_space IS NOT NULL
               AND ((SELECT COUNT(*) FROM vector_store_chunks c WHERE c.attach_id = f.attach_id)
                    != (SELECT COUNT(*) FROM vector_store_embeddings e
                        JOIN vector_store_chunks c
                          ON c.attach_id = e.attach_id AND c.ordinal = e.ordinal
                        WHERE e.attach_id = f.attach_id)
                    OR (SELECT COUNT(DISTINCT length(e.vector)) FROM vector_store_embeddings e
                        WHERE e.attach_id = f.attach_id) != 1
                    OR EXISTS (SELECT 1 FROM vector_store_embeddings e
                               WHERE e.attach_id = f.attach_id
                                 AND e.ordinal NOT IN (SELECT ordinal FROM vector_store_chunks c
                                                       WHERE c.attach_id = f.attach_id)))",
        )?;
        statement
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?
    };
    for attach_id in &mismatched {
        transaction.execute(
            "DELETE FROM vector_store_embeddings WHERE attach_id = ?1",
            [attach_id],
        )?;
        transaction.execute(
            "UPDATE vector_store_files SET embedding_space = NULL WHERE attach_id = ?1",
            [attach_id],
        )?;
    }
    transaction.commit()?;
    Ok(count(stray) + count(mismatched.len()))
}

/// A vector as stored: F16, little-endian.
fn encode(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_le_bytes())
        .collect()
}

/// Decode a stored vector into `out`; false if the width differs.
fn decode(bytes: &[u8], out: &mut [f32]) -> bool {
    if bytes.len() != out.len() * 2 {
        return false;
    }
    for (value, pair) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
        *value = f16::from_le_bytes(*pair).to_f32();
    }
    true
}

/// Up to `limit` attachments of the garbage outbox.
fn garbage(connection: &rusqlite::Connection, limit: usize) -> Result<Vec<String>> {
    let mut statement =
        connection.prepare("SELECT attach_id FROM vector_store_index_garbage LIMIT ?1")?;
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let ids = statement
        .query_map([limit], |row| row.get(0))?
        .collect::<rusqlite::Result<Vec<String>>>()?;
    Ok(ids)
}

/// Open or create the index in `directory` (created owner-only; never a
/// symlink), refusing another schema.
fn open_index(directory: &Path, schema: &Schema) -> Result<Index> {
    create_private_dir(directory)?;
    let mmap =
        MmapDirectory::open(directory).map_err(|error| Error::Io(std::io::Error::other(error)))?;
    let index = Index::open_or_create(mmap, schema.clone()).map_err(index_error)?;
    if index.schema() == *schema {
        Ok(index)
    } else {
        Err(Error::Conflict(
            "the vector store index has another schema".to_owned(),
        ))
    }
}

/// Open (owner-only) and exclusively lock `path` without waiting.
fn lock_owner(path: &Path) -> Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    // The store directory is owner-only, so nobody else can swap this path
    // between the check and the open.
    if fs::symlink_metadata(path).is_ok_and(|metadata| !metadata.is_file()) {
        return Err(Error::Io(std::io::Error::new(
            ErrorKind::PermissionDenied,
            format!("{} is not a regular file", path.display()),
        )));
    }
    let file = options.open(path)?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(fs::TryLockError::WouldBlock) => Err(Error::Conflict(format!(
            "the vector store index of {} is in use by another process",
            path.parent().unwrap_or(path).display()
        ))),
        Err(fs::TryLockError::Error(error)) => Err(error.into()),
    }
}

fn create_private_dir(directory: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(0o700);
    }
    match builder.create(directory) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    let metadata = fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(Error::Io(std::io::Error::new(
            ErrorKind::PermissionDenied,
            format!("{} is not a directory", directory.display()),
        )));
    }
    Ok(())
}

/// Remove the (derived) index directory before recreating it; refuses a
/// symlink.
fn remove_index_directory(directory: &PathBuf) -> Result<()> {
    match fs::symlink_metadata(directory) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            fs::remove_dir_all(directory)?;
            Ok(())
        }
        Ok(_) => Err(Error::Io(std::io::Error::new(
            ErrorKind::PermissionDenied,
            format!("{} is not a directory", directory.display()),
        ))),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn index_error(error: TantivyError) -> Error {
    Error::Io(std::io::Error::other(error))
}
