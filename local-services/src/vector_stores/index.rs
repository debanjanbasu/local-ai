//! The ranked index of vector store chunks: one persistent Tantivy index
//! beside the database, derived from it and reconciled with it.
//!
//! See the parent module docs ("Index and consistency") for the protocol.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

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
    ContentPart, DEFAULT_NUM_RESULTS, IndexCheck, MAX_NUM_RESULTS, MAX_QUERIES, MAX_QUERY_CHARS,
    Query, Ranker, SearchRequest, SearchResult, SearchResults, check_chars, corrupt, count,
    invalid, next_pending, searchable_files, signed, touch, unsigned, validate_filter,
    writable_store,
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
        };
        let check = opened.check()?;
        if check.rebuilt {
            eprintln!(
                "warning: the vector store index disagreed with the database and was rebuilt \
                 ({} chunks)",
                check.chunks
            );
        }
        Ok(opened)
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
        let searcher = self.reader.searcher();
        let allowed: Box<dyn tantivy::query::Query> = Box::new(ConstScoreQuery::new(
            Box::new(TermSetQuery::new(
                files
                    .iter()
                    .map(|file| Term::from_field_text(self.fields.attach, file)),
            )),
            0.0,
        ));
        let mut best: BTreeMap<(String, u64), f64> = BTreeMap::new();
        for query in &plan.queries {
            let words = self.words(query)?;
            if words.is_empty() || files.is_empty() {
                continue;
            }
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
                (Occur::Must, allowed.box_clone()),
                (Occur::Must, Box::new(any_word)),
            ]);
            let top = TopDocs::with_limit(plan.limit).order_by_score();
            for (score, address) in searcher.search(&query, &top).map_err(index_error)? {
                let key = self.chunk_key(&searcher, address)?;
                let reported = normalize(score);
                let entry = best.entry(key).or_insert(reported);
                *entry = entry.max(reported);
            }
        }
        let mut hits: Vec<((String, u64), f64)> = best
            .into_iter()
            .filter(|(_, score)| *score >= plan.threshold)
            .collect();
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
            search_query: plan.queries,
            data,
            has_more: false,
            next_page: None,
        })
    }

    /// Index the oldest files still `in_progress` (at most
    /// [`INDEX_BATCH_FILES`], stopping once their text reaches
    /// [`INDEX_BATCH_BYTES`]); how many were processed, 0 when none was
    /// waiting.
    ///
    /// Order: record the attachments in the garbage outbox, write and commit
    /// their index documents, then, in one database transaction, mark each
    /// `completed`, insert its chunk rows and remove it from the outbox. A
    /// crash between the steps leaves documents the outbox removes; a file
    /// deleted, cancelled or re-attached meanwhile keeps its outbox entry.
    // The writer lock spans the whole protocol: the outbox entries, the
    // index commit and the confirming transaction must not interleave with
    // another indexing run or a collection.
    #[allow(clippy::significant_drop_tightening)]
    pub fn index_next(&self) -> Result<usize> {
        let mut writer = self.lock();
        let pending = next_pending(&self.store.connection()?, INDEX_BATCH_FILES)?;
        let mut parsed = Vec::new();
        let mut failed = Vec::new();
        let mut bytes = 0;
        for file in pending {
            if bytes >= INDEX_BATCH_BYTES {
                break;
            }
            match self.store.parse_source(&file) {
                Ok((text, ranges)) => {
                    bytes += text.len();
                    parsed.push((file, text, ranges));
                }
                Err(error) => failed.push((file, error)),
            }
        }
        let processed = parsed.len() + failed.len();
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
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for (file, _, _) in &parsed {
            transaction.execute(
                "INSERT OR IGNORE INTO vector_store_index_garbage (attach_id) VALUES (?1)",
                [&file.attach_id],
            )?;
        }
        transaction.commit()?;
        for (file, text, ranges) in &parsed {
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
                "UPDATE vector_store_files SET status = 'completed', usage_bytes = ?2, text = ?3
                 WHERE attach_id = ?1 AND status = 'in_progress'",
            )?;
            let mut insert = transaction.prepare(
                "INSERT INTO vector_store_chunks (attach_id, ordinal, start, end)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;
            let mut confirm = transaction
                .prepare("DELETE FROM vector_store_index_garbage WHERE attach_id = ?1")?;
            for (file, text, ranges) in &parsed {
                let changed = complete.execute(params![
                    file.attach_id,
                    signed(count(text.len()))?,
                    text.as_bytes()
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
                confirm.execute([&file.attach_id])?;
            }
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
        Ok(IndexCheck {
            chunks,
            collected,
            rebuilt: !consistent,
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

    fn chunk_key(
        &self,
        searcher: &tantivy::Searcher,
        address: DocAddress,
    ) -> Result<(String, u64)> {
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

/// A validated search request.
struct SearchPlan {
    queries: Vec<String>,
    limit: usize,
    threshold: f64,
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
                "rewrite_query is not supported: this server ranks with lexical BM25 and has \
                 no query rewriter",
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
                "ranker default-2024-11-15 is not available; use auto or none (both are plain \
                 BM25 here)",
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
        })
    }
}

/// The reported score of a BM25 score `s`: `s / (1 + s)`.
fn normalize(score: f32) -> f64 {
    let score = f64::from(score).max(0.0);
    score / (1.0 + score)
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
