//! Vector stores: `OpenAI`-compatible, durable collections of parsed files
//! with ranked **lexical** search.
//!
//! # What "vector store" means here
//!
//! The API shape is `OpenAI`'s, but ranking is BM25 over a persistent
//! [Tantivy](tantivy) index ([`VectorIndex`]; Unicode words, lowercased,
//! folded to ASCII, English Porter-2 stemming). No embedding model is
//! involved: pooled hidden states of the server's one Bonsai checkpoint were
//! measured as retrieval embeddings and lost to BM25 while costing orders of
//! magnitude more compute (see `docs/BONSAI.md`), so they are not used.
//! Search therefore matches words, not meaning: a query with no word (after
//! stemming) in common with a chunk never finds it.
//!
//! # Files and chunks
//!
//! A vector store file references a live File of the Files API (any purpose)
//! and becomes `in_progress` at once; the background indexer
//! ([`VectorStoreIndexer`], or [`VectorIndex::index_next`]) reads its bytes,
//! and the file ends `completed`, `failed` (with `last_error`) or, when its
//! file batch is cancelled first, `cancelled`. Only UTF-8 text without NUL
//! bytes is indexed (an optional BOM is dropped); anything else fails as
//! `unsupported_file`. A file larger than [`MAX_INDEXED_FILE_BYTES`],
//! holding no text, or needing more than [`MAX_FILE_CHUNKS`] chunks fails as
//! `invalid_file`.
//!
//! The parsed text is copied into the database, so the content endpoint and
//! search keep working while the source File lives. Deleting the source File
//! (directly, or when an expired one is purged) removes it from every vector
//! store in the same transaction (a trigger on the `files` table).
//!
//! Chunking follows the requested strategy, but sizes are counted in
//! whitespace-separated **words**, not model tokens (`auto` is `static` with
//! 800 and 400). Every file reports the static strategy it was chunked with.
//!
//! # Index and consistency
//!
//! The database is the source of truth: file rows, their parsed text and
//! their chunk rows (byte ranges). The Tantivy index in the store directory
//! is derived from it: one document per chunk of a completed file, keyed by
//! attachment and ordinal, holding postings only (search reads chunk text
//! from the database). Two stores cannot share one transaction, so the
//! protocol keeps one invariant instead: **every chunk of a completed file
//! has its index document; every other document belongs to an attachment
//! listed in the garbage outbox** (`vector_store_index_garbage`).
//!
//! * Indexing ([`VectorIndex::index_next`]) lists the attachment in the
//!   outbox, commits its documents to the index, then marks the file
//!   `completed`, inserts its chunks and removes it from the outbox in one
//!   database transaction. A crash before that transaction leaves the file
//!   `in_progress` (indexed again from the start on the next run, replacing
//!   its documents) and its documents listed for removal.
//! * Deleting a file, a store or a source File deletes the rows in one
//!   transaction, and a trigger lists each removed attachment in the outbox;
//!   the indexer removes their documents later ([`VectorIndex::collect_garbage`],
//!   on every run and at start) and only then clears the outbox entries.
//! * Search only considers attachments that are `completed` in its own
//!   database snapshot, so documents awaiting removal are never returned.
//!
//! Opening the index ([`VectorIndex::open`]) collects the outbox and then
//! compares the live documents with the chunk rows; any disagreement (an
//! index lost, damaged, of another schema or unopenable, or edited outside
//! this module) rebuilds the whole index from the stored text. One process
//! owns the index (an exclusive lock file); a second server on the same
//! data directory cannot open it.
//!
//! An indexing attempt only completes a file that is still the same
//! attachment and still `in_progress`, so a file deleted, cancelled or
//! re-attached meanwhile discards the attempt (its documents stay listed for
//! removal).
//!
//! # Search and scores
//!
//! Queries are analyzed like the chunks and matched with OR semantics; an
//! array of queries is searched query by query and each chunk keeps its best
//! score. BM25 term statistics (Tantivy's, `k1 = 1.2`, `b = 0.75`) are
//! computed over every store's chunks on this server, not per store. The raw
//! BM25 score `s` (larger is better, never negative) is reported as
//! `s / (1 + s)`: in `[0, 1)`, monotonic in BM25 within one query, **not** a
//! calibrated similarity and not comparable across queries.
//! `score_threshold` applies to that value. `rewrite_query: true` and the
//! named `default-2024-11-15` reranker are refused (nothing here can rewrite
//! queries or rerank); the rankers `auto` and `none` both mean plain BM25.
//! Results are chunk-level, one item per chunk, at most `max_num_results`
//! (1 to 50), never paged.
//!
//! # Attribute filters
//!
//! `eq`/`ne` compare strings, numbers and booleans by type and value; `gt`,
//! `gte`, `lt` and `lte` compare numbers numerically and strings by code
//! point; `in`/`nin` test membership in an array of strings and numbers. A
//! comparison against a missing key or a value of another type is false,
//! except `ne` and `nin`, which are then true. `and`/`or` nest up to
//! [`MAX_FILTER_DEPTH`] levels.
//!
//! # Expiry
//!
//! A store with `expires_after` expires `days` after its `last_active_at`,
//! which creating it, modifying it, attaching files and searching it update
//! (searches at most once per [`ACTIVITY_GRANULARITY_SECONDS`], so a
//! burst of searches costs one database write).
//! An expired store reports `status: expired`; it can still be read,
//! modified and deleted, but searching it or attaching files is a conflict.

use std::collections::HashSet;
use std::io::Read as _;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use rusqlite::{Connection, OptionalExtension as _, Row, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::conversations::{
    MAX_METADATA_KEY_CHARS, MAX_METADATA_PAIRS, MAX_METADATA_VALUE_CHARS, Metadata, Page,
};
use crate::{Error, Result, Store, migrate, new_id, now};

mod index;

pub use index::{ACTIVITY_GRANULARITY_SECONDS, INDEX_BATCH_BYTES, INDEX_BATCH_FILES, VectorIndex};

/// Largest source file the indexer reads.
pub const MAX_INDEXED_FILE_BYTES: u64 = 16 * 1024 * 1024;
/// Most chunks one file may produce.
pub const MAX_FILE_CHUNKS: usize = 50_000;
/// Most files one vector store may hold.
pub const MAX_STORE_FILES: usize = 10_000;
/// Most `file_ids` when creating a store.
pub const MAX_CREATE_FILE_IDS: usize = 500;
/// Most files in one file batch.
pub const MAX_BATCH_FILES: usize = 2_000;
/// Longest single query, in characters.
pub const MAX_QUERY_CHARS: usize = 4_096;
/// Most queries in one search.
pub const MAX_QUERIES: usize = 16;
/// Largest `max_num_results`.
pub const MAX_NUM_RESULTS: u32 = 50;
/// `max_num_results` when absent.
pub const DEFAULT_NUM_RESULTS: u32 = 10;
/// Deepest nesting of compound filters.
pub const MAX_FILTER_DEPTH: usize = 8;
/// Most comparisons in one filter.
pub const MAX_FILTER_COMPARISONS: usize = 64;
/// Most values in one `in`/`nin` comparison.
pub const MAX_FILTER_VALUES: usize = 256;
/// Most attributes per file.
pub const MAX_ATTRIBUTES: usize = 16;
/// Longest attribute key, in characters.
pub const MAX_ATTRIBUTE_KEY_CHARS: usize = 64;
/// Longest string attribute value, in characters.
pub const MAX_ATTRIBUTE_VALUE_CHARS: usize = 512;
/// Longest store name or description, in characters.
pub const MAX_NAME_CHARS: usize = 1_024;
/// Page size when `limit` is absent.
pub const DEFAULT_LIST_LIMIT: u32 = 20;
/// Largest page size.
pub const MAX_LIST_LIMIT: u32 = 100;
/// The `auto` strategy's chunk size, in words.
pub const AUTO_MAX_CHUNK_TOKENS: u32 = 800;
/// The `auto` strategy's overlap, in words.
pub const AUTO_CHUNK_OVERLAP_TOKENS: u32 = 400;
/// Smallest static chunk size.
pub const MIN_CHUNK_TOKENS: u32 = 100;
/// Largest static chunk size.
pub const MAX_CHUNK_TOKENS: u32 = 4_096;
/// Longest expiry, in days.
pub const MAX_EXPIRES_AFTER_DAYS: u32 = 365;

const STORE_PREFIX: &str = "vs_";
const BATCH_PREFIX: &str = "vsfb_";
const ATTACH_PREFIX: &str = "vsfa_";
const DAY_SECONDS: u64 = 86_400;

/// Version 1. Chunk text is sliced from the file's stored `text` by byte
/// range; the ranked index itself is the Tantivy directory of [`index`].
/// `vector_store_index_garbage` is the outbox of attachments whose index
/// documents may have to be removed (see the module docs).
const MIGRATIONS: &[&str] = &["
CREATE TABLE vector_stores (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT,
    metadata TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    last_active_at INTEGER NOT NULL,
    expires_after_days INTEGER
) STRICT;
CREATE TABLE vector_store_file_batches (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    store_id TEXT NOT NULL REFERENCES vector_stores (id) ON DELETE CASCADE,
    created_at INTEGER NOT NULL,
    total INTEGER NOT NULL,
    cancelled_at INTEGER
) STRICT;
CREATE INDEX vector_store_file_batches_store ON vector_store_file_batches (store_id);
CREATE TABLE vector_store_files (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    attach_id TEXT NOT NULL UNIQUE,
    store_id TEXT NOT NULL REFERENCES vector_stores (id) ON DELETE CASCADE,
    file_id TEXT NOT NULL,
    batch_id TEXT,
    filename TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('in_progress', 'completed', 'failed', 'cancelled')),
    error_code TEXT,
    error_message TEXT,
    max_chunk_tokens INTEGER NOT NULL,
    chunk_overlap_tokens INTEGER NOT NULL,
    attributes TEXT NOT NULL,
    usage_bytes INTEGER NOT NULL DEFAULT 0,
    text BLOB,
    UNIQUE (store_id, file_id)
) STRICT;
CREATE INDEX vector_store_files_status ON vector_store_files (status, seq);
CREATE INDEX vector_store_files_file ON vector_store_files (file_id);
CREATE INDEX vector_store_files_batch ON vector_store_files (batch_id, seq);
CREATE TABLE vector_store_chunks (
    attach_id TEXT NOT NULL REFERENCES vector_store_files (attach_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    PRIMARY KEY (attach_id, ordinal)
) STRICT, WITHOUT ROWID;
CREATE TABLE vector_store_index_garbage (
    attach_id TEXT PRIMARY KEY NOT NULL
) STRICT, WITHOUT ROWID;
CREATE TRIGGER vector_store_files_deleted AFTER DELETE ON vector_store_files BEGIN
    INSERT OR IGNORE INTO vector_store_index_garbage (attach_id) VALUES (OLD.attach_id);
END;
CREATE TRIGGER vector_store_files_source_deleted AFTER DELETE ON files BEGIN
    DELETE FROM vector_store_files WHERE file_id = OLD.id;
END;
"];

const STORE_COLUMNS: &str = "id, name, metadata, created_at, last_active_at, expires_after_days";
const FILE_COLUMNS: &str = "file_id, store_id, created_at, status, error_code, error_message, \
     max_chunk_tokens, chunk_overlap_tokens, attributes, usage_bytes";

/// Create or upgrade the vector store tables (after the Files tables, whose
/// deletions they follow).
pub(crate) fn initialize(connection: &mut Connection) -> Result<()> {
    migrate(connection, "vector_stores", MIGRATIONS)
}

/// File attributes: at most [`MAX_ATTRIBUTES`] keys, each value a string,
/// a finite number or a boolean.
pub type Attributes = Map<String, Value>;

/// `expires_after`: the only anchor is `last_active_at`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpirationPolicy {
    pub anchor: ExpirationAnchor,
    /// 1 to [`MAX_EXPIRES_AFTER_DAYS`].
    pub days: u32,
}

/// The one expiration anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpirationAnchor {
    LastActiveAt,
}

/// A requested chunking strategy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChunkingStrategy {
    /// [`AUTO_MAX_CHUNK_TOKENS`] with [`AUTO_CHUNK_OVERLAP_TOKENS`].
    #[default]
    Auto,
    Static {
        #[serde(rename = "static")]
        config: StaticChunking,
    },
}

impl ChunkingStrategy {
    /// The static sizes this strategy chunks with, validated.
    fn resolve(self) -> Result<StaticChunking> {
        match self {
            Self::Auto => Ok(StaticChunking {
                max_chunk_size_tokens: AUTO_MAX_CHUNK_TOKENS,
                chunk_overlap_tokens: AUTO_CHUNK_OVERLAP_TOKENS,
            }),
            Self::Static { config } => {
                if !(MIN_CHUNK_TOKENS..=MAX_CHUNK_TOKENS).contains(&config.max_chunk_size_tokens) {
                    return Err(invalid(format!(
                        "chunking_strategy.static.max_chunk_size_tokens must be between \
                         {MIN_CHUNK_TOKENS} and {MAX_CHUNK_TOKENS}"
                    )));
                }
                if config.chunk_overlap_tokens > config.max_chunk_size_tokens / 2 {
                    return Err(invalid(
                        "chunking_strategy.static.chunk_overlap_tokens must not exceed half \
                         of max_chunk_size_tokens",
                    ));
                }
                Ok(config)
            }
        }
    }
}

/// Static chunk sizes (in words; see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticChunking {
    pub max_chunk_size_tokens: u32,
    pub chunk_overlap_tokens: u32,
}

/// The strategy a file was chunked with, as reported: always static.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "static")]
pub struct ReportedChunking {
    #[serde(rename = "static")]
    pub config: StaticChunking,
}

/// File counts of a store or a batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileCounts {
    pub in_progress: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub total: u64,
}

/// A store's status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreStatus {
    Expired,
    InProgress,
    Completed,
}

/// A vector store, serialized as the public `vector_store` object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "vector_store")]
pub struct VectorStore {
    pub id: String,
    pub created_at: u64,
    /// Empty when none was given.
    pub name: String,
    /// Bytes of parsed text held for its completed files.
    pub usage_bytes: u64,
    pub file_counts: FileCounts,
    pub status: StoreStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_after: Option<ExpirationPolicy>,
    pub expires_at: Option<u64>,
    pub last_active_at: Option<u64>,
    pub metadata: Metadata,
}

/// Input for [`Store::create_vector_store`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CreateVectorStore {
    /// At most [`MAX_CREATE_FILE_IDS`], attached with `chunking_strategy`.
    pub file_ids: Vec<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub expires_after: Option<ExpirationPolicy>,
    pub chunking_strategy: Option<ChunkingStrategy>,
    pub metadata: Option<Metadata>,
}

/// Input for [`Store::update_vector_store`]; an absent field is unchanged and
/// an explicit `null` clears it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UpdateVectorStore {
    #[serde(deserialize_with = "present")]
    pub name: Option<Option<String>>,
    #[serde(deserialize_with = "present")]
    pub expires_after: Option<Option<ExpirationPolicy>>,
    #[serde(deserialize_with = "present")]
    pub metadata: Option<Option<Metadata>>,
}

/// Reads a present field, `null` included, as `Some`: absent, `null` and a
/// value are three different updates.
#[allow(clippy::option_option)]
fn present<'de, D, T>(deserializer: D) -> std::result::Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

/// `vector_store.deleted`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "vector_store.deleted")]
pub struct VectorStoreDeleted {
    pub id: String,
    pub deleted: bool,
}

/// A vector store file's status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileStatus {
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

impl FileStatus {
    /// The wire and stored name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            _ => return Err(corrupt("vector store file status")),
        })
    }
}

/// Why a file failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    ServerError,
    UnsupportedFile,
    InvalidFile,
}

impl ErrorCode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ServerError => "server_error",
            Self::UnsupportedFile => "unsupported_file",
            Self::InvalidFile => "invalid_file",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "server_error" => Self::ServerError,
            "unsupported_file" => Self::UnsupportedFile,
            "invalid_file" => Self::InvalidFile,
            _ => return Err(corrupt("vector store file error code")),
        })
    }
}

/// A failed file's `last_error`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LastError {
    pub code: ErrorCode,
    pub message: String,
}

/// A file in a store, serialized as the public `vector_store.file` object.
/// Its `id` is the File's ID.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "vector_store.file")]
pub struct VectorStoreFile {
    pub id: String,
    /// Bytes of parsed text held for it (0 until completed).
    pub usage_bytes: u64,
    pub created_at: u64,
    pub vector_store_id: String,
    pub status: FileStatus,
    pub last_error: Option<LastError>,
    pub chunking_strategy: ReportedChunking,
    pub attributes: Attributes,
}

/// Input for [`Store::create_vector_store_file`], and one entry of a batch's
/// `files`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateVectorStoreFile {
    pub file_id: String,
    #[serde(default)]
    pub chunking_strategy: Option<ChunkingStrategy>,
    #[serde(default)]
    pub attributes: Option<Attributes>,
}

/// `vector_store.file.deleted`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "vector_store.file.deleted")]
pub struct VectorStoreFileDeleted {
    pub id: String,
    pub deleted: bool,
}

/// One text part of a file's content.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
}

/// `vector_store.file_content.page`: the whole parsed text in one part.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "vector_store.file_content.page")]
pub struct VectorStoreFileContent {
    pub data: Vec<ContentPart>,
    pub has_more: bool,
    pub next_page: Option<String>,
}

/// A file batch's status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    InProgress,
    Completed,
    Cancelled,
    Failed,
}

/// A file batch, serialized as the public `vector_store.file_batch` object.
///
/// It is `in_progress` while any of its files is, `cancelled` once cancelled,
/// then `failed` if every file failed and `completed` otherwise. A file
/// attached again later (alone or in another batch) leaves this batch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "vector_store.file_batch")]
pub struct FileBatch {
    pub id: String,
    pub created_at: u64,
    pub vector_store_id: String,
    pub status: BatchStatus,
    pub file_counts: FileCounts,
}

/// Input for [`Store::create_vector_store_file_batch`]: exactly one of
/// `file_ids` (sharing `chunking_strategy` and `attributes`) or `files`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CreateFileBatch {
    pub file_ids: Option<Vec<String>>,
    pub files: Option<Vec<CreateVectorStoreFile>>,
    pub chunking_strategy: Option<ChunkingStrategy>,
    pub attributes: Option<Attributes>,
}

/// List order; newest first by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ListOrder {
    Asc,
    #[default]
    Desc,
}

/// A list query: `after`/`before` are IDs in the listed collection.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ListQuery {
    /// 1 to [`MAX_LIST_LIMIT`]; [`DEFAULT_LIST_LIMIT`] when absent.
    pub limit: Option<u32>,
    pub order: ListOrder,
    pub after: Option<String>,
    pub before: Option<String>,
    /// Files only: just those in this status.
    pub filter: Option<FileStatus>,
}

/// The search query: one string or several.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Query {
    One(String),
    Many(Vec<String>),
}

/// A comparison operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Comparison {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    Nin,
}

/// A compound operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Compound {
    And,
    Or,
}

/// An attribute filter (see the module docs for the semantics).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Filter {
    Comparison {
        #[serde(rename = "type")]
        op: Comparison,
        key: String,
        value: Value,
    },
    Compound {
        #[serde(rename = "type")]
        op: Compound,
        filters: Vec<Self>,
    },
}

/// A named reranker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ranker {
    #[serde(rename = "none")]
    None,
    #[default]
    #[serde(rename = "auto")]
    Auto,
    /// Refused: no such reranker exists here.
    #[serde(rename = "default-2024-11-15")]
    Default20241115,
}

/// `ranking_options`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RankingOptions {
    pub ranker: Ranker,
    /// 0 to 1, applied to the reported score.
    pub score_threshold: f64,
}

/// Input for [`Store::search_vector_store`], the body of
/// `POST /vector_stores/{id}/search`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    pub query: Query,
    #[serde(default)]
    pub rewrite_query: bool,
    /// 1 to [`MAX_NUM_RESULTS`]; [`DEFAULT_NUM_RESULTS`] when absent.
    #[serde(default)]
    pub max_num_results: Option<u32>,
    #[serde(default)]
    pub filters: Option<Filter>,
    #[serde(default)]
    pub ranking_options: Option<RankingOptions>,
}

impl SearchRequest {
    /// A plain search for `query` with the defaults.
    #[must_use]
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: Query::One(query.into()),
            rewrite_query: false,
            max_num_results: None,
            filters: None,
            ranking_options: None,
        }
    }
}

/// One matching chunk.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_id: String,
    pub filename: String,
    /// `s / (1 + s)` of the BM25 score `s` (see the module docs).
    pub score: f64,
    pub attributes: Attributes,
    pub content: Vec<ContentPart>,
}

/// `vector_store.search_results.page`, best first.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "vector_store.search_results.page")]
pub struct SearchResults {
    pub search_query: Vec<String>,
    pub data: Vec<SearchResult>,
    pub has_more: bool,
    pub next_page: Option<String>,
}

/// What [`VectorIndex::check`] found and did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexCheck {
    /// Chunks the index holds after the check.
    pub chunks: u64,
    /// Index documents of deleted or unconfirmed attachments removed first.
    pub collected: u64,
    /// Whether the index disagreed with the chunk table and was rebuilt.
    pub rebuilt: bool,
}

/// A parsed file (its text and chunk byte ranges), or why it failed.
type Parsed = std::result::Result<(String, Vec<(usize, usize)>), (ErrorCode, String)>;

/// One file the indexer is to process.
struct Pending {
    attach_id: String,
    file_id: String,
    chunking: StaticChunking,
}

impl Store {
    /// Create a store, attaching `file_ids` (each must be a live File).
    pub fn create_vector_store(&self, request: &CreateVectorStore) -> Result<VectorStore> {
        let name = request.name.clone().unwrap_or_default();
        check_chars("name", &name, MAX_NAME_CHARS)?;
        if let Some(description) = &request.description {
            check_chars("description", description, MAX_NAME_CHARS)?;
        }
        let metadata = request.metadata.clone().unwrap_or_default();
        validate_metadata(&metadata)?;
        let days = request.expires_after.map(validate_expiry).transpose()?;
        if request.file_ids.len() > MAX_CREATE_FILE_IDS {
            return Err(invalid(format!(
                "file_ids may hold at most {MAX_CREATE_FILE_IDS} files"
            )));
        }
        let chunking = request.chunking_strategy.unwrap_or_default().resolve()?;
        let files: Vec<CreateVectorStoreFile> = request
            .file_ids
            .iter()
            .map(|file_id| CreateVectorStoreFile {
                file_id: file_id.clone(),
                chunking_strategy: Some(ChunkingStrategy::Static { config: chunking }),
                attributes: None,
            })
            .collect();
        let attachments = resolve_files(&files)?;
        let id = new_id(STORE_PREFIX)?;
        let at = signed(now()?)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "INSERT INTO vector_stores (id, name, description, metadata, created_at,
                                        last_active_at, expires_after_days)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5, ?6)",
            params![
                id,
                name,
                request.description,
                serde_json::to_string(&metadata)?,
                at,
                days
            ],
        )?;
        attach(&transaction, &id, None, &attachments, at)?;
        transaction.commit()?;
        load_store(&connection, &id)
    }

    /// The store `id`.
    pub fn get_vector_store(&self, id: &str) -> Result<VectorStore> {
        load_store(&self.connection()?, id)
    }

    /// Stores, newest first by default.
    pub fn list_vector_stores(&self, query: &ListQuery) -> Result<Page<VectorStore>> {
        if query.filter.is_some() {
            return Err(invalid("filter applies only to vector store files"));
        }
        let connection = self.connection()?;
        let seq = |id: &str| {
            connection
                .query_row("SELECT seq FROM vector_stores WHERE id = ?1", [id], |row| {
                    row.get::<_, i64>(0)
                })
                .optional()?
                .ok_or_else(|| store_not_found(id))
        };
        let window = Window::new(query, seq)?;
        let ids = window.ids(
            &connection,
            "SELECT id FROM vector_stores WHERE seq > ?1 AND seq < ?2",
            &[],
        )?;
        let (ids, has_more) = window.finish(ids);
        let data = ids
            .iter()
            .map(|id| load_store(&connection, id))
            .collect::<Result<Vec<_>>>()?;
        Ok(page(data, has_more, |store| store.id.clone()))
    }

    /// Change a store's name, expiry or metadata (an explicit `null` clears
    /// it) and mark it active.
    pub fn update_vector_store(
        &self,
        id: &str,
        request: &UpdateVectorStore,
    ) -> Result<VectorStore> {
        if let Some(Some(name)) = &request.name {
            check_chars("name", name, MAX_NAME_CHARS)?;
        }
        if let Some(Some(metadata)) = &request.metadata {
            validate_metadata(metadata)?;
        }
        let days = match request.expires_after {
            Some(Some(policy)) => Some(Some(validate_expiry(policy)?)),
            Some(None) => Some(None),
            None => None,
        };
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        load_store(&transaction, id)?;
        if let Some(name) = &request.name {
            transaction.execute(
                "UPDATE vector_stores SET name = ?2 WHERE id = ?1",
                params![id, name.clone().unwrap_or_default()],
            )?;
        }
        if let Some(metadata) = &request.metadata {
            transaction.execute(
                "UPDATE vector_stores SET metadata = ?2 WHERE id = ?1",
                params![
                    id,
                    serde_json::to_string(&metadata.clone().unwrap_or_default())?
                ],
            )?;
        }
        if let Some(days) = days {
            transaction.execute(
                "UPDATE vector_stores SET expires_after_days = ?2 WHERE id = ?1",
                params![id, days],
            )?;
        }
        touch(&transaction, id)?;
        transaction.commit()?;
        load_store(&connection, id)
    }

    /// Delete a store, its files, batches and index entries, atomically.
    pub fn delete_vector_store(&self, id: &str) -> Result<VectorStoreDeleted> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        load_store(&transaction, id)?;
        transaction.execute("DELETE FROM vector_store_files WHERE store_id = ?1", [id])?;
        transaction.execute(
            "DELETE FROM vector_store_file_batches WHERE store_id = ?1",
            [id],
        )?;
        transaction.execute("DELETE FROM vector_stores WHERE id = ?1", [id])?;
        transaction.commit()?;
        Ok(VectorStoreDeleted {
            id: id.to_owned(),
            deleted: true,
        })
    }

    /// Attach a live File to a store; it is indexed in the background. A file
    /// already attached is replaced (new attributes and chunking, indexed
    /// again).
    pub fn create_vector_store_file(
        &self,
        store_id: &str,
        request: &CreateVectorStoreFile,
    ) -> Result<VectorStoreFile> {
        let attachments = resolve_files(std::slice::from_ref(request))?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        writable_store(&transaction, store_id)?;
        let at = signed(now()?)?;
        attach(&transaction, store_id, None, &attachments, at)?;
        touch(&transaction, store_id)?;
        transaction.commit()?;
        load_file(&connection, store_id, &request.file_id)
    }

    /// The file `file_id` of store `store_id`.
    pub fn get_vector_store_file(&self, store_id: &str, file_id: &str) -> Result<VectorStoreFile> {
        let connection = self.connection()?;
        load_store(&connection, store_id)?;
        load_file(&connection, store_id, file_id)
    }

    /// A store's files, newest first by default.
    pub fn list_vector_store_files(
        &self,
        store_id: &str,
        query: &ListQuery,
    ) -> Result<Page<VectorStoreFile>> {
        let connection = self.connection()?;
        load_store(&connection, store_id)?;
        list_files(&connection, store_id, None, query)
    }

    /// Replace a file's attributes.
    pub fn update_vector_store_file(
        &self,
        store_id: &str,
        file_id: &str,
        attributes: &Attributes,
    ) -> Result<VectorStoreFile> {
        validate_attributes(attributes)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        load_store(&transaction, store_id)?;
        let changed = transaction.execute(
            "UPDATE vector_store_files SET attributes = ?3 WHERE store_id = ?1 AND file_id = ?2",
            params![store_id, file_id, serde_json::to_string(attributes)?],
        )?;
        if changed == 0 {
            return Err(file_not_found(store_id, file_id));
        }
        transaction.commit()?;
        load_file(&connection, store_id, file_id)
    }

    /// Remove a file from a store (the File itself stays), with its index
    /// entries, atomically.
    pub fn delete_vector_store_file(
        &self,
        store_id: &str,
        file_id: &str,
    ) -> Result<VectorStoreFileDeleted> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        load_store(&transaction, store_id)?;
        if !detach(&transaction, store_id, file_id)? {
            return Err(file_not_found(store_id, file_id));
        }
        transaction.commit()?;
        Ok(VectorStoreFileDeleted {
            id: file_id.to_owned(),
            deleted: true,
        })
    }

    /// The parsed text of a completed file, in one part; empty otherwise.
    pub fn vector_store_file_content(
        &self,
        store_id: &str,
        file_id: &str,
    ) -> Result<VectorStoreFileContent> {
        let connection = self.connection()?;
        load_store(&connection, store_id)?;
        let text: Option<Option<Vec<u8>>> = connection
            .query_row(
                "SELECT text FROM vector_store_files WHERE store_id = ?1 AND file_id = ?2",
                params![store_id, file_id],
                |row| row.get(0),
            )
            .optional()?;
        let text = text.ok_or_else(|| file_not_found(store_id, file_id))?;
        let data = match text {
            Some(bytes) => vec![ContentPart {
                kind: "text".to_owned(),
                text: String::from_utf8(bytes).map_err(|_| corrupt("vector store file text"))?,
            }],
            None => Vec::new(),
        };
        Ok(VectorStoreFileContent {
            data,
            has_more: false,
            next_page: None,
        })
    }

    /// Attach up to [`MAX_BATCH_FILES`] live Files as one batch, atomically.
    pub fn create_vector_store_file_batch(
        &self,
        store_id: &str,
        request: &CreateFileBatch,
    ) -> Result<FileBatch> {
        let files = match (&request.file_ids, &request.files) {
            (Some(ids), None) => ids
                .iter()
                .map(|file_id| CreateVectorStoreFile {
                    file_id: file_id.clone(),
                    chunking_strategy: request.chunking_strategy,
                    attributes: request.attributes.clone(),
                })
                .collect::<Vec<_>>(),
            (None, Some(files)) => {
                if request.chunking_strategy.is_some() || request.attributes.is_some() {
                    return Err(invalid(
                        "chunking_strategy and attributes go inside each entry of files",
                    ));
                }
                files.clone()
            }
            _ => return Err(invalid("give exactly one of file_ids and files")),
        };
        if files.is_empty() || files.len() > MAX_BATCH_FILES {
            return Err(invalid(format!(
                "a file batch holds 1 to {MAX_BATCH_FILES} files"
            )));
        }
        let attachments = resolve_files(&files)?;
        let id = new_id(BATCH_PREFIX)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        writable_store(&transaction, store_id)?;
        let at = signed(now()?)?;
        transaction.execute(
            "INSERT INTO vector_store_file_batches (id, store_id, created_at, total)
             VALUES (?1, ?2, ?3, ?4)",
            params![id, store_id, at, signed(count(attachments.len()))?],
        )?;
        attach(&transaction, store_id, Some(&id), &attachments, at)?;
        touch(&transaction, store_id)?;
        transaction.commit()?;
        load_batch(&connection, store_id, &id)
    }

    /// The batch `batch_id` of store `store_id`.
    pub fn get_vector_store_file_batch(&self, store_id: &str, batch_id: &str) -> Result<FileBatch> {
        let connection = self.connection()?;
        load_store(&connection, store_id)?;
        load_batch(&connection, store_id, batch_id)
    }

    /// Cancel an in-progress batch: its files still in progress become
    /// `cancelled` (and are not indexed). Cancelling a cancelled batch
    /// changes nothing; a finished one is a conflict.
    pub fn cancel_vector_store_file_batch(
        &self,
        store_id: &str,
        batch_id: &str,
    ) -> Result<FileBatch> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        load_store(&transaction, store_id)?;
        let batch = load_batch(&transaction, store_id, batch_id)?;
        match batch.status {
            BatchStatus::InProgress => {
                transaction.execute(
                    "UPDATE vector_store_file_batches SET cancelled_at = ?2 WHERE id = ?1",
                    params![batch_id, signed(now()?)?],
                )?;
                transaction.execute(
                    "UPDATE vector_store_files SET status = 'cancelled'
                     WHERE batch_id = ?1 AND status = 'in_progress'",
                    [batch_id],
                )?;
            }
            BatchStatus::Cancelled => {}
            BatchStatus::Completed | BatchStatus::Failed => {
                return Err(Error::Conflict(format!(
                    "file batch {batch_id} has finished; only an in-progress batch can be \
                     cancelled"
                )));
            }
        }
        transaction.commit()?;
        load_batch(&connection, store_id, batch_id)
    }

    /// The files of a batch, newest first by default.
    pub fn list_vector_store_file_batch_files(
        &self,
        store_id: &str,
        batch_id: &str,
        query: &ListQuery,
    ) -> Result<Page<VectorStoreFile>> {
        let connection = self.connection()?;
        load_store(&connection, store_id)?;
        load_batch(&connection, store_id, batch_id)?;
        list_files(&connection, store_id, Some(batch_id), query)
    }

    /// Read and chunk a pending file, or say why it fails.
    fn parse_source(&self, pending: &Pending) -> Parsed {
        let content = match self.open_file_content(&pending.file_id) {
            Ok(content) => content,
            Err(Error::NotFound(_)) => {
                return Err((
                    ErrorCode::InvalidFile,
                    format!("file {} no longer exists", pending.file_id),
                ));
            }
            Err(error) => return Err((ErrorCode::ServerError, error.to_string())),
        };
        if content.file.bytes > MAX_INDEXED_FILE_BYTES {
            return Err((
                ErrorCode::InvalidFile,
                format!("the file is larger than {MAX_INDEXED_FILE_BYTES} bytes"),
            ));
        }
        let mut bytes = Vec::new();
        content
            .content
            .take(MAX_INDEXED_FILE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| (ErrorCode::ServerError, error.to_string()))?;
        if count(bytes.len()) > MAX_INDEXED_FILE_BYTES {
            return Err((
                ErrorCode::InvalidFile,
                format!("the file is larger than {MAX_INDEXED_FILE_BYTES} bytes"),
            ));
        }
        let unsupported = || {
            (
                ErrorCode::UnsupportedFile,
                "only UTF-8 text files can be indexed".to_owned(),
            )
        };
        if bytes.contains(&0) {
            return Err(unsupported());
        }
        let mut text = String::from_utf8(bytes).map_err(|_| unsupported())?;
        if text.starts_with('\u{feff}') {
            text.drain(..'\u{feff}'.len_utf8());
        }
        let ranges = chunk(&text, pending.chunking)?;
        Ok((text, ranges))
    }
}

/// The byte ranges of `text`'s chunks: windows of whole words.
fn chunk(
    text: &str,
    chunking: StaticChunking,
) -> std::result::Result<Vec<(usize, usize)>, (ErrorCode, String)> {
    let mut words = Vec::new();
    let mut start = None;
    for (index, character) in text.char_indices() {
        match (character.is_whitespace(), start) {
            (true, Some(begin)) => {
                words.push((begin, index));
                start = None;
            }
            (false, None) => start = Some(index),
            _ => {}
        }
    }
    if let Some(begin) = start {
        words.push((begin, text.len()));
    }
    if words.is_empty() {
        return Err((ErrorCode::InvalidFile, "the file holds no text".to_owned()));
    }
    let size = chunking.max_chunk_size_tokens as usize;
    let step = size - chunking.chunk_overlap_tokens as usize;
    let chunks = 1 + words.len().saturating_sub(size).div_ceil(step);
    if chunks > MAX_FILE_CHUNKS {
        return Err((
            ErrorCode::InvalidFile,
            format!("the file would need {chunks} chunks; at most {MAX_FILE_CHUNKS} are allowed"),
        ));
    }
    let mut ranges = Vec::with_capacity(chunks);
    let mut first = 0;
    loop {
        let last = (first + size).min(words.len());
        if let (Some(&(begin, _)), Some(&(_, end))) = (words.get(first), words.get(last - 1)) {
            ranges.push((begin, end));
        }
        if last == words.len() {
            break;
        }
        first += step;
    }
    Ok(ranges)
}

/// A validated attachment: the File, its name and how to index it.
struct Attachment {
    file_id: String,
    chunking: StaticChunking,
    attributes: Attributes,
}

/// Validate the requested files (each once).
fn resolve_files(files: &[CreateVectorStoreFile]) -> Result<Vec<Attachment>> {
    let mut seen = HashSet::new();
    files
        .iter()
        .map(|file| {
            if !seen.insert(file.file_id.as_str()) {
                return Err(invalid(format!("file {} is listed twice", file.file_id)));
            }
            let attributes = file.attributes.clone().unwrap_or_default();
            validate_attributes(&attributes)?;
            Ok(Attachment {
                file_id: file.file_id.clone(),
                chunking: file.chunking_strategy.unwrap_or_default().resolve()?,
                attributes,
            })
        })
        .collect()
}

/// Attach (or re-attach) files to a store inside `transaction`.
fn attach(
    transaction: &Transaction<'_>,
    store_id: &str,
    batch_id: Option<&str>,
    attachments: &[Attachment],
    at: i64,
) -> Result<()> {
    let now = at;
    for attachment in attachments {
        let filename: Option<String> = transaction
            .query_row(
                "SELECT filename FROM files
                 WHERE id = ?1 AND (expires_at IS NULL OR expires_at > ?2)",
                params![attachment.file_id, now],
                |row| row.get(0),
            )
            .optional()?;
        let filename = filename
            .ok_or_else(|| Error::NotFound(format!("file {} not found", attachment.file_id)))?;
        detach(transaction, store_id, &attachment.file_id)?;
        transaction.execute(
            "INSERT INTO vector_store_files (attach_id, store_id, file_id, batch_id, filename,
                                             created_at, status, max_chunk_tokens,
                                             chunk_overlap_tokens, attributes)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'in_progress', ?7, ?8, ?9)",
            params![
                new_id(ATTACH_PREFIX)?,
                store_id,
                attachment.file_id,
                batch_id,
                filename,
                at,
                attachment.chunking.max_chunk_size_tokens,
                attachment.chunking.chunk_overlap_tokens,
                serde_json::to_string(&attachment.attributes)?,
            ],
        )?;
    }
    let files: i64 = transaction.query_row(
        "SELECT COUNT(*) FROM vector_store_files WHERE store_id = ?1",
        [store_id],
        |row| row.get(0),
    )?;
    if unsigned(files)? > count(MAX_STORE_FILES) {
        return Err(invalid(format!(
            "a vector store holds at most {MAX_STORE_FILES} files"
        )));
    }
    Ok(())
}

/// Remove one file (its chunks cascade; a trigger queues its index
/// documents for removal); whether it was attached.
fn detach(transaction: &Transaction<'_>, store_id: &str, file_id: &str) -> Result<bool> {
    let removed = transaction.execute(
        "DELETE FROM vector_store_files WHERE store_id = ?1 AND file_id = ?2",
        params![store_id, file_id],
    )?;
    Ok(removed > 0)
}

/// Mark a store active now.
fn touch(connection: &Connection, store_id: &str) -> Result<()> {
    connection.execute(
        "UPDATE vector_stores SET last_active_at = ?2 WHERE id = ?1",
        params![store_id, signed(now()?)?],
    )?;
    Ok(())
}

/// The store, refusing an expired one.
fn writable_store(connection: &Connection, store_id: &str) -> Result<VectorStore> {
    let store = load_store(connection, store_id)?;
    if store.status == StoreStatus::Expired {
        return Err(Error::Conflict(format!(
            "vector store {store_id} has expired"
        )));
    }
    Ok(store)
}

/// A completed file of a store, as search sees it.
struct Searchable {
    attach_id: String,
    file_id: String,
    filename: String,
    attributes: Attributes,
}

/// The completed files of `store_id` that `filter` admits.
fn searchable_files(
    connection: &Connection,
    store_id: &str,
    filter: Option<&Filter>,
) -> Result<Vec<Searchable>> {
    let mut statement = connection.prepare(
        "SELECT attach_id, file_id, filename, attributes FROM vector_store_files
         WHERE store_id = ?1 AND status = 'completed'",
    )?;
    let rows = statement
        .query_map([store_id], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get::<_, String>(3)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<(String, String, String, String)>>>()?;
    let mut files = Vec::new();
    for (attach_id, file_id, filename, attributes) in rows {
        let attributes: Attributes = serde_json::from_str(&attributes)?;
        if filter.is_none_or(|filter| admits(filter, &attributes)) {
            files.push(Searchable {
                attach_id,
                file_id,
                filename,
                attributes,
            });
        }
    }
    Ok(files)
}

/// The oldest files still `in_progress`, at most `limit`.
fn next_pending(connection: &Connection, limit: usize) -> Result<Vec<Pending>> {
    let mut statement = connection.prepare(
        "SELECT attach_id, file_id, max_chunk_tokens, chunk_overlap_tokens
         FROM vector_store_files WHERE status = 'in_progress' ORDER BY seq LIMIT ?1",
    )?;
    let rows = statement
        .query_map([i64::try_from(limit).unwrap_or(i64::MAX)], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    rows.into_iter()
        .map(|(attach_id, file_id, max, overlap)| {
            Ok(Pending {
                attach_id,
                file_id,
                chunking: StaticChunking {
                    max_chunk_size_tokens: u32::try_from(max).map_err(|_| corrupt("chunk size"))?,
                    chunk_overlap_tokens: u32::try_from(overlap)
                        .map_err(|_| corrupt("chunk overlap"))?,
                },
            })
        })
        .collect()
}

/// Whether `filter` admits a file with `attributes`.
fn admits(filter: &Filter, attributes: &Attributes) -> bool {
    match filter {
        Filter::Compound { op, filters } => match op {
            Compound::And => filters.iter().all(|inner| admits(inner, attributes)),
            Compound::Or => filters.iter().any(|inner| admits(inner, attributes)),
        },
        Filter::Comparison { op, key, value } => {
            let actual = attributes.get(key);
            match op {
                Comparison::Eq => actual.is_some_and(|actual| same(actual, value)),
                Comparison::Ne => !actual.is_some_and(|actual| same(actual, value)),
                Comparison::In => actual.is_some_and(|actual| member(actual, value)),
                Comparison::Nin => !actual.is_some_and(|actual| member(actual, value)),
                Comparison::Gt | Comparison::Gte | Comparison::Lt | Comparison::Lte => {
                    let ordering = actual.and_then(|actual| order(actual, value));
                    ordering.is_some_and(|ordering| match op {
                        Comparison::Gt => ordering.is_gt(),
                        Comparison::Gte => ordering.is_ge(),
                        Comparison::Lt => ordering.is_lt(),
                        _ => ordering.is_le(),
                    })
                }
            }
        }
    }
}

fn same(actual: &Value, expected: &Value) -> bool {
    match (actual, expected) {
        (Value::Number(left), Value::Number(right)) => left
            .as_f64()
            .zip(right.as_f64())
            .is_some_and(|(l, r)| l.total_cmp(&r).is_eq()),
        (Value::String(left), Value::String(right)) => left == right,
        (Value::Bool(left), Value::Bool(right)) => left == right,
        _ => false,
    }
}

fn member(actual: &Value, set: &Value) -> bool {
    set.as_array()
        .is_some_and(|values| values.iter().any(|value| same(actual, value)))
}

fn order(actual: &Value, expected: &Value) -> Option<std::cmp::Ordering> {
    match (actual, expected) {
        (Value::Number(left), Value::Number(right)) => {
            Some(left.as_f64()?.total_cmp(&right.as_f64()?))
        }
        (Value::String(left), Value::String(right)) => Some(left.cmp(right)),
        _ => None,
    }
}

fn validate_filter(filter: &Filter) -> Result<()> {
    fn walk(filter: &Filter, depth: usize, comparisons: &mut usize) -> Result<()> {
        if depth > MAX_FILTER_DEPTH {
            return Err(invalid(format!(
                "filters nest at most {MAX_FILTER_DEPTH} levels"
            )));
        }
        match filter {
            Filter::Compound { filters, .. } => {
                if filters.is_empty() {
                    return Err(invalid("a compound filter needs at least one filter"));
                }
                filters
                    .iter()
                    .try_for_each(|inner| walk(inner, depth + 1, comparisons))
            }
            Filter::Comparison { op, key, value } => {
                *comparisons += 1;
                if *comparisons > MAX_FILTER_COMPARISONS {
                    return Err(invalid(format!(
                        "a filter holds at most {MAX_FILTER_COMPARISONS} comparisons"
                    )));
                }
                check_chars("filter key", key, MAX_ATTRIBUTE_KEY_CHARS)?;
                let scalar = |value: &Value| {
                    matches!(value, Value::String(_) | Value::Number(_) | Value::Bool(_))
                };
                let valid = match (op, value) {
                    (Comparison::In | Comparison::Nin, Value::Array(values)) => {
                        values.len() <= MAX_FILTER_VALUES
                            && values
                                .iter()
                                .all(|value| matches!(value, Value::String(_) | Value::Number(_)))
                    }
                    (Comparison::In | Comparison::Nin, _) => false,
                    (_, value) => scalar(value),
                };
                if valid {
                    Ok(())
                } else {
                    Err(invalid(format!(
                        "filter {key:?}: in and nin take an array of at most \
                         {MAX_FILTER_VALUES} strings or numbers; other comparisons take a \
                         string, number or boolean"
                    )))
                }
            }
        }
    }
    walk(filter, 1, &mut 0)
}

fn validate_attributes(attributes: &Attributes) -> Result<()> {
    if attributes.len() > MAX_ATTRIBUTES {
        return Err(invalid(format!(
            "attributes may hold at most {MAX_ATTRIBUTES} keys"
        )));
    }
    for (key, value) in attributes {
        check_chars("attribute key", key, MAX_ATTRIBUTE_KEY_CHARS)?;
        match value {
            Value::String(text) => check_chars("attribute value", text, MAX_ATTRIBUTE_VALUE_CHARS)?,
            Value::Number(_) | Value::Bool(_) => {}
            _ => {
                return Err(invalid(format!(
                    "attribute {key:?} must be a string, number or boolean"
                )));
            }
        }
    }
    Ok(())
}

fn validate_metadata(metadata: &Metadata) -> Result<()> {
    if metadata.len() > MAX_METADATA_PAIRS {
        return Err(invalid(format!(
            "metadata may hold at most {MAX_METADATA_PAIRS} pairs"
        )));
    }
    for (key, value) in metadata {
        check_chars("metadata key", key, MAX_METADATA_KEY_CHARS)?;
        check_chars("metadata value", value, MAX_METADATA_VALUE_CHARS)?;
    }
    Ok(())
}

fn validate_expiry(policy: ExpirationPolicy) -> Result<u32> {
    if (1..=MAX_EXPIRES_AFTER_DAYS).contains(&policy.days) {
        Ok(policy.days)
    } else {
        Err(invalid(format!(
            "expires_after.days must be between 1 and {MAX_EXPIRES_AFTER_DAYS}"
        )))
    }
}

fn check_chars(what: &str, text: &str, max: usize) -> Result<()> {
    if text.chars().count() > max {
        Err(invalid(format!("{what} may be at most {max} characters")))
    } else {
        Ok(())
    }
}

/// A cursor window over a `seq`-ordered collection (see [`ListQuery`]).
struct Window {
    limit: usize,
    low: i64,
    high: i64,
    /// Scan in ascending `seq` order.
    ascending: bool,
    /// The scan runs against the list order (only `before` was given).
    reversed: bool,
}

impl Window {
    fn new(query: &ListQuery, seq: impl Fn(&str) -> Result<i64>) -> Result<Self> {
        let limit = query.limit.unwrap_or(DEFAULT_LIST_LIMIT);
        if !(1..=MAX_LIST_LIMIT).contains(&limit) {
            return Err(invalid(format!(
                "limit must be between 1 and {MAX_LIST_LIMIT}"
            )));
        }
        let after = query.after.as_deref().map(&seq).transpose()?;
        let before = query.before.as_deref().map(&seq).transpose()?;
        let descending = query.order == ListOrder::Desc;
        // In list order, `after` bounds the start and `before` the end.
        let (low, high) = if descending {
            (before.unwrap_or(i64::MIN), after.unwrap_or(i64::MAX))
        } else {
            (after.unwrap_or(i64::MIN), before.unwrap_or(i64::MAX))
        };
        let reversed = before.is_some() && after.is_none();
        Ok(Self {
            limit: usize::try_from(limit).unwrap_or(usize::MAX),
            low,
            high,
            ascending: descending == reversed,
            reversed,
        })
    }

    /// Run `sql` (selecting one ID, with `seq > ?1 AND seq < ?2` and the
    /// `extra` parameters from `?3`) over the window, one row past the page.
    fn ids(
        &self,
        connection: &Connection,
        sql: &str,
        extra: &[&dyn rusqlite::ToSql],
    ) -> Result<Vec<String>> {
        let direction = if self.ascending { "ASC" } else { "DESC" };
        let limit = i64::try_from(self.limit)
            .unwrap_or(i64::MAX)
            .saturating_add(1);
        let sql = format!("{sql} ORDER BY seq {direction} LIMIT {limit}");
        let mut parameters: Vec<&dyn rusqlite::ToSql> = vec![&self.low, &self.high];
        parameters.extend_from_slice(extra);
        let mut statement = connection.prepare(&sql)?;
        let ids = statement
            .query_map(parameters.as_slice(), |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(ids)
    }

    /// The page in list order and whether more follow in the scan direction.
    fn finish(&self, mut ids: Vec<String>) -> (Vec<String>, bool) {
        let has_more = ids.len() > self.limit;
        ids.truncate(self.limit);
        if self.reversed {
            ids.reverse();
        }
        (ids, has_more)
    }
}

fn page<T>(data: Vec<T>, has_more: bool, id: impl Fn(&T) -> String) -> Page<T> {
    Page {
        first_id: data.first().map(&id),
        last_id: data.last().map(&id),
        data,
        has_more,
    }
}

fn list_files(
    connection: &Connection,
    store_id: &str,
    batch_id: Option<&str>,
    query: &ListQuery,
) -> Result<Page<VectorStoreFile>> {
    let seq = |file_id: &str| {
        connection
            .query_row(
                "SELECT seq FROM vector_store_files WHERE store_id = ?1 AND file_id = ?2",
                params![store_id, file_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .ok_or_else(|| file_not_found(store_id, file_id))
    };
    let window = Window::new(query, seq)?;
    let status = query.filter.map(FileStatus::as_str);
    let ids = window.ids(
        connection,
        "SELECT file_id FROM vector_store_files
         WHERE seq > ?1 AND seq < ?2 AND store_id = ?3 AND (?4 IS NULL OR batch_id = ?4)
           AND (?5 IS NULL OR status = ?5)",
        &[&store_id, &batch_id, &status],
    )?;
    let (ids, has_more) = window.finish(ids);
    let data = ids
        .iter()
        .map(|file_id| load_file(connection, store_id, file_id))
        .collect::<Result<Vec<_>>>()?;
    Ok(page(data, has_more, |file| file.id.clone()))
}

fn counts(connection: &Connection, column: &str, id: &str) -> Result<FileCounts> {
    let mut statement = connection.prepare(&format!(
        "SELECT status, COUNT(*) FROM vector_store_files WHERE {column} = ?1 GROUP BY status"
    ))?;
    let rows = statement
        .query_map([id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut counts = FileCounts::default();
    for (status, number) in rows {
        let number = unsigned(number)?;
        match FileStatus::parse(&status)? {
            FileStatus::InProgress => counts.in_progress = number,
            FileStatus::Completed => counts.completed = number,
            FileStatus::Failed => counts.failed = number,
            FileStatus::Cancelled => counts.cancelled = number,
        }
        counts.total += number;
    }
    Ok(counts)
}

fn load_store(connection: &Connection, id: &str) -> Result<VectorStore> {
    let row = connection
        .query_row(
            &format!("SELECT {STORE_COLUMNS} FROM vector_stores WHERE id = ?1"),
            [id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, Option<i64>>(5)?,
                ))
            },
        )
        .optional()?
        .ok_or_else(|| store_not_found(id))?;
    let (id, name, metadata, created_at, last_active_at, days) = row;
    let file_counts = counts(connection, "store_id", &id)?;
    let usage: i64 = connection.query_row(
        "SELECT COALESCE(SUM(usage_bytes), 0) FROM vector_store_files WHERE store_id = ?1",
        [&id],
        |row| row.get(0),
    )?;
    let last_active_at = unsigned(last_active_at)?;
    let days = days
        .map(|days| u32::try_from(days).map_err(|_| corrupt("expiry")))
        .transpose()?;
    let expires_at = days.map(|days| last_active_at.saturating_add(u64::from(days) * DAY_SECONDS));
    let status = if expires_at.is_some_and(|at| now().is_ok_and(|now| now >= at)) {
        StoreStatus::Expired
    } else if file_counts.in_progress > 0 {
        StoreStatus::InProgress
    } else {
        StoreStatus::Completed
    };
    Ok(VectorStore {
        created_at: unsigned(created_at)?,
        name,
        usage_bytes: unsigned(usage)?,
        file_counts,
        status,
        expires_after: days.map(|days| ExpirationPolicy {
            anchor: ExpirationAnchor::LastActiveAt,
            days,
        }),
        expires_at,
        last_active_at: Some(last_active_at),
        metadata: serde_json::from_str(&metadata)?,
        id,
    })
}

fn load_batch(connection: &Connection, store_id: &str, id: &str) -> Result<FileBatch> {
    let (created_at, cancelled): (i64, Option<i64>) = connection
        .query_row(
            "SELECT created_at, cancelled_at FROM vector_store_file_batches
             WHERE id = ?1 AND store_id = ?2",
            params![id, store_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?
        .ok_or_else(|| Error::NotFound(format!("file batch {id} not found in {store_id}")))?;
    let file_counts = counts(connection, "batch_id", id)?;
    let status = if cancelled.is_some() {
        BatchStatus::Cancelled
    } else if file_counts.in_progress > 0 {
        BatchStatus::InProgress
    } else if file_counts.total > 0 && file_counts.failed == file_counts.total {
        BatchStatus::Failed
    } else {
        BatchStatus::Completed
    };
    Ok(FileBatch {
        id: id.to_owned(),
        created_at: unsigned(created_at)?,
        vector_store_id: store_id.to_owned(),
        status,
        file_counts,
    })
}

struct RawFile {
    file_id: String,
    store_id: String,
    created_at: i64,
    status: String,
    error_code: Option<String>,
    error_message: Option<String>,
    max: i64,
    overlap: i64,
    attributes: String,
    usage: i64,
}

fn raw_file(row: &Row<'_>) -> rusqlite::Result<RawFile> {
    Ok(RawFile {
        file_id: row.get(0)?,
        store_id: row.get(1)?,
        created_at: row.get(2)?,
        status: row.get(3)?,
        error_code: row.get(4)?,
        error_message: row.get(5)?,
        max: row.get(6)?,
        overlap: row.get(7)?,
        attributes: row.get(8)?,
        usage: row.get(9)?,
    })
}

impl RawFile {
    fn into_file(self) -> Result<VectorStoreFile> {
        let last_error = match self.error_code {
            Some(code) => Some(LastError {
                code: ErrorCode::parse(&code)?,
                message: self.error_message.unwrap_or_default(),
            }),
            None => None,
        };
        Ok(VectorStoreFile {
            id: self.file_id,
            usage_bytes: unsigned(self.usage)?,
            created_at: unsigned(self.created_at)?,
            vector_store_id: self.store_id,
            status: FileStatus::parse(&self.status)?,
            last_error,
            chunking_strategy: ReportedChunking {
                config: StaticChunking {
                    max_chunk_size_tokens: u32::try_from(self.max)
                        .map_err(|_| corrupt("chunk size"))?,
                    chunk_overlap_tokens: u32::try_from(self.overlap)
                        .map_err(|_| corrupt("chunk overlap"))?,
                },
            },
            attributes: serde_json::from_str(&self.attributes)?,
        })
    }
}

fn load_file(connection: &Connection, store_id: &str, file_id: &str) -> Result<VectorStoreFile> {
    connection
        .query_row(
            &format!(
                "SELECT {FILE_COLUMNS} FROM vector_store_files WHERE store_id = ?1 AND file_id = ?2"
            ),
            params![store_id, file_id],
            raw_file,
        )
        .optional()?
        .ok_or_else(|| file_not_found(store_id, file_id))?
        .into_file()
}

/// The background indexer: one thread that indexes `in_progress` files when
/// notified, never polling.
///
/// It runs once when started (resuming files a stopped process left
/// `in_progress`), then once per [`Self::notify`] (coalesced). Each run
/// first drops the index documents of deleted attachments, then indexes
/// files until none is waiting. Closing it waits for the file being indexed;
/// a process killed meanwhile leaves that file `in_progress` for the next
/// start.
#[derive(Debug)]
pub struct VectorStoreIndexer {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Debug, Default)]
struct Shared {
    signals: Mutex<Signals>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct Signals {
    requested: u64,
    finished: u64,
    stopping: bool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Signals> {
        self.signals.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl VectorStoreIndexer {
    /// Start the indexer for `index`.
    pub fn start(index: Arc<VectorIndex>) -> Result<Self> {
        let shared = Arc::new(Shared::default());
        shared.lock().requested = 1;
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("vector-store-indexer".to_owned())
            .spawn(move || run_indexer(&index, &worker))?;
        Ok(Self {
            shared,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Ask for a run (after attaching or removing files).
    pub fn notify(&self) {
        self.shared.lock().requested += 1;
        self.shared.changed.notify_all();
    }

    /// Block until every run requested before this call has finished, or
    /// the indexer has stopped.
    pub fn wait_idle(&self) {
        let mut signals = self.shared.lock();
        let target = signals.requested;
        while signals.finished < target && !signals.stopping {
            signals = self
                .shared
                .changed
                .wait(signals)
                .unwrap_or_else(PoisonError::into_inner);
        }
        drop(signals);
    }

    /// Stop after the current file and wait for the thread to end.
    pub fn close(&self) {
        self.shared.lock().stopping = true;
        self.shared.changed.notify_all();
        let thread = self
            .thread
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(thread) = thread {
            let _ = thread.join();
        }
    }
}

impl Drop for VectorStoreIndexer {
    fn drop(&mut self) {
        self.close();
    }
}

fn run_indexer(index: &VectorIndex, shared: &Shared) {
    loop {
        let target = {
            let mut signals = shared.lock();
            while signals.finished == signals.requested && !signals.stopping {
                signals = shared
                    .changed
                    .wait(signals)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if signals.stopping {
                return;
            }
            signals.requested
        };
        if let Err(error) = index.collect_garbage() {
            // Still queued; the next run or start retries.
            eprintln!(
                "warning: removing deleted vector store files from the index failed: {error}"
            );
        }
        while !shared.lock().stopping {
            match index.index_next() {
                Ok(0) => break,
                Ok(_) => {}
                Err(error) => {
                    // Left in progress; the next notification or start retries.
                    eprintln!("warning: vector store indexing stopped: {error}");
                    break;
                }
            }
        }
        let mut signals = shared.lock();
        signals.finished = target;
        drop(signals);
        shared.changed.notify_all();
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidArgument(message.into())
}

fn store_not_found(id: &str) -> Error {
    Error::NotFound(format!("vector store {id} not found"))
}

fn file_not_found(store_id: &str, file_id: &str) -> Error {
    Error::NotFound(format!(
        "file {file_id} not found in vector store {store_id}"
    ))
}

fn corrupt(what: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("stored {what} is invalid"),
    ))
}

fn signed(value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| corrupt("integer"))
}

fn unsigned(value: i64) -> Result<u64> {
    u64::try_from(value).map_err(|_| corrupt("integer"))
}

fn count(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}
