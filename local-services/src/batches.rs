//! Batches: durable, `OpenAI`-compatible batch jobs over one local model.
//!
//! A batch reads a `purpose=batch` JSONL input file (at most
//! [`MAX_BATCH_FILE_BYTES`](crate::files::MAX_BATCH_FILE_BYTES) and
//! [`MAX_LINES`] requests). The whole file is validated before anything is
//! stored: every line is `{custom_id, method: "POST", url, body}` with a
//! unique `custom_id`, the batch's own `url`, one shared `body.model`, and
//! neither `stream` nor `background` set. A refused file stores nothing, so a
//! batch exists only once it is valid; it is created `in_progress` (there is
//! no asynchronous `validating` phase to report).
//!
//! Only [`Endpoint`]s the server generates for locally are accepted; every
//! other endpoint is refused before anything is queued. No remote API is ever
//! called.
//!
//! # Execution protocol
//!
//! The queue lives in the store's database, one row per input line. A worker
//! (in this process, another one, or a later run) takes the batch's
//! [`BatchLease`], an exclusive advisory lock on an owner-only file under
//! `<root>/batch-locks/` that is held for as long as the lease value lives
//! and is released by the operating system when the process dies. While it
//! holds it the worker repeats:
//!
//! 1. [`Store::next_batch_line`]: the lowest pending line, or `Finish`;
//! 2. generate the line;
//! 3. [`Store::settle_batch_line`]: one transaction records the result and
//!    the counter, and only if the line is still pending.
//!
//! Then [`Store::finish_batch`] writes the output and error JSONL files and the
//! terminal status. Each line's output record ID is fixed when the batch is
//! created, so a re-run produces the same record ID. Settlement is exactly
//! once; generation is not: a crash between generating and settling a line
//! generates it again on resume.
//!
//! # Result files
//!
//! Once finishing starts the lines are frozen (no more settlements), and the
//! result files have reserved, deterministic IDs (`file_<batch id>_output`
//! and `file_<batch id>_error`) with the internal `batch_output` purpose, at
//! most [`MAX_BATCH_OUTPUT_BYTES`](crate::files::MAX_BATCH_OUTPUT_BYTES) each,
//! expiring [`BATCH_OUTPUT_EXPIRES_AFTER_SECONDS`](crate::files::BATCH_OUTPUT_EXPIRES_AFTER_SECONDS)
//! after creation. Each is streamed
//! from the database and committed idempotently, so a finish interrupted
//! after committing a file reuses that file on resume instead of writing a
//! duplicate; one deleted or expired in between is not recreated, and the
//! batch still names it. A result file that cannot be written fails the
//! batch with `output_failed`.
//!
//! Every line names the same `body.model`, and it must be the installed model
//! the caller names when creating the batch; a batch for any other model is
//! refused before anything is stored.
//!
//! Expiry (`completion_window` is `24h`) and cancellation are observed at line
//! boundaries: unfinished lines of an expired batch are failed with
//! `batch_expired`, while those of a cancelled batch are left out, and in both
//! cases every line settled before is in the output.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead as _, BufReader, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use rusqlite::{Connection, OptionalExtension as _, Row, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::conversations::{
    MAX_METADATA_KEY_CHARS, MAX_METADATA_PAIRS, MAX_METADATA_VALUE_CHARS, Metadata, Page,
};
use crate::files::{FilePurpose, InternalFile};
use crate::{Error, Result, Store, migrate, new_id, now};

/// Most requests one batch may hold.
pub const MAX_LINES: usize = 50_000;
/// Largest single input line, the same as the HTTP server's request limit.
pub const MAX_LINE_BYTES: usize = 2 << 20;
/// Longest `custom_id`, in characters.
pub const MAX_CUSTOM_ID_CHARS: usize = 512;
/// The only completion window.
pub const COMPLETION_WINDOW: &str = "24h";
/// [`COMPLETION_WINDOW`] in seconds.
pub const COMPLETION_WINDOW_SECONDS: u64 = 86_400;
/// Page size when [`ListBatches::limit`] is absent.
pub const DEFAULT_LIST_LIMIT: u32 = 20;
/// Largest page size.
pub const MAX_LIST_LIMIT: u32 = 100;

const ID_PREFIX: &str = "batch_";
const RECORD_PREFIX: &str = "batch_req_";
const LOCK_DIR: &str = "batch-locks";
const MAX_ID_TAIL: usize = 64;
/// Records read per query while streaming a result file.
const RECORD_PAGE: i64 = 256;
const EXPIRED_MESSAGE: &str =
    "this request was not executed before the batch's completion window expired";

/// Version 1: batches and their lines. `ending` is the terminal status a
/// finalizing or cancelling batch is moving to.
const MIGRATIONS: &[&str] = &["
CREATE TABLE batches (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    endpoint TEXT NOT NULL,
    model TEXT NOT NULL,
    input_file_id TEXT NOT NULL,
    completion_window TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('in_progress', 'finalizing', 'completed', 'failed',
                                           'expired', 'cancelling', 'cancelled')),
    ending TEXT,
    metadata TEXT NOT NULL,
    errors TEXT,
    output_file_id TEXT,
    error_file_id TEXT,
    created_at INTEGER NOT NULL,
    in_progress_at INTEGER,
    expires_at INTEGER NOT NULL,
    finalizing_at INTEGER,
    completed_at INTEGER,
    failed_at INTEGER,
    expired_at INTEGER,
    cancelling_at INTEGER,
    cancelled_at INTEGER,
    total INTEGER NOT NULL,
    completed INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0
) STRICT;
CREATE INDEX batches_status ON batches (status, seq);
CREATE TABLE batch_lines (
    batch_id TEXT NOT NULL REFERENCES batches (id),
    line INTEGER NOT NULL,
    custom_id TEXT NOT NULL,
    record_id TEXT NOT NULL UNIQUE,
    body TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'succeeded', 'failed')),
    result TEXT,
    PRIMARY KEY (batch_id, line),
    UNIQUE (batch_id, custom_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX batch_lines_state ON batch_lines (batch_id, state, line);
"];

const COLUMNS: &str = "id, endpoint, model, input_file_id, completion_window, status, metadata, \
     errors, output_file_id, error_file_id, created_at, in_progress_at, expires_at, \
     finalizing_at, completed_at, failed_at, expired_at, cancelling_at, cancelled_at, total, \
     completed, failed";

/// Create or upgrade the batch tables.
pub(crate) fn initialize(connection: &mut Connection) -> Result<()> {
    migrate(connection, "batches", MIGRATIONS)
}

/// An endpoint a batch line may target: exactly the local generation APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Endpoint {
    #[serde(rename = "/v1/responses")]
    Responses,
    #[serde(rename = "/v1/chat/completions")]
    ChatCompletions,
    #[serde(rename = "/v1/completions")]
    Completions,
}

impl Endpoint {
    /// The URL path, as lines and the batch name it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Responses => "/v1/responses",
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
        }
    }
}

impl FromStr for Endpoint {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "/v1/responses" => Ok(Self::Responses),
            "/v1/chat/completions" => Ok(Self::ChatCompletions),
            "/v1/completions" => Ok(Self::Completions),
            other => Err(invalid(format!(
                "endpoint {other:?} is not supported; batches run only /v1/responses, \
                 /v1/chat/completions and /v1/completions on the local model"
            ))),
        }
    }
}

/// A batch's lifecycle state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    InProgress,
    Finalizing,
    Completed,
    Failed,
    Expired,
    Cancelling,
    Cancelled,
}

impl BatchStatus {
    /// The wire and stored name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Finalizing => "finalizing",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Expired => "expired",
            Self::Cancelling => "cancelling",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether nothing more will happen to the batch.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Expired | Self::Cancelled
        )
    }

    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "in_progress" => Self::InProgress,
            "finalizing" => Self::Finalizing,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "expired" => Self::Expired,
            "cancelling" => Self::Cancelling,
            "cancelled" => Self::Cancelled,
            _ => return Err(corrupt("batch status")),
        })
    }

    /// The column recording when the batch reached this status.
    const fn time_column(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress_at",
            Self::Finalizing => "finalizing_at",
            Self::Completed => "completed_at",
            Self::Failed => "failed_at",
            Self::Expired => "expired_at",
            Self::Cancelling => "cancelling_at",
            Self::Cancelled => "cancelled_at",
        }
    }
}

/// Line counts. `completed` and `failed` count settled lines only, so a
/// cancelled batch may have fewer than `total`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestCounts {
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
}

/// One batch-level error.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchError {
    pub code: String,
    pub message: String,
    pub param: Option<String>,
    pub line: Option<u64>,
}

/// Batch-level errors, as the public `list` object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "list")]
pub struct BatchErrors {
    pub data: Vec<BatchError>,
}

/// A batch, serialized as the public `batch` object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "batch")]
pub struct Batch {
    pub id: String,
    pub endpoint: Endpoint,
    /// The `body.model` every line names; the server's one model runs them.
    pub model: String,
    pub errors: Option<BatchErrors>,
    pub input_file_id: String,
    pub completion_window: String,
    pub status: BatchStatus,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub created_at: u64,
    pub in_progress_at: Option<u64>,
    pub expires_at: u64,
    pub finalizing_at: Option<u64>,
    pub completed_at: Option<u64>,
    pub failed_at: Option<u64>,
    pub expired_at: Option<u64>,
    pub cancelling_at: Option<u64>,
    pub cancelled_at: Option<u64>,
    pub request_counts: RequestCounts,
    pub metadata: Metadata,
}

/// Input for [`Store::create_batch`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateBatch {
    pub input_file_id: String,
    /// One of the [`Endpoint`] paths; anything else is refused.
    pub endpoint: String,
    /// Must be [`COMPLETION_WINDOW`].
    pub completion_window: String,
    pub metadata: Metadata,
}

/// Query for [`Store::list_batches`]: newest first.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ListBatches {
    /// Return only batches created before this batch ID.
    pub after: Option<String>,
    /// 1 to [`MAX_LIST_LIMIT`]; [`DEFAULT_LIST_LIMIT`] when absent.
    pub limit: Option<u32>,
}

/// Exclusive ownership of one batch's execution (see the module docs).
///
/// Dropping it releases the lock; so does the death of the process.
#[derive(Debug)]
pub struct BatchLease {
    id: String,
    /// The root of the store that granted it; other stores refuse it.
    root: Arc<PathBuf>,
    _file: File,
}

impl BatchLease {
    /// The batch this lease owns.
    #[must_use]
    pub const fn batch_id(&self) -> &str {
        self.id.as_str()
    }
}

/// What a lease holder does next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NextLine {
    /// Generate this line, then settle it.
    Line(PendingLine),
    /// No line is to run (all settled, cancelled or expired): call
    /// [`Store::finish_batch`].
    Finish,
    /// The batch is already terminal.
    Done,
}

/// A line still to be generated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingLine {
    /// Zero-based position in the input file.
    pub line: u64,
    pub custom_id: String,
    /// The stable output record ID of this line.
    pub record_id: String,
    pub endpoint: Endpoint,
    /// The request body as given (with `model`).
    pub body: Map<String, Value>,
    /// When the batch expires; generation past it is wasted.
    pub expires_at: u64,
}

/// The result of one line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LineOutcome {
    /// What the endpoint answered: a `2xx` counts as completed and goes to the
    /// output file, anything else as failed and goes to the error file.
    Response { status_code: u16, body: Value },
    /// The line produced no response at all (for example `batch_expired`).
    Error { code: String, message: String },
}

impl Store {
    /// Validate the whole input file and create an `in_progress` batch with
    /// one pending line per request, atomically. Nothing is stored when any
    /// line is refused.
    ///
    /// Every line must name `installed_model` (the model that will run the
    /// lines) as its `body.model`; any other model is refused as invalid
    /// before anything is stored.
    pub fn create_batch(&self, request: &CreateBatch, installed_model: &str) -> Result<Batch> {
        let endpoint: Endpoint = request.endpoint.parse()?;
        if request.completion_window != COMPLETION_WINDOW {
            return Err(invalid(format!(
                "completion_window must be {COMPLETION_WINDOW:?}"
            )));
        }
        validate_metadata(&request.metadata)?;
        let content =
            self.open_file_content(&request.input_file_id)
                .map_err(|error| match error {
                    Error::NotFound(message) => invalid(format!("input_file_id: {message}")),
                    other => other,
                })?;
        if content.file.purpose != FilePurpose::Batch {
            return Err(invalid(format!(
                "input file {} has purpose {}; batches read only purpose=batch files",
                content.file.id,
                content.file.purpose.as_str()
            )));
        }
        let (model, lines) = parse_input(content.content, endpoint)?;
        if model != installed_model {
            return Err(invalid(format!(
                "body.model {model:?} is not the installed model {installed_model:?}"
            )));
        }
        let id = new_id(ID_PREFIX)?;
        let created_at = now()?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "INSERT INTO batches (id, endpoint, model, input_file_id, completion_window, status,
                                  metadata, created_at, in_progress_at, expires_at, total)
             VALUES (?1, ?2, ?3, ?4, ?5, 'in_progress', ?6, ?7, ?7, ?8, ?9)",
            params![
                id,
                endpoint.as_str(),
                model,
                request.input_file_id,
                COMPLETION_WINDOW,
                serde_json::to_string(&request.metadata)?,
                signed(created_at)?,
                signed(created_at.saturating_add(COMPLETION_WINDOW_SECONDS))?,
                signed(count(lines.len()))?,
            ],
        )?;
        {
            let mut insert = transaction.prepare(
                "INSERT INTO batch_lines (batch_id, line, custom_id, record_id, body, state)
                 VALUES (?1, ?2, ?3, ?4, ?5, 'pending')",
            )?;
            for (index, line) in lines.iter().enumerate() {
                insert.execute(params![
                    id,
                    signed(count(index))?,
                    line.custom_id,
                    new_id(RECORD_PREFIX)?,
                    line.body
                ])?;
            }
        }
        transaction.commit()?;
        load(&connection, &id)
    }

    /// The batch `id`.
    pub fn get_batch(&self, id: &str) -> Result<Batch> {
        load(&self.connection()?, id)
    }

    /// Batches, newest first.
    pub fn list_batches(&self, query: &ListBatches) -> Result<Page<Batch>> {
        let limit = query.limit.unwrap_or(DEFAULT_LIST_LIMIT);
        if !(1..=MAX_LIST_LIMIT).contains(&limit) {
            return Err(invalid(format!(
                "limit must be between 1 and {MAX_LIST_LIMIT}"
            )));
        }
        let connection = self.connection()?;
        let cursor = match &query.after {
            Some(after) => connection
                .query_row("SELECT seq FROM batches WHERE id = ?1", [after], |row| {
                    row.get::<_, i64>(0)
                })
                .optional()?
                .ok_or_else(|| not_found(after))?,
            None => i64::MAX,
        };
        let mut statement = connection.prepare(&format!(
            "SELECT {COLUMNS} FROM batches WHERE seq < ?1 ORDER BY seq DESC LIMIT ?2"
        ))?;
        let rows = statement
            .query_map(params![cursor, i64::from(limit) + 1], raw)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let page = usize::try_from(limit).unwrap_or(usize::MAX);
        let has_more = rows.len() > page;
        let data = rows
            .into_iter()
            .take(page)
            .map(Raw::into_batch)
            .collect::<Result<Vec<_>>>()?;
        Ok(Page {
            first_id: data.first().map(|batch| batch.id.clone()),
            last_id: data.last().map(|batch| batch.id.clone()),
            data,
            has_more,
        })
    }

    /// Ask an `in_progress` batch to stop: it becomes `cancelling`, and its
    /// worker finishes it as `cancelled` at the next line boundary. A batch
    /// already cancelling or cancelled is returned unchanged; any other
    /// terminal or finalizing batch is a [`Error::Conflict`].
    pub fn cancel_batch(&self, id: &str) -> Result<Batch> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let batch = load(&transaction, id)?;
        match batch.status {
            BatchStatus::InProgress => {
                transaction.execute(
                    "UPDATE batches SET status = 'cancelling', cancelling_at = ?2 WHERE id = ?1",
                    params![id, signed(now()?)?],
                )?;
            }
            BatchStatus::Cancelling | BatchStatus::Cancelled => {}
            other => {
                return Err(Error::Conflict(format!(
                    "batch {id} is {}; only an in-progress batch can be cancelled",
                    other.as_str()
                )));
            }
        }
        transaction.commit()?;
        load(&connection, id)
    }

    /// Batches a worker still has to run or finish, oldest first.
    pub fn active_batches(&self) -> Result<Vec<String>> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            "SELECT id FROM batches WHERE status IN ('in_progress', 'cancelling', 'finalizing')
             ORDER BY seq",
        )?;
        let ids = statement
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(ids)
    }

    /// Take the execution lease of batch `id` without waiting; `None` when
    /// another holder (in this process or another) has it.
    pub fn lease_batch(&self, id: &str) -> Result<Option<BatchLease>> {
        validate_id(id)?;
        load(&self.connection()?, id)?;
        let dir = self.root().join(LOCK_DIR);
        private_dir(&dir)?;
        let file = private_file(&dir.join(format!("{id}.lock")))?;
        match file.try_lock() {
            Ok(()) => Ok(Some(BatchLease {
                id: id.to_owned(),
                root: Arc::clone(&self.root),
                _file: file,
            })),
            Err(fs::TryLockError::WouldBlock) => Ok(None),
            Err(fs::TryLockError::Error(error)) => Err(error.into()),
        }
    }

    /// The next thing the holder of `lease` should do.
    pub fn next_batch_line(&self, lease: &BatchLease) -> Result<NextLine> {
        self.check_lease(lease)?;
        let connection = self.connection()?;
        let batch = load(&connection, &lease.id)?;
        match batch.status {
            BatchStatus::InProgress if now()? < batch.expires_at => {}
            BatchStatus::InProgress | BatchStatus::Cancelling | BatchStatus::Finalizing => {
                return Ok(NextLine::Finish);
            }
            _ => return Ok(NextLine::Done),
        }
        let row = connection
            .query_row(
                "SELECT line, custom_id, record_id, body FROM batch_lines
                 WHERE batch_id = ?1 AND state = 'pending' ORDER BY line LIMIT 1",
                [&lease.id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                },
            )
            .optional()?;
        let Some((line, custom_id, record_id, body)) = row else {
            return Ok(NextLine::Finish);
        };
        Ok(NextLine::Line(PendingLine {
            line: unsigned(line)?,
            custom_id,
            record_id,
            endpoint: batch.endpoint,
            body: serde_json::from_str(&body)?,
            expires_at: batch.expires_at,
        }))
    }

    /// Record `outcome` for pending `line` and count it, in one transaction.
    ///
    /// `Ok(false)` when the line was already settled: nothing changes, so a
    /// repeated settlement never double-counts. Settling is allowed while the
    /// batch is `in_progress` or `cancelling`, until [`Store::finish_batch`]
    /// starts: from then on the lines are frozen.
    pub fn settle_batch_line(
        &self,
        lease: &BatchLease,
        line: u64,
        outcome: &LineOutcome,
    ) -> Result<bool> {
        self.check_lease(lease)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let status = load(&transaction, &lease.id)?.status;
        let finishing = transaction
            .query_row(
                "SELECT ending FROM batches WHERE id = ?1",
                [&lease.id],
                |row| row.get::<_, Option<String>>(0),
            )?
            .is_some();
        if finishing || !matches!(status, BatchStatus::InProgress | BatchStatus::Cancelling) {
            return Err(Error::Conflict(format!(
                "batch {} is {}; its lines can no longer be settled",
                lease.id,
                status.as_str()
            )));
        }
        let index = signed(line)?;
        let row: Option<(String, String, String)> = transaction
            .query_row(
                "SELECT custom_id, record_id, state FROM batch_lines
                 WHERE batch_id = ?1 AND line = ?2",
                params![lease.id, index],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let (custom_id, record_id, state) =
            row.ok_or_else(|| Error::NotFound(format!("batch {} has no line {line}", lease.id)))?;
        if state != "pending" {
            return Ok(false);
        }
        let (succeeded, result) = record(&record_id, &custom_id, outcome);
        transaction.execute(
            "UPDATE batch_lines SET state = ?3, result = ?4
             WHERE batch_id = ?1 AND line = ?2 AND state = 'pending'",
            params![
                lease.id,
                index,
                if succeeded { "succeeded" } else { "failed" },
                result
            ],
        )?;
        transaction.execute(
            if succeeded {
                "UPDATE batches SET completed = completed + 1 WHERE id = ?1"
            } else {
                "UPDATE batches SET failed = failed + 1 WHERE id = ?1"
            },
            [&lease.id],
        )?;
        transaction.commit()?;
        Ok(true)
    }

    /// End the batch: fail the unfinished lines of an expired batch, write
    /// the output and error files from the settled lines in input order, and
    /// record the terminal status. Safe to repeat after a crash; an already
    /// terminal batch is returned unchanged.
    ///
    /// The result files get reserved IDs, so a repeat reuses a file an
    /// interrupted attempt already committed (see the module docs). A file
    /// that cannot be written fails the batch (`output_failed`).
    pub fn finish_batch(&self, lease: &BatchLease) -> Result<Batch> {
        self.check_lease(lease)?;
        let Some(ending) = self.begin_finish(lease)? else {
            return self.get_batch(&lease.id);
        };
        let written = self.write_results(&lease.id);
        let connection = self.connection()?;
        let at = signed(now()?)?;
        match written {
            Ok((output, error)) => {
                // Matches nothing only if the batch already ended; the result
                // files carry reserved IDs, so they are then that ending's own.
                connection.execute(
                    &format!(
                        "UPDATE batches SET status = ?2, {} = ?3, output_file_id = ?4,
                                            error_file_id = ?5
                         WHERE id = ?1 AND status IN ('finalizing', 'cancelling')",
                        ending.time_column()
                    ),
                    params![lease.id, ending.as_str(), at, output, error],
                )?;
            }
            Err(error) => {
                let errors = BatchErrors {
                    data: vec![BatchError {
                        code: "output_failed".to_owned(),
                        message: format!("the batch result files could not be written: {error}"),
                        param: None,
                        line: None,
                    }],
                };
                connection.execute(
                    "UPDATE batches SET status = 'failed', failed_at = ?2, errors = ?3
                     WHERE id = ?1 AND status IN ('finalizing', 'cancelling')",
                    params![lease.id, at, serde_json::to_string(&errors)?],
                )?;
            }
        }
        load(&connection, &lease.id)
    }

    /// Move the batch to finalizing (or keep it cancelling) and return the
    /// terminal status it is ending in; `None` when it already ended.
    fn begin_finish(&self, lease: &BatchLease) -> Result<Option<BatchStatus>> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let batch = load(&transaction, &lease.id)?;
        let ending = match batch.status {
            status if status.is_terminal() => return Ok(None),
            BatchStatus::Cancelling => {
                // Freeze the lines: settlement stops once finishing starts.
                transaction.execute(
                    "UPDATE batches SET ending = 'cancelled' WHERE id = ?1 AND ending IS NULL",
                    [&lease.id],
                )?;
                BatchStatus::Cancelled
            }
            BatchStatus::Finalizing => {
                let ending: Option<String> = transaction.query_row(
                    "SELECT ending FROM batches WHERE id = ?1",
                    [&lease.id],
                    |row| row.get(0),
                )?;
                ending
                    .as_deref()
                    .map_or(Ok(BatchStatus::Completed), BatchStatus::parse)?
            }
            _ => {
                let pending: i64 = transaction.query_row(
                    "SELECT COUNT(*) FROM batch_lines WHERE batch_id = ?1 AND state = 'pending'",
                    [&lease.id],
                    |row| row.get(0),
                )?;
                let at = now()?;
                let ending = if at >= batch.expires_at {
                    BatchStatus::Expired
                } else if pending == 0 {
                    BatchStatus::Completed
                } else {
                    return Err(Error::Conflict(format!(
                        "batch {} still has {pending} pending requests",
                        lease.id
                    )));
                };
                if ending == BatchStatus::Expired {
                    expire_pending(&transaction, &lease.id)?;
                }
                transaction.execute(
                    "UPDATE batches SET status = 'finalizing', finalizing_at = ?2, ending = ?3
                     WHERE id = ?1",
                    params![lease.id, signed(at)?, ending.as_str()],
                )?;
                ending
            }
        };
        transaction.commit()?;
        Ok(Some(ending))
    }

    /// Refuse a lease another store (another root) granted.
    fn check_lease(&self, lease: &BatchLease) -> Result<()> {
        if Arc::ptr_eq(&self.root, &lease.root) || self.root == lease.root {
            Ok(())
        } else {
            Err(invalid(format!(
                "the lease on batch {} was granted by another store",
                lease.id
            )))
        }
    }

    /// Write the output and error files; on failure none is left behind.
    fn write_results(&self, id: &str) -> Result<(Option<String>, Option<String>)> {
        let output = self.write_records(id, "succeeded", "output")?;
        match self.write_records(id, "failed", "error") {
            Ok(error) => Ok((output, error)),
            Err(error) => {
                if let Some(file) = &output {
                    let _ = self.delete_file(file);
                }
                Err(error)
            }
        }
    }

    /// The result file `file_<id>_<kind>` of the lines in `state`, streamed
    /// from the database, or the one an earlier attempt committed under that
    /// reserved ID (even if since deleted or expired); `None` when no line is
    /// in `state`.
    fn write_records(&self, id: &str, state: &'static str, kind: &str) -> Result<Option<String>> {
        let connection = self.connection()?;
        let any: bool = connection.query_row(
            "SELECT EXISTS (SELECT 1 FROM batch_lines WHERE batch_id = ?1 AND state = ?2)",
            params![id, state],
            |row| row.get(0),
        )?;
        if !any {
            return Ok(None);
        }
        let records = Records {
            connection,
            batch: id.to_owned(),
            state,
            after: -1,
            buffer: Vec::new(),
            position: 0,
            done: false,
        };
        let file_id = format!("file_{id}_{kind}");
        let file = self.create_internal_file(
            &file_id,
            &format!("{id}_{kind}.jsonl"),
            FilePurpose::BatchOutput,
            records,
        )?;
        Ok(Some(match file {
            InternalFile::Live(file) => file.id,
            InternalFile::Gone => file_id,
        }))
    }
}

/// Streams the result records of one state as JSONL, a page at a time.
struct Records {
    connection: Connection,
    batch: String,
    state: &'static str,
    after: i64,
    buffer: Vec<u8>,
    position: usize,
    done: bool,
}

impl Records {
    fn refill(&mut self) -> rusqlite::Result<()> {
        self.buffer.clear();
        self.position = 0;
        let mut statement = self.connection.prepare(
            "SELECT line, result FROM batch_lines
             WHERE batch_id = ?1 AND state = ?2 AND line > ?3 ORDER BY line LIMIT ?4",
        )?;
        let rows = statement.query_map(
            params![self.batch, self.state, self.after, RECORD_PAGE],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?)),
        )?;
        let mut any = false;
        for row in rows {
            let (line, result) = row?;
            any = true;
            self.after = line;
            self.buffer
                .extend_from_slice(result.unwrap_or_default().as_bytes());
            self.buffer.push(b'\n');
        }
        self.done = !any;
        Ok(())
    }
}

impl Read for Records {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        while self.position >= self.buffer.len() {
            if self.done {
                return Ok(0);
            }
            self.refill().map_err(io::Error::other)?;
        }
        let available = self.buffer.get(self.position..).unwrap_or_default();
        let copied = available.len().min(out.len());
        if let (Some(target), Some(source)) = (out.get_mut(..copied), available.get(..copied)) {
            target.copy_from_slice(source);
        }
        self.position += copied;
        Ok(copied)
    }
}

/// A validated input line.
struct InputLine {
    custom_id: String,
    body: String,
}

/// Validate every line of an input file; returns the one model and the lines.
fn parse_input(content: impl Read, endpoint: Endpoint) -> Result<(String, Vec<InputLine>)> {
    let mut reader = BufReader::new(content);
    let mut lines = Vec::new();
    let mut seen = HashSet::new();
    let mut model = None;
    let mut buffer = Vec::new();
    let mut number = 0_usize;
    let mut blank = None;
    loop {
        buffer.clear();
        if reader.read_until(b'\n', &mut buffer)? == 0 {
            break;
        }
        number += 1;
        let text = buffer.strip_suffix(b"\n").unwrap_or(buffer.as_slice());
        let text = text.strip_suffix(b"\r").unwrap_or(text);
        if text.iter().all(u8::is_ascii_whitespace) {
            blank.get_or_insert(number);
            continue;
        }
        if let Some(blank) = blank {
            return Err(line_error(blank, "blank lines are not allowed"));
        }
        if lines.len() == MAX_LINES {
            return Err(invalid(format!(
                "a batch may hold at most {MAX_LINES} requests"
            )));
        }
        if text.len() > MAX_LINE_BYTES {
            return Err(line_error(
                number,
                &format!("a request may be at most {MAX_LINE_BYTES} bytes"),
            ));
        }
        let line = parse_line(text, endpoint, &mut model)
            .map_err(|message| line_error(number, &message))?;
        if !seen.insert(line.custom_id.clone()) {
            return Err(line_error(
                number,
                &format!("duplicate custom_id {:?}", line.custom_id),
            ));
        }
        lines.push(line);
    }
    let model = model.ok_or_else(|| invalid("the input file holds no requests"))?;
    Ok((model, lines))
}

fn parse_line(
    text: &[u8],
    endpoint: Endpoint,
    model: &mut Option<String>,
) -> std::result::Result<InputLine, String> {
    let value: Value =
        serde_json::from_slice(text).map_err(|error| format!("invalid JSON: {error}"))?;
    let Value::Object(object) = value else {
        return Err("each line must be a JSON object".to_owned());
    };
    if let Some(key) = object
        .keys()
        .find(|key| !matches!(key.as_str(), "custom_id" | "method" | "url" | "body"))
    {
        return Err(format!("field {key:?} is not supported"));
    }
    let custom_id = match object.get("custom_id") {
        Some(Value::String(id)) if !id.is_empty() && id.chars().count() <= MAX_CUSTOM_ID_CHARS => {
            id.clone()
        }
        _ => {
            return Err(format!(
                "custom_id must be a string of 1 to {MAX_CUSTOM_ID_CHARS} characters"
            ));
        }
    };
    if object.get("method").and_then(Value::as_str) != Some("POST") {
        return Err("method must be \"POST\"".to_owned());
    }
    match object.get("url").and_then(Value::as_str) {
        Some(url) if url == endpoint.as_str() => {}
        Some(url) => {
            return Err(format!(
                "url {url:?} does not match the batch endpoint {}",
                endpoint.as_str()
            ));
        }
        None => return Err("url must be a string".to_owned()),
    }
    let Some(Value::Object(body)) = object.get("body") else {
        return Err("body must be a JSON object".to_owned());
    };
    let Some(name) = body
        .get("model")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
    else {
        return Err("body.model must be a non-empty string".to_owned());
    };
    match model.as_deref() {
        Some(expected) if expected != name => {
            return Err(format!(
                "body.model {name:?} differs from {expected:?}; a batch runs one model"
            ));
        }
        Some(_) => {}
        None => *model = Some(name.to_owned()),
    }
    for flag in ["stream", "background"] {
        if !matches!(
            body.get(flag),
            None | Some(Value::Null | Value::Bool(false))
        ) {
            return Err(format!("body.{flag} must be false in a batch"));
        }
    }
    Ok(InputLine {
        custom_id,
        body: serde_json::to_string(body).map_err(|error| error.to_string())?,
    })
}

/// Fail every pending line of an expiring batch with `batch_expired`.
fn expire_pending(connection: &Connection, id: &str) -> Result<()> {
    let rows = {
        let mut statement = connection.prepare(
            "SELECT line, custom_id, record_id FROM batch_lines
             WHERE batch_id = ?1 AND state = 'pending'",
        )?;
        statement
            .query_map([id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
    };
    let outcome = LineOutcome::Error {
        code: "batch_expired".to_owned(),
        message: EXPIRED_MESSAGE.to_owned(),
    };
    let mut update = connection.prepare(
        "UPDATE batch_lines SET state = 'failed', result = ?3 WHERE batch_id = ?1 AND line = ?2",
    )?;
    for (line, custom_id, record_id) in &rows {
        let (_, result) = record(record_id, custom_id, &outcome);
        update.execute(params![id, line, result])?;
    }
    connection.execute(
        "UPDATE batches SET failed = failed + ?2 WHERE id = ?1",
        params![id, signed(count(rows.len()))?],
    )?;
    Ok(())
}

/// The output record of a line, and whether it counts as completed.
fn record(record_id: &str, custom_id: &str, outcome: &LineOutcome) -> (bool, String) {
    let (succeeded, value) = match outcome {
        LineOutcome::Response { status_code, body } => (
            (200..300).contains(status_code),
            json!({
                "id": record_id,
                "custom_id": custom_id,
                "response": {
                    "status_code": status_code,
                    "request_id": format!(
                        "req_{}",
                        record_id.strip_prefix(RECORD_PREFIX).unwrap_or(record_id)
                    ),
                    "body": body,
                },
                "error": null,
            }),
        ),
        LineOutcome::Error { code, message } => (
            false,
            json!({
                "id": record_id,
                "custom_id": custom_id,
                "response": null,
                "error": {"code": code, "message": message},
            }),
        ),
    };
    (succeeded, value.to_string())
}

/// A stored batch row before decoding.
struct Raw {
    id: String,
    endpoint: String,
    model: String,
    input_file_id: String,
    completion_window: String,
    status: String,
    metadata: String,
    errors: Option<String>,
    output_file_id: Option<String>,
    error_file_id: Option<String>,
    times: [Option<i64>; 9],
    counts: [i64; 3],
}

fn raw(row: &Row<'_>) -> rusqlite::Result<Raw> {
    let mut times = [None; 9];
    for (offset, slot) in times.iter_mut().enumerate() {
        *slot = row.get(10 + offset)?;
    }
    Ok(Raw {
        id: row.get(0)?,
        endpoint: row.get(1)?,
        model: row.get(2)?,
        input_file_id: row.get(3)?,
        completion_window: row.get(4)?,
        status: row.get(5)?,
        metadata: row.get(6)?,
        errors: row.get(7)?,
        output_file_id: row.get(8)?,
        error_file_id: row.get(9)?,
        times,
        counts: [row.get(19)?, row.get(20)?, row.get(21)?],
    })
}

impl Raw {
    // The bindings are the public `batch` field names (`expires_at` and
    // `expired_at` both appear in the spec).
    #[allow(clippy::similar_names)]
    fn into_batch(self) -> Result<Batch> {
        let time = |value: Option<i64>| value.map(unsigned).transpose();
        let [
            created_at,
            in_progress_at,
            expires_at,
            finalizing_at,
            completed_at,
            failed_at,
            expired_at,
            cancelling_at,
            cancelled_at,
        ] = self.times;
        let [total, completed, failed] = self.counts;
        Ok(Batch {
            endpoint: self.endpoint.parse()?,
            status: BatchStatus::parse(&self.status)?,
            errors: self
                .errors
                .as_deref()
                .map(serde_json::from_str)
                .transpose()?,
            metadata: serde_json::from_str(&self.metadata)?,
            created_at: time(created_at)?.unwrap_or_default(),
            in_progress_at: time(in_progress_at)?,
            expires_at: time(expires_at)?.unwrap_or_default(),
            finalizing_at: time(finalizing_at)?,
            completed_at: time(completed_at)?,
            failed_at: time(failed_at)?,
            expired_at: time(expired_at)?,
            cancelling_at: time(cancelling_at)?,
            cancelled_at: time(cancelled_at)?,
            request_counts: RequestCounts {
                total: unsigned(total)?,
                completed: unsigned(completed)?,
                failed: unsigned(failed)?,
            },
            id: self.id,
            model: self.model,
            input_file_id: self.input_file_id,
            completion_window: self.completion_window,
            output_file_id: self.output_file_id,
            error_file_id: self.error_file_id,
        })
    }
}

fn load(connection: &Connection, id: &str) -> Result<Batch> {
    connection
        .query_row(
            &format!("SELECT {COLUMNS} FROM batches WHERE id = ?1"),
            [id],
            raw,
        )
        .optional()?
        .ok_or_else(|| not_found(id))?
        .into_batch()
}

fn validate_metadata(metadata: &Metadata) -> Result<()> {
    if metadata.len() > MAX_METADATA_PAIRS {
        return Err(invalid(format!(
            "metadata may hold at most {MAX_METADATA_PAIRS} pairs"
        )));
    }
    for (key, value) in metadata {
        if key.chars().count() > MAX_METADATA_KEY_CHARS {
            return Err(invalid(format!(
                "metadata keys may be at most {MAX_METADATA_KEY_CHARS} characters"
            )));
        }
        if value.chars().count() > MAX_METADATA_VALUE_CHARS {
            return Err(invalid(format!(
                "metadata values may be at most {MAX_METADATA_VALUE_CHARS} characters"
            )));
        }
    }
    Ok(())
}

/// Only `batch_` and ASCII letters and digits ever become a lock path.
fn validate_id(id: &str) -> Result<()> {
    let valid = id.strip_prefix(ID_PREFIX).is_some_and(|tail| {
        !tail.is_empty()
            && tail.len() <= MAX_ID_TAIL
            && tail.bytes().all(|byte| byte.is_ascii_alphanumeric())
    });
    if valid { Ok(()) } else { Err(not_found(id)) }
}

/// Create `path` owner-only if needed; refuse a symlink, a non-directory or
/// one anyone else may use.
fn private_dir(path: &Path) -> Result<()> {
    use std::os::unix::fs::{DirBuilderExt as _, PermissionsExt as _};
    match fs::DirBuilder::new().mode(0o700).create(path) {
        Err(error) if error.kind() != ErrorKind::AlreadyExists => return Err(error.into()),
        _ => {}
    }
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_dir()
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(refused(path));
    }
    Ok(())
}

/// Open (creating owner-only) a lock file, refusing a symlink or a file that
/// is not the one checked.
fn private_file(path: &Path) -> Result<File> {
    use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _};
    if fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        return Err(refused(path));
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)?;
    let opened = file.metadata()?;
    let named = fs::symlink_metadata(path)?;
    if !opened.is_file()
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
        || opened.permissions().mode() & 0o077 != 0
    {
        return Err(refused(path));
    }
    Ok(file)
}

fn refused(path: &Path) -> Error {
    Error::Io(io::Error::new(
        ErrorKind::PermissionDenied,
        format!(
            "batch lock path {} must be a regular file or directory only its owner can use",
            path.display()
        ),
    ))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidArgument(message.into())
}

fn line_error(line: usize, message: &str) -> Error {
    invalid(format!("input file line {line}: {message}"))
}

fn not_found(id: &str) -> Error {
    Error::NotFound(format!("batch {id} not found"))
}

fn corrupt(what: &str) -> Error {
    Error::Io(io::Error::new(
        ErrorKind::InvalidData,
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
