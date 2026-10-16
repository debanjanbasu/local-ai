//! Evals: durable, `OpenAI`-compatible evaluations of the one local model.
//!
//! An eval is a `custom` data source config (the JSON Schema of each row's
//! `item`, optionally with the `sample` namespace) and testing criteria. A
//! run snapshots its data source rows when it is created (inline
//! `file_content` or an uploaded file's JSONL, each row `{item, sample?}`,
//! every item checked against the eval's item schema), so later file changes
//! never affect it.
//!
//! # Graders
//!
//! * `string_check` (`eq`, `ne`, `like`, `ilike`) and `text_similarity`
//!   (`fuzzy_match`, `bleu`, `gleu`, `rouge_1` to `rouge_5`, `rouge_l`) are
//!   computed exactly as [`graders`] documents.
//! * `label_model` and `score_model` run on the loaded model, which is the
//!   only model a grader may name. The grader's own messages are sent as
//!   they are, and the answer is constrained (structured output) to
//!   `{"result": ...}`: one of `labels`, or a number within `range`
//!   (default `[0, 1]`). A label passes when it is a passing label; a score
//!   when it is at least `pass_threshold`, which `score_model` must set here.
//! * Refused when the eval is created: `python` (no sandbox), `multi` (not an
//!   eval criterion), and the `cosine` and `meteor` metrics.
//!
//! Template references (`{{item.x}}`, `{{sample.output_text}}`, see
//! [`template`]) that do not exist for a row make that row's result an
//! error. `logs` and `stored_completions` configs and run sources are
//! refused: evals here cannot query stored request logs.
//!
//! # Execution
//!
//! Generation reuses the Batches service (see [`crate::batches`]): every
//! model call of a run is a line of an ordinary batch the run creates, so it
//! runs, resumes after a restart and is cancelled exactly as batch lines
//! are. A run moves through three phases:
//!
//! 1. `sampling` (only `completions` and `responses` data sources): one
//!    batch with a request per row, built from `input_messages` and
//!    `sampling_params` (the spec's defaults: `temperature` 1, `top_p` 1,
//!    `seed` 42; `max_completion_tokens` defaults to the server's 8192). A
//!    `completions` source samples through Chat Completions, which
//!    deprecated `seed`: choosing one is refused, and its samples report
//!    [`CHAT_COMPLETIONS_SEED`], the server's fixed seed;
//!    `score_model` graders, which also run through Chat, refuse
//!    `sampling_params.seed` likewise;
//! 2. `grading`: one batch with a request per row and model grader;
//! 3. `scoring`: every row is graded and the output items, result counts,
//!    per-criterion results and usage are written in one transaction.
//!
//! The holder of a run's [`EvalRunLease`] (an exclusive advisory lock that
//! the operating system releases when the process dies) drives it with
//! [`Store::advance_eval_run`], which creates the phase's batch or collects
//! its results once the batch has ended, and returns when it has to wait for
//! a batch. Each step is idempotent: a batch is found again (by its
//! metadata) rather than recreated after a crash, results are collected with
//! a conditional phase change, and scoring commits all at once. A crash
//! between storing a batch's input file and creating the batch leaves that
//! (expiring) input file behind.
//!
//! The batches are visible in `/v1/batches` with the metadata `eval_batch`
//! (`<run id>:<phase>`), `eval_id` and `eval_run_id`. Cancelling a run marks
//! it `canceled` at once and cancels its current batch; a run whose batch is
//! cancelled by anyone else fails. A canceled or failed run has no output
//! items. A batch line that failed or expired makes its row errored, not the
//! run.
//!
//! # Results
//!
//! An output item passes when every criterion passed. A row is `errored`
//! (and its item has status `fail`) when its sample could not be generated
//! or any grader produced no result; `result_counts` counts each row once as
//! passed, failed or errored. `per_testing_criteria_results` counts, per
//! criterion (by name, which must be unique within an eval), the rows that
//! passed it and those that did not. `per_model_usage` sums every generation
//! of the run. `report_url` is filled in by the HTTP layer.

mod definitions;
pub mod graders;
mod scoring;
mod template;

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead as _, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

use rusqlite::{Connection, OptionalExtension as _, Row as SqlRow, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::batches::{
    self, BatchStatus, COMPLETION_WINDOW, CreateBatch, Endpoint, MAX_LINE_BYTES, MAX_LINES, Owner,
};
use crate::conversations::{Metadata, Order, Page};
use crate::files::{CreateFile, FilePurpose};
use crate::{Error, Result, Store, migrate, new_id, now};

pub use definitions::CHAT_COMPLETIONS_SEED;
use definitions::{Criterion, ItemSource, Row, invalid};
pub use scoring::EvalApiError;
use scoring::{StoredSample, Usage};

/// Most rows one run may hold (one batch line each).
pub const MAX_ITEMS: usize = MAX_LINES;
/// Page size when a list's `limit` is absent.
pub const DEFAULT_LIST_LIMIT: u32 = 20;
/// Largest page size.
pub const MAX_LIST_LIMIT: u32 = 100;

const EVAL_PREFIX: &str = "eval_";
const RUN_PREFIX: &str = "evalrun_";
const OUTPUT_PREFIX: &str = "outputitem_";
const LOCK_DIR: &str = "eval-locks";
const MAX_ID_TAIL: usize = 64;
/// The batch metadata key naming the run phase a batch belongs to.
const BATCH_KEY: &str = "eval_batch";

/// Version 1: evals, runs, their rows, model grades and output items.
/// Version 2: each run's batch [`Owner`], given to the batches it creates.
const MIGRATIONS: &[&str] = &[
    "
CREATE TABLE evals (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    data_source_config TEXT NOT NULL,
    testing_criteria TEXT NOT NULL,
    metadata TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
) STRICT;
CREATE INDEX evals_updated ON evals (updated_at, seq);
CREATE TABLE eval_runs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    eval_id TEXT NOT NULL REFERENCES evals (id),
    name TEXT,
    model TEXT,
    data_source TEXT NOT NULL,
    metadata TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'in_progress', 'completed', 'canceled',
                                           'failed')),
    phase TEXT NOT NULL CHECK (phase IN ('sampling', 'grading', 'scoring')),
    batch_id TEXT,
    error TEXT,
    result_counts TEXT,
    per_model_usage TEXT,
    per_testing_criteria_results TEXT,
    created_at INTEGER NOT NULL
) STRICT;
CREATE INDEX eval_runs_eval ON eval_runs (eval_id, seq);
CREATE INDEX eval_runs_status ON eval_runs (status, seq);
CREATE TABLE eval_run_items (
    run_id TEXT NOT NULL REFERENCES eval_runs (id),
    item INTEGER NOT NULL,
    datasource_item TEXT NOT NULL,
    sample TEXT,
    PRIMARY KEY (run_id, item)
) STRICT, WITHOUT ROWID;
CREATE TABLE eval_run_grades (
    run_id TEXT NOT NULL REFERENCES eval_runs (id),
    item INTEGER NOT NULL,
    criterion INTEGER NOT NULL,
    record TEXT NOT NULL,
    PRIMARY KEY (run_id, item, criterion)
) STRICT, WITHOUT ROWID;
CREATE TABLE eval_output_items (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES eval_runs (id),
    item INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pass', 'fail')),
    body TEXT NOT NULL,
    UNIQUE (run_id, item)
) STRICT;
CREATE INDEX eval_output_items_run ON eval_output_items (run_id, status, seq);
",
    "
ALTER TABLE eval_runs ADD COLUMN project_id TEXT;
ALTER TABLE eval_runs ADD COLUMN api_key_id TEXT;
",
];

/// Create or upgrade the eval tables.
pub(crate) fn initialize(connection: &mut Connection) -> Result<()> {
    migrate(connection, "evals", MIGRATIONS)
}

// ------------------------------------------------------------ public shapes

/// An eval, serialized as the public `eval` object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "eval")]
pub struct Eval {
    pub id: String,
    pub name: String,
    pub data_source_config: Value,
    pub testing_criteria: Vec<Value>,
    pub created_at: u64,
    pub metadata: Metadata,
}

/// The answer to deleting an eval.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "eval.deleted")]
pub struct EvalDeleted {
    pub deleted: bool,
    pub eval_id: String,
}

/// The answer to deleting a run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "eval.run.deleted")]
pub struct EvalRunDeleted {
    pub deleted: bool,
    pub run_id: String,
}

/// A run's lifecycle state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Queued,
    InProgress,
    Completed,
    Canceled,
    Failed,
}

impl RunStatus {
    /// The wire and stored name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Canceled => "canceled",
            Self::Failed => "failed",
        }
    }

    /// Whether nothing more will happen to the run.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Canceled | Self::Failed)
    }

    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "queued" => Self::Queued,
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "canceled" => Self::Canceled,
            "failed" => Self::Failed,
            _ => return Err(corrupt("run status")),
        })
    }
}

/// How a run's rows came out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResultCounts {
    pub total: u64,
    pub errored: u64,
    pub failed: u64,
    pub passed: u64,
}

/// What one model used during a run.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelUsage {
    pub model_name: String,
    pub invocation_count: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cached_tokens: u64,
}

/// How many rows passed and did not pass one criterion.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CriterionResult {
    pub testing_criteria: String,
    pub passed: u64,
    pub failed: u64,
}

/// A run, serialized as the public `eval.run` object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "eval.run")]
pub struct EvalRun {
    pub id: String,
    pub eval_id: String,
    pub status: RunStatus,
    /// The model that samples, for `completions` and `responses` sources.
    pub model: Option<String>,
    pub name: Option<String>,
    pub created_at: u64,
    /// Empty here; the HTTP layer sets the run's own URL.
    pub report_url: String,
    pub result_counts: ResultCounts,
    pub per_model_usage: Option<Vec<ModelUsage>>,
    pub per_testing_criteria_results: Option<Vec<CriterionResult>>,
    pub data_source: Value,
    pub metadata: Metadata,
    pub error: Option<EvalApiError>,
}

/// Input for [`Store::create_eval`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateEval {
    pub name: Option<String>,
    pub metadata: Option<Metadata>,
    pub data_source_config: Value,
    pub testing_criteria: Vec<Value>,
}

/// Input for [`Store::update_eval`]: each field present replaces the stored one.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateEval {
    pub name: Option<String>,
    pub metadata: Option<Metadata>,
}

/// What evals are listed by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalOrderBy {
    #[default]
    CreatedAt,
    UpdatedAt,
}

/// Query for [`Store::list_evals`]; oldest first by default.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ListEvals {
    pub after: Option<String>,
    pub limit: Option<u32>,
    pub order: Option<Order>,
    pub order_by: Option<EvalOrderBy>,
}

/// Input for [`Store::create_eval_run`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateEvalRun {
    pub name: Option<String>,
    pub metadata: Option<Metadata>,
    pub data_source: Value,
}

/// Query for [`Store::list_eval_runs`]; oldest first by default.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ListEvalRuns {
    pub after: Option<String>,
    pub limit: Option<u32>,
    pub order: Option<Order>,
    pub status: Option<RunStatus>,
}

/// An output item's status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputItemStatus {
    Pass,
    Fail,
}

impl OutputItemStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
        }
    }
}

/// Query for [`Store::list_eval_output_items`]; in row order by default.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ListOutputItems {
    pub after: Option<String>,
    pub limit: Option<u32>,
    pub order: Option<Order>,
    pub status: Option<OutputItemStatus>,
}

/// A canceled run and the batch whose current line should be interrupted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanceledRun {
    pub run: EvalRun,
    pub batch: Option<String>,
}

/// What [`Store::advance_eval_run`] left the run waiting for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Advance {
    /// It created this batch: a batch worker should be told.
    Started(String),
    /// It waits for this batch to end.
    Waiting(String),
    /// The run is terminal.
    Finished,
}

/// Exclusive ownership of one run's execution; dropping it (or the death of
/// the process) releases it.
#[derive(Debug)]
pub struct EvalRunLease {
    id: String,
    root: Arc<PathBuf>,
    _file: File,
}

impl EvalRunLease {
    /// The run this lease owns.
    #[must_use]
    pub const fn run_id(&self) -> &str {
        self.id.as_str()
    }
}

/// The phase a run is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Sampling,
    Grading,
    Scoring,
}

impl Phase {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Sampling => "sampling",
            Self::Grading => "grading",
            Self::Scoring => "scoring",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "sampling" => Self::Sampling,
            "grading" => Self::Grading,
            "scoring" => Self::Scoring,
            _ => return Err(corrupt("run phase")),
        })
    }
}

/// A run row with its execution state.
struct RunRow {
    run: EvalRun,
    phase: Phase,
    batch_id: Option<String>,
}

// ------------------------------------------------------------------- store

impl Store {
    /// Validate and store an eval.
    ///
    /// Model graders must name `installed_model`, the one model this server
    /// runs.
    pub fn create_eval(&self, request: &CreateEval, installed_model: &str) -> Result<Eval> {
        let metadata = request.metadata.clone().unwrap_or_default();
        batches::validate_metadata(&metadata)?;
        let config = definitions::parse_config(&request.data_source_config)?;
        if request.testing_criteria.is_empty() {
            return Err(invalid("testing_criteria must hold at least one grader"));
        }
        let mut criteria = Vec::with_capacity(request.testing_criteria.len());
        for (index, value) in request.testing_criteria.iter().enumerate() {
            let criterion = Criterion::parse(value, index, installed_model)?;
            if criteria
                .iter()
                .any(|other: &Criterion| other.name() == criterion.name())
            {
                return Err(invalid(format!(
                    "testing_criteria[{index}]: name {:?} is used twice; results are reported \
                     by name",
                    criterion.name()
                )));
            }
            criteria.push(criterion);
        }
        let criteria = criteria
            .iter()
            .map(serde_json::to_value)
            .collect::<serde_json::Result<Vec<_>>>()?;
        let id = new_id(EVAL_PREFIX)?;
        let name = request.name.clone().unwrap_or_else(|| id.clone());
        let at = signed(now()?)?;
        let connection = self.connection()?;
        connection.execute(
            "INSERT INTO evals (id, name, data_source_config, testing_criteria, metadata,
                                created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
            params![
                id,
                name,
                config.to_string(),
                Value::Array(criteria).to_string(),
                serde_json::to_string(&metadata)?,
                at
            ],
        )?;
        load_eval(&connection, &id)
    }

    /// The eval `id`.
    pub fn get_eval(&self, id: &str) -> Result<Eval> {
        load_eval(&self.connection()?, id)
    }

    /// Evals, oldest first by default.
    pub fn list_evals(&self, query: &ListEvals) -> Result<Page<Eval>> {
        let limit = page_limit(query.limit)?;
        let order = query.order.unwrap_or(Order::Asc);
        let by_update = query.order_by.unwrap_or_default() == EvalOrderBy::UpdatedAt;
        let connection = self.connection()?;
        let cursor = match &query.after {
            Some(after) => connection
                .query_row(
                    "SELECT updated_at, seq FROM evals WHERE id = ?1",
                    [after],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                )
                .optional()?
                .ok_or_else(|| Error::NotFound(format!("eval {after} not found")))?,
            None if order == Order::Asc => (i64::MIN, i64::MIN),
            None => (i64::MAX, i64::MAX),
        };
        let (compare, direction) = match order {
            Order::Asc => (">", "ASC"),
            Order::Desc => ("<", "DESC"),
        };
        let sql = if by_update {
            format!(
                "SELECT {EVAL_COLUMNS} FROM evals WHERE (updated_at, seq) {compare} (?1, ?2)
                 ORDER BY updated_at {direction}, seq {direction} LIMIT ?3"
            )
        } else {
            format!(
                "SELECT {EVAL_COLUMNS} FROM evals WHERE seq {compare} ?2 AND ?1 = ?1
                 ORDER BY seq {direction} LIMIT ?3"
            )
        };
        let mut statement = connection.prepare(&sql)?;
        let rows = statement
            .query_map(params![cursor.0, cursor.1, i64::from(limit) + 1], raw_eval)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let rows = rows
            .into_iter()
            .map(RawEval::into_eval)
            .collect::<Result<Vec<_>>>()?;
        Ok(page(rows, limit, |eval| eval.id.clone()))
    }

    /// Rename an eval or replace its metadata.
    pub fn update_eval(&self, id: &str, request: &UpdateEval) -> Result<Eval> {
        if let Some(metadata) = &request.metadata {
            batches::validate_metadata(metadata)?;
        }
        let connection = self.connection()?;
        load_eval(&connection, id)?;
        connection.execute(
            "UPDATE evals SET name = COALESCE(?2, name), metadata = COALESCE(?3, metadata),
                              updated_at = ?4
             WHERE id = ?1",
            params![
                id,
                request.name,
                request
                    .metadata
                    .as_ref()
                    .map(serde_json::to_string)
                    .transpose()?,
                signed(now()?)?
            ],
        )?;
        load_eval(&connection, id)
    }

    /// Delete an eval and all its runs. Refused while any run is unfinished.
    pub fn delete_eval(&self, id: &str) -> Result<EvalDeleted> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        load_eval(&transaction, id)?;
        let active: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM eval_runs
             WHERE eval_id = ?1 AND status IN ('queued', 'in_progress')",
            [id],
            |row| row.get(0),
        )?;
        if active > 0 {
            return Err(invalid(format!(
                "eval {id} has {active} unfinished runs; cancel them before deleting it"
            )));
        }
        for table in ["eval_output_items", "eval_run_grades", "eval_run_items"] {
            transaction.execute(
                &format!(
                    "DELETE FROM {table} WHERE run_id IN (SELECT id FROM eval_runs WHERE eval_id = ?1)"
                ),
                [id],
            )?;
        }
        transaction.execute("DELETE FROM eval_runs WHERE eval_id = ?1", [id])?;
        transaction.execute("DELETE FROM evals WHERE id = ?1", [id])?;
        transaction.commit()?;
        Ok(EvalDeleted {
            deleted: true,
            eval_id: id.to_owned(),
        })
    }

    /// Validate a run's data source, snapshot its rows and queue the run.
    pub fn create_eval_run(
        &self,
        eval_id: &str,
        request: &CreateEvalRun,
        installed_model: &str,
    ) -> Result<EvalRun> {
        self.create_eval_run_for(eval_id, request, installed_model, None)
    }

    /// [`Self::create_eval_run`] for `owner`: the batches the run creates
    /// are theirs, so its generations' usage is attributed to them.
    pub fn create_eval_run_for(
        &self,
        eval_id: &str,
        request: &CreateEvalRun,
        installed_model: &str,
        owner: Option<&Owner>,
    ) -> Result<EvalRun> {
        let eval = self.get_eval(eval_id)?;
        let criteria = criteria(&eval)?;
        let metadata = request.metadata.clone().unwrap_or_default();
        batches::validate_metadata(&metadata)?;
        let source = definitions::parse_run_source(&request.data_source, installed_model)?;
        let rows = match &source.source {
            ItemSource::FileContent { content } => content.clone(),
            ItemSource::FileId { id } => self.read_rows(id)?,
        };
        if rows.is_empty() {
            return Err(invalid("the data source holds no rows"));
        }
        if rows.len() > MAX_ITEMS {
            return Err(invalid(format!("a run may hold at most {MAX_ITEMS} rows")));
        }
        let model_graders = criteria.iter().filter(|c| c.uses_model()).count();
        if rows.len().saturating_mul(model_graders) > MAX_LINES {
            return Err(invalid(format!(
                "rows times model graders must be at most {MAX_LINES}"
            )));
        }
        let validator = definitions::compile(&definitions::item_schema(&eval.data_source_config))
            .map_err(invalid)?;
        for (index, row) in rows.iter().enumerate() {
            let item = Value::Object(row.item.clone());
            if let Err(error) = validator.validate(&item) {
                return Err(invalid(format!(
                    "data source row {index}: the item does not match the eval's item_schema: \
                     {error}"
                )));
            }
        }
        let sampling = source.sampler.is_some();
        let model = source
            .sampler
            .as_ref()
            .map(|sampler| sampler.model().to_owned());
        let id = new_id(RUN_PREFIX)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        load_eval(&transaction, eval_id)?;
        transaction.execute(
            "INSERT INTO eval_runs (id, eval_id, name, model, data_source, metadata, status, phase,
                                    created_at, project_id, api_key_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'queued', ?7, ?8, ?9, ?10)",
            params![
                id,
                eval_id,
                request.name,
                model,
                request.data_source.to_string(),
                serde_json::to_string(&metadata)?,
                if sampling {
                    Phase::Sampling.as_str()
                } else {
                    Phase::Grading.as_str()
                },
                signed(now()?)?,
                owner.map(|owner| owner.project_id.as_str()),
                owner.map(|owner| owner.api_key_id.as_str()),
            ],
        )?;
        {
            let mut insert = transaction.prepare(
                "INSERT INTO eval_run_items (run_id, item, datasource_item, sample)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;
            for (index, row) in rows.iter().enumerate() {
                let sample = if sampling {
                    None
                } else {
                    Some(serde_json::to_string(&scoring::provided_sample(
                        row.sample.as_ref(),
                    ))?)
                };
                insert.execute(params![
                    id,
                    signed(count(index))?,
                    Value::Object(row.item.clone()).to_string(),
                    sample
                ])?;
            }
        }
        transaction.commit()?;
        Ok(load_run(&connection, &id)?.run)
    }

    /// The rows of an uploaded JSONL file.
    fn read_rows(&self, file_id: &str) -> Result<Vec<Row>> {
        let content = self
            .open_file_content(file_id)
            .map_err(|error| match error {
                Error::NotFound(message) => invalid(format!("data_source.source.id: {message}")),
                other => other,
            })?;
        let mut reader = BufReader::new(content.content);
        let mut rows = Vec::new();
        let mut buffer = Vec::new();
        let mut number = 0_usize;
        loop {
            buffer.clear();
            if reader.read_until(b'\n', &mut buffer)? == 0 {
                break;
            }
            number += 1;
            if buffer.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            if buffer.len() > MAX_LINE_BYTES {
                return Err(invalid(format!(
                    "data source file line {number}: a row may be at most {MAX_LINE_BYTES} bytes"
                )));
            }
            if rows.len() == MAX_ITEMS {
                return Err(invalid(format!("a run may hold at most {MAX_ITEMS} rows")));
            }
            let row: Row = serde_json::from_slice(&buffer)
                .map_err(|error| invalid(format!("data source file line {number}: {error}")))?;
            rows.push(row);
        }
        Ok(rows)
    }

    /// The run `run_id` of eval `eval_id`.
    pub fn get_eval_run(&self, eval_id: &str, run_id: &str) -> Result<EvalRun> {
        let connection = self.connection()?;
        load_eval(&connection, eval_id)?;
        Ok(load_scoped_run(&connection, eval_id, run_id)?.run)
    }

    /// An eval's runs, oldest first by default.
    pub fn list_eval_runs(&self, eval_id: &str, query: &ListEvalRuns) -> Result<Page<EvalRun>> {
        let limit = page_limit(query.limit)?;
        let order = query.order.unwrap_or(Order::Asc);
        let connection = self.connection()?;
        load_eval(&connection, eval_id)?;
        let cursor = match &query.after {
            Some(after) => connection
                .query_row(
                    "SELECT seq FROM eval_runs WHERE id = ?1 AND eval_id = ?2",
                    params![after, eval_id],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
                .ok_or_else(|| run_not_found(after))?,
            None if order == Order::Asc => i64::MIN,
            None => i64::MAX,
        };
        let (compare, direction) = match order {
            Order::Asc => (">", "ASC"),
            Order::Desc => ("<", "DESC"),
        };
        let mut statement = connection.prepare(&format!(
            "SELECT {RUN_COLUMNS} FROM eval_runs
             WHERE eval_id = ?1 AND seq {compare} ?2 AND (?3 IS NULL OR status = ?3)
             ORDER BY seq {direction} LIMIT ?4"
        ))?;
        let rows = statement
            .query_map(
                params![
                    eval_id,
                    cursor,
                    query.status.map(RunStatus::as_str),
                    i64::from(limit) + 1
                ],
                raw_run,
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let runs = rows
            .into_iter()
            .map(|raw| raw.into_row().map(|row| row.run))
            .collect::<Result<Vec<_>>>()?;
        Ok(page(runs, limit, |run| run.id.clone()))
    }

    /// Cancel a queued or in-progress run: it is `canceled` at once and its
    /// current batch is cancelled. A canceled run is returned unchanged; a
    /// completed or failed one is a [`Error::Conflict`].
    pub fn cancel_eval_run(&self, eval_id: &str, run_id: &str) -> Result<CanceledRun> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        load_eval(&transaction, eval_id)?;
        let row = load_scoped_run(&transaction, eval_id, run_id)?;
        let batch = match row.run.status {
            RunStatus::Queued | RunStatus::InProgress => {
                transaction.execute(
                    "UPDATE eval_runs SET status = 'canceled' WHERE id = ?1",
                    [run_id],
                )?;
                row.batch_id
            }
            RunStatus::Canceled => None,
            other => {
                return Err(Error::Conflict(format!(
                    "run {run_id} is {}; only a queued or in-progress run can be canceled",
                    other.as_str()
                )));
            }
        };
        transaction.commit()?;
        if let Some(batch) = &batch {
            self.stop_batch(batch)?;
        }
        Ok(CanceledRun {
            run: load_run(&connection, run_id)?.run,
            batch,
        })
    }

    /// Delete a finished run and its output items. Refused while unfinished.
    pub fn delete_eval_run(&self, eval_id: &str, run_id: &str) -> Result<EvalRunDeleted> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        load_eval(&transaction, eval_id)?;
        let row = load_scoped_run(&transaction, eval_id, run_id)?;
        if !row.run.status.is_terminal() {
            return Err(invalid(format!(
                "run {run_id} is {}; cancel it before deleting it",
                row.run.status.as_str()
            )));
        }
        for table in ["eval_output_items", "eval_run_grades", "eval_run_items"] {
            transaction.execute(&format!("DELETE FROM {table} WHERE run_id = ?1"), [run_id])?;
        }
        transaction.execute("DELETE FROM eval_runs WHERE id = ?1", [run_id])?;
        transaction.commit()?;
        Ok(EvalRunDeleted {
            deleted: true,
            run_id: run_id.to_owned(),
        })
    }

    /// A run's output items, in row order by default.
    pub fn list_eval_output_items(
        &self,
        eval_id: &str,
        run_id: &str,
        query: &ListOutputItems,
    ) -> Result<Page<Value>> {
        let limit = page_limit(query.limit)?;
        let order = query.order.unwrap_or(Order::Asc);
        let connection = self.connection()?;
        load_eval(&connection, eval_id)?;
        load_scoped_run(&connection, eval_id, run_id)?;
        let cursor = match &query.after {
            Some(after) => connection
                .query_row(
                    "SELECT seq FROM eval_output_items WHERE id = ?1 AND run_id = ?2",
                    params![after, run_id],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
                .ok_or_else(|| output_not_found(after))?,
            None if order == Order::Asc => i64::MIN,
            None => i64::MAX,
        };
        let (compare, direction) = match order {
            Order::Asc => (">", "ASC"),
            Order::Desc => ("<", "DESC"),
        };
        let mut statement = connection.prepare(&format!(
            "SELECT body FROM eval_output_items
             WHERE run_id = ?1 AND seq {compare} ?2 AND (?3 IS NULL OR status = ?3)
             ORDER BY seq {direction} LIMIT ?4"
        ))?;
        let rows = statement
            .query_map(
                params![
                    run_id,
                    cursor,
                    query.status.map(OutputItemStatus::as_str),
                    i64::from(limit) + 1
                ],
                |row| row.get::<_, String>(0),
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let items = rows
            .iter()
            .map(|body| serde_json::from_str(body).map_err(Error::from))
            .collect::<Result<Vec<Value>>>()?;
        Ok(page(items, limit, |item| {
            item.get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        }))
    }

    /// One output item of a run.
    pub fn get_eval_output_item(
        &self,
        eval_id: &str,
        run_id: &str,
        item_id: &str,
    ) -> Result<Value> {
        let connection = self.connection()?;
        load_eval(&connection, eval_id)?;
        load_scoped_run(&connection, eval_id, run_id)?;
        let body: String = connection
            .query_row(
                "SELECT body FROM eval_output_items WHERE id = ?1 AND run_id = ?2",
                params![item_id, run_id],
                |row| row.get(0),
            )
            .optional()?
            .ok_or_else(|| output_not_found(item_id))?;
        Ok(serde_json::from_str(&body)?)
    }

    /// Runs a worker still has to drive, oldest first.
    pub fn active_eval_runs(&self) -> Result<Vec<String>> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            "SELECT id FROM eval_runs WHERE status IN ('queued', 'in_progress') ORDER BY seq",
        )?;
        let ids = statement
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(ids)
    }

    /// Take the execution lease of run `id` without waiting; `None` when
    /// another holder (in this process or another) has it.
    pub fn lease_eval_run(&self, id: &str) -> Result<Option<EvalRunLease>> {
        let valid = id.strip_prefix(RUN_PREFIX).is_some_and(|tail| {
            !tail.is_empty()
                && tail.len() <= MAX_ID_TAIL
                && tail.bytes().all(|byte| byte.is_ascii_alphanumeric())
        });
        if !valid {
            return Err(run_not_found(id));
        }
        load_run(&self.connection()?, id)?;
        let dir = self.root().join(LOCK_DIR);
        batches::private_dir(&dir)?;
        let file = batches::private_file(&dir.join(format!("{id}.lock")))?;
        match file.try_lock() {
            Ok(()) => Ok(Some(EvalRunLease {
                id: id.to_owned(),
                root: Arc::clone(&self.root),
                _file: file,
            })),
            Err(std::fs::TryLockError::WouldBlock) => Ok(None),
            Err(std::fs::TryLockError::Error(error)) => Err(error.into()),
        }
    }

    /// Drive the leased run as far as it can go without waiting: create the
    /// current phase's batch, or collect it once it has ended, and finally
    /// score the run. `installed_model` is the model batches will run on.
    pub fn advance_eval_run(&self, lease: &EvalRunLease, installed_model: &str) -> Result<Advance> {
        if !(Arc::ptr_eq(&self.root, &lease.root) || self.root == lease.root) {
            return Err(invalid(format!(
                "the lease on run {} was granted by another store",
                lease.id
            )));
        }
        loop {
            let connection = self.connection()?;
            let row = load_run(&connection, &lease.id)?;
            if row.run.status.is_terminal() {
                return Ok(Advance::Finished);
            }
            connection.execute(
                "UPDATE eval_runs SET status = 'in_progress' WHERE id = ?1 AND status = 'queued'",
                [&lease.id],
            )?;
            drop(connection);
            let eval = self.get_eval(&row.run.eval_id)?;
            let criteria = criteria(&eval)?;
            let step = match row.phase {
                Phase::Sampling => self.advance_sampling(&row, installed_model)?,
                Phase::Grading => self.advance_grading(&row, &criteria, installed_model)?,
                Phase::Scoring => {
                    self.score_run(&row, &criteria)?;
                    return Ok(Advance::Finished);
                }
            };
            if let Some(step) = step {
                return Ok(step);
            }
        }
    }

    /// One step of the sampling phase; `None` when the phase moved on.
    fn advance_sampling(&self, row: &RunRow, installed_model: &str) -> Result<Option<Advance>> {
        let model = row.run.model.clone().unwrap_or_default();
        let source = definitions::parse_run_source(&row.run.data_source, &model)?;
        let sampler = source
            .sampler
            .ok_or_else(|| corrupt("sampling run without a sampler"))?;
        let items = self.run_items(&row.run.id)?;
        let custom_id = |index: u64| format!("item-{index}");
        let Some(batch_id) = &row.batch_id else {
            let mut lines = Vec::new();
            let mut unbuilt = Vec::new();
            for (index, item, sample) in &items {
                if sample.is_some() {
                    continue;
                }
                match sampler.request(item) {
                    Ok((body, _)) => lines.push((custom_id(*index), body)),
                    Err(message) => unbuilt.push((
                        *index,
                        scoring::unbuilt_sample(message, sampler.model(), sampler.params()),
                    )),
                }
            }
            self.store_samples(&row.run.id, &unbuilt)?;
            if lines.is_empty() {
                self.next_phase(&row.run.id, Phase::Sampling, None)?;
                return Ok(None);
            }
            return self.start_batch(
                row,
                Phase::Sampling,
                sampler.endpoint(),
                &lines,
                installed_model,
            );
        };
        let Some(records) = self.ended_batch(row, batch_id)? else {
            return Ok(Some(self.waiting_or_finished(row, batch_id)?));
        };
        let generated: Vec<(u64, StoredSample)> = items
            .iter()
            .filter(|(_, _, sample)| sample.is_none())
            .map(|(index, item, _)| {
                let sample = match sampler.request(item) {
                    Ok((_, input)) => scoring::generated_sample(
                        records.get(&custom_id(*index)),
                        sampler.endpoint(),
                        sampler.model(),
                        &input,
                        sampler.params(),
                    ),
                    Err(message) => {
                        scoring::unbuilt_sample(message, sampler.model(), sampler.params())
                    }
                };
                (*index, sample)
            })
            .collect();
        self.store_samples(&row.run.id, &generated)?;
        self.next_phase(&row.run.id, Phase::Sampling, Some(batch_id))?;
        Ok(None)
    }

    /// One step of the grading phase; `None` when the phase moved on.
    fn advance_grading(
        &self,
        row: &RunRow,
        criteria: &[Criterion],
        installed_model: &str,
    ) -> Result<Option<Advance>> {
        let custom_id = |item: u64, criterion: usize| format!("item-{item}-criterion-{criterion}");
        let Some(batch_id) = &row.batch_id else {
            let mut lines = Vec::new();
            for (index, item, sample) in self.run_items(&row.run.id)? {
                let sample = sample.ok_or_else(|| corrupt("grading row without a sample"))?;
                if sample.error.is_some() {
                    continue;
                }
                let namespaces = template::Namespaces {
                    item: &item,
                    sample: sample.namespace.as_ref(),
                };
                for (position, criterion) in criteria.iter().enumerate() {
                    if let Some(Ok((body, _))) = criterion.grader_request(namespaces) {
                        lines.push((custom_id(index, position), body));
                    }
                }
            }
            if lines.is_empty() {
                self.next_phase(&row.run.id, Phase::Grading, None)?;
                return Ok(None);
            }
            return self.start_batch(
                row,
                Phase::Grading,
                Endpoint::ChatCompletions,
                &lines,
                installed_model,
            );
        };
        let Some(records) = self.ended_batch(row, batch_id)? else {
            return Ok(Some(self.waiting_or_finished(row, batch_id)?));
        };
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if !advance_phase(&transaction, &row.run.id, Phase::Grading, Some(batch_id))? {
            return Ok(None);
        }
        {
            let mut insert = transaction.prepare(
                "INSERT OR REPLACE INTO eval_run_grades (run_id, item, criterion, record)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;
            for (custom_id, record) in &records {
                let parsed = custom_id
                    .strip_prefix("item-")
                    .and_then(|rest| rest.split_once("-criterion-"))
                    .and_then(|(item, criterion)| {
                        Some((item.parse::<i64>().ok()?, criterion.parse::<i64>().ok()?))
                    });
                if let Some((item, criterion)) = parsed {
                    insert.execute(params![row.run.id, item, criterion, record.to_string()])?;
                }
            }
        }
        transaction.commit()?;
        Ok(None)
    }

    /// The results of `batch_id` by `custom_id` once it has ended; `None`
    /// while it runs (or after the run failed because the batch was
    /// cancelled or is gone).
    fn ended_batch(&self, row: &RunRow, batch_id: &str) -> Result<Option<HashMap<String, Value>>> {
        let batch = match self.get_batch(batch_id) {
            Ok(batch) => batch,
            Err(Error::NotFound(_)) => {
                self.fail_run(
                    &row.run.id,
                    &EvalApiError::new(
                        "batch_missing",
                        format!("batch {batch_id} no longer exists"),
                    ),
                )?;
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if !batch.status.is_terminal() {
            return Ok(None);
        }
        if batch.status == BatchStatus::Cancelled {
            self.fail_run(
                &row.run.id,
                &EvalApiError::new(
                    "batch_cancelled",
                    format!("batch {batch_id} was cancelled before the run finished"),
                ),
            )?;
            return Ok(None);
        }
        Ok(Some(self.batch_results(batch_id)?.into_iter().collect()))
    }

    fn waiting_or_finished(&self, row: &RunRow, batch_id: &str) -> Result<Advance> {
        let status = load_run(&self.connection()?, &row.run.id)?.run.status;
        Ok(if status.is_terminal() {
            Advance::Finished
        } else {
            Advance::Waiting(batch_id.to_owned())
        })
    }

    /// Create the batch of `phase` (or find the one created before a crash)
    /// and record it on the run.
    fn start_batch(
        &self,
        row: &RunRow,
        phase: Phase,
        endpoint: Endpoint,
        lines: &[(String, Map<String, Value>)],
        installed_model: &str,
    ) -> Result<Option<Advance>> {
        let key = format!("{}:{}", row.run.id, phase.as_str());
        let batch_id = if let Some(id) = self.find_batch(BATCH_KEY, &key)? {
            id
        } else {
            let mut content = Vec::new();
            for (custom_id, body) in lines {
                serde_json::to_writer(
                    &mut content,
                    &json!({"custom_id": custom_id, "method": "POST", "url": endpoint.as_str(),
                            "body": body}),
                )?;
                content.push(b'\n');
            }
            let refused = |message: String| EvalApiError::new("batch_refused", message);
            let file = match self.create_file(
                &CreateFile {
                    filename: format!("{}_{}.jsonl", row.run.id, phase.as_str()),
                    purpose: FilePurpose::Batch,
                    expires_after: None,
                },
                content.as_slice(),
            ) {
                Ok(file) => file,
                Err(Error::InvalidArgument(message)) => {
                    self.fail_run(&row.run.id, &refused(message))?;
                    return Ok(Some(Advance::Finished));
                }
                Err(error) => return Err(error),
            };
            let mut metadata = Metadata::new();
            metadata.insert(BATCH_KEY.to_owned(), key);
            metadata.insert("eval_id".to_owned(), row.run.eval_id.clone());
            metadata.insert("eval_run_id".to_owned(), row.run.id.clone());
            let owner = self.connection()?.query_row(
                "SELECT project_id, api_key_id FROM eval_runs WHERE id = ?1",
                [&row.run.id],
                |row| {
                    Ok(match (row.get(0)?, row.get(1)?) {
                        (Some(project_id), Some(api_key_id)) => Some(Owner {
                            project_id,
                            api_key_id,
                        }),
                        _ => None,
                    })
                },
            )?;
            let created = self.create_batch_for(
                &CreateBatch {
                    input_file_id: file.id.clone(),
                    endpoint: endpoint.as_str().to_owned(),
                    completion_window: COMPLETION_WINDOW.to_owned(),
                    metadata,
                },
                installed_model,
                owner.as_ref(),
            );
            match created {
                Ok(batch) => batch.id,
                Err(Error::InvalidArgument(message) | Error::Conflict(message)) => {
                    let _ = self.delete_file(&file.id);
                    self.fail_run(&row.run.id, &refused(message))?;
                    return Ok(Some(Advance::Finished));
                }
                Err(error) => {
                    let _ = self.delete_file(&file.id);
                    return Err(error);
                }
            }
        };
        let recorded = self.connection()?.execute(
            "UPDATE eval_runs SET batch_id = ?2
             WHERE id = ?1 AND phase = ?3 AND batch_id IS NULL
               AND status IN ('queued', 'in_progress')",
            params![row.run.id, batch_id, phase.as_str()],
        )?;
        if recorded == 0 {
            // Canceled meanwhile: its cancel did not see this batch.
            self.stop_batch(&batch_id)?;
            return Ok(Some(Advance::Finished));
        }
        Ok(Some(Advance::Started(batch_id)))
    }

    /// Cancel a batch unless it has already ended.
    fn stop_batch(&self, batch_id: &str) -> Result<()> {
        match self.cancel_batch(batch_id) {
            Ok(_) | Err(Error::Conflict(_) | Error::NotFound(_)) => Ok(()),
            Err(error) => Err(error),
        }
    }

    /// Every row of a run: index, item and stored sample.
    fn run_items(&self, run_id: &str) -> Result<Vec<(u64, Value, Option<StoredSample>)>> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            "SELECT item, datasource_item, sample FROM eval_run_items WHERE run_id = ?1
             ORDER BY item",
        )?;
        let rows = statement
            .query_map([run_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows.into_iter()
            .map(|(index, item, sample)| {
                Ok((
                    unsigned(index)?,
                    serde_json::from_str(&item)?,
                    sample.as_deref().map(serde_json::from_str).transpose()?,
                ))
            })
            .collect()
    }

    fn store_samples(&self, run_id: &str, samples: &[(u64, StoredSample)]) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        {
            let mut update = transaction
                .prepare("UPDATE eval_run_items SET sample = ?3 WHERE run_id = ?1 AND item = ?2")?;
            for (index, sample) in samples {
                update.execute(params![
                    run_id,
                    signed(*index)?,
                    serde_json::to_string(sample)?
                ])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    fn next_phase(&self, run_id: &str, from: Phase, batch_id: Option<&str>) -> Result<bool> {
        advance_phase(&self.connection()?, run_id, from, batch_id)
    }

    fn fail_run(&self, run_id: &str, error: &EvalApiError) -> Result<()> {
        self.connection()?.execute(
            "UPDATE eval_runs SET status = 'failed', error = ?2
             WHERE id = ?1 AND status IN ('queued', 'in_progress')",
            params![run_id, serde_json::to_string(error)?],
        )?;
        Ok(())
    }

    /// The batch result record of every model grader request of a run, by
    /// row and criterion.
    fn model_records(&self, run_id: &str) -> Result<HashMap<u64, HashMap<usize, Value>>> {
        let connection = self.connection()?;
        let mut statement = connection
            .prepare("SELECT item, criterion, record FROM eval_run_grades WHERE run_id = ?1")?;
        let rows = statement
            .query_map([run_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut records: HashMap<u64, HashMap<usize, Value>> = HashMap::new();
        for (item, criterion, record) in rows {
            records.entry(unsigned(item)?).or_default().insert(
                unsigned(criterion)? as usize,
                serde_json::from_str(&record)?,
            );
        }
        Ok(records)
    }

    /// Grade every row and complete the run, all in one transaction.
    #[allow(clippy::too_many_lines)]
    fn score_run(&self, row: &RunRow, criteria: &[Criterion]) -> Result<()> {
        let run = &row.run;
        let items = self.run_items(&run.id)?;
        let records = self.model_records(&run.id)?;
        let created_at = now()?;
        let mut counts = ResultCounts::default();
        let mut per_criterion: Vec<CriterionResult> = criteria
            .iter()
            .map(|criterion| CriterionResult {
                testing_criteria: criterion.name().to_owned(),
                ..CriterionResult::default()
            })
            .collect();
        let mut usage = UsageTable::default();
        let mut outputs = Vec::with_capacity(items.len());
        let empty = HashMap::new();
        for (index, item, sample) in &items {
            let sample = sample
                .clone()
                .ok_or_else(|| corrupt("scoring row without a sample"))?;
            if sample.invoked {
                usage.add(run.model.as_deref().unwrap_or_default(), 1, sample.usage);
            }
            let graded = scoring::grade_row(
                criteria,
                item,
                &sample,
                records.get(index).unwrap_or(&empty),
            );
            if let Some(model) = criteria.iter().find_map(Criterion::model) {
                usage.add(model, graded.invocations, graded.usage);
            }
            for (result, passed) in per_criterion.iter_mut().zip(&graded.passed) {
                *if *passed {
                    &mut result.passed
                } else {
                    &mut result.failed
                } += 1;
            }
            let all_passed = !graded.errored && graded.passed.iter().all(|passed| *passed);
            counts.total += 1;
            *match (graded.errored, all_passed) {
                (true, _) => &mut counts.errored,
                (false, true) => &mut counts.passed,
                (false, false) => &mut counts.failed,
            } += 1;
            let status = if all_passed {
                OutputItemStatus::Pass
            } else {
                OutputItemStatus::Fail
            };
            let id = new_id(OUTPUT_PREFIX)?;
            let body = json!({
                "object": "eval.run.output_item",
                "id": id,
                "run_id": run.id,
                "eval_id": run.eval_id,
                "created_at": created_at,
                "status": status.as_str(),
                "datasource_item_id": index,
                "datasource_item": item,
                "results": graded.results,
                "sample": sample.record,
            });
            outputs.push((id, *index, status, body));
        }
        let per_model_usage = usage.into_models();
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let completed = transaction.execute(
            "UPDATE eval_runs SET status = 'completed', result_counts = ?2, per_model_usage = ?3,
                                  per_testing_criteria_results = ?4
             WHERE id = ?1 AND status = 'in_progress' AND phase = 'scoring'",
            params![
                run.id,
                serde_json::to_string(&counts)?,
                (!per_model_usage.is_empty())
                    .then(|| serde_json::to_string(&per_model_usage))
                    .transpose()?,
                serde_json::to_string(&per_criterion)?
            ],
        )?;
        if completed == 0 {
            return Ok(());
        }
        {
            let mut insert = transaction.prepare(
                "INSERT INTO eval_output_items (id, run_id, item, status, body)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for (id, index, status, body) in &outputs {
                insert.execute(params![
                    id,
                    run.id,
                    signed(*index)?,
                    status.as_str(),
                    body.to_string()
                ])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }
}

/// Generation usage summed per model.
#[derive(Default)]
struct UsageTable(BTreeMap<String, (u64, Usage)>);

impl UsageTable {
    /// Count `invocations` generations of `model` that used `usage`.
    fn add(&mut self, model: &str, invocations: u64, usage: Usage) {
        if invocations == 0 {
            return;
        }
        let entry = self.0.entry(model.to_owned()).or_default();
        entry.0 += invocations;
        entry.1.add(usage);
    }

    fn into_models(self) -> Vec<ModelUsage> {
        self.0
            .into_iter()
            .map(|(model_name, (invocation_count, usage))| ModelUsage {
                model_name,
                invocation_count,
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                cached_tokens: usage.cached_tokens,
            })
            .collect()
    }
}

/// Move a run from `from` to the next phase, clearing its batch, if it is
/// still in progress in `from` with `batch_id`. Whether it moved.
fn advance_phase(
    connection: &Connection,
    run_id: &str,
    from: Phase,
    batch_id: Option<&str>,
) -> Result<bool> {
    let next = match from {
        Phase::Sampling => Phase::Grading,
        Phase::Grading | Phase::Scoring => Phase::Scoring,
    };
    let moved = connection.execute(
        "UPDATE eval_runs SET phase = ?3, batch_id = NULL
         WHERE id = ?1 AND phase = ?2 AND batch_id IS ?4 AND status = 'in_progress'",
        params![run_id, from.as_str(), next.as_str(), batch_id],
    )?;
    Ok(moved > 0)
}

fn criteria(eval: &Eval) -> Result<Vec<Criterion>> {
    eval.testing_criteria
        .iter()
        .map(|value| {
            serde_json::from_value(value.clone()).map_err(|_| corrupt("testing criterion"))
        })
        .collect()
}

fn page<T>(mut rows: Vec<T>, limit: u32, id: impl Fn(&T) -> String) -> Page<T> {
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    let has_more = rows.len() > limit;
    rows.truncate(limit);
    Page {
        first_id: rows.first().map(&id),
        last_id: rows.last().map(&id),
        data: rows,
        has_more,
    }
}

fn page_limit(limit: Option<u32>) -> Result<u32> {
    let limit = limit.unwrap_or(DEFAULT_LIST_LIMIT);
    if (1..=MAX_LIST_LIMIT).contains(&limit) {
        Ok(limit)
    } else {
        Err(invalid(format!(
            "limit must be between 1 and {MAX_LIST_LIMIT}"
        )))
    }
}

const EVAL_COLUMNS: &str = "id, name, data_source_config, testing_criteria, metadata, created_at";

struct RawEval {
    id: String,
    name: String,
    config: String,
    criteria: String,
    metadata: String,
    created_at: i64,
}

fn raw_eval(row: &SqlRow<'_>) -> rusqlite::Result<RawEval> {
    Ok(RawEval {
        id: row.get(0)?,
        name: row.get(1)?,
        config: row.get(2)?,
        criteria: row.get(3)?,
        metadata: row.get(4)?,
        created_at: row.get(5)?,
    })
}

impl RawEval {
    fn into_eval(self) -> Result<Eval> {
        Ok(Eval {
            id: self.id,
            name: self.name,
            data_source_config: serde_json::from_str(&self.config)?,
            testing_criteria: serde_json::from_str(&self.criteria)?,
            created_at: unsigned(self.created_at)?,
            metadata: serde_json::from_str(&self.metadata)?,
        })
    }
}

fn load_eval(connection: &Connection, id: &str) -> Result<Eval> {
    connection
        .query_row(
            &format!("SELECT {EVAL_COLUMNS} FROM evals WHERE id = ?1"),
            [id],
            raw_eval,
        )
        .optional()?
        .ok_or_else(|| Error::NotFound(format!("eval {id} not found")))?
        .into_eval()
}

const RUN_COLUMNS: &str = "id, eval_id, name, model, data_source, metadata, status, phase, \
     batch_id, error, result_counts, per_model_usage, per_testing_criteria_results, created_at";

struct RawRun {
    text: [Option<String>; 13],
    created_at: i64,
}

fn raw_run(row: &SqlRow<'_>) -> rusqlite::Result<RawRun> {
    let mut text: [Option<String>; 13] = Default::default();
    for (index, slot) in text.iter_mut().enumerate() {
        *slot = row.get(index)?;
    }
    Ok(RawRun {
        text,
        created_at: row.get(13)?,
    })
}

impl RawRun {
    fn into_row(self) -> Result<RunRow> {
        let [
            id,
            eval_id,
            name,
            model,
            data_source,
            metadata,
            status,
            phase,
            batch_id,
            error,
            result_counts,
            per_model_usage,
            per_criteria,
        ] = self.text;
        let json = |text: Option<String>| -> Result<Option<Value>> {
            Ok(text.as_deref().map(serde_json::from_str).transpose()?)
        };
        let required = |text: Option<String>, what: &str| text.ok_or_else(|| corrupt(what));
        Ok(RunRow {
            run: EvalRun {
                id: required(id, "run id")?,
                eval_id: required(eval_id, "run eval")?,
                status: RunStatus::parse(&required(status, "run status")?)?,
                model,
                name,
                created_at: unsigned(self.created_at)?,
                report_url: String::new(),
                result_counts: json(result_counts)?
                    .map(serde_json::from_value)
                    .transpose()?
                    .unwrap_or_default(),
                per_model_usage: json(per_model_usage)?
                    .map(serde_json::from_value)
                    .transpose()?,
                per_testing_criteria_results: json(per_criteria)?
                    .map(serde_json::from_value)
                    .transpose()?,
                data_source: json(data_source)?.unwrap_or(Value::Null),
                metadata: json(metadata)?
                    .map(serde_json::from_value)
                    .transpose()?
                    .unwrap_or_default(),
                error: json(error)?.map(serde_json::from_value).transpose()?,
            },
            phase: Phase::parse(&required(phase, "run phase")?)?,
            batch_id,
        })
    }
}

fn load_run(connection: &Connection, id: &str) -> Result<RunRow> {
    connection
        .query_row(
            &format!("SELECT {RUN_COLUMNS} FROM eval_runs WHERE id = ?1"),
            [id],
            raw_run,
        )
        .optional()?
        .ok_or_else(|| run_not_found(id))?
        .into_row()
}

fn load_scoped_run(connection: &Connection, eval_id: &str, run_id: &str) -> Result<RunRow> {
    let row = load_run(connection, run_id)?;
    if row.run.eval_id == eval_id {
        Ok(row)
    } else {
        Err(run_not_found(run_id))
    }
}

fn run_not_found(id: &str) -> Error {
    Error::NotFound(format!("run {id} not found"))
}

fn output_not_found(id: &str) -> Error {
    Error::NotFound(format!("output item {id} not found"))
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

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
