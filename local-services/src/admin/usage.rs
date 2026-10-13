//! Per-request usage records and their time-bucketed aggregation for the
//! `OpenAI` Usage API (`GET /organization/usage/*`).
//!
//! The server writes one record per finished model request (see
//! [`Admin::record_usage`]). A query walks UTC-aligned buckets of one width
//! from `start_time` (inclusive) to `end_time` (exclusive); every bucket in
//! the page is returned, an empty one with no results, and the results of a
//! bucket are the records in it summed per distinct value of the `group_by`
//! fields. Fields not grouped by are `null`.

use rusqlite::types::Value as SqlValue;
use serde::Serialize;
use serde_json::{Map, Value, json};

use super::{Admin, now_signed};
use crate::{Error, Result};

/// Seconds in each supported bucket width.
pub const MINUTE: i64 = 60;
pub const HOUR: i64 = 3600;
pub const DAY: i64 = 86_400;

/// What a usage record or query is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsageKind {
    Completions,
    Embeddings,
    Moderations,
    AudioSpeeches,
    AudioTranscriptions,
    Images,
    CodeInterpreterSessions,
    FileSearchCalls,
    WebSearchCalls,
    VectorStores,
}

impl UsageKind {
    /// The kind served at `/organization/usage/{segment}`.
    #[must_use]
    pub fn from_segment(segment: &str) -> Option<Self> {
        Some(match segment {
            "completions" => Self::Completions,
            "embeddings" => Self::Embeddings,
            "moderations" => Self::Moderations,
            "audio_speeches" => Self::AudioSpeeches,
            "audio_transcriptions" => Self::AudioTranscriptions,
            "images" => Self::Images,
            "code_interpreter_sessions" => Self::CodeInterpreterSessions,
            "file_search_calls" => Self::FileSearchCalls,
            "web_search_calls" => Self::WebSearchCalls,
            "vector_stores" => Self::VectorStores,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn segment(self) -> &'static str {
        match self {
            Self::Completions => "completions",
            Self::Embeddings => "embeddings",
            Self::Moderations => "moderations",
            Self::AudioSpeeches => "audio_speeches",
            Self::AudioTranscriptions => "audio_transcriptions",
            Self::Images => "images",
            Self::CodeInterpreterSessions => "code_interpreter_sessions",
            Self::FileSearchCalls => "file_search_calls",
            Self::WebSearchCalls => "web_search_calls",
            Self::VectorStores => "vector_stores",
        }
    }

    /// The `group_by` values the spec allows for this kind.
    #[must_use]
    pub const fn group_fields(self) -> &'static [&'static str] {
        match self {
            Self::Completions => &[
                "project_id",
                "user_id",
                "api_key_id",
                "model",
                "batch",
                "service_tier",
                "api_source",
            ],
            Self::Embeddings
            | Self::Moderations
            | Self::AudioSpeeches
            | Self::AudioTranscriptions => &["project_id", "user_id", "api_key_id", "model"],
            Self::Images => &[
                "project_id",
                "user_id",
                "api_key_id",
                "model",
                "size",
                "source",
            ],
            Self::CodeInterpreterSessions | Self::VectorStores => &["project_id"],
            Self::FileSearchCalls => &["project_id", "user_id", "api_key_id", "vector_store_id"],
            Self::WebSearchCalls => &[
                "project_id",
                "user_id",
                "api_key_id",
                "model",
                "context_level",
                "api_source",
            ],
        }
    }
}

/// One finished request's usage.
#[derive(Clone, Debug)]
pub struct UsageRecord {
    pub kind: UsageKind,
    /// Unix seconds; `None` records it now.
    pub at: Option<i64>,
    pub project_id: Option<String>,
    pub user_id: Option<String>,
    pub api_key_id: Option<String>,
    pub model: Option<String>,
    pub batch: bool,
    pub input_tokens: i64,
    pub input_cached_tokens: i64,
    pub output_tokens: i64,
    /// Characters (speech), seconds (transcription), images or sessions.
    pub units: i64,
    pub requests: i64,
}

impl UsageRecord {
    /// A record of `kind` with every count zero and no attribution.
    #[must_use]
    pub const fn new(kind: UsageKind) -> Self {
        Self {
            kind,
            at: None,
            project_id: None,
            user_id: None,
            api_key_id: None,
            model: None,
            batch: false,
            input_tokens: 0,
            input_cached_tokens: 0,
            output_tokens: 0,
            units: 0,
            requests: 1,
        }
    }
}

/// A usage query, already validated by [`UsageQuery::bucket_spans`].
#[derive(Clone, Debug)]
pub struct UsageQuery {
    pub kind: UsageKind,
    pub start_time: i64,
    /// Exclusive; `None` is the end of the bucket holding now.
    pub end_time: Option<i64>,
    /// Bucket width in seconds: [`MINUTE`], [`HOUR`] or [`DAY`].
    pub bucket_width: i64,
    /// Buckets per page; the width's default when `None`.
    pub limit: Option<usize>,
    /// The `next_page` cursor of a previous page.
    pub page: Option<String>,
    pub project_ids: Vec<String>,
    pub user_ids: Vec<String>,
    pub api_key_ids: Vec<String>,
    pub models: Vec<String>,
    pub batch: Option<bool>,
    pub group_by: Vec<String>,
}

/// `UsageResponse`: `object: "page"`.
#[derive(Clone, Debug, Serialize)]
pub struct UsagePage {
    pub object: &'static str,
    pub data: Vec<Bucket>,
    pub has_more: bool,
    pub next_page: Option<String>,
}

/// `UsageTimeBucket`.
#[derive(Clone, Debug, Serialize)]
pub struct Bucket {
    pub object: &'static str,
    pub start_time: i64,
    pub end_time: i64,
    pub results: Vec<Value>,
}

/// The default and largest number of buckets per page for `width`.
#[must_use]
pub const fn bucket_limits(width: i64) -> (usize, usize) {
    match width {
        MINUTE => (60, 1440),
        HOUR => (24, 168),
        _ => (7, 31),
    }
}

/// The bucket width named `text` (`1m`, `1h`, `1d`).
#[must_use]
pub fn parse_width(text: &str) -> Option<i64> {
    match text {
        "1m" => Some(MINUTE),
        "1h" => Some(HOUR),
        "1d" => Some(DAY),
        _ => None,
    }
}

/// A page of `[start, end)` bucket spans and the next page's cursor.
pub type Spans = (Vec<(i64, i64)>, Option<String>);

/// The page of bucket spans a query covers: `[start, end)` pairs aligned to
/// multiples of `width` (UTC) and clipped to `[start_time, end_time)`, and the
/// cursor of the next page if buckets remain.
pub fn bucket_spans(
    start_time: i64,
    end_time: i64,
    width: i64,
    limit: usize,
    page: Option<&str>,
) -> Result<Spans> {
    if start_time < 0 {
        return Err(Error::InvalidArgument(
            "start_time must be a Unix time in seconds".into(),
        ));
    }
    if end_time <= start_time {
        return Err(Error::InvalidArgument(
            "end_time must be after start_time".into(),
        ));
    }
    let first = start_time.div_euclid(width) * width;
    let mut cursor = match page {
        None => first,
        Some(page) => page
            .strip_prefix("page_")
            .and_then(|digits| digits.parse::<i64>().ok())
            .filter(|cursor| (first..end_time).contains(cursor) && cursor % width == 0)
            .ok_or_else(|| Error::InvalidArgument(format!("invalid page cursor {page:?}")))?,
    };
    let mut spans = Vec::new();
    while cursor < end_time && spans.len() < limit {
        spans.push((cursor.max(start_time), (cursor + width).min(end_time)));
        cursor += width;
    }
    let next = (cursor < end_time).then(|| format!("page_{cursor}"));
    Ok((spans, next))
}

/// The columns a `group_by` field reads; `None` for fields this server does
/// not record, which group everything under one `null` (or `unlabeled`).
fn column(field: &str) -> Option<&'static str> {
    Some(match field {
        "project_id" => "project_id",
        "user_id" => "user_id",
        "api_key_id" => "api_key_id",
        "model" => "model",
        "batch" => "batch",
        _ => return None,
    })
}

impl Admin {
    /// Write one usage record.
    pub fn record_usage(&self, record: &UsageRecord) -> Result<()> {
        let at = match record.at {
            Some(at) => at,
            None => now_signed()?,
        };
        self.store().connection()?.execute(
            "INSERT INTO usage_records (kind, at, project_id, user_id, api_key_id, model, batch,
                 input_tokens, input_cached_tokens, output_tokens, units, requests)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            rusqlite::params![
                record.kind.segment(),
                at,
                record.project_id,
                record.user_id,
                record.api_key_id,
                record.model,
                record.batch,
                record.input_tokens,
                record.input_cached_tokens,
                record.output_tokens,
                record.units,
                record.requests
            ],
        )?;
        Ok(())
    }

    /// One page of usage buckets for `query`.
    pub fn usage(&self, query: &UsageQuery) -> Result<UsagePage> {
        let (default, max) = bucket_limits(query.bucket_width);
        let limit = query.limit.unwrap_or(default);
        if !(1..=max).contains(&limit) {
            return Err(Error::InvalidArgument(format!(
                "limit must be between 1 and {max} for this bucket_width"
            )));
        }
        for field in &query.group_by {
            if !query.kind.group_fields().contains(&field.as_str()) {
                return Err(Error::InvalidArgument(format!(
                    "group_by {field:?} is not supported for {} usage; supported: {}",
                    query.kind.segment(),
                    query.kind.group_fields().join(", ")
                )));
            }
        }
        // Without an end, through the end of the bucket holding now.
        let end_time = match query.end_time {
            Some(end) => end,
            None => (now_signed()?.div_euclid(query.bucket_width) + 1) * query.bucket_width,
        };
        let (spans, next_page) = bucket_spans(
            query.start_time,
            end_time,
            query.bucket_width,
            limit,
            query.page.as_deref(),
        )?;
        let mut data: Vec<Bucket> = spans
            .iter()
            .map(|&(start_time, end_time)| Bucket {
                object: "bucket",
                start_time,
                end_time,
                results: Vec::new(),
            })
            .collect();
        if let (Some(&(from, _)), Some(&(_, to))) = (spans.first(), spans.last()) {
            if query.kind == UsageKind::VectorStores {
                self.vector_store_snapshot(&mut data)?;
            } else {
                self.aggregate(query, from, to, &mut data)?;
            }
        }
        Ok(UsagePage {
            object: "page",
            has_more: next_page.is_some(),
            data,
            next_page,
        })
    }

    /// Vector store storage is a level, not a count of requests, and only
    /// its current value is known: it is reported in the bucket holding now.
    fn vector_store_snapshot(&self, data: &mut [Bucket]) -> Result<()> {
        let now = now_signed()?;
        if let Some(bucket) = data
            .iter_mut()
            .find(|bucket| (bucket.start_time..bucket.end_time).contains(&now))
        {
            let bytes: i64 = self.store().connection()?.query_row(
                "SELECT COALESCE(SUM(usage_bytes), 0) FROM vector_store_files",
                [],
                |row| row.get(0),
            )?;
            bucket.results.push(
                json!({"object":"organization.usage.vector_stores.result","usage_bytes":bytes,
                       "project_id":null}),
            );
        }
        Ok(())
    }

    fn aggregate(&self, query: &UsageQuery, from: i64, to: i64, data: &mut [Bucket]) -> Result<()> {
        let width = query.bucket_width;
        let grouped: Vec<&'static str> = query
            .group_by
            .iter()
            .filter_map(|field| column(field))
            .collect();
        let mut clauses = vec!["kind = ?".to_owned(), "at >= ?".into(), "at < ?".into()];
        let mut values = vec![
            SqlValue::Text(query.kind.segment().into()),
            SqlValue::Integer(from),
            SqlValue::Integer(to),
        ];
        for (column, wanted) in [
            ("project_id", &query.project_ids),
            ("user_id", &query.user_ids),
            ("api_key_id", &query.api_key_ids),
            ("model", &query.models),
        ] {
            super::in_clause(&mut clauses, &mut values, column, wanted);
        }
        if let Some(batch) = query.batch {
            clauses.push("batch = ?".into());
            values.push(SqlValue::Integer(i64::from(batch)));
        }
        let group_columns: String = grouped.iter().flat_map(|column| [", ", column]).collect();
        let sql = format!(
            "SELECT (at / {width}) * {width} AS bucket{group_columns},
                    SUM(input_tokens), SUM(input_cached_tokens), SUM(output_tokens),
                    SUM(units), SUM(requests)
             FROM usage_records WHERE {}
             GROUP BY bucket{group_columns} ORDER BY bucket{group_columns}",
            clauses.join(" AND ")
        );
        let connection = self.store().connection()?;
        let mut statement = connection.prepare(&sql)?;
        let mut rows = statement.query(rusqlite::params_from_iter(values))?;
        while let Some(row) = rows.next()? {
            let bucket: i64 = row.get(0)?;
            let mut groups = Map::new();
            for (index, column) in grouped.iter().enumerate() {
                let value: SqlValue = row.get(index + 1)?;
                let value = match value {
                    SqlValue::Integer(flag) if *column == "batch" => json!(flag != 0),
                    SqlValue::Text(text) => json!(text),
                    _ => Value::Null,
                };
                groups.insert((*column).to_owned(), value);
            }
            let base = grouped.len() + 1;
            let sums: [i64; 5] = [
                row.get(base)?,
                row.get(base + 1)?,
                row.get(base + 2)?,
                row.get(base + 3)?,
                row.get(base + 4)?,
            ];
            if let Some(target) = data
                .iter_mut()
                .find(|slot| (slot.start_time..slot.end_time).contains(&bucket.max(from)))
            {
                target
                    .results
                    .push(result(query.kind, &query.group_by, &groups, sums));
            }
        }
        Ok(())
    }
}

/// One aggregated result in the kind's `OpenAI` shape.
fn result(
    kind: UsageKind,
    group_by: &[String],
    groups: &Map<String, Value>,
    sums: [i64; 5],
) -> Value {
    let [input, cached, output, units, requests] = sums;
    let mut value = match kind {
        UsageKind::Completions => json!({
            "object":"organization.usage.completions.result",
            "input_tokens":input,"input_cached_tokens":cached,
            "input_uncached_tokens":input - cached,"output_tokens":output,
            "input_text_tokens":input,"output_text_tokens":output,
            "input_cached_text_tokens":cached,
            "input_audio_tokens":0,"input_cached_audio_tokens":0,"output_audio_tokens":0,
            "input_image_tokens":0,"input_cached_image_tokens":0,"output_image_tokens":0,
            "num_model_requests":requests,
            "project_id":null,"user_id":null,"api_key_id":null,"model":null,"batch":null,
            "service_tier":null,"api_source":null,
        }),
        UsageKind::Embeddings | UsageKind::Moderations => json!({
            "object":format!("organization.usage.{}.result", kind.segment()),
            "input_tokens":input,"num_model_requests":requests,
            "project_id":null,"user_id":null,"api_key_id":null,"model":null,
        }),
        UsageKind::AudioSpeeches => json!({
            "object":"organization.usage.audio_speeches.result","characters":units,
            "num_model_requests":requests,
            "project_id":null,"user_id":null,"api_key_id":null,"model":null,
        }),
        UsageKind::AudioTranscriptions => json!({
            "object":"organization.usage.audio_transcriptions.result","seconds":units,
            "num_model_requests":requests,
            "project_id":null,"user_id":null,"api_key_id":null,"model":null,
        }),
        UsageKind::Images => json!({
            "object":"organization.usage.images.result","images":units,
            "num_model_requests":requests,"source":null,"size":null,
            "project_id":null,"user_id":null,"api_key_id":null,"model":null,
        }),
        UsageKind::CodeInterpreterSessions => json!({
            "object":"organization.usage.code_interpreter_sessions.result",
            "num_sessions":units,"project_id":null,
        }),
        UsageKind::FileSearchCalls => json!({
            "object":"organization.usage.file_search_calls.result","num_requests":requests,
            "project_id":null,"user_id":null,"api_key_id":null,"vector_store_id":null,
        }),
        UsageKind::WebSearchCalls => json!({
            "object":"organization.usage.web_search_calls.result",
            "num_model_requests":requests,"num_requests":units,
            "project_id":null,"user_id":null,"api_key_id":null,"model":null,
            "context_level":null,"api_source":null,
        }),
        UsageKind::VectorStores => json!({
            "object":"organization.usage.vector_stores.result","usage_bytes":0,"project_id":null,
        }),
    };
    for (field, group) in groups {
        value[field.as_str()] = group.clone();
    }
    if group_by.iter().any(|field| field == "api_source") {
        value["api_source"] = json!("unlabeled");
    }
    value
}
