//! The Responses `file_search` tool, run natively over this server's vector
//! stores ([`local_services::vector_stores`]).
//!
//! It needs no server permission: it only reads the vector stores of this
//! server's own data directory, which the same callers can already search
//! through `POST /v1/vector_stores/{id}/search`. A request declares
//! `{"type":"file_search","vector_store_ids":[...]}` with optional
//! `max_num_results` (1 to 50, default [`DEFAULT_MAX_NUM_RESULTS`]),
//! `ranking_options` (`ranker` `auto` or `none`, `score_threshold`, and
//! `hybrid_search` weights) and `filters` (attribute filters); the stores
//! must exist and not be expired, or the request is refused before
//! generation.
//!
//! The model sees one strict native function, `file_search`, taking
//! `queries` (1 to 16 strings). One call ranks the chunks of every declared
//! store together with [`VectorIndex::search_stores`](local_services::vector_stores::VectorIndex::search_stores),
//! the same search the REST endpoint runs: each query embedded once with the
//! server's loaded `EmbeddingGemma` 2 model (`auto` with the model
//! installed), or BM25 (`none`, or no model), or, with `hybrid_search`,
//! weighted reciprocal rank fusion of the two. Chunk text comes from the
//! database snapshot, never from the source files.
//!
//! The call is reported as a `file_search_call` item with its `queries` and
//! `status` (`completed`, or `failed` with the error the model read), and
//! with its `results` only when the request's `include` lists
//! `file_search_call.results`. Streaming emits `output_item.added`,
//! `response.file_search_call.in_progress` and `.searching` when the call
//! is generated, then `.completed` and `output_item.done` when it ran.
//!
//! **Citations.** The model reads each result behind a numbered marker,
//! `【n】`, numbered across the Response's searches, and is asked to cite a
//! passage by writing its marker after the text it supports. Markers naming
//! a result of this Response are removed from the answer text (with the
//! whitespace before them) and become `file_citation` annotations at that
//! position, counted in Unicode scalar values of the final text, with
//! `response.output_text.annotation.added` events. Anything else in lenticular
//! brackets is left as written. Replayed `file_search_call` items (input,
//! `previous_response_id`, conversations) show the model their results
//! without markers, or, when they carry none, say so, so earlier passages
//! are never cited as if they were this Response's.

use std::collections::HashMap;
use std::fmt::Write as _;

use serde_json::{Map, Value, json};

use local_engine::{ToolCall, ToolDefinition, ToolOutput};
use local_services::tools::ToolError;
use local_services::vector_stores::{
    Filter, MAX_NUM_RESULTS, MAX_QUERIES, Query, RRF_K, Ranker, Ranking, RankingOptions,
    SearchRequest, SearchResults,
};

use super::hosted::Rendered;

/// The tool's name, to the model and in the request.
pub(super) const NAME: &str = "file_search";
/// Results per call when the request does not set `max_num_results`: each
/// result is a chunk (about 800 words with the default chunking) the model
/// must read, so more results cost prefill time (see `docs/TOOLS.md`).
pub(super) const DEFAULT_MAX_NUM_RESULTS: u32 = 5;
/// Stores one `file_search` tool may name.
pub(super) const MAX_STORES: usize = 16;
/// Bytes after an opening `【` that may still become a citation marker.
const MAX_MARKER_BYTES: usize = 64;

/// A declared `file_search` tool.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Config {
    pub(super) vector_store_ids: Vec<String>,
    /// Every field of a search but the query.
    template: SearchRequest,
    /// The `hybrid_search` ranking, when one was asked for.
    hybrid: Option<Ranking>,
}

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::InvalidArgument(message.into())
}

impl Config {
    /// Parse a `file_search` tool declaration, returning it, the native
    /// function the model sees and the tool as the Response echoes it.
    pub(super) fn declare(
        object: &Map<String, Value>,
    ) -> crate::Result<(Self, ToolDefinition, Value)> {
        if let Some(key) = object.keys().find(|key| {
            !matches!(
                key.as_str(),
                "type" | "vector_store_ids" | "max_num_results" | "ranking_options" | "filters"
            )
        }) {
            return Err(invalid(format!(
                "file_search tool member {key:?} is not supported"
            )));
        }
        let ids = object
            .get("vector_store_ids")
            .and_then(Value::as_array)
            .filter(|ids| (1..=MAX_STORES).contains(&ids.len()))
            .ok_or_else(|| {
                invalid(format!(
                    "a file_search tool requires vector_store_ids: 1 to {MAX_STORES} vector store IDs"
                ))
            })?;
        let vector_store_ids = ids
            .iter()
            .map(|id| {
                id.as_str()
                    .filter(|id| !id.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        invalid("file_search vector_store_ids must be non-empty strings")
                    })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let max_num_results = match object.get("max_num_results") {
            None | Some(Value::Null) => DEFAULT_MAX_NUM_RESULTS,
            Some(value) => value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| (1..=MAX_NUM_RESULTS).contains(value))
                .ok_or_else(|| {
                    invalid(format!(
                        "file_search max_num_results must be an integer from 1 to {MAX_NUM_RESULTS}"
                    ))
                })?,
        };
        let (ranking_options, hybrid, hybrid_echo) = ranking(object.get("ranking_options"))?;
        let filters = match object.get("filters") {
            None | Some(Value::Null) => None,
            Some(filters) => Some(
                serde_json::from_value::<Filter>(filters.clone())
                    .map_err(|error| invalid(format!("file_search filters: {error}")))?,
            ),
        };
        let template = SearchRequest {
            query: Query::One(String::new()),
            rewrite_query: false,
            max_num_results: Some(max_num_results),
            filters,
            ranking_options: Some(ranking_options),
        };
        // Everything but the query is checked now, as a search would.
        SearchRequest {
            query: Query::One("x".to_owned()),
            ..template.clone()
        }
        .validate()
        .map_err(|error| invalid(format!("file_search: {}", service_message(&error))))?;
        let mut options = json!({
            "ranker": match ranking_options.ranker {
                Ranker::None => "none",
                _ => "auto",
            },
            "score_threshold": ranking_options.score_threshold,
        });
        if let Some(weights) = hybrid_echo {
            options["hybrid_search"] = weights;
        }
        let echo = json!({
            "type": NAME,
            "vector_store_ids": vector_store_ids,
            "max_num_results": max_num_results,
            "ranking_options": options,
            "filters": object.get("filters").cloned().unwrap_or(Value::Null),
        });
        Ok((
            Self {
                vector_store_ids,
                template,
                hybrid,
            },
            definition(),
            echo,
        ))
    }

    /// Whether searching needs the embedding model (`hybrid_search`).
    pub(super) const fn needs_embeddings(&self) -> bool {
        self.hybrid.is_some()
    }

    /// The ranking to force, if not the one the ranker selects.
    pub(super) const fn ranking(&self) -> Option<Ranking> {
        self.hybrid
    }

    /// The search a call with `arguments` runs, or why its arguments are
    /// unusable.
    pub(super) fn request(&self, arguments: &Value) -> Result<SearchRequest, String> {
        let queries = queries(arguments).ok_or("queries must be an array of strings")?;
        if queries.is_empty() || queries.len() > MAX_QUERIES {
            return Err(format!("queries must hold 1 to {MAX_QUERIES} strings"));
        }
        Ok(SearchRequest {
            query: Query::Many(queries),
            ..self.template.clone()
        })
    }
}

/// `ranking_options`: the native options, the `hybrid_search` ranking and
/// its echo.
fn ranking(
    value: Option<&Value>,
) -> crate::Result<(RankingOptions, Option<Ranking>, Option<Value>)> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok((RankingOptions::default(), None, None));
    };
    let object = value
        .as_object()
        .ok_or_else(|| invalid("file_search ranking_options must be an object"))?;
    if let Some(key) = object
        .keys()
        .find(|key| !matches!(key.as_str(), "ranker" | "score_threshold" | "hybrid_search"))
    {
        return Err(invalid(format!(
            "file_search ranking_options member {key:?} is not supported"
        )));
    }
    let mut options = RankingOptions::default();
    if let Some(ranker) = object.get("ranker").filter(|value| !value.is_null()) {
        options.ranker = serde_json::from_value(ranker.clone())
            .map_err(|_| invalid("file_search ranking_options.ranker must be auto or none"))?;
    }
    if let Some(threshold) = object
        .get("score_threshold")
        .filter(|value| !value.is_null())
    {
        options.score_threshold = threshold.as_f64().ok_or_else(|| {
            invalid("file_search ranking_options.score_threshold must be a number")
        })?;
    }
    let Some(hybrid) = object.get("hybrid_search").filter(|value| !value.is_null()) else {
        return Ok((options, None, None));
    };
    let weight = |name: &str| {
        hybrid
            .get(name)
            .and_then(Value::as_f64)
            .filter(|weight| weight.is_finite() && *weight >= 0.0)
            .ok_or_else(|| {
                invalid(format!(
                    "file_search ranking_options.hybrid_search requires {name}, a number of at \
                     least 0"
                ))
            })
    };
    let (embedding_weight, text_weight) = (weight("embedding_weight")?, weight("text_weight")?);
    if hybrid.as_object().is_some_and(|object| object.len() != 2) {
        return Err(invalid(
            "file_search ranking_options.hybrid_search holds only embedding_weight and text_weight",
        ));
    }
    if embedding_weight + text_weight <= 0.0 {
        return Err(invalid(
            "file_search ranking_options.hybrid_search needs a positive weight",
        ));
    }
    if options.ranker == Ranker::None {
        return Err(invalid(
            "file_search ranking_options.hybrid_search needs ranker auto: ranker none is BM25 alone",
        ));
    }
    Ok((
        options,
        Some(Ranking::Fused {
            k: RRF_K,
            text_weight,
            embedding_weight,
        }),
        Some(json!({"embedding_weight":embedding_weight,"text_weight":text_weight})),
    ))
}

/// The native function the model calls.
fn definition() -> ToolDefinition {
    let mut parameters =
        json!({"type":"object","required":["queries"],"additionalProperties":false});
    parameters["properties"] = json!({"queries":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":MAX_QUERIES}});
    ToolDefinition {
        name: NAME.to_owned(),
        description: Some(
            "Search the user's files for passages relevant to the request and read the best \
             ones, numbered. Pass one or more short search queries (several are searched at \
             once). Cite a passage you use by writing its marker, such as 【1】, right after \
             the sentence it supports."
                .to_owned(),
        ),
        parameters,
        strict: true,
    }
}

/// The message of a vector store error, without its category prefix.
pub(super) fn service_message(error: &local_services::Error) -> String {
    match error {
        local_services::Error::InvalidArgument(message)
        | local_services::Error::NotFound(message)
        | local_services::Error::Conflict(message) => message.clone(),
        other => other.to_string(),
    }
}

/// A call's queries, when they are an array of strings.
fn queries(arguments: &Value) -> Option<Vec<String>> {
    arguments
        .get("queries")?
        .as_array()?
        .iter()
        .map(|query| query.as_str().map(str::to_owned))
        .collect()
}

/// The queries a call (or replayed item) names, for reporting.
fn reported_queries(value: Option<&Value>) -> Value {
    value.and_then(Value::as_array).map_or_else(
        || json!([]),
        |queries| {
            Value::Array(
                queries
                    .iter()
                    .filter(|query| query.is_string())
                    .cloned()
                    .collect(),
            )
        },
    )
}

/// One result as the spec's `file_search_call` item lists it.
fn result_item(result: &local_services::vector_stores::SearchResult) -> Value {
    let text: String = result
        .content
        .iter()
        .map(|part| part.text.as_str())
        .collect();
    json!({
        "file_id": result.file_id,
        "filename": result.filename,
        "score": result.score,
        "text": text,
        "attributes": result.attributes,
    })
}

/// What the model reads for `results` (items as [`result_item`] makes
/// them), numbered from `first` behind citation markers, or unnumbered.
fn passages(queries: &Value, results: &[Value], first: Option<usize>) -> String {
    let quoted: Vec<String> = queries
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|query| format!("{query:?}"))
        .collect();
    let quoted = quoted.join(", ");
    if results.is_empty() {
        return format!("No passages matched {quoted} in the searched files.");
    }
    let mut text = format!("Passages for {quoted}:");
    for (offset, result) in results.iter().enumerate() {
        let filename = result
            .get("filename")
            .and_then(Value::as_str)
            .unwrap_or_default();
        text.push_str("\n\n");
        if let Some(first) = first {
            let _ = write!(text, "【{}】 ", first + offset);
        }
        text.push_str(filename);
        if let Some(attributes) = result
            .get("attributes")
            .and_then(Value::as_object)
            .filter(|attributes| !attributes.is_empty())
        {
            let _ = write!(text, " {}", Value::Object(attributes.clone()));
        }
        text.push('\n');
        text.push_str(
            result
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
    }
    if first.is_some() {
        text.push_str("\n\nCite a passage where you use it with its marker, such as 【");
        let _ = write!(text, "{}】.", first.unwrap_or(1));
    }
    text
}

/// The event a call's search reports: what the model reads, and the
/// structured record the Response renders (`status`, `queries`, `results`
/// and `first`, the number of the first result's marker).
pub(super) fn output(
    call: &ToolCall,
    result: &Result<SearchResults, ToolError>,
    first: usize,
) -> ToolOutput {
    let queries = reported_queries(call.arguments.get("queries"));
    let (content, detail) = match result {
        Ok(found) => {
            let results: Vec<Value> = found.data.iter().map(result_item).collect();
            (
                passages(&queries, &results, Some(first)),
                json!({"status":"completed","queries":queries,"results":results,"first":first}),
            )
        }
        Err(error) => (
            format!("error: {error}"),
            json!({"status":"failed","queries":queries,"results":[],"first":first}),
        ),
    };
    ToolOutput {
        call_id: call.id.clone(),
        name: call.name.clone(),
        content,
        detail,
    }
}

/// How many result markers `output` numbered.
pub(super) fn numbered(output: &ToolOutput) -> usize {
    output
        .detail
        .get("results")
        .and_then(Value::as_array)
        .map_or(0, Vec::len)
}

/// The `file_search_call` item `id` at `index` for a call the model just
/// made, `searching`, with the events announcing it.
pub(super) fn started(id: &str, index: usize, call: &ToolCall) -> Rendered {
    let queries = reported_queries(call.arguments.get("queries"));
    let added = json!({"id":id,"type":"file_search_call","status":"in_progress","queries":queries,"results":null});
    let mut item = added.clone();
    item["status"] = json!("searching");
    Rendered {
        item,
        events: vec![
            (
                "response.output_item.added",
                json!({"output_index":index,"item":added}),
            ),
            (
                "response.file_search_call.in_progress",
                json!({"output_index":index,"item_id":id}),
            ),
            (
                "response.file_search_call.searching",
                json!({"output_index":index,"item_id":id}),
            ),
        ],
    }
}

/// Finish the item at `index` with its search's `output`, returning the
/// events that report it. `include` says whether the item lists results.
pub(super) fn finished(
    item: &mut Value,
    index: usize,
    output: &ToolOutput,
    include: bool,
) -> Vec<(&'static str, Value)> {
    let status = output
        .detail
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("failed");
    item["status"] = json!(status);
    item["results"] = if include {
        output.detail.get("results").cloned().unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    let id = item.get("id").cloned().unwrap_or(Value::Null);
    let mut events = Vec::new();
    if status == "completed" {
        events.push((
            "response.file_search_call.completed",
            json!({"output_index":index,"item_id":id}),
        ));
    }
    events.push((
        "response.output_item.done",
        json!({"output_index":index,"item":item.clone()}),
    ));
    events
}

/// A replayed `file_search_call` item: the call the model made (its call ID
/// is the item's ID) and what it read back.
pub(super) fn input(item: &Value) -> crate::Result<(ToolCall, String)> {
    let id = item
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("a file_search_call item requires id"))?;
    let queries = item
        .get("queries")
        .filter(|queries| {
            queries
                .as_array()
                .is_some_and(|queries| queries.iter().all(Value::is_string))
        })
        .cloned()
        .ok_or_else(|| invalid("a file_search_call item requires queries, an array of strings"))?;
    let content = match (
        item.get("status").and_then(Value::as_str),
        item.get("results"),
    ) {
        (Some("failed"), _) => "error: this search failed".to_owned(),
        (_, Some(Value::Array(results))) => passages(&queries, results, None),
        _ => "The results of this search are not part of the conversation; search again to \
              read them."
            .to_owned(),
    };
    Ok((
        ToolCall {
            id: id.to_owned(),
            name: NAME.to_owned(),
            arguments: json!({"queries":queries}),
        },
        content,
    ))
}

/// A piece of answer text after citation markers are taken out.
#[derive(Debug, PartialEq)]
pub(super) enum Piece {
    Text(String),
    /// The cited results' `(file_id, filename)`, in marker order.
    Cite(Vec<(String, String)>),
}

/// Turns `【n】` markers in answer text into citations of this Response's
/// search results, as the text streams.
#[derive(Debug, Default)]
pub(super) struct Citations {
    /// Each result's marker number and `(file_id, filename)`.
    sources: HashMap<usize, (String, String)>,
    /// Text not yet passed on: trailing whitespace (dropped when a marker
    /// follows) and an unfinished marker.
    held: String,
}

impl Citations {
    /// Record the results of a search's `output` as citable.
    pub(super) fn add(&mut self, output: &ToolOutput) {
        let first = output
            .detail
            .get("first")
            .and_then(Value::as_u64)
            .and_then(|first| usize::try_from(first).ok())
            .unwrap_or(1);
        for (offset, result) in output
            .detail
            .get("results")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            let field = |name: &str| {
                result
                    .get(name)
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned()
            };
            self.sources
                .insert(first + offset, (field("file_id"), field("filename")));
        }
    }

    /// Whether any result can be cited (otherwise text passes unchanged).
    pub(super) fn active(&self) -> bool {
        !self.sources.is_empty()
    }

    /// The text of `piece` to pass on now, and the citations within it.
    pub(super) fn filter(&mut self, piece: &str) -> Vec<Piece> {
        let mut pieces = Vec::new();
        if !self.active() {
            pieces.push(Piece::Text(piece.to_owned()));
            return pieces;
        }
        let mut rest = std::mem::take(&mut self.held);
        rest.push_str(piece);
        let mut text = String::new();
        // Move `text`'s trailing whitespace to the front of `held`.
        let hold_whitespace = |text: &mut String, held: &mut String| {
            let kept = text.trim_end().len();
            held.insert_str(0, text.get(kept..).unwrap_or_default());
            text.truncate(kept);
        };
        loop {
            let Some(open) = rest.find('【') else {
                text.push_str(&rest);
                hold_whitespace(&mut text, &mut self.held);
                break;
            };
            let (before, marker) = rest.split_at(open);
            text.push_str(before);
            let body = marker.get('【'.len_utf8()..).unwrap_or_default();
            match body.split_once('】') {
                Some((inside, after)) => {
                    match self.cited(inside) {
                        Some(cited) => {
                            text.truncate(text.trim_end().len());
                            if !text.is_empty() {
                                pieces.push(Piece::Text(std::mem::take(&mut text)));
                            }
                            pieces.push(Piece::Cite(cited));
                        }
                        None => {
                            let _ = write!(text, "【{inside}】");
                        }
                    }
                    rest = after.to_owned();
                }
                None if body.len() <= MAX_MARKER_BYTES => {
                    marker.clone_into(&mut self.held);
                    hold_whitespace(&mut text, &mut self.held);
                    break;
                }
                None => {
                    text.push('【');
                    rest = body.to_owned();
                }
            }
        }
        if !text.is_empty() {
            pieces.push(Piece::Text(text));
        }
        pieces
    }

    /// The held text, at the end of the answer.
    pub(super) fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// The results a marker's body (`1`, `1, 2`, `3†notes.md`) cites, if
    /// every number in it is one of this Response's.
    fn cited(&self, body: &str) -> Option<Vec<(String, String)>> {
        let mut cited = Vec::new();
        for token in body.split([',', ' ']).filter(|token| !token.is_empty()) {
            let number = token.split('†').next().unwrap_or_default();
            if number.is_empty() || !number.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            cited.push(self.sources.get(&number.parse().ok()?)?.clone());
        }
        (!cited.is_empty()).then_some(cited)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
#[path = "file_search_tests.rs"]
mod tests;
