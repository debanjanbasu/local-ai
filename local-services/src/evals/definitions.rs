//! The typed, validated forms of an eval's data source config and testing
//! criteria and of a run's data source, and the generation requests they
//! turn into.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::graders::{Metric, Operation};
use super::template::{self, Namespaces};
use crate::batches::Endpoint;
use crate::{Error, Result};

/// The default sampling values the spec gives for run data sources.
const DEFAULT_TEMPERATURE: f64 = 1.0;
const DEFAULT_TOP_P: f64 = 1.0;
const DEFAULT_SEED: i64 = 42;
/// The seed every Chat Completions request is sampled with on this server.
///
/// The server refuses Chat's deprecated `seed` member, so a sample generated
/// through Chat Completions reports this one. It equals the engine's default
/// sampling seed (`local-ai`'s tests check that).
pub const CHAT_COMPLETIONS_SEED: i64 = 0x5eed_5eed_cafe_f00d;
/// Why a `seed` cannot be chosen for a generation that runs through Chat
/// Completions.
const CHAT_SEED_REFUSED: &str = "is not supported: this server runs it through Chat Completions, \
     which deprecated seed, so it samples with the server's fixed seed";
/// The server's own default output limit, sent explicitly so a sample
/// reports the limit it ran with.
pub const DEFAULT_MAX_COMPLETION_TOKENS: u64 = 8_192;

pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidArgument(message.into())
}

/// A message role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
    Developer,
}

impl Role {
    const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::Developer => "developer",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageType {
    Message,
}

/// A message of a sampling template or a model grader's input. Its content
/// is text: a string, an `input_text` or `output_text` part, or a list of
/// those; every text may hold template references.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    role: Role,
    content: Value,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    kind: Option<MessageType>,
}

impl Message {
    /// The text parts of the content, refusing media.
    fn texts(&self) -> Result<Vec<&str>> {
        fn part(value: &Value) -> Result<&str> {
            match value {
                Value::String(text) => Ok(text),
                Value::Object(object) => {
                    let kind = object
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    match kind {
                        "input_text" | "output_text" => {}
                        "input_image" | "input_audio" | "input_file" => {
                            return Err(invalid(format!(
                                "message content of type {kind:?} is not supported: Bonsai is \
                                 text-only"
                            )));
                        }
                        other => {
                            return Err(invalid(format!(
                                "message content of type {other:?} is not supported"
                            )));
                        }
                    }
                    if let Some(key) = object
                        .keys()
                        .find(|key| !matches!(key.as_str(), "type" | "text"))
                    {
                        return Err(invalid(format!(
                            "message content field {key:?} is not supported"
                        )));
                    }
                    object
                        .get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| invalid("a text content part needs a text string"))
                }
                _ => Err(invalid("message content must be text")),
            }
        }
        match &self.content {
            Value::Array(parts) => parts.iter().map(part).collect(),
            other => Ok(vec![part(other)?]),
        }
    }

    fn check(&self) -> Result<()> {
        for text in self.texts()? {
            template::check(text).map_err(invalid)?;
        }
        Ok(())
    }

    /// The role and the rendered text (parts concatenated, as the chat
    /// endpoints join them).
    fn render(&self, namespaces: Namespaces<'_>) -> std::result::Result<RenderedMessage, String> {
        let mut text = String::new();
        for part in self.texts().map_err(|error| error.to_string())? {
            text.push_str(&template::render(part, namespaces)?);
        }
        Ok(RenderedMessage {
            role: self.role.as_str().to_owned(),
            content: text,
        })
    }
}

/// A generation request body and the rendered messages it sends, or why it
/// cannot be built for a row.
pub type Built = std::result::Result<(Map<String, Value>, Vec<RenderedMessage>), String>;

/// A message after rendering.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderedMessage {
    pub role: String,
    pub content: String,
}

fn check_messages(messages: &[Message], what: &str) -> Result<()> {
    if messages.is_empty() {
        return Err(invalid(format!("{what} must hold at least one message")));
    }
    messages.iter().try_for_each(Message::check)
}

fn render_messages(
    messages: &[Message],
    namespaces: Namespaces<'_>,
) -> std::result::Result<Vec<RenderedMessage>, String> {
    messages
        .iter()
        .map(|message| message.render(namespaces))
        .collect()
}

// ------------------------------------------------------------------ criteria

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OperationName {
    Eq,
    Ne,
    Like,
    Ilike,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricName {
    Cosine,
    FuzzyMatch,
    Bleu,
    Gleu,
    Meteor,
    #[serde(rename = "rouge_1")]
    Rouge1,
    #[serde(rename = "rouge_2")]
    Rouge2,
    #[serde(rename = "rouge_3")]
    Rouge3,
    #[serde(rename = "rouge_4")]
    Rouge4,
    #[serde(rename = "rouge_5")]
    Rouge5,
    RougeL,
}

impl MetricName {
    fn metric(self) -> Result<Metric> {
        Ok(match self {
            Self::FuzzyMatch => Metric::FuzzyMatch,
            Self::Bleu => Metric::Bleu,
            Self::Gleu => Metric::Gleu,
            Self::Rouge1 => Metric::Rouge(1),
            Self::Rouge2 => Metric::Rouge(2),
            Self::Rouge3 => Metric::Rouge(3),
            Self::Rouge4 => Metric::Rouge(4),
            Self::Rouge5 => Metric::Rouge(5),
            Self::RougeL => Metric::RougeL,
            Self::Cosine => {
                return Err(invalid(
                    "evaluation_metric \"cosine\" is not supported: it needs a validated \
                     embedding model, and this server has none",
                ));
            }
            Self::Meteor => {
                return Err(invalid(
                    "evaluation_metric \"meteor\" is not supported: it needs WordNet synonyms \
                     and stemming, which this server does not implement",
                ));
            }
        })
    }
}

/// Sampling parameters of a `score_model` grader (the spec's names).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraderSampling {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_completions_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

impl GraderSampling {
    /// Refuse what a grader, which runs through Chat Completions, cannot use.
    fn check(&self) -> Result<()> {
        if self.seed.is_some() {
            return Err(invalid(format!(
                "sampling_params.seed of a score_model grader {CHAT_SEED_REFUSED}"
            )));
        }
        if self.max_completions_tokens == Some(0) {
            return Err(invalid(
                "sampling_params.max_completions_tokens must be at least 1",
            ));
        }
        Ok(())
    }
}

/// One testing criterion, as stored and as the `Eval` object shows it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Criterion {
    StringCheck {
        name: String,
        input: String,
        reference: String,
        operation: OperationName,
    },
    TextSimilarity {
        name: String,
        input: String,
        reference: String,
        evaluation_metric: MetricName,
        pass_threshold: f64,
    },
    LabelModel {
        name: String,
        model: String,
        input: Vec<Message>,
        labels: Vec<String>,
        passing_labels: Vec<String>,
    },
    ScoreModel {
        name: String,
        model: String,
        input: Vec<Message>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sampling_params: Option<GraderSampling>,
        pass_threshold: f64,
    },
}

/// What a deterministic grader needs to grade one row.
pub enum Check<'a> {
    String {
        input: &'a str,
        reference: &'a str,
        operation: Operation,
    },
    Similarity {
        input: &'a str,
        reference: &'a str,
        metric: Metric,
        pass_threshold: f64,
    },
}

impl Criterion {
    /// Parse and validate one testing criterion of a create request.
    pub fn parse(value: &Value, index: usize, installed_model: &str) -> Result<Self> {
        let at = format!("testing_criteria[{index}]");
        match value.get("type").and_then(Value::as_str) {
            Some("string_check" | "text_similarity" | "label_model" | "score_model") => {}
            Some("python") => {
                return Err(invalid(format!(
                    "{at}: python graders are not supported: this server has no sandbox to run \
                     grader code in"
                )));
            }
            Some("multi") => {
                return Err(invalid(format!(
                    "{at}: multi graders are not eval testing criteria (they combine graders for \
                     reinforcement fine-tuning)"
                )));
            }
            Some(other) => {
                return Err(invalid(format!(
                    "{at}: testing criterion type {other:?} is not supported"
                )));
            }
            None => return Err(invalid(format!("{at}: type must be a string"))),
        }
        let criterion: Self = serde_json::from_value(value.clone())
            .map_err(|error| invalid(format!("{at}: {error}")))?;
        criterion
            .check(installed_model)
            .map_err(|error| invalid(format!("{at}: {error}")))?;
        Ok(criterion)
    }

    fn check(&self, installed_model: &str) -> Result<()> {
        if self.name().is_empty() {
            return Err(invalid("name must not be empty"));
        }
        let model_check = |model: &str| {
            if model == installed_model {
                Ok(())
            } else {
                Err(invalid(format!(
                    "grader model {model:?} is not the loaded model {installed_model:?}; this \
                     server runs one model"
                )))
            }
        };
        match self {
            Self::StringCheck {
                input, reference, ..
            } => {
                template::check(input).map_err(invalid)?;
                template::check(reference).map_err(invalid)?;
            }
            Self::TextSimilarity {
                input,
                reference,
                evaluation_metric,
                pass_threshold,
                ..
            } => {
                evaluation_metric.metric()?;
                template::check(input).map_err(invalid)?;
                template::check(reference).map_err(invalid)?;
                if !pass_threshold.is_finite() {
                    return Err(invalid("pass_threshold must be a finite number"));
                }
            }
            Self::LabelModel {
                model,
                input,
                labels,
                passing_labels,
                ..
            } => {
                model_check(model)?;
                check_messages(input, "input")?;
                if labels.is_empty() || labels.iter().any(String::is_empty) {
                    return Err(invalid("labels must be one or more non-empty strings"));
                }
                if labels
                    .iter()
                    .enumerate()
                    .any(|(i, label)| labels[..i].contains(label))
                {
                    return Err(invalid("labels must be unique"));
                }
                if passing_labels.is_empty()
                    || passing_labels.iter().any(|label| !labels.contains(label))
                {
                    return Err(invalid(
                        "passing_labels must be a non-empty subset of labels",
                    ));
                }
            }
            Self::ScoreModel {
                model,
                input,
                range,
                sampling_params,
                pass_threshold,
                ..
            } => {
                model_check(model)?;
                check_messages(input, "input")?;
                if let Some(range) = range {
                    match range.as_slice() {
                        [low, high] if low.is_finite() && high.is_finite() && low < high => {}
                        _ => {
                            return Err(invalid(
                                "range must be two finite numbers, the lower one first",
                            ));
                        }
                    }
                }
                if !pass_threshold.is_finite() {
                    return Err(invalid("pass_threshold must be a finite number"));
                }
                if let Some(params) = sampling_params {
                    params.check()?;
                }
            }
        }
        Ok(())
    }

    pub fn name(&self) -> &str {
        match self {
            Self::StringCheck { name, .. }
            | Self::TextSimilarity { name, .. }
            | Self::LabelModel { name, .. }
            | Self::ScoreModel { name, .. } => name,
        }
    }

    pub const fn kind(&self) -> &'static str {
        match self {
            Self::StringCheck { .. } => "string_check",
            Self::TextSimilarity { .. } => "text_similarity",
            Self::LabelModel { .. } => "label_model",
            Self::ScoreModel { .. } => "score_model",
        }
    }

    pub fn uses_model(&self) -> bool {
        self.model().is_some()
    }

    /// The model a model grader runs on.
    pub fn model(&self) -> Option<&str> {
        match self {
            Self::LabelModel { model, .. } | Self::ScoreModel { model, .. } => Some(model),
            Self::StringCheck { .. } | Self::TextSimilarity { .. } => None,
        }
    }

    /// The deterministic check, rendered for one row; `None` for a model
    /// grader.
    pub fn deterministic(&self) -> Option<Check<'_>> {
        match self {
            Self::StringCheck {
                input,
                reference,
                operation,
                ..
            } => Some(Check::String {
                input,
                reference,
                operation: match operation {
                    OperationName::Eq => Operation::Eq,
                    OperationName::Ne => Operation::Ne,
                    OperationName::Like => Operation::Like,
                    OperationName::Ilike => Operation::Ilike,
                },
            }),
            Self::TextSimilarity {
                input,
                reference,
                evaluation_metric,
                pass_threshold,
                ..
            } => evaluation_metric
                .metric()
                .ok()
                .map(|metric| Check::Similarity {
                    input,
                    reference,
                    metric,
                    pass_threshold: *pass_threshold,
                }),
            Self::LabelModel { .. } | Self::ScoreModel { .. } => None,
        }
    }

    /// The `/v1/chat/completions` body that asks the grader model about one
    /// row, and the rendered messages; `Err` when a template cannot be
    /// rendered for it. The answer is constrained to `{"result": ...}`: a
    /// label from `labels`, or a number within `range`.
    pub fn grader_request(&self, namespaces: Namespaces<'_>) -> Option<Built> {
        let (model, input, result, sampling) = match self {
            Self::LabelModel {
                model,
                input,
                labels,
                ..
            } => (
                model,
                input,
                json!({"type": "string", "enum": labels}),
                None,
            ),
            Self::ScoreModel {
                model,
                input,
                sampling_params,
                ..
            } => {
                let (low, high) = self.range();
                (
                    model,
                    input,
                    json!({"type": "number", "minimum": low, "maximum": high}),
                    sampling_params.as_ref(),
                )
            }
            Self::StringCheck { .. } | Self::TextSimilarity { .. } => return None,
        };
        Some(render_messages(input, namespaces).map(|messages| {
            let mut body = Map::new();
            body.insert("model".into(), Value::from(model.as_str()));
            body.insert("messages".into(), json!(messages));
            body.insert(
                "response_format".into(),
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "grader_result",
                        "strict": true,
                        "schema": {
                            "type": "object",
                            "properties": {"result": result},
                            "required": ["result"],
                            "additionalProperties": false,
                        },
                    },
                }),
            );
            if let Some(params) = sampling {
                let mut put = |key: &str, value: Option<Value>| {
                    if let Some(value) = value {
                        body.insert(key.to_owned(), value);
                    }
                };
                put("top_p", params.top_p.map(Value::from));
                put("temperature", params.temperature.map(Value::from));
                put(
                    "max_completion_tokens",
                    params.max_completions_tokens.map(Value::from),
                );
                put(
                    "reasoning_effort",
                    params.reasoning_effort.clone().map(Value::from),
                );
            }
            (body, messages)
        }))
    }

    /// A `score_model`'s range (default 0 to 1).
    pub const fn range(&self) -> (f64, f64) {
        match self {
            Self::ScoreModel {
                range: Some(range), ..
            } => match range.as_slice() {
                [low, high] => (*low, *high),
                _ => (0.0, 1.0),
            },
            _ => (0.0, 1.0),
        }
    }
}

// ------------------------------------------------------- data source config

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CustomConfig {
    #[serde(rename = "type")]
    _kind: String,
    item_schema: Map<String, Value>,
    #[serde(default)]
    include_sample_schema: bool,
}

/// The `schema` of the `sample` namespace a sampling run fills.
fn sample_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "output_text": {"type": "string"},
            "output_json": {"type": "object"},
            "output_tools": {"type": "array", "items": {"type": "object"}},
            "finish_reason": {"type": "string"},
            "usage": {"type": "object"},
        },
        "required": ["model", "output_text", "finish_reason", "usage"],
    })
}

/// Validate a create request's `data_source_config` and return the stored
/// (public) form: `{"type": "custom", "schema": ...}`.
pub fn parse_config(value: &Value) -> Result<Value> {
    match value.get("type").and_then(Value::as_str) {
        Some("custom") => {}
        Some(kind @ ("logs" | "stored_completions")) => {
            return Err(invalid(format!(
                "data_source_config type {kind:?} is not supported: evals on this server cannot \
                 query stored request logs; use a \"custom\" data source"
            )));
        }
        Some(other) => {
            return Err(invalid(format!(
                "data_source_config type {other:?} is not supported"
            )));
        }
        None => return Err(invalid("data_source_config.type must be a string")),
    }
    let config: CustomConfig = serde_json::from_value(value.clone())
        .map_err(|error| invalid(format!("data_source_config: {error}")))?;
    let item_schema = Value::Object(config.item_schema);
    compile(&item_schema)
        .map_err(|error| invalid(format!("data_source_config.item_schema: {error}")))?;
    let mut properties = Map::new();
    properties.insert("item".into(), item_schema);
    let mut required = vec![Value::from("item")];
    if config.include_sample_schema {
        properties.insert("sample".into(), sample_schema());
        required.push(Value::from("sample"));
    }
    Ok(json!({
        "type": "custom",
        "schema": {"type": "object", "properties": properties, "required": required},
    }))
}

/// Compile a JSON Schema offline.
pub fn compile(schema: &Value) -> std::result::Result<jsonschema::Validator, String> {
    jsonschema::options()
        .offline()
        .build(schema)
        .map_err(|error| format!("invalid JSON Schema: {error}"))
}

/// The item schema of a stored config.
pub fn item_schema(config: &Value) -> Value {
    config
        .pointer("/schema/properties/item")
        .cloned()
        .unwrap_or_else(|| json!({}))
}

// ---------------------------------------------------------- run data source

/// One data source row.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub item: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ItemSource {
    FileContent { content: Vec<Row> },
    FileId { id: String },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum InputMessages {
    Template { template: Vec<Message> },
    ItemReference { item_reference: String },
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionsSampling {
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    max_completion_tokens: Option<u64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponsesSampling {
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    max_completion_tokens: Option<u64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    text: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum RunSourceIn {
    Jsonl {
        source: ItemSource,
    },
    Completions {
        #[serde(default)]
        input_messages: Option<InputMessages>,
        #[serde(default)]
        sampling_params: Option<CompletionsSampling>,
        #[serde(default)]
        model: Option<String>,
        source: ItemSource,
    },
    Responses {
        #[serde(default)]
        input_messages: Option<InputMessages>,
        #[serde(default)]
        sampling_params: Option<ResponsesSampling>,
        #[serde(default)]
        model: Option<String>,
        source: ItemSource,
    },
}

/// How a run generates its samples.
#[derive(Clone, Debug)]
pub struct Sampler {
    endpoint: Endpoint,
    model: String,
    messages: Messages,
    temperature: f64,
    top_p: f64,
    seed: i64,
    max_tokens: u64,
    reasoning_effort: Option<String>,
    /// `response_format` and `tools` (Chat) or `text` and `tools`
    /// (Responses), passed through for the endpoint to validate.
    extra: Map<String, Value>,
}

#[derive(Clone, Debug)]
enum Messages {
    Template(Vec<Message>),
    Reference(String),
}

/// A validated run data source.
#[derive(Clone, Debug)]
pub struct RunSource {
    pub source: ItemSource,
    pub sampler: Option<Sampler>,
}

/// Validate a run's `data_source`.
pub fn parse_run_source(value: &Value, installed_model: &str) -> Result<RunSource> {
    if let Some(kind @ ("stored_completions" | "responses")) =
        value.pointer("/source/type").and_then(Value::as_str)
    {
        return Err(invalid(format!(
            "data_source.source type {kind:?} is not supported: evals on this server cannot \
             query stored request logs; use \"file_content\" or \"file_id\""
        )));
    }
    let parsed: RunSourceIn = serde_json::from_value(value.clone())
        .map_err(|error| invalid(format!("data_source: {error}")))?;
    let (endpoint, model, input_messages, source, common, extra) = match parsed {
        RunSourceIn::Jsonl { source } => {
            return Ok(RunSource {
                source,
                sampler: None,
            });
        }
        RunSourceIn::Completions {
            input_messages,
            sampling_params,
            model,
            source,
        } => {
            let params = sampling_params.unwrap_or_default();
            if params.seed.is_some() {
                return Err(invalid(format!(
                    "data_source.sampling_params.seed of a completions data source \
                     {CHAT_SEED_REFUSED}; use a responses data source to choose one"
                )));
            }
            let mut extra = Map::new();
            if let Some(format) = params.response_format {
                extra.insert("response_format".into(), format);
            }
            if let Some(tools) = params.tools {
                extra.insert("tools".into(), Value::Array(tools));
            }
            let common = (
                params.temperature,
                params.top_p,
                Some(CHAT_COMPLETIONS_SEED),
                params.max_completion_tokens,
                params.reasoning_effort,
            );
            let endpoint = Endpoint::ChatCompletions;
            (endpoint, model, input_messages, source, common, extra)
        }
        RunSourceIn::Responses {
            input_messages,
            sampling_params,
            model,
            source,
        } => {
            let params = sampling_params.unwrap_or_default();
            let mut extra = Map::new();
            if let Some(text) = params.text {
                extra.insert("text".into(), text);
            }
            if let Some(tools) = params.tools {
                extra.insert("tools".into(), Value::Array(tools));
            }
            let common = (
                params.temperature,
                params.top_p,
                params.seed,
                params.max_completion_tokens,
                params.reasoning_effort,
            );
            let endpoint = Endpoint::Responses;
            (endpoint, model, input_messages, source, common, extra)
        }
    };
    let (temperature, top_p, seed, max_tokens, reasoning_effort) = common;
    Ok(RunSource {
        source,
        sampler: Some(Sampler {
            endpoint,
            model: sampling_model(model, installed_model)?,
            messages: sampling_messages(input_messages)?,
            temperature: temperature.unwrap_or(DEFAULT_TEMPERATURE),
            top_p: top_p.unwrap_or(DEFAULT_TOP_P),
            seed: seed.unwrap_or(DEFAULT_SEED),
            max_tokens: match max_tokens {
                Some(0) => {
                    return Err(invalid(
                        "data_source.sampling_params.max_completion_tokens must be at least 1",
                    ));
                }
                other => other.unwrap_or(DEFAULT_MAX_COMPLETION_TOKENS),
            },
            reasoning_effort,
            extra,
        }),
    })
}

/// The model a run samples: the loaded one, the only one it may name.
fn sampling_model(model: Option<String>, installed_model: &str) -> Result<String> {
    match model {
        None => Ok(installed_model.to_owned()),
        Some(model) if model == installed_model => Ok(model),
        Some(model) => Err(invalid(format!(
            "data_source.model {model:?} is not the loaded model {installed_model:?}; this \
             server runs one model"
        ))),
    }
}

/// The validated `input_messages` of a sampling run.
fn sampling_messages(messages: Option<InputMessages>) -> Result<Messages> {
    match messages {
        None => Err(invalid(
            "data_source.input_messages is required to generate samples",
        )),
        Some(InputMessages::Template { template }) => {
            check_messages(&template, "data_source.input_messages.template")?;
            Ok(Messages::Template(template))
        }
        Some(InputMessages::ItemReference { item_reference }) => {
            let path = item_reference.trim();
            template::check_path(path).map_err(invalid)?;
            if !path.starts_with("item.") {
                return Err(invalid(
                    "data_source.input_messages.item_reference must name a path in the item \
                     namespace",
                ));
            }
            Ok(Messages::Reference(path.to_owned()))
        }
    }
}

/// The sampling parameters a sample reports.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub temperature: f64,
    pub top_p: f64,
    pub seed: i64,
    pub max_tokens: u64,
}

impl Sampler {
    pub const fn endpoint(&self) -> Endpoint {
        self.endpoint
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub const fn params(&self) -> Params {
        Params {
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed,
            max_tokens: self.max_tokens,
        }
    }

    /// The request body that samples `item`, and the input the sample
    /// reports; `Err` when the messages cannot be built for this row.
    pub fn request(&self, item: &Value) -> Built {
        let namespaces = Namespaces { item, sample: None };
        let (messages, rendered) = match &self.messages {
            Messages::Template(template) => {
                let rendered = render_messages(template, namespaces)?;
                (json!(rendered), rendered)
            }
            Messages::Reference(path) => {
                let value = template::lookup(path, namespaces)?;
                let Value::Array(messages) = value else {
                    return Err(format!("{path} must be an array of messages"));
                };
                let rendered = messages
                    .iter()
                    .map(|message| {
                        let role = message.get("role").and_then(Value::as_str);
                        let content = message.get("content");
                        match (role, content) {
                            (Some(role), Some(content)) => Ok(RenderedMessage {
                                role: role.to_owned(),
                                content: content
                                    .as_str()
                                    .map_or_else(|| content.to_string(), str::to_owned),
                            }),
                            _ => Err(format!("{path} must hold messages with a role and content")),
                        }
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                (value.clone(), rendered)
            }
        };
        let mut body = Map::new();
        body.insert("model".into(), Value::from(self.model.as_str()));
        body.insert("temperature".into(), Value::from(self.temperature));
        body.insert("top_p".into(), Value::from(self.top_p));
        match self.endpoint {
            Endpoint::Responses => {
                body.insert("seed".into(), Value::from(self.seed));
                body.insert("input".into(), messages);
                body.insert("max_output_tokens".into(), Value::from(self.max_tokens));
                body.insert("store".into(), Value::Bool(false));
                if let Some(effort) = &self.reasoning_effort {
                    body.insert("reasoning".into(), json!({"effort": effort}));
                }
            }
            Endpoint::ChatCompletions | Endpoint::RetiredCompletions => {
                body.insert("messages".into(), messages);
                body.insert("max_completion_tokens".into(), Value::from(self.max_tokens));
                if let Some(effort) = &self.reasoning_effort {
                    body.insert("reasoning_effort".into(), Value::from(effort.as_str()));
                }
            }
        }
        for (key, value) in &self.extra {
            body.insert(key.clone(), value.clone());
        }
        Ok((body, rendered))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    const MODEL: &str = "local";

    fn criterion(value: &Value) -> Result<Criterion> {
        Criterion::parse(value, 0, MODEL)
    }

    #[test]
    fn refused_graders_explain_why() {
        let cases = [
            (
                json!({"type": "python", "name": "p", "source": "def grade(s, i): return 1"}),
                "sandbox",
            ),
            (
                json!({"type": "multi", "name": "m", "graders": {}, "calculate_output": "1"}),
                "multi",
            ),
            (
                json!({"type": "text_similarity", "name": "c", "input": "a", "reference": "b",
                       "evaluation_metric": "cosine", "pass_threshold": 0.5}),
                "embedding",
            ),
            (
                json!({"type": "text_similarity", "name": "m", "input": "a", "reference": "b",
                       "evaluation_metric": "meteor", "pass_threshold": 0.5}),
                "WordNet",
            ),
            (
                json!({"type": "score_model", "name": "s", "model": "gpt-4o", "pass_threshold": 0.5,
                       "input": [{"role": "user", "content": "rate"}]}),
                "loaded model",
            ),
            (
                json!({"type": "score_model", "name": "s", "model": MODEL, "pass_threshold": 0.5,
                       "sampling_params": {"seed": 7}, "input": [{"role": "user", "content": "rate"}]}),
                "deprecated seed",
            ),
            (
                json!({"type": "label_model", "name": "l", "model": MODEL, "labels": ["a"],
                       "passing_labels": ["b"], "input": [{"role": "user", "content": "x"}]}),
                "subset",
            ),
            (
                json!({"type": "string_check", "name": "s", "input": "{{items.x}}", "reference": "a",
                       "operation": "eq"}),
                "item. or sample.",
            ),
            (
                json!({"type": "string_check", "name": "s", "input": "a", "reference": "a",
                       "operation": "eq", "extra": 1}),
                "unknown field",
            ),
            (
                json!({"type": "score_model", "name": "s", "model": MODEL, "pass_threshold": 0.5,
                       "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]}),
                "text-only",
            ),
        ];
        for (value, why) in cases {
            let error = criterion(&value).expect_err("refused").to_string();
            assert!(error.contains(why), "{error:?} should mention {why:?}");
        }
    }

    #[test]
    fn criteria_round_trip_in_their_public_shape() {
        let value = json!({"type": "score_model", "name": "s", "model": MODEL, "range": [1.0, 5.0],
                           "pass_threshold": 3.0, "sampling_params": {"temperature": 0.5},
                           "input": [{"role": "system", "content": "rate"},
                                     {"role": "user", "content": [{"type": "input_text", "text": "{{item.q}}"}]}]});
        let parsed = criterion(&value).expect("accepted");
        assert_eq!(serde_json::to_value(&parsed).expect("json"), value);
        let item = json!({"q": "why?"});
        let (body, messages) = parsed
            .grader_request(Namespaces {
                item: &item,
                sample: None,
            })
            .expect("model grader")
            .expect("renders");
        assert_eq!(messages[1].content, "why?");
        assert_eq!(body["temperature"], 0.5);
        // Graders run through Chat Completions, which deprecated `seed`.
        assert!(!body.contains_key("seed"));
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["properties"]["result"],
            json!({"type": "number", "minimum": 1.0, "maximum": 5.0})
        );
    }

    #[test]
    fn custom_configs_wrap_the_item_schema() {
        let config = parse_config(&json!({"type": "custom", "include_sample_schema": true,
                                          "item_schema": {"type": "object"}}))
        .expect("accepted");
        assert_eq!(
            config["schema"]["properties"]["item"],
            json!({"type": "object"})
        );
        assert_eq!(config["schema"]["required"], json!(["item", "sample"]));
        for refused in [
            json!({"type": "logs"}),
            json!({"type": "stored_completions"}),
            json!({"type": "custom", "item_schema": {"type": 5}}),
        ] {
            assert!(parse_config(&refused).is_err(), "{refused} refused");
        }
    }

    #[test]
    fn sampling_requests_carry_the_spec_defaults() {
        let source = parse_run_source(
            &json!({"type": "completions",
                    "input_messages": {"type": "template",
                                       "template": [{"role": "user", "content": "Say {{item.word}}"}]},
                    "source": {"type": "file_content", "content": [{"item": {"word": "hi"}}]}}),
            MODEL,
        )
        .expect("accepted");
        let sampler = source.sampler.expect("samples");
        let (body, input) = sampler.request(&json!({"word": "hi"})).expect("renders");
        assert_eq!(
            Value::Object(body),
            json!({"model": MODEL, "messages": [{"role": "user", "content": "Say hi"}],
                   "temperature": 1.0, "top_p": 1.0, "max_completion_tokens": 8192})
        );
        assert_eq!(input[0].content, "Say hi");
        assert_eq!(sampler.params().seed, CHAT_COMPLETIONS_SEED);
        let source = parse_run_source(
            &json!({"type": "responses",
                    "input_messages": {"type": "template",
                                       "template": [{"role": "user", "content": "Say {{item.word}}"}]},
                    "source": {"type": "file_content", "content": [{"item": {"word": "hi"}}]}}),
            MODEL,
        )
        .expect("accepted");
        let sampler = source.sampler.expect("samples");
        let (body, _) = sampler.request(&json!({"word": "hi"})).expect("renders");
        assert_eq!(body["seed"], 42, "Responses takes the spec's default seed");
        let error = parse_run_source(
            &json!({"type": "completions", "sampling_params": {"seed": 7},
                    "input_messages": {"type": "template", "template": [{"role": "user", "content": "x"}]},
                    "source": {"type": "file_content", "content": []}}),
            MODEL,
        )
        .expect_err("a Chat seed is refused")
        .to_string();
        assert!(error.contains("deprecated seed"), "{error}");
        for refused in [
            json!({"type": "completions", "model": "gpt-4o",
                   "input_messages": {"type": "template", "template": [{"role": "user", "content": "x"}]},
                   "source": {"type": "file_content", "content": []}}),
            json!({"type": "completions", "source": {"type": "file_content", "content": []}}),
            json!({"type": "completions", "source": {"type": "stored_completions"}}),
            json!({"type": "responses", "source": {"type": "responses"}}),
        ] {
            assert!(
                parse_run_source(&refused, MODEL).is_err(),
                "{refused} refused"
            );
        }
    }
}
