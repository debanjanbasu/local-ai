//! Chat Completions request parsing.
//!
//! Chat `response_format` is mapped onto the engine's native
//! [`ResponseFormat`]: `text`, `json_object`, and `json_schema` with its
//! nested `json_schema` object (see [`chat_response_format`]). The engine
//! compiles the schema before queueing and enforces it on every answer token,
//! whatever `strict` says.
//!
//! Chat `tool_choice` (`auto`, `none`, `required` or
//! `{"type":"function","function":{"name"}}`), `parallel_tool_calls` and a
//! tool's `function.strict` map one to one onto the engine's native
//! [`ToolChoice`], `parallel_tool_calls` and [`ToolDefinition::strict`],
//! which the engine enforces with its tool grammar (see [`tool_policy`]).
//!
//! Members the `OpenAI` specification marks deprecated are honoured, because
//! harnesses and older SDKs still send them: `max_tokens` is
//! `max_completion_tokens`, `seed` seeds the sampler, `functions` and
//! `function_call` are function `tools` and `tool_choice` (answered in their
//! own legacy shape, see [`PreparedGeneration::functions`]), `session_id` is
//! `prompt_cache_key`, and `user` and `prompt_cache_retention`, which change
//! nothing here, are ignored like any other unknown member.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use local_engine::{
    ChatMessage, ChatRequest, ReasoningEffort, ResponseFormat, Sampling, ToolCall, ToolChoice,
    ToolDefinition,
};

use crate::GenerateParams;

#[derive(Deserialize)]
struct GenerateRequest {
    /// Checked by [`super::models::check_generation_value`]; the loaded model
    /// answers.
    model: Option<Value>,
    messages: Option<Vec<Message>>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    stream: Option<bool>,
    stream_options: Option<StreamOptions>,
    prompt_cache_key: Option<String>,
    tools: Option<Vec<ChatTool>>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
    reasoning_effort: Option<String>,
    /// Deprecated spelling of `max_completion_tokens`, which wins when both
    /// are set.
    max_tokens: Option<usize>,
    /// Any JSON integer; a negative one is reinterpreted as its 64 bits.
    seed: Option<serde_json::Number>,
    /// Deprecated spelling of `tools` (functions only).
    functions: Option<Vec<ChatFunction>>,
    /// Deprecated spelling of `tool_choice`: `"none"`, `"auto"` or
    /// `{"name"}`.
    function_call: Option<Value>,
    /// This server's former spelling of `prompt_cache_key`.
    session_id: Option<String>,
    /// vLLM's and llama.cpp's multiplicative penalty, used as given; it
    /// replaces the one `frequency_penalty` maps to.
    repetition_penalty: Option<f32>,
    /// vLLM's and llama.cpp's template switches; `enable_thinking` turns
    /// reasoning off (`false`) or leaves it to `reasoning_effort` (`true`).
    chat_template_kwargs: Option<ChatTemplateKwargs>,
    // Options this server cannot honour. They are named so a request that
    // sets one fails instead of being answered as if it had not.
    n: Option<u64>,
    stop: Option<Value>,
    logprobs: Option<Value>,
    top_logprobs: Option<u64>,
    logit_bias: Option<Value>,
    store: Option<bool>,
    verbosity: Option<String>,
    response_format: Option<Value>,
    audio: Option<Value>,
    modalities: Option<Vec<String>>,
    prediction: Option<Value>,
    web_search_options: Option<Value>,
    service_tier: Option<String>,
}

#[derive(Deserialize)]
struct ChatTemplateKwargs {
    enable_thinking: Option<bool>,
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: Option<bool>,
}

impl GenerateRequest {
    fn params(&self) -> GenerateParams {
        let defaults = GenerateParams::default();
        GenerateParams {
            max_tokens: self
                .max_completion_tokens
                .or(self.max_tokens)
                .unwrap_or(defaults.max_tokens),
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            min_p: self.min_p.unwrap_or(defaults.min_p),
            presence_penalty: self.presence_penalty.unwrap_or(defaults.presence_penalty),
            repetition_penalty: repetition_penalty(self.repetition_penalty, self.frequency_penalty),
            seed: self
                .seed
                .as_ref()
                .and_then(|seed| {
                    seed.as_u64()
                        .or_else(|| seed.as_i64().map(i64::cast_unsigned))
                })
                .unwrap_or(defaults.seed),
        }
    }

    /// Fail on any named option whose effect this server cannot produce.
    fn reject_unsupported(&self) -> crate::Result<()> {
        let unsupported = |name: &str, why: &str| {
            Err(crate::Error::InvalidArgument(format!(
                "{name} is not supported: {why}"
            )))
        };
        if self
            .seed
            .as_ref()
            .is_some_and(|seed| seed.as_u64().is_none() && seed.as_i64().is_none())
        {
            return Err(crate::Error::InvalidArgument(
                "seed must be an integer".into(),
            ));
        }
        if self.n.is_some_and(|n| n != 1) {
            return unsupported("n", "this server generates exactly one choice");
        }
        if self.stop.as_ref().is_some_and(|stop| !is_empty(stop)) {
            return unsupported("stop", "stop sequences are not implemented");
        }
        let logprobs = match self.logprobs.as_ref() {
            None | Some(Value::Null) => false,
            Some(Value::Bool(enabled)) => *enabled,
            _ => {
                return Err(crate::Error::InvalidArgument(
                    "logprobs must be a boolean".into(),
                ));
            }
        };
        if logprobs || self.top_logprobs.is_some_and(|n| n > 0) {
            return unsupported("logprobs", "token log-probabilities are not reported");
        }
        if self.logit_bias.as_ref().is_some_and(|bias| !is_empty(bias)) {
            return unsupported("logit_bias", "the sampler has no per-token bias");
        }
        if self.store == Some(true) {
            return unsupported("store", "this server stores no completions");
        }
        if !matches!(self.verbosity.as_deref(), None | Some("medium")) {
            return unsupported("verbosity", "only the default, medium, is produced");
        }
        if self.audio.as_ref().is_some_and(|audio| !audio.is_null()) {
            return unsupported("audio", "Bonsai 2 is text-only");
        }
        if self
            .modalities
            .as_ref()
            .is_some_and(|modalities| modalities.iter().any(|modality| modality != "text"))
        {
            return unsupported("modalities", "Bonsai 2 only produces text");
        }
        if self
            .prediction
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            return unsupported("prediction", "predicted outputs are not implemented");
        }
        if self
            .web_search_options
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            return unsupported("web_search_options", "there is no built-in web search");
        }
        service_tier(self.service_tier.as_deref())
    }
}

/// The sampler's multiplicative repetition penalty: vLLM's
/// `repetition_penalty` as given, else `OpenAI`'s additive
/// `frequency_penalty` as `1 + max(value, 0)`, else none.
pub(super) fn repetition_penalty(repetition: Option<f32>, frequency: Option<f32>) -> f32 {
    repetition
        .or_else(|| frequency.map(|value| 1.0 + value.max(0.0)))
        .unwrap_or_else(|| GenerateParams::default().repetition_penalty)
}

fn is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(text) => text.is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(map) => map.is_empty(),
        _ => false,
    }
}

/// Accept only the service tiers that mean "however this server runs".
pub(super) fn service_tier(tier: Option<&str>) -> crate::Result<()> {
    match tier {
        None | Some("auto" | "default") => Ok(()),
        Some(other) => Err(crate::Error::InvalidArgument(format!(
            "service_tier {other:?} is not supported; this server has one tier (use \"auto\" or \
             \"default\")"
        ))),
    }
}

#[derive(Deserialize)]
struct Message {
    role: String,
    /// Absent and `null` are the same: an assistant turn that only called tools,
    /// or a tool that returned nothing.
    #[serde(default)]
    content: Value,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ChatToolCall>>,
    tool_call_id: Option<String>,
    /// Deprecated spelling of a single `tool_calls` entry: `{"name","arguments"}`.
    function_call: Option<Value>,
}

#[derive(Deserialize)]
struct ChatToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: Option<String>,
    function: ChatFunctionCall,
}

#[derive(Deserialize)]
struct ChatFunctionCall {
    name: String,
    arguments: Value,
}

#[derive(Deserialize)]
struct ChatTool {
    #[serde(rename = "type")]
    kind: String,
    function: Option<ChatFunction>,
}

#[derive(Deserialize)]
struct ChatFunction {
    name: String,
    description: Option<String>,
    parameters: Option<Value>,
    strict: Option<bool>,
}

pub(super) struct PreparedGeneration {
    pub(super) stream: bool,
    /// `stream_options.include_usage`.
    pub(super) include_usage: bool,
    /// The request declared deprecated `functions`, so its call is answered
    /// as `message.function_call` with finish reason `function_call`, the
    /// shape such a client reads.
    pub(super) functions: bool,
    pub(super) request: ChatRequest,
}

/// Parse a Chat Completions request body.
#[cfg(test)]
pub(super) fn prepare_generation(body: &[u8], thinking: bool) -> crate::Result<PreparedGeneration> {
    prepare_generation_with(body, thinking, None)
}

/// Parse a Chat Completions request body, resolving `file_id` images
/// against `files`.
pub(super) fn prepare_generation_with(
    body: &[u8],
    thinking: bool,
    files: Option<&local_services::Store>,
) -> crate::Result<PreparedGeneration> {
    let request: GenerateRequest = serde_json::from_slice(body)
        .map_err(|error| crate::Error::InvalidArgument(format!("invalid JSON request: {error}")))?;
    let params = request.params();
    let stream = request.stream.unwrap_or(false);
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    request.reject_unsupported()?;
    super::models::check_generation_value(request.model.as_ref())?;
    let messages = request
        .messages
        .as_deref()
        .ok_or_else(|| crate::Error::InvalidArgument("messages are required".into()))?;
    // The ID of the last deprecated assistant `function_call`, which the
    // `function` message after it answers.
    let mut pending = None;
    let messages = messages
        .iter()
        .enumerate()
        .map(|(index, message)| chat_message(message, index, &mut pending, files))
        .collect::<crate::Result<Vec<_>>>()?;
    let functions = request.functions.is_some();
    let (tools, tool_choice) = match (&request.functions, &request.tools) {
        (Some(_), Some(_)) => {
            return Err(crate::Error::InvalidArgument(
                "use tools or the deprecated functions, not both".into(),
            ));
        }
        (Some(functions), None) => {
            if request.tool_choice.is_some() {
                return Err(crate::Error::InvalidArgument(
                    "the deprecated functions take function_call, not tool_choice".into(),
                ));
            }
            let tools = functions
                .iter()
                .map(|function| {
                    function_definition(
                        function.name.clone(),
                        function.description.clone(),
                        function.parameters.clone(),
                        function.strict,
                    )
                })
                .collect::<crate::Result<Vec<_>>>()?;
            (tools, function_call_choice(request.function_call.as_ref())?)
        }
        (None, tools) => {
            if request
                .function_call
                .as_ref()
                .is_some_and(|value| !value.is_null())
            {
                return Err(crate::Error::InvalidArgument(
                    "function_call selects one of the deprecated functions; with tools use \
                     tool_choice"
                        .into(),
                ));
            }
            let tools = tools
                .as_deref()
                .unwrap_or_default()
                .iter()
                .map(chat_tool)
                .collect::<crate::Result<Vec<_>>>()?;
            (tools, chat_tool_choice(request.tool_choice.as_ref())?)
        }
    };
    tool_policy(&tools, &tool_choice)?;
    let response_format = chat_response_format(request.response_format.as_ref())?;
    let effort = match &request.chat_template_kwargs {
        Some(ChatTemplateKwargs {
            enable_thinking: Some(false),
        }) => Some("none"),
        _ => request.reasoning_effort.as_deref(),
    };
    let reasoning = effective_reasoning(effort, thinking)?;
    Ok(PreparedGeneration {
        stream,
        include_usage: stream && include_usage,
        functions,
        request: ChatRequest {
            messages,
            max_tokens: params.max_tokens,
            sampling: Sampling(params),
            thinking: reasoning.is_some(),
            effort: reasoning.unwrap_or_default(),
            preserve_thinking: true,
            session: request.prompt_cache_key.or(request.session_id),
            tools,
            tool_choice,
            // A deprecated function call is always the only one.
            parallel_tool_calls: !functions && request.parallel_tool_calls.unwrap_or(true),
            response_format,
        },
    })
}

/// The deprecated `function_call`: absent, `null` or `"auto"` is `auto`,
/// `"none"` is none, and `{"name":"NAME"}` names the function to call.
fn function_call_choice(choice: Option<&Value>) -> crate::Result<ToolChoice> {
    match choice {
        None | Some(Value::Null) => Ok(ToolChoice::Auto),
        Some(Value::String(mode)) if mode == "auto" => Ok(ToolChoice::Auto),
        Some(Value::String(mode)) if mode == "none" => Ok(ToolChoice::None),
        Some(Value::Object(object)) => {
            only_members(object, &["name"], "function_call")?;
            Ok(ToolChoice::Function(choice_name(object, "function_call")?))
        }
        Some(other) => Err(crate::Error::InvalidArgument(format!(
            "invalid function_call {other}; expected \"auto\", \"none\" or {{\"name\":...}}"
        ))),
    }
}

/// Chat's `response_format`: absent or `null` is text; otherwise exactly
/// `{"type":"text"}`, `{"type":"json_object"}`, or
/// `{"type":"json_schema","json_schema":{"name",["description"],"schema",["strict"]}}`.
///
/// Unknown members, wrong types, an invalid `name` and a missing `schema`
/// fail here; the schema itself is validated by the engine's compiler, which
/// enforces it whether `strict` is `true`, `false` or absent.
pub(super) fn chat_response_format(value: Option<&Value>) -> crate::Result<ResponseFormat> {
    const FIELD: &str = "response_format";
    let Some((kind, object)) = format_object(value, FIELD)? else {
        return Ok(ResponseFormat::Text);
    };
    match kind {
        "text" | "json_object" => {
            only_members(object, &["type"], FIELD)?;
            Ok(if kind == "text" {
                ResponseFormat::Text
            } else {
                ResponseFormat::JsonObject
            })
        }
        "json_schema" => {
            only_members(object, &["type", "json_schema"], FIELD)?;
            let nested = object
                .get("json_schema")
                .ok_or_else(|| invalid_format(format!("{FIELD}.json_schema is required")))?
                .as_object()
                .ok_or_else(|| invalid_format(format!("{FIELD}.json_schema must be an object")))?;
            let field = format!("{FIELD}.json_schema");
            only_members(nested, &["name", "description", "schema", "strict"], &field)?;
            Ok(ResponseFormat::JsonSchema(
                SchemaFormat::parse(nested, &field)?.schema,
            ))
        }
        other => Err(unsupported_format(FIELD, other)),
    }
}

/// The Responses `text.format`: absent or `null` is text; otherwise exactly
/// `{"type":"text"}`, `{"type":"json_object"}`, or the flat
/// `{"type":"json_schema","name",["description"],"schema",["strict"]}`.
///
/// Returns the native format and the format the Response object echoes: the
/// accepted one, with `strict` reported as sent (`false` when omitted, its
/// documented default). The schema is enforced either way.
pub(super) fn responses_text_format(
    value: Option<&Value>,
) -> crate::Result<(ResponseFormat, Value)> {
    const FIELD: &str = "text.format";
    let Some((kind, object)) = format_object(value, FIELD)? else {
        return Ok((ResponseFormat::Text, json!({"type":"text"})));
    };
    match kind {
        "text" => {
            only_members(object, &["type"], FIELD)?;
            Ok((ResponseFormat::Text, json!({"type":"text"})))
        }
        "json_object" => {
            only_members(object, &["type"], FIELD)?;
            Ok((ResponseFormat::JsonObject, json!({"type":"json_object"})))
        }
        "json_schema" => {
            only_members(
                object,
                &["type", "name", "description", "schema", "strict"],
                FIELD,
            )?;
            let format = SchemaFormat::parse(object, FIELD)?;
            let mut echo = json!({
                "type":"json_schema",
                "name":format.name,
                "schema":format.schema,
                "strict":format.strict.unwrap_or(false),
            });
            if let Some(description) = format.description {
                echo["description"] = json!(description);
            }
            Ok((ResponseFormat::JsonSchema(format.schema), echo))
        }
        other => Err(unsupported_format(FIELD, other)),
    }
}

/// A `json_schema` format's members, the same in both APIs.
struct SchemaFormat {
    name: String,
    description: Option<String>,
    schema: Value,
    strict: Option<bool>,
}

impl SchemaFormat {
    fn parse(object: &Map<String, Value>, field: &str) -> crate::Result<Self> {
        let name = match object.get("name") {
            Some(Value::String(name)) => name.clone(),
            None | Some(Value::Null) => {
                return Err(invalid_format(format!("{field}.name is required")));
            }
            Some(_) => return Err(invalid_format(format!("{field}.name must be a string"))),
        };
        if name.is_empty()
            || name.len() > 64
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(invalid_format(format!(
                "{field}.name {name:?} must be 1 to 64 characters of a-z, A-Z, 0-9, _ or -"
            )));
        }
        let description = match object.get("description") {
            None | Some(Value::Null) => None,
            Some(Value::String(text)) => Some(text.clone()),
            Some(_) => {
                return Err(invalid_format(format!(
                    "{field}.description must be a string"
                )));
            }
        };
        let schema = match object.get("schema") {
            Some(schema @ Value::Object(_)) => schema.clone(),
            None | Some(Value::Null) => {
                return Err(invalid_format(format!(
                    "{field}.schema is required: without one there is nothing to enforce (use \
                     json_object for any JSON object)"
                )));
            }
            Some(_) => {
                return Err(invalid_format(format!(
                    "{field}.schema must be a JSON Schema object"
                )));
            }
        };
        let strict = match object.get("strict") {
            None | Some(Value::Null) => None,
            Some(Value::Bool(strict)) => Some(*strict),
            Some(_) => return Err(invalid_format(format!("{field}.strict must be a boolean"))),
        };
        Ok(Self {
            name,
            description,
            schema,
            strict,
        })
    }
}

/// A format's `type` and members, or `None` for an absent or `null` format.
fn format_object<'a>(
    value: Option<&'a Value>,
    field: &str,
) -> crate::Result<Option<(&'a str, &'a Map<String, Value>)>> {
    let object = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(object)) => object,
        Some(_) => return Err(invalid_format(format!("{field} must be an object"))),
    };
    match object.get("type") {
        Some(Value::String(kind)) => Ok(Some((kind.as_str(), object))),
        None | Some(Value::Null) => Err(invalid_format(format!("{field}.type is required"))),
        Some(_) => Err(invalid_format(format!("{field}.type must be a string"))),
    }
}

fn only_members(object: &Map<String, Value>, allowed: &[&str], field: &str) -> crate::Result<()> {
    object
        .keys()
        .find(|key| !allowed.contains(&key.as_str()))
        .map_or(Ok(()), |key| {
            Err(invalid_format(format!(
                "unknown field {field}.{key}; expected only {}",
                allowed.join(", ")
            )))
        })
}

fn unsupported_format(field: &str, kind: &str) -> crate::Error {
    crate::Error::InvalidArgument(format!(
        "{field} type {kind:?} is not supported; use \"text\", \"json_object\" or \"json_schema\""
    ))
}

const fn invalid_format(message: String) -> crate::Error {
    crate::Error::InvalidArgument(message)
}

/// Resolve a reasoning effort against the server's `--no-thinking` choice:
/// the level the model reasons at, or `None` for no reasoning.
///
/// The checkpoint template knows `low`, `medium` and `xhigh`, or closes the
/// thought block empty (`none`). `OpenAI`'s other levels run at the nearest
/// one, `minimal` at `low` and `high` at `xhigh`, and a Response reports the
/// level that ran. An operator who started the server with `--no-thinking`
/// has decided, so a request cannot turn reasoning back on.
pub(super) fn effective_reasoning(
    effort: Option<&str>,
    server: bool,
) -> crate::Result<Option<ReasoningEffort>> {
    let level = match effort {
        None => return Ok(server.then_some(ReasoningEffort::XHigh)),
        Some("none") => return Ok(None),
        Some("minimal" | "low") => ReasoningEffort::Low,
        Some("medium") => ReasoningEffort::Medium,
        Some("high" | "xhigh") => ReasoningEffort::XHigh,
        Some(other) => {
            return Err(crate::Error::InvalidArgument(format!(
                "reasoning effort {other:?} is not supported; use none, minimal, low, medium, \
                 high or xhigh"
            )));
        }
    };
    if !server {
        return Err(crate::Error::InvalidArgument(format!(
            "reasoning effort {:?} was requested but this server was started with --no-thinking",
            effort.unwrap_or_default()
        )));
    }
    Ok(Some(level))
}

/// Check `choice` against the declared `tools` before anything is queued.
///
/// The engine enforces every choice with its native tool grammar: `none`
/// shows the model the tools (the prompt is the same for every choice) but
/// its answer can never contain a call, `required` makes at least one call
/// mandatory, and a named function makes exactly that one mandatory. A
/// mandatory call needs a tool to call, and a named one must be declared;
/// both are refused here with the request's own field names.
pub(super) fn tool_policy(tools: &[ToolDefinition], choice: &ToolChoice) -> crate::Result<()> {
    match choice {
        ToolChoice::Required if tools.is_empty() => Err(crate::Error::InvalidArgument(
            "tool_choice \"required\" needs at least one tool in tools".into(),
        )),
        ToolChoice::Function(name) if !tools.iter().any(|tool| tool.name == *name) => {
            Err(crate::Error::InvalidArgument(format!(
                "tool_choice names function {name:?}, which is not declared in tools"
            )))
        }
        ToolChoice::Allowed { tools: allowed, .. } => allowed
            .iter()
            .find(|name| !tools.iter().any(|tool| tool.name == **name))
            .map_or(Ok(()), |name| {
                Err(crate::Error::InvalidArgument(format!(
                    "tool_choice allowed_tools names {name:?}, which is not declared in tools"
                )))
            }),
        _ => Ok(()),
    }
}

/// An `allowed_tools` choice: `mode` (`auto`, the default, or `required`)
/// and a non-empty `tools` list, each entry's tool name given by `name`.
pub(super) fn allowed_tools(
    mode: Option<&Value>,
    tools: Option<&Value>,
    name: impl Fn(&Value) -> crate::Result<String>,
) -> crate::Result<ToolChoice> {
    let required = match mode {
        None | Some(Value::Null) => false,
        Some(Value::String(mode)) if mode == "auto" => false,
        Some(Value::String(mode)) if mode == "required" => true,
        Some(other) => {
            return Err(crate::Error::InvalidArgument(format!(
                "allowed_tools mode must be \"auto\" or \"required\", not {other}"
            )));
        }
    };
    let tools = match tools {
        Some(Value::Array(tools)) if !tools.is_empty() => {
            tools.iter().map(name).collect::<crate::Result<Vec<_>>>()?
        }
        _ => {
            return Err(crate::Error::InvalidArgument(
                "allowed_tools needs a non-empty tools list".into(),
            ));
        }
    };
    Ok(ToolChoice::Allowed { tools, required })
}

/// A `tool_choice` mode string shared by both APIs.
fn choice_mode(mode: &str, named: &str) -> crate::Result<ToolChoice> {
    match mode {
        "auto" => Ok(ToolChoice::Auto),
        "none" => Ok(ToolChoice::None),
        "required" => Ok(ToolChoice::Required),
        other => Err(crate::Error::InvalidArgument(format!(
            "invalid tool_choice {other:?}; expected \"auto\", \"none\", \"required\" or {named}"
        ))),
    }
}

/// A named function's `name`, which must be a non-empty string.
fn choice_name(object: &Map<String, Value>, field: &str) -> crate::Result<String> {
    match object.get("name") {
        Some(Value::String(name)) if !name.is_empty() => Ok(name.clone()),
        Some(Value::String(_)) => Err(crate::Error::InvalidArgument(format!(
            "{field}.name must not be empty"
        ))),
        None | Some(Value::Null) => Err(crate::Error::InvalidArgument(format!(
            "{field}.name is required"
        ))),
        Some(_) => Err(crate::Error::InvalidArgument(format!(
            "{field}.name must be a string"
        ))),
    }
}

/// The `type` of a `tool_choice` object, refusing the kinds this server
/// cannot honour by name.
fn choice_kind<'a>(object: &'a Map<String, Value>, function_shape: &str) -> crate::Result<&'a str> {
    match object.get("type") {
        Some(Value::String(kind)) if kind == "function" => Ok(kind),
        Some(Value::String(kind)) => Err(crate::Error::InvalidArgument(format!(
            "tool_choice type {kind:?} is not supported; only function tools run here, named as \
             {function_shape}"
        ))),
        None | Some(Value::Null) => Err(crate::Error::InvalidArgument(
            "tool_choice.type is required".into(),
        )),
        Some(_) => Err(crate::Error::InvalidArgument(
            "tool_choice.type must be a string".into(),
        )),
    }
}

const CHAT_NAMED: &str = "{\"type\":\"function\",\"function\":{\"name\":...}}";
const RESPONSES_NAMED: &str = "{\"type\":\"function\",\"name\":...}";

/// Chat's `tool_choice`: absent or `null` is `auto`; otherwise exactly
/// `"auto"`, `"none"`, `"required"` or
/// `{"type":"function","function":{"name":"NAME"}}`. The Responses shape,
/// unknown members and other tool kinds fail.
pub(super) fn chat_tool_choice(choice: Option<&Value>) -> crate::Result<ToolChoice> {
    let object = match choice {
        None | Some(Value::Null) => return Ok(ToolChoice::Auto),
        Some(Value::String(mode)) => return choice_mode(mode, CHAT_NAMED),
        Some(Value::Object(object)) => object,
        Some(other) => {
            return Err(crate::Error::InvalidArgument(format!(
                "invalid tool_choice {other}; expected \"auto\", \"none\", \"required\" or \
                 {CHAT_NAMED}"
            )));
        }
    };
    if object.get("type").and_then(Value::as_str) == Some("allowed_tools") {
        only_members(object, &["type", "allowed_tools"], "tool_choice")?;
        let allowed = object
            .get("allowed_tools")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                crate::Error::InvalidArgument("tool_choice.allowed_tools must be an object".into())
            })?;
        only_members(allowed, &["mode", "tools"], "tool_choice.allowed_tools")?;
        return allowed_tools(allowed.get("mode"), allowed.get("tools"), |tool| {
            let function = tool
                .get("function")
                .and_then(Value::as_object)
                .filter(|_| tool.get("type").and_then(Value::as_str) == Some("function"))
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "allowed_tools entries must be {CHAT_NAMED}"
                    ))
                })?;
            choice_name(function, "allowed_tools.tools[].function")
        });
    }
    choice_kind(object, CHAT_NAMED)?;
    only_members(object, &["type", "function"], "tool_choice")?;
    let function = match object.get("function") {
        Some(Value::Object(function)) => function,
        None | Some(Value::Null) => {
            return Err(crate::Error::InvalidArgument(format!(
                "tool_choice.function is required: name a function as {CHAT_NAMED}"
            )));
        }
        Some(_) => {
            return Err(crate::Error::InvalidArgument(
                "tool_choice.function must be an object".into(),
            ));
        }
    };
    only_members(function, &["name"], "tool_choice.function")?;
    Ok(ToolChoice::Function(choice_name(
        function,
        "tool_choice.function",
    )?))
}

/// The Responses `tool_choice`: absent or `null` is `auto`; otherwise
/// exactly `"auto"`, `"none"`, `"required"` or
/// `{"type":"function","name":"NAME"}`. Returns the native choice and the
/// value the Response object echoes: the mode string, or the named object.
pub(super) fn responses_tool_choice(choice: Option<&Value>) -> crate::Result<(ToolChoice, Value)> {
    let object = match choice {
        None | Some(Value::Null) => return Ok((ToolChoice::Auto, json!("auto"))),
        Some(Value::String(mode)) => {
            return choice_mode(mode, RESPONSES_NAMED).map(|choice| (choice, json!(mode)));
        }
        Some(Value::Object(object)) => object,
        Some(other) => {
            return Err(crate::Error::InvalidArgument(format!(
                "invalid tool_choice {other}; expected \"auto\", \"none\", \"required\" or \
                 {RESPONSES_NAMED}"
            )));
        }
    };
    if object.get("type").and_then(Value::as_str) == Some("allowed_tools") {
        only_members(object, &["type", "mode", "tools"], "tool_choice")?;
        // Each entry names a tool as a `tool_choice` would: a function, a
        // custom tool, or a built-in type such as `shell`.
        let choice = allowed_tools(object.get("mode"), object.get("tools"), |tool| {
            let named = match super::hosted::tool_choice(Some(tool)) {
                Some(named) => named?.0,
                None => responses_tool_choice(Some(tool))?.0,
            };
            match named {
                ToolChoice::Function(name) => Ok(name),
                _ => Err(crate::Error::InvalidArgument(format!(
                    "allowed_tools entries must name a tool, as {RESPONSES_NAMED} does"
                ))),
            }
        })?;
        return Ok((choice, Value::Object(object.clone())));
    }
    choice_kind(object, RESPONSES_NAMED)?;
    only_members(object, &["type", "name"], "tool_choice")?;
    let name = choice_name(object, "tool_choice")?;
    let echo = json!({"type":"function","name":name});
    Ok((ToolChoice::Function(name), echo))
}

/// A function tool definition, or a clear refusal of anything else.
///
/// `strict` (absent or `null` is `false`) asks the engine to enforce
/// `parameters` while the call is generated; a schema it cannot enforce
/// exactly is refused when the request is submitted, never weakened.
pub(super) fn function_definition(
    name: String,
    description: Option<String>,
    parameters: Option<Value>,
    strict: Option<bool>,
) -> crate::Result<ToolDefinition> {
    let parameters = match parameters {
        // The engine reads `null` as "takes no arguments", which is what an
        // omitted `parameters` means in both OpenAI APIs.
        None | Some(Value::Null) => Value::Null,
        Some(schema @ Value::Object(_)) => schema,
        Some(_) => {
            return Err(crate::Error::InvalidArgument(format!(
                "tool {name:?} parameters must be a JSON Schema object"
            )));
        }
    };
    Ok(ToolDefinition {
        name,
        description,
        parameters,
        strict: strict.unwrap_or(false),
    })
}

fn chat_tool(tool: &ChatTool) -> crate::Result<ToolDefinition> {
    if tool.kind != "function" {
        return Err(crate::Error::InvalidArgument(format!(
            "tool type {:?} is not supported; only function tools are",
            tool.kind
        )));
    }
    let function = tool.function.as_ref().ok_or_else(|| {
        crate::Error::InvalidArgument("a function tool requires a function object".into())
    })?;
    function_definition(
        function.name.clone(),
        function.description.clone(),
        function.parameters.clone(),
        function.strict,
    )
}

/// One history message. `index` names the call a deprecated assistant
/// `function_call` becomes, which `pending` keeps for the deprecated
/// `function` message answering it.
fn chat_message(
    message: &Message,
    index: usize,
    pending: &mut Option<String>,
    files: Option<&local_services::Store>,
) -> crate::Result<ChatMessage> {
    let nullable = matches!(message.role.as_str(), "assistant" | "tool" | "function");
    let parts = if nullable && message.content.is_null() {
        super::images::Parts::default()
    } else {
        message_parts(&message.content, files)?
    };
    let mut tool_calls = message
        .tool_calls
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|call| {
            if call.kind.as_deref().is_some_and(|kind| kind != "function") {
                return Err(crate::Error::InvalidArgument(format!(
                    "tool call {:?} has unsupported type {:?}",
                    call.id, call.kind
                )));
            }
            tool_call(
                call.id.clone(),
                call.function.name.clone(),
                &call.function.arguments,
            )
        })
        .collect::<crate::Result<Vec<_>>>()?;
    if let Some(call) = message
        .function_call
        .as_ref()
        .filter(|call| !call.is_null())
    {
        let call: ChatFunctionCall = serde_json::from_value(call.clone()).map_err(|error| {
            crate::Error::InvalidArgument(format!(
                "messages[{index}].function_call must be {{\"name\",\"arguments\"}}: {error}"
            ))
        })?;
        let id = format!("call_function_{index}");
        tool_calls.push(tool_call(id.clone(), call.name, &call.arguments)?);
        *pending = Some(id);
    }
    if !tool_calls.is_empty() && message.role != "assistant" {
        return Err(crate::Error::InvalidArgument(
            "only assistant messages can carry tool_calls".into(),
        ));
    }
    let (role, tool_call_id) = if message.role == "function" {
        // A deprecated `function` message is the result of the assistant's
        // last `function_call`: the tool message answering that call.
        let id = pending.take().ok_or_else(|| {
            crate::Error::InvalidArgument(format!(
                "messages[{index}] has the function role but answers no assistant function_call"
            ))
        })?;
        ("tool".to_owned(), Some(id))
    } else {
        (message.role.clone(), message.tool_call_id.clone())
    };
    if role == "tool" && tool_call_id.is_none() {
        return Err(crate::Error::InvalidArgument(
            "a tool message requires tool_call_id".into(),
        ));
    }
    Ok(ChatMessage {
        role,
        content: parts.text,
        reasoning_content: message.reasoning_content.clone(),
        tool_calls,
        tool_call_id,
        images: parts.images,
    })
}

/// A prior call from the history. `arguments` is a JSON string on the wire; an
/// already-parsed object is accepted too, since that is what it denotes.
pub(super) fn tool_call(id: String, name: String, arguments: &Value) -> crate::Result<ToolCall> {
    let arguments = match arguments {
        Value::String(text) => serde_json::from_str(text).map_err(|error| {
            crate::Error::InvalidArgument(format!(
                "tool call {id:?} arguments are not valid JSON: {error}"
            ))
        })?,
        other => other.clone(),
    };
    Ok(ToolCall {
        id,
        name,
        arguments,
    })
}

/// A message's `content`: a string, or `text`, `image_url` (a data or
/// `http(s)` URL, with `detail`), `video_url` (likewise; vLLM's and
/// llama.cpp's extension, also llama.cpp's `input_video` alias) and `file`
/// (an uploaded image's or video's `file_id`, or its base64 `file_data`)
/// parts in order.
pub(super) fn message_parts(
    value: &Value,
    files: Option<&local_services::Store>,
) -> crate::Result<super::images::Parts> {
    use super::images::Kind;
    let invalid = |message: &str| crate::Error::InvalidArgument(message.into());
    let mut parts = super::images::Parts::default();
    if let Some(text) = value.as_str() {
        parts.push_text(text);
        return Ok(parts);
    }
    let items = value
        .as_array()
        .ok_or_else(|| invalid("message content must be a text string or an array of parts"))?;
    // `{"url": …}` (with `detail`) or a bare URL string.
    let source = |item: &Value, field: &str| -> crate::Result<(String, Option<Value>)> {
        match item.get(field) {
            Some(Value::String(url)) => Ok((url.clone(), None)),
            Some(Value::Object(object)) => {
                let url = object
                    .get("url")
                    .or_else(|| object.get("data"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        crate::Error::InvalidArgument(format!("{field} requires a url string"))
                    })?;
                Ok((url.to_owned(), object.get("detail").cloned()))
            }
            _ => Err(crate::Error::InvalidArgument(format!(
                "a {field} part requires {field}, an object with a url"
            ))),
        }
    };
    for item in items {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => parts.push_text(
                item.get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("text content requires a text string"))?,
            ),
            Some("image_url") => {
                let (url, detail) = source(item, "image_url")?;
                parts.push_image(
                    super::images::url(&url, Kind::Image)?,
                    super::images::detail(detail.as_ref())?,
                );
            }
            Some(kind @ ("video_url" | "input_video")) => {
                let (url, _) = source(item, kind)?;
                let data = if url.starts_with("data:")
                    || url.starts_with("http://")
                    || url.starts_with("https://")
                {
                    super::images::url(&url, Kind::Video)?
                } else {
                    super::images::data_url(&url, true)?
                };
                parts.push_video(data);
            }
            Some("file") => {
                let file = item
                    .get("file")
                    .ok_or_else(|| invalid("a file part requires file"))?;
                let data = match (
                    file.get("file_id").and_then(Value::as_str),
                    file.get("file_data").and_then(Value::as_str),
                ) {
                    (Some(id), None) => super::images::file(files, id)?,
                    (None, Some(data)) => super::images::data_url(data, true)?,
                    _ => {
                        return Err(invalid(
                            "a file part needs exactly one of file_id and file_data",
                        ));
                    }
                };
                parts.push_file(data, local_engine::ImageDetail::Auto);
            }
            Some("input_audio") => {
                return Err(invalid(
                    "audio content is not supported; send text, images or video",
                ));
            }
            _ => {
                return Err(invalid(
                    "content parts must be text, image_url, video_url or file",
                ));
            }
        }
    }
    Ok(parts)
}
