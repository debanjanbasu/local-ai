//! Chat Completions and legacy Completions request parsing.
//!
//! Chat `response_format` is mapped onto the engine's native
//! [`ResponseFormat`]: `text`, `json_object`, and `json_schema` with its
//! nested `json_schema` object (see [`chat_response_format`]). The engine
//! compiles the schema before queueing and enforces it on every answer token,
//! whatever `strict` says. Legacy Completions keep refusing every format but
//! `text`: structured output is not part of that API.
//!
//! Chat `tool_choice` (`auto`, `none`, `required` or
//! `{"type":"function","function":{"name"}}`), `parallel_tool_calls` and a
//! tool's `function.strict` map one to one onto the engine's native
//! [`ToolChoice`], `parallel_tool_calls` and [`ToolDefinition::strict`],
//! which the engine enforces with its tool grammar (see [`tool_policy`]).

use serde::Deserialize;
use serde_json::{Map, Value, json};

use local_engine::{
    ChatMessage, ChatRequest, CompletionRequest, ResponseFormat, Sampling, ToolCall, ToolChoice,
    ToolDefinition,
};

use crate::GenerateParams;

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: Option<String>,
    messages: Option<Vec<Message>>,
    max_tokens: Option<usize>,
    /// Chat's current name for `max_tokens`; it wins when both are sent.
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
    stream: Option<bool>,
    stream_options: Option<StreamOptions>,
    user: Option<String>,
    session_id: Option<String>,
    prompt_cache_key: Option<String>,
    tools: Option<Vec<ChatTool>>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
    reasoning_effort: Option<String>,
    // Options this server cannot honour. They are named so a request that
    // sets one fails instead of being answered as if it had not.
    n: Option<u64>,
    stop: Option<Value>,
    /// Boolean for Chat, an integer for legacy Completions.
    logprobs: Option<Value>,
    top_logprobs: Option<u64>,
    logit_bias: Option<Value>,
    best_of: Option<u64>,
    echo: Option<bool>,
    suffix: Option<String>,
    store: Option<bool>,
    verbosity: Option<String>,
    response_format: Option<Value>,
    audio: Option<Value>,
    modalities: Option<Vec<String>>,
    functions: Option<Value>,
    function_call: Option<Value>,
    prediction: Option<Value>,
    web_search_options: Option<Value>,
    service_tier: Option<String>,
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
            repetition_penalty: self
                .frequency_penalty
                .map_or(defaults.repetition_penalty, |value| 1.0 + value.max(0.0)),
            seed: self.seed.unwrap_or(defaults.seed),
        }
    }

    /// Fail on any named option whose effect this server cannot produce.
    fn reject_unsupported(&self, chat: bool) -> crate::Result<()> {
        let unsupported = |name: &str, why: &str| {
            Err(crate::Error::InvalidArgument(format!(
                "{name} is not supported: {why}"
            )))
        };
        if self.n.is_some_and(|n| n != 1) {
            return unsupported("n", "this server generates exactly one choice");
        }
        if self.stop.as_ref().is_some_and(|stop| !is_empty(stop)) {
            return unsupported("stop", "stop sequences are not implemented");
        }
        let logprobs = match self.logprobs.as_ref() {
            None | Some(Value::Null) => false,
            Some(Value::Bool(enabled)) if chat => *enabled,
            Some(Value::Number(count)) if !chat && count.as_u64().is_some() => true,
            _ => {
                return Err(crate::Error::InvalidArgument(
                    "logprobs must be a boolean for Chat or a non-negative integer for Completions"
                        .into(),
                ));
            }
        };
        if logprobs || self.top_logprobs.is_some_and(|n| n > 0) {
            return unsupported("logprobs", "token log-probabilities are not reported");
        }
        if self.logit_bias.as_ref().is_some_and(|bias| !is_empty(bias)) {
            return unsupported("logit_bias", "the sampler has no per-token bias");
        }
        if self.best_of.is_some_and(|count| count != 1) {
            return unsupported("best_of", "this server generates exactly one candidate");
        }
        if self.echo == Some(true) {
            return unsupported("echo", "prompt echo is not implemented");
        }
        if self
            .suffix
            .as_ref()
            .is_some_and(|suffix| !suffix.is_empty())
        {
            return unsupported("suffix", "fill-in-the-middle generation is not implemented");
        }
        if self.store == Some(true) {
            return unsupported("store", "this server stores no completions");
        }
        if !matches!(self.verbosity.as_deref(), None | Some("medium")) {
            return unsupported("verbosity", "only the default, medium, is produced");
        }
        // Chat parses its format strictly in `chat_response_format`; legacy
        // Completions keep their old, lenient refusal of anything but text.
        if !chat
            && let Some(format) = &self.response_format
            && !format.is_null()
            && format.get("type").and_then(Value::as_str) != Some("text")
        {
            return unsupported(
                "response_format",
                "JSON mode and structured outputs are Chat Completions and Responses options, \
                 not legacy Completions ones; only {\"type\":\"text\"} is accepted here",
            );
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
            .functions
            .as_ref()
            .is_some_and(|value| !value.is_null())
            || self
                .function_call
                .as_ref()
                .is_some_and(|value| !value.is_null())
        {
            return unsupported("functions", "use tools and tool_choice instead");
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

pub(super) enum GenerationRequest {
    Chat(ChatRequest),
    Completion(CompletionRequest),
}

pub(super) struct PreparedGeneration {
    pub(super) stream: bool,
    /// `stream_options.include_usage`, Chat only.
    pub(super) include_usage: bool,
    pub(super) request: GenerationRequest,
}

pub(super) fn prepare_generation(
    body: &[u8],
    chat: bool,
    thinking: bool,
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
    request.reject_unsupported(chat)?;
    if !chat
        && (request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
            || request
                .tool_choice
                .as_ref()
                .is_some_and(|choice| !choice.is_null())
            || request.parallel_tool_calls == Some(false))
    {
        return Err(crate::Error::InvalidArgument(
            "tools is not supported: tools, tool_choice and parallel_tool_calls are Chat \
             Completions and Responses options, not legacy Completions ones"
                .into(),
        ));
    }
    if !chat && include_usage {
        return Err(crate::Error::InvalidArgument(
            "stream_options.include_usage is not supported for legacy Completions".into(),
        ));
    }
    let generation = if chat {
        let messages = request
            .messages
            .as_deref()
            .ok_or_else(|| crate::Error::InvalidArgument("messages are required".into()))?;
        let messages = messages
            .iter()
            .map(chat_message)
            .collect::<crate::Result<Vec<_>>>()?;
        let tools = request
            .tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(chat_tool)
            .collect::<crate::Result<Vec<_>>>()?;
        let tool_choice = chat_tool_choice(request.tool_choice.as_ref())?;
        tool_policy(&tools, &tool_choice)?;
        let response_format = chat_response_format(request.response_format.as_ref())?;
        GenerationRequest::Chat(ChatRequest {
            messages,
            max_tokens: params.max_tokens,
            sampling: Sampling(params),
            thinking: effective_thinking(request.reasoning_effort.as_deref(), thinking)?,
            session: session(request.prompt_cache_key, request.session_id, request.user),
            tools,
            tool_choice,
            parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
            response_format,
        })
    } else {
        let prompt = request
            .prompt
            .ok_or_else(|| crate::Error::InvalidArgument("prompt is required".into()))?;
        GenerationRequest::Completion(CompletionRequest {
            prompt,
            max_tokens: params.max_tokens,
            sampling: Sampling(params),
            session: session(request.prompt_cache_key, request.session_id, request.user),
            response_format: ResponseFormat::Text,
        })
    };
    Ok(PreparedGeneration {
        stream,
        include_usage: chat && stream && include_usage,
        request: generation,
    })
}

/// The prompt-cache affinity hint: `prompt_cache_key` is the `OpenAI` name for
/// it, `session_id` this server's older one, and `user` the field it replaced.
pub(super) fn session(
    prompt_cache_key: Option<String>,
    session_id: Option<String>,
    user: Option<String>,
) -> Option<String> {
    prompt_cache_key.or(session_id).or(user)
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

/// Resolve a reasoning effort against the server's `--no-thinking` choice.
///
/// The checkpoint has exactly two modes: its template either asks for `xhigh`
/// reasoning or closes the thought block empty. Those are the only two efforts
/// that describe what will really run, so every other level is refused rather
/// than silently rounded to one of them. An operator who started the server
/// with `--no-thinking` has decided, so a request cannot turn reasoning back on.
pub(super) fn effective_thinking(effort: Option<&str>, server: bool) -> crate::Result<bool> {
    match effort {
        None => Ok(server),
        Some("none") => Ok(false),
        Some("xhigh") if server => Ok(true),
        Some("xhigh") => Err(crate::Error::InvalidArgument(
            "reasoning effort \"xhigh\" was requested but this server was started with \
             --no-thinking"
                .into(),
        )),
        Some(other) => Err(crate::Error::InvalidArgument(format!(
            "reasoning effort {other:?} is not supported; Bonsai 2 reasons at \"xhigh\" or not at \
             all (\"none\")"
        ))),
    }
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
        _ => Ok(()),
    }
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
        Some(Value::String(kind)) if kind == "allowed_tools" => Err(crate::Error::InvalidArgument(
            "tool_choice type \"allowed_tools\" is not supported; send only the allowed tools \
             in tools instead"
                .into(),
        )),
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

fn chat_message(message: &Message) -> crate::Result<ChatMessage> {
    let nullable = matches!(message.role.as_str(), "assistant" | "tool");
    let content = if nullable && message.content.is_null() {
        String::new()
    } else {
        message_text(&message.content)?
    };
    let tool_calls = message
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
    if !tool_calls.is_empty() && message.role != "assistant" {
        return Err(crate::Error::InvalidArgument(
            "only assistant messages can carry tool_calls".into(),
        ));
    }
    if message.role == "tool" && message.tool_call_id.is_none() {
        return Err(crate::Error::InvalidArgument(
            "a tool message requires tool_call_id".into(),
        ));
    }
    if message.role == "function" {
        return Err(crate::Error::InvalidArgument(
            "the deprecated function role is not supported; use tool messages".into(),
        ));
    }
    Ok(ChatMessage {
        role: message.role.clone(),
        content,
        reasoning_content: message.reasoning_content.clone(),
        tool_calls,
        tool_call_id: message.tool_call_id.clone(),
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

pub(super) fn message_text(value: &Value) -> crate::Result<String> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_owned());
    }
    let items = value
        .as_array()
        .ok_or_else(|| crate::Error::InvalidArgument("message content must be text".into()))?;
    let mut text = String::new();
    for item in items {
        if item.get("type").and_then(Value::as_str) != Some("text") {
            return Err(crate::Error::InvalidArgument(
                "media content is not supported; Bonsai 2 is text-only".into(),
            ));
        }
        text.push_str(item.get("text").and_then(Value::as_str).ok_or_else(|| {
            crate::Error::InvalidArgument("text content requires a text string".into())
        })?);
    }
    Ok(text)
}
