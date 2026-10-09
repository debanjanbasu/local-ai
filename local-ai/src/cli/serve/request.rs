use serde::Deserialize;
use serde_json::Value;

use local_engine::{
    ChatMessage, ChatRequest, CompletionRequest, Sampling, ToolCall, ToolDefinition,
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
        if let Some(format) = &self.response_format
            && !format.is_null()
            && format.get("type").and_then(Value::as_str) != Some("text")
        {
            return unsupported(
                "response_format",
                "JSON mode and structured outputs need constrained decoding, which this \
                 server does not have; only {\"type\":\"text\"} is accepted",
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
        if self.parallel_tool_calls == Some(false) {
            return unsupported(
                "parallel_tool_calls=false",
                "a single call per turn cannot be guaranteed without constrained decoding",
            );
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
        let tools = apply_tool_choice(tools, request.tool_choice.as_ref())?;
        GenerationRequest::Chat(ChatRequest {
            messages,
            max_tokens: params.max_tokens,
            sampling: Sampling(params),
            thinking: effective_thinking(request.reasoning_effort.as_deref(), thinking)?,
            session: session(request.prompt_cache_key, request.session_id, request.user),
            tools,
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

/// Apply `tool_choice` to the declared tools.
///
/// `auto` hands the tools to the template. `none` withholds them, so the model
/// is never shown a tool it may not call; prior calls and results in the
/// history are still rendered. `required` and naming a function are promises
/// that need constrained decoding, so they fail rather than being treated as a
/// hint the model is free to ignore.
pub(super) fn apply_tool_choice(
    tools: Vec<ToolDefinition>,
    choice: Option<&Value>,
) -> crate::Result<Vec<ToolDefinition>> {
    match tool_choice(choice)? {
        ToolChoice::Auto => Ok(tools),
        ToolChoice::None => Ok(Vec::new()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ToolChoice {
    Auto,
    None,
}

impl ToolChoice {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::None => "none",
        }
    }
}

pub(super) fn tool_choice(choice: Option<&Value>) -> crate::Result<ToolChoice> {
    match choice {
        None | Some(Value::Null) => Ok(ToolChoice::Auto),
        Some(Value::String(mode)) if mode == "auto" => Ok(ToolChoice::Auto),
        Some(Value::String(mode)) if mode == "none" => Ok(ToolChoice::None),
        Some(Value::String(mode)) if mode == "required" => Err(forced_choice()),
        Some(Value::Object(_)) => Err(forced_choice()),
        Some(other) => Err(crate::Error::InvalidArgument(format!(
            "invalid tool_choice {other}; expected \"auto\" or \"none\""
        ))),
    }
}

fn forced_choice() -> crate::Error {
    crate::Error::InvalidArgument(
        "tool_choice \"required\" and forcing a named tool are not supported: a guaranteed call \
         needs constrained decoding, which this server does not have; use \"auto\" or \"none\""
            .into(),
    )
}

/// A function tool definition, or a clear refusal of anything else.
pub(super) fn function_definition(
    name: String,
    description: Option<String>,
    parameters: Option<Value>,
    strict: Option<bool>,
) -> crate::Result<ToolDefinition> {
    if strict == Some(true) {
        return Err(crate::Error::InvalidArgument(format!(
            "tool {name:?} sets strict=true, which needs constrained decoding; this server only \
             validates calls after generation, so set strict to false or omit it"
        )));
    }
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
