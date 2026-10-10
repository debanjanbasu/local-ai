//! `POST /v1/responses`: the text and function-calling subset of the `OpenAI`
//! Responses API.
//!
//! Schema source: <https://developers.openai.com/api/reference/resources/responses/methods/create>
//! and its streaming-events page. What is implemented is implemented to that
//! shape; everything else is refused with a 400 naming the option, because a
//! request answered as if an option were honoured when it was not is worse
//! than one that fails:
//!
//! - Storage is opt-in. Without `--response-store` nothing is kept: `store`
//!   is reported as `false`, and `store: true` and `previous_response_id`
//!   fail. With it, `store` takes `OpenAI`'s default of `true`, completed and
//!   incomplete responses are persisted (see [`super::store`]) before their
//!   terminal event is sent, and `previous_response_id` continues one. Only
//!   conversation items carry over; `instructions`, tools and sampling
//!   controls are this request's alone. `background`, `conversation` and
//!   `item_reference` inputs still fail. Clients may instead carry history
//!   explicitly in `input`, which is supported in full for messages,
//!   reasoning, `function_call` and `function_call_output` items.
//! - Function tools only. Built-in, MCP and custom tools, `tool_choice` forcing
//!   (`required` or a named tool), `strict: true` and `text.format` other than
//!   `text` all need constrained decoding or hosted services this server lacks.
//! - Text only: image, file and audio parts fail.
//! - `reasoning.effort` is `none` or `xhigh`, the checkpoint's only two modes,
//!   and reasoning is returned as `reasoning_text` content, never a summary.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use local_engine::bonsai_model::StopReason;
use local_engine::{ChatMessage, ChatRequest, Event, Sampling, Stats, ToolCall, ToolDefinition};

use super::reasoning_crypto::ReasoningCipher;
use super::request::{
    effective_thinking, function_definition, service_tier, session, tool_call, tool_choice,
};
use super::response::{Reply, arguments_text, new_id, unix_now};
use super::store::ResponseStore;
use crate::GenerateParams;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponsesRequest {
    #[allow(dead_code)]
    model: Option<String>,
    input: Option<Value>,
    instructions: Option<String>,
    max_output_tokens: Option<usize>,
    /// Bounds built-in tool calls, of which this server makes none.
    #[allow(dead_code)]
    max_tool_calls: Option<u64>,
    metadata: Option<Value>,
    parallel_tool_calls: Option<bool>,
    previous_response_id: Option<String>,
    reasoning: Option<Reasoning>,
    store: Option<bool>,
    stream: Option<bool>,
    stream_options: Option<StreamOptions>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    text: Option<TextConfig>,
    tool_choice: Option<Value>,
    tools: Option<Vec<Value>>,
    truncation: Option<String>,
    user: Option<String>,
    safety_identifier: Option<String>,
    prompt_cache_key: Option<String>,
    service_tier: Option<String>,
    include: Option<Vec<String>>,
    background: Option<bool>,
    top_logprobs: Option<u64>,
    // Each of these needs server-side state or a hosted service; only `null`
    // is accepted, so a client relying on one learns that it is not honoured.
    conversation: Option<Value>,
    prompt: Option<Value>,
    context_management: Option<Value>,
    prompt_cache_options: Option<Value>,
    prompt_cache_retention: Option<Value>,
    moderation: Option<Value>,
    access_programs: Option<Value>,
    // This server's sampling extensions, the same as on Chat Completions.
    top_k: Option<usize>,
    min_p: Option<f32>,
    seed: Option<u64>,
    session_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reasoning {
    effort: Option<String>,
    summary: Option<String>,
    generate_summary: Option<String>,
    context: Option<String>,
    mode: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    include_obfuscation: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextConfig {
    format: Option<Value>,
    verbosity: Option<String>,
}

/// The request members a Response object reports back.
#[derive(Debug)]
pub(super) struct Echo {
    instructions: Option<String>,
    max_output_tokens: Option<usize>,
    metadata: Value,
    tool_choice: &'static str,
    tools: Vec<Value>,
    temperature: f32,
    top_p: f32,
    effort: &'static str,
    user: Option<String>,
    prompt_cache_key: Option<String>,
    safety_identifier: Option<String>,
    previous_response_id: Option<String>,
    cipher: Option<Arc<ReasoningCipher>>,
    /// Where the final Response is persisted, when it is to be stored.
    persist: Option<Persist>,
}

/// A Response that is to be stored, and the resolved input it answers.
#[derive(Debug)]
struct Persist {
    store: Arc<ResponseStore>,
    input_items: Vec<Value>,
}

pub(super) struct PreparedResponses {
    pub(super) stream: bool,
    pub(super) request: ChatRequest,
    pub(super) echo: Echo,
}

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::InvalidArgument(message.into())
}

fn unsupported(name: &str, why: &str) -> crate::Error {
    invalid(format!("{name} is not supported: {why}"))
}

fn present(value: Option<&Value>) -> bool {
    value.is_some_and(|value| !value.is_null())
}

impl ResponsesRequest {
    /// Fail on any option whose effect this server cannot produce. `stored`
    /// says whether a response store was configured.
    #[allow(clippy::too_many_lines)]
    fn reject_unsupported(&self, stored: bool, encrypted: bool) -> crate::Result<()> {
        if !stored && self.store == Some(true) {
            return Err(unsupported(
                "store=true",
                "this server was started without --response-store, so it keeps no responses; \
                 send store=false and carry history in input",
            ));
        }
        if self.background == Some(true) {
            return Err(unsupported(
                "background",
                "responses are generated in-request",
            ));
        }
        if !stored && self.previous_response_id.is_some() {
            return Err(unsupported(
                "previous_response_id",
                "this server was started without --response-store, so it keeps no responses; \
                 send the earlier items in input instead",
            ));
        }
        if self.previous_response_id.is_some() && present(self.conversation.as_ref()) {
            return Err(invalid(
                "previous_response_id and conversation cannot be used together",
            ));
        }
        if present(self.conversation.as_ref()) {
            return Err(unsupported(
                "conversation",
                "the Conversations API is not implemented; use previous_response_id or send the \
                 earlier items in input",
            ));
        }
        for (name, value) in [
            ("prompt", &self.prompt),
            ("context_management", &self.context_management),
            ("prompt_cache_options", &self.prompt_cache_options),
            ("prompt_cache_retention", &self.prompt_cache_retention),
            ("moderation", &self.moderation),
            ("access_programs", &self.access_programs),
        ] {
            if present(value.as_ref()) {
                return Err(unsupported(name, "it needs a hosted OpenAI service"));
            }
        }
        if self.include.as_ref().is_some_and(|include| {
            include
                .iter()
                .any(|value| value != "reasoning.encrypted_content" || !encrypted)
        }) {
            return Err(unsupported(
                "include",
                "only reasoning.encrypted_content is implemented, and it requires --reasoning-key",
            ));
        }
        if self.top_logprobs.is_some_and(|n| n > 0) {
            return Err(unsupported(
                "top_logprobs",
                "log-probabilities are not reported",
            ));
        }
        if self.parallel_tool_calls == Some(false) {
            return Err(unsupported(
                "parallel_tool_calls=false",
                "a single call per turn cannot be guaranteed without constrained decoding",
            ));
        }
        if !matches!(self.truncation.as_deref(), None | Some("disabled")) {
            return Err(unsupported(
                "truncation=auto",
                "the engine never drops input; an over-long request fails instead",
            ));
        }
        if self
            .stream_options
            .as_ref()
            .is_some_and(|options| options.include_obfuscation == Some(true))
        {
            return Err(unsupported(
                "stream_options.include_obfuscation",
                "not implemented",
            ));
        }
        if let Some(text) = &self.text {
            if let Some(format) = &text.format
                && !format.is_null()
                && format.get("type").and_then(Value::as_str) != Some("text")
            {
                return Err(unsupported(
                    "text.format",
                    "JSON mode and structured outputs need constrained decoding; only \
                     {\"type\":\"text\"} is accepted",
                ));
            }
            if !matches!(text.verbosity.as_deref(), None | Some("medium")) {
                return Err(unsupported(
                    "text.verbosity",
                    "only the default, medium, is produced",
                ));
            }
        }
        service_tier(self.service_tier.as_deref())
    }

    fn params(&self) -> GenerateParams {
        let defaults = GenerateParams::default();
        GenerateParams {
            max_tokens: self.max_output_tokens.unwrap_or(defaults.max_tokens),
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            min_p: self.min_p.unwrap_or(defaults.min_p),
            seed: self.seed.unwrap_or(defaults.seed),
            ..defaults
        }
    }
}

/// The effort to run at, refusing reasoning options this server cannot honour.
fn reasoning_effort(reasoning: Option<&Reasoning>) -> crate::Result<Option<&str>> {
    let Some(reasoning) = reasoning else {
        return Ok(None);
    };
    if reasoning.summary.is_some() || reasoning.generate_summary.is_some() {
        return Err(unsupported(
            "reasoning.summary",
            "no summarizer runs; the model's reasoning is returned verbatim as reasoning_text",
        ));
    }
    if !matches!(
        reasoning.context.as_deref(),
        None | Some("auto" | "all_turns")
    ) {
        return Err(unsupported(
            "reasoning.context",
            "earlier reasoning is rendered exactly as it is sent in input",
        ));
    }
    if !matches!(reasoning.mode.as_deref(), None | Some("standard")) {
        return Err(unsupported(
            "reasoning.mode",
            "only standard execution exists",
        ));
    }
    Ok(reasoning.effort.as_deref())
}

/// [`prepare_responses_with`] on a server with no response store.
#[cfg(test)]
pub(super) fn prepare_responses(body: &[u8], thinking: bool) -> crate::Result<PreparedResponses> {
    prepare_responses_with(body, thinking, None, None, "m")
}

/// Count requests use the generation input contract, but never persist a
/// response or admit generation. Reject generation-only options rather than
/// silently accepting a misspelled count request.
pub(super) fn prepare_input_tokens(
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<ChatRequest> {
    let mut value: Value = serde_json::from_slice(body)
        .map_err(|error| invalid(format!("invalid JSON request: {error}")))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("token count request must be an object"))?;
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "model"
                | "input"
                | "previous_response_id"
                | "tools"
                | "text"
                | "reasoning"
                | "truncation"
                | "instructions"
                | "conversation"
                | "tool_choice"
                | "parallel_tool_calls"
        ) {
            return Err(unsupported(
                key,
                "not an implemented input-token counting option",
            ));
        }
    }
    if object.get("input").is_none_or(Value::is_null) {
        object.insert("input".into(), json!([]));
    }
    object.insert("store".into(), json!(false));
    let body = serde_json::to_vec(&value)
        .map_err(|error| invalid(format!("invalid token count request: {error}")))?;
    Ok(prepare_responses_with(&body, thinking, store, cipher, model)?.request)
}

/// Parse a `POST /v1/responses` body, resolving `previous_response_id`
/// against `store` and deciding whether the result will be stored there.
pub(super) fn prepare_responses_with(
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<PreparedResponses> {
    let request: ResponsesRequest = serde_json::from_slice(body)
        .map_err(|error| invalid(format!("invalid JSON request: {error}")))?;
    request.reject_unsupported(store.is_some(), cipher.is_some())?;
    let thinking = effective_thinking(reasoning_effort(request.reasoning.as_ref())?, thinking)?;
    let choice = tool_choice(request.tool_choice.as_ref())?;
    let mut echoed_tools = Vec::new();
    let mut tools = Vec::new();
    for tool in request.tools.as_deref().unwrap_or_default() {
        let definition = function_tool(tool)?;
        echoed_tools.push(json!({"type":"function","name":definition.name,"description":definition.description,"parameters":definition.parameters,"strict":false}));
        tools.push(definition);
    }
    if choice == super::request::ToolChoice::None {
        tools.clear();
    }
    let empty_input = json!([]);
    let input = request
        .input
        .as_ref()
        .or_else(|| request.previous_response_id.as_ref().map(|_| &empty_input))
        .ok_or_else(|| invalid("input is required"))?;
    // The resolved input: the earlier response's stored history, which never
    // includes its instructions, then this request's items. Each record keeps
    // its full history, so this is one read however long the chain is.
    let mut items = match (&request.previous_response_id, store) {
        (Some(id), Some(store)) => store
            .load(id)
            .map_err(|error| {
                crate::Error::Generation(format!("could not read stored response {id}: {error}"))
            })?
            .ok_or_else(|| invalid(format!("previous response with id {id:?} not found")))?
            .history(),
        _ => Vec::new(),
    };
    normalize_input(input, &mut items)?;
    for item in &mut items {
        if item.get("type").and_then(Value::as_str) != Some("reasoning") {
            continue;
        }
        if let Some(encrypted) = item
            .get("encrypted_content")
            .filter(|value| !value.is_null())
        {
            let cipher = cipher
                .ok_or_else(|| unsupported("encrypted reasoning", "requires --reasoning-key"))?;
            let envelope = encrypted
                .as_str()
                .ok_or_else(|| invalid("encrypted_content must be a string"))?;
            let id = item
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("encrypted reasoning requires its original item id"))?;
            let text = cipher
                .unseal(model, id, envelope)
                .map_err(|error| invalid(error.to_string()))?;
            // The authenticated payload is authoritative over any plaintext
            // supplied alongside it. Keep the envelope in stored input items
            // so pagination and subsequent stateless replay preserve it.
            item["content"] = json!([{"type":"reasoning_text","text":text}]);
        }
    }
    let messages = input_messages(
        &Value::Array(items.clone()),
        request.instructions.as_deref(),
    )?;
    // Storing follows OpenAI's default of `true` once a store exists; without
    // one, `reject_unsupported` has already refused an explicit `true`.
    let persist = store
        .filter(|_| request.store.unwrap_or(true))
        .map(|store| Persist {
            store: Arc::clone(store),
            input_items: items,
        });
    let params = request.params();
    let echo = Echo {
        instructions: request.instructions.clone(),
        max_output_tokens: request.max_output_tokens,
        metadata: request.metadata.clone().unwrap_or_else(|| json!({})),
        tool_choice: choice.as_str(),
        tools: echoed_tools,
        temperature: params.temperature,
        top_p: params.top_p,
        effort: if thinking { "xhigh" } else { "none" },
        user: request.user.clone(),
        prompt_cache_key: request.prompt_cache_key.clone(),
        safety_identifier: request.safety_identifier.clone(),
        previous_response_id: request.previous_response_id.clone(),
        cipher: cipher.cloned(),
        persist,
    };
    Ok(PreparedResponses {
        stream: request.stream.unwrap_or(false),
        request: ChatRequest {
            messages,
            max_tokens: params.max_tokens,
            sampling: Sampling(params),
            thinking,
            session: session(request.prompt_cache_key, request.session_id, request.user),
            tools,
        },
        echo,
    })
}

fn function_tool(tool: &Value) -> crate::Result<ToolDefinition> {
    let kind = tool.get("type").and_then(Value::as_str).unwrap_or_default();
    if kind != "function" {
        return Err(unsupported(
            &format!("tool type {kind:?}"),
            "only function tools run here; built-in, MCP and custom tools need hosted services",
        ));
    }
    let name = tool
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("a function tool requires a name"))?;
    for flag in ["defer_loading", "async"] {
        if tool.get(flag).and_then(Value::as_bool) == Some(true) {
            return Err(unsupported(
                &format!("tool {flag}"),
                "tool search is not implemented",
            ));
        }
    }
    let description = match tool.get("description") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => {
            return Err(invalid(format!(
                "tool {name:?} description must be a string"
            )));
        }
    };
    function_definition(
        name.to_owned(),
        description,
        tool.get("parameters").cloned(),
        tool.get("strict").and_then(Value::as_bool),
    )
}

/// The assistant turn being assembled from consecutive output items.
///
/// One generated turn comes back as several items — reasoning, a message,
/// then any calls — and the template renders it as one assistant message, so
/// consecutive items are folded together until something ends the turn.
#[derive(Default)]
struct Turn {
    reasoning: Option<String>,
    content: Option<String>,
    calls: Vec<ToolCall>,
}

impl Turn {
    const fn is_empty(&self) -> bool {
        self.reasoning.is_none() && self.content.is_none() && self.calls.is_empty()
    }

    fn into_message(self) -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            content: self.content.unwrap_or_default(),
            reasoning_content: self.reasoning,
            tool_calls: self.calls,
            tool_call_id: None,
        }
    }
}

fn flush(turn: &mut Turn, messages: &mut Vec<ChatMessage>) {
    if !turn.is_empty() {
        messages.push(std::mem::take(turn).into_message());
    }
}

fn plain(role: &str, content: String, tool_call_id: Option<String>) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content,
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id,
    }
}

/// Append `input` to `items` as stored items: each with a unique `id`, its
/// `type`, text content as a part list, and the status the item resources
/// require.
///
/// Only the shape is normalised; [`input_messages`] still validates every item,
/// so an invalid one fails with the same message it would without a store.
/// A client-supplied `id` is kept unless it repeats one already present, so
/// `input_items` pagination by `after` always names one item.
fn normalize_input(input: &Value, items: &mut Vec<Value>) -> crate::Result<()> {
    let new: Vec<Value> = match input {
        Value::String(text) => {
            vec![
                json!({"type":"message","role":"user","content":[{"type":"input_text","text":text}]}),
            ]
        }
        Value::Array(new) => new.clone(),
        _ => return Err(invalid("input must be a string or an array of items")),
    };
    let mut seen: std::collections::HashSet<String> = items
        .iter()
        .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect();
    for item in new {
        let mut item = match item {
            Value::Object(item) => item,
            other => {
                items.push(other);
                continue;
            }
        };
        let kind = item
            .entry("type")
            .or_insert_with(|| json!("message"))
            .as_str()
            .unwrap_or_default()
            .to_owned();
        let assistant = item.get("role").and_then(Value::as_str) == Some("assistant");
        let prefix = match kind.as_str() {
            "message" => "msg_",
            "reasoning" => "rs_",
            "function_call" => "fc_",
            "function_call_output" => "fco_",
            _ => "item_",
        };
        if kind == "message"
            && let Some(Value::String(text)) = item.get("content")
        {
            let part = if assistant {
                json!({"type":"output_text","text":text,"annotations":[]})
            } else {
                json!({"type":"input_text","text":text})
            };
            item.insert("content".into(), json!([part]));
        }
        if kind == "message"
            && assistant
            && let Some(Value::Array(parts)) = item.get_mut("content")
        {
            for part in parts {
                if matches!(
                    part.get("type").and_then(Value::as_str),
                    Some("input_text" | "output_text")
                ) && let Some(part) = part.as_object_mut()
                {
                    part.insert("type".into(), json!("output_text"));
                    part.entry("annotations").or_insert_with(|| json!([]));
                }
            }
        }
        let needs_status = matches!(kind.as_str(), "function_call" | "function_call_output")
            || (kind == "message" && assistant);
        if needs_status && !item.contains_key("status") {
            item.insert("status".into(), json!("completed"));
        }
        if kind == "reasoning" && !item.contains_key("summary") {
            item.insert("summary".into(), json!([]));
        }
        let id = match item.get("id").and_then(Value::as_str) {
            Some(id) if !seen.contains(id) => id.to_owned(),
            _ if kind == "reasoning" && present(item.get("encrypted_content")) => {
                return Err(invalid(
                    "encrypted reasoning requires its original unique item id",
                ));
            }
            _ => new_id(prefix),
        };
        seen.insert(id.clone());
        item.insert("id".into(), json!(id));
        items.push(Value::Object(item));
    }
    Ok(())
}

/// Convert already-authenticated `input` into the chat history the engine renders.
fn input_messages(input: &Value, instructions: Option<&str>) -> crate::Result<Vec<ChatMessage>> {
    let mut messages = Vec::new();
    match input {
        Value::String(text) => messages.push(plain("user", text.clone(), None)),
        Value::Array(items) => {
            let mut turn = Turn::default();
            for item in items {
                input_item(item, &mut turn, &mut messages)?;
            }
            flush(&mut turn, &mut messages);
        }
        _ => return Err(invalid("input must be a string or an array of items")),
    }
    if let Some(instructions) = instructions.filter(|text| !text.is_empty()) {
        // The template takes one system message and only first, so the
        // instructions join a leading system message instead of preceding it.
        match messages.first_mut() {
            Some(first) if first.role == "system" => {
                first.content = format!("{instructions}\n\n{}", first.content);
            }
            _ => messages.insert(0, plain("system", instructions.to_owned(), None)),
        }
    }
    Ok(messages)
}

fn input_item(item: &Value, turn: &mut Turn, messages: &mut Vec<ChatMessage>) -> crate::Result<()> {
    let kind = item.get("type").and_then(Value::as_str);
    match kind {
        None | Some("message") => {
            let role = item
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("a message input item requires a role"))?;
            let content = item_text(item.get("content").unwrap_or(&Value::Null), role)?;
            match role {
                "assistant" => {
                    if turn.content.is_some() || !turn.calls.is_empty() {
                        flush(turn, messages);
                    }
                    turn.content = Some(content);
                }
                "user" | "system" | "developer" => {
                    flush(turn, messages);
                    messages.push(plain(role, content, None));
                }
                other => return Err(invalid(format!("unsupported message role {other:?}"))),
            }
        }
        Some("reasoning") => {
            flush(turn, messages);
            turn.reasoning = Some(reasoning_text(item)?);
        }
        Some("function_call") => {
            let field = |name: &str| {
                item.get(name)
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid(format!("a function_call item requires {name}")))
            };
            let call = tool_call(
                field("call_id")?.to_owned(),
                field("name")?.to_owned(),
                item.get("arguments").unwrap_or(&Value::Null),
            )?;
            turn.calls.push(call);
        }
        Some("function_call_output") => {
            flush(turn, messages);
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("a function_call_output item requires call_id"))?;
            let output = match item.get("output") {
                Some(Value::String(text)) => text.clone(),
                Some(parts @ Value::Array(_)) => item_text(parts, "tool")?,
                _ => return Err(invalid("function_call_output output must be text")),
            };
            messages.push(plain("tool", output, Some(call_id.to_owned())));
        }
        Some("item_reference") => {
            return Err(unsupported(
                "item_reference",
                "item lookup is not implemented; send the item itself",
            ));
        }
        Some(other) => {
            return Err(unsupported(
                &format!("input item type {other:?}"),
                "only messages, reasoning, function_call and function_call_output are accepted",
            ));
        }
    }
    Ok(())
}

/// Text of a message or tool output: a string, or text parts concatenated.
fn item_text(content: &Value, role: &str) -> crate::Result<String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let kind = part.get("type").and_then(Value::as_str).unwrap_or_default();
                let field = match kind {
                    "input_text" | "output_text" => "text",
                    "refusal" if role == "assistant" => "refusal",
                    "input_image" | "input_file" | "input_audio" => {
                        return Err(unsupported(
                            &format!("content part {kind:?}"),
                            "Bonsai 2 is text-only",
                        ));
                    }
                    other => {
                        return Err(invalid(format!("unsupported content part type {other:?}")));
                    }
                };
                text.push_str(part.get(field).and_then(Value::as_str).ok_or_else(|| {
                    invalid(format!("a {kind} content part requires a {field} string"))
                })?);
            }
            Ok(text)
        }
        _ => Err(invalid(
            "message content must be a string or an array of parts",
        )),
    }
}

/// Earlier reasoning: its verbatim `content` when present, else its summary.
fn reasoning_text(item: &Value) -> crate::Result<String> {
    let parts = |name: &str, kind: &str| -> crate::Result<Option<String>> {
        let Some(parts) = item.get(name).and_then(Value::as_array) else {
            return Ok(None);
        };
        if parts.is_empty() {
            return Ok(None);
        }
        let mut text = Vec::new();
        for part in parts {
            if part.get("type").and_then(Value::as_str) != Some(kind) {
                return Err(invalid(format!("reasoning {name} parts must be {kind}")));
            }
            text.push(
                part.get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid(format!("a {kind} part requires text")))?,
            );
        }
        Ok(Some(text.join("\n\n")))
    };
    Ok(parts("content", "reasoning_text")?
        .or(parts("summary", "summary_text")?)
        .unwrap_or_default())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ItemKind {
    Reasoning,
    Message,
}

/// The output item text is currently being written into.
struct Open {
    kind: ItemKind,
    id: String,
    index: usize,
    text: String,
}

/// One Response being built from engine events.
///
/// The same state machine serves the streaming and the buffered body, so the
/// document a non-streaming client receives is exactly the `response` of the
/// terminal event a streaming client would have. Events are only materialised
/// when streaming.
pub(super) struct ResponsesState {
    id: String,
    /// Hex shared by every item ID of this response.
    stem: String,
    created: u64,
    model: Arc<str>,
    echo: Arc<Echo>,
    streaming: bool,
    sequence: u64,
    events: Vec<Value>,
    output: Vec<Value>,
    open: Option<Open>,
    encryption_error: Option<String>,
}

impl ResponsesState {
    pub(super) fn new(reply: &Reply, echo: Arc<Echo>, streaming: bool) -> Self {
        let stem = reply
            .id
            .strip_prefix("resp_")
            .unwrap_or(&reply.id)
            .to_owned();
        Self {
            id: reply.id.clone(),
            stem,
            created: reply.created,
            model: Arc::clone(&reply.model),
            echo,
            streaming,
            sequence: 0,
            events: Vec::new(),
            output: Vec::new(),
            open: None,
            encryption_error: None,
        }
    }

    /// Queue one streaming event, numbering it.
    fn emit(&mut self, kind: &str, fields: Value) {
        if !self.streaming {
            return;
        }
        let mut event = Map::new();
        event.insert("type".into(), json!(kind));
        if let Value::Object(fields) = fields {
            event.extend(fields);
        }
        event.insert("sequence_number".into(), json!(self.sequence));
        self.sequence += 1;
        self.events.push(Value::Object(event));
    }

    /// The events produced since the last call, in order.
    pub(super) fn take_events(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.events)
    }

    /// `response.created` and `response.in_progress`.
    pub(super) fn start(&mut self) {
        let response = self.response("in_progress", None, None, None);
        self.emit("response.created", json!({"response":response}));
        let response = self.response("in_progress", None, None, None);
        self.emit("response.in_progress", json!({"response":response}));
    }

    fn item_id(&self, prefix: &str) -> String {
        format!("{prefix}_{}{:02}", self.stem, self.output.len())
    }

    /// Append text to the item of `kind`, opening it (and closing any other).
    fn text(&mut self, kind: ItemKind, piece: &str) {
        if self.open.as_ref().is_none_or(|open| open.kind != kind) {
            self.close("completed");
            let index = self.output.len();
            let (id, item, part) = match kind {
                ItemKind::Reasoning => {
                    let id = self.item_id("rs");
                    (
                        id.clone(),
                        json!({"id":id,"type":"reasoning","summary":[],"content":[],"status":"in_progress"}),
                        json!({"type":"reasoning_text","text":""}),
                    )
                }
                ItemKind::Message => {
                    let id = self.item_id("msg");
                    (
                        id.clone(),
                        json!({"id":id,"type":"message","status":"in_progress","role":"assistant","content":[]}),
                        json!({"type":"output_text","text":"","annotations":[],"logprobs":[]}),
                    )
                }
            };
            self.emit(
                "response.output_item.added",
                json!({"output_index":index,"item":item}),
            );
            self.emit(
                "response.content_part.added",
                json!({"item_id":id,"output_index":index,"content_index":0,"part":part}),
            );
            // The slot is reserved now so the next item's index follows it.
            self.output.push(Value::Null);
            self.open = Some(Open {
                kind,
                id,
                index,
                text: String::new(),
            });
        }
        let Some(open) = self.open.as_mut() else {
            return;
        };
        open.text.push_str(piece);
        let (id, index) = (open.id.clone(), open.index);
        match kind {
            ItemKind::Reasoning => self.emit(
                "response.reasoning_text.delta",
                json!({"item_id":id,"output_index":index,"content_index":0,"delta":piece}),
            ),
            ItemKind::Message => self.emit(
                "response.output_text.delta",
                json!({"item_id":id,"output_index":index,"content_index":0,"delta":piece,"logprobs":[]}),
            ),
        }
    }

    /// Finish the open text item, if any, with `status`.
    fn close(&mut self, status: &str) {
        let Some(open) = self.open.take() else {
            return;
        };
        let Open {
            kind,
            id,
            index,
            text,
        } = open;
        let item = match kind {
            ItemKind::Reasoning => {
                let part = json!({"type":"reasoning_text","text":text});
                self.emit(
                    "response.reasoning_text.done",
                    json!({"item_id":id,"output_index":index,"content_index":0,"text":text}),
                );
                self.emit(
                    "response.content_part.done",
                    json!({"item_id":id,"output_index":index,"content_index":0,"part":part}),
                );
                let mut item = json!({"id":id,"type":"reasoning","summary":[],"content":[part],"status":status});
                if let Some(cipher) = &self.echo.cipher {
                    match cipher.seal(&self.model, &id, &text) {
                        Ok(envelope) => item["encrypted_content"] = json!(envelope),
                        Err(error) => self.encryption_error = Some(error.to_string()),
                    }
                }
                item
            }
            ItemKind::Message => {
                let part = json!({"type":"output_text","text":text,"annotations":[],"logprobs":[]});
                self.emit(
                    "response.output_text.done",
                    json!({"item_id":id,"output_index":index,"content_index":0,"text":text,"logprobs":[]}),
                );
                self.emit(
                    "response.content_part.done",
                    json!({"item_id":id,"output_index":index,"content_index":0,"part":part}),
                );
                json!({"id":id,"type":"message","status":status,"role":"assistant","content":[part]})
            }
        };
        self.emit(
            "response.output_item.done",
            json!({"output_index":index,"item":item}),
        );
        if let Some(slot) = self.output.get_mut(index) {
            *slot = item;
        }
    }

    /// A complete, validated call from the engine.
    ///
    /// The engine only reports a call once the whole of it has been generated
    /// and validated, so the arguments arrive as one delta followed at once by
    /// `done`. That is the honest shape of what happened: the arguments were
    /// buffered by the engine, not streamed token by token.
    fn call(&mut self, call: &ToolCall) {
        self.close("completed");
        let index = self.output.len();
        let id = self.item_id("fc");
        let arguments = arguments_text(call);
        self.emit(
            "response.output_item.added",
            json!({"output_index":index,"item":{"id":id,"type":"function_call","status":"in_progress","call_id":call.id,"name":call.name,"arguments":""}}),
        );
        self.emit(
            "response.function_call_arguments.delta",
            json!({"item_id":id,"output_index":index,"delta":arguments}),
        );
        self.emit(
            "response.function_call_arguments.done",
            json!({"item_id":id,"output_index":index,"arguments":arguments}),
        );
        let item = json!({"id":id,"type":"function_call","status":"completed","call_id":call.id,"name":call.name,"arguments":arguments});
        self.emit(
            "response.output_item.done",
            json!({"output_index":index,"item":item}),
        );
        self.output.push(item);
    }

    /// Fold one engine event in. Returns the final Response on a terminal
    /// event: `Finished`, or `Error` once generation had started.
    pub(super) fn event(&mut self, event: Event) -> Option<Value> {
        match event {
            Event::Reasoning(piece) => self.text(ItemKind::Reasoning, &piece),
            Event::Content(piece) => self.text(ItemKind::Message, &piece),
            Event::ToolCall(call) => self.call(&call),
            Event::TokenIds(_) => {}
            Event::Finished(stats) => return Some(self.finish(&stats)),
            Event::Error(message) => return Some(self.fail(&message)),
        }
        None
    }

    fn finish(&mut self, stats: &Stats) -> Value {
        let (kind, overall, item_status, incomplete, error) = match stats.stop_reason {
            StopReason::Eos => ("response.completed", "completed", "completed", None, None),
            StopReason::TokenLimit => (
                "response.incomplete",
                "incomplete",
                "incomplete",
                Some(json!({"reason":"max_output_tokens"})),
                None,
            ),
            StopReason::Cancelled => (
                "response.failed",
                "failed",
                "incomplete",
                None,
                Some(json!({"code":"server_error","message":"generation was cancelled"})),
            ),
        };
        self.close(item_status);
        if let Some(message) = self.encryption_error.take() {
            return self.fail(&message);
        }
        let response = self.response(overall, incomplete.as_ref(), error.as_ref(), Some(stats));
        // A stored response is on disk before any client is told it finished,
        // so an ID a client has seen complete can always be continued. A
        // response that could not be stored fails rather than claim `store`.
        if error.is_none()
            && let Some(persist) = &self.echo.persist
            && let Err(failure) = persist.store.save(&response, &persist.input_items)
        {
            eprintln!("response {} could not be stored: {failure}", self.id);
            let message = format!("the response could not be stored: {failure}");
            self.emit(
                "error",
                json!({"code":"server_error","message":message,"param":null}),
            );
            let error = json!({"code":"server_error","message":message});
            let failed = self.response("failed", None, Some(&error), Some(stats));
            self.emit("response.failed", json!({"response":failed}));
            return failed;
        }
        self.emit(kind, json!({"response":response}));
        response
    }

    fn fail(&mut self, message: &str) -> Value {
        self.close("incomplete");
        self.emit(
            "error",
            json!({"code":"server_error","message":message,"param":null}),
        );
        let error = json!({"code":"server_error","message":message});
        let response = self.response("failed", None, Some(&error), None);
        self.emit("response.failed", json!({"response":response}));
        response
    }

    /// The Response object in its current state.
    fn response(
        &self,
        status: &str,
        incomplete: Option<&Value>,
        error: Option<&Value>,
        measured: Option<&Stats>,
    ) -> Value {
        let echo = &self.echo;
        let usage = measured.map(|stats| {
            let generation = &stats.generation;
            // No separately accounted cache-creation tokens: cache writes are
            // automatic local state, not OpenAI's charged cache-write tier.
            json!({"input_tokens":generation.prompt_tokens,"input_tokens_details":{"cached_tokens":generation.reused_prompt_tokens,"cache_write_tokens":0},"output_tokens":generation.generated_tokens,"output_tokens_details":{"reasoning_tokens":stats.reasoning_tokens},"total_tokens":generation.prompt_tokens+generation.generated_tokens})
        });
        let completed_at = (status == "completed").then(unix_now);
        let output: Vec<&Value> = self.output.iter().filter(|item| !item.is_null()).collect();
        json!({
            "id":self.id,
            "object":"response",
            "created_at":self.created,
            "status":status,
            "completed_at":completed_at,
            "error":error,
            "incomplete_details":incomplete,
            "instructions":echo.instructions,
            "max_output_tokens":echo.max_output_tokens,
            "model":self.model.as_ref(),
            "output":output,
            "parallel_tool_calls":true,
            "previous_response_id":echo.previous_response_id,
            "reasoning":{"effort":echo.effort,"summary":null,"context":"all_turns"},
            // Only a response that reached the store says so; a failed one,
            // including one whose write failed, was never stored.
            "store":echo.persist.is_some() && status != "failed",
            "background":false,
            "access_programs":null,
            "service_tier":"default",
            "temperature":echo.temperature,
            "text":{"format":{"type":"text"}},
            "tool_choice":echo.tool_choice,
            "tools":echo.tools,
            "top_p":echo.top_p,
            "truncation":"disabled",
            "usage":usage,
            "user":echo.user,
            "prompt_cache_key":echo.prompt_cache_key,
            "safety_identifier":echo.safety_identifier,
            "metadata":echo.metadata,
        })
    }
}

/// One Responses SSE frame: the event type is repeated in the `event:` line,
/// as `OpenAI` sends it.
pub(super) fn sse_frame(event: &Value) -> String {
    let kind = event.get("type").and_then(Value::as_str).unwrap_or("error");
    format!("event: {kind}\ndata: {event}\n\n")
}
