//! `POST /v1/responses`: the text and function-calling subset of the `OpenAI`
//! Responses API, served statelessly.
//!
//! Schema source: <https://developers.openai.com/api/reference/resources/responses/methods/create>
//! and its streaming-events page. What is implemented is implemented to that
//! shape; everything else is refused with a 400 naming the option, because a
//! request answered as if an option were honoured when it was not is worse
//! than one that fails:
//!
//! - Nothing is stored. `store` is reported as `false`, `store: true` and
//!   `background: true` fail, and so do `previous_response_id`, `conversation`
//!   and `item_reference` inputs, which all need a response store to resolve.
//!   Clients carry history explicitly in `input`, which is supported in full
//!   for messages, reasoning, `function_call` and `function_call_output` items.
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

use super::request::{
    effective_thinking, function_definition, service_tier, session, tool_call, tool_choice,
};
use super::response::{Reply, arguments_text, unix_now};
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
    fn reject_unsupported(&self) -> crate::Result<()> {
        if self.store == Some(true) {
            return Err(unsupported(
                "store=true",
                "this server keeps no responses; send store=false and carry history in input",
            ));
        }
        if self.background == Some(true) {
            return Err(unsupported(
                "background",
                "responses are generated in-request",
            ));
        }
        if self.previous_response_id.is_some() || present(self.conversation.as_ref()) {
            return Err(unsupported(
                "previous_response_id and conversation",
                "they need stored responses; send the earlier items in input instead",
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
        if self
            .include
            .as_ref()
            .is_some_and(|include| !include.is_empty())
        {
            return Err(unsupported(
                "include",
                "there are no logprobs, encrypted reasoning or tool outputs to add",
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

pub(super) fn prepare_responses(body: &[u8], thinking: bool) -> crate::Result<PreparedResponses> {
    let request: ResponsesRequest = serde_json::from_slice(body)
        .map_err(|error| invalid(format!("invalid JSON request: {error}")))?;
    request.reject_unsupported()?;
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
    let input = request
        .input
        .as_ref()
        .ok_or_else(|| invalid("input is required"))?;
    let messages = input_messages(input, request.instructions.as_deref())?;
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

/// Convert `instructions` and `input` into the chat history the engine renders.
pub(super) fn input_messages(
    input: &Value,
    instructions: Option<&str>,
) -> crate::Result<Vec<ChatMessage>> {
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
            if present(item.get("encrypted_content")) {
                return Err(unsupported(
                    "encrypted reasoning",
                    "this server never issues encrypted_content, so it cannot read one",
                ));
            }
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
                "this server stores no items; send the item itself",
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
                json!({"id":id,"type":"reasoning","summary":[],"content":[part],"status":status})
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
        let response = self.response(overall, incomplete.as_ref(), error.as_ref(), Some(stats));
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
            "previous_response_id":null,
            "reasoning":{"effort":echo.effort,"summary":null,"context":"all_turns"},
            "store":false,
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
