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
//!   controls are this request's alone. `background: true` is stored in the
//!   response store when `store` is `true` (the default once a store exists);
//!   with `store: false`, or with `store` omitted on a server without one, it
//!   is kept only temporarily, in a private store of this process, until
//!   [`super::background::TEMPORARY_RETENTION`] after it ends, and reports
//!   `store: false` (see [`super::background::Temporary`]). With
//!   `stream: true` its events are journaled so a stream can be resumed (see
//!   [`super::background`] and [`super::resume`]).
//!   `item_reference` inputs still fail. Clients may instead carry history
//!   explicitly in `input`, which is supported in full for messages,
//!   reasoning, `function_call` and `function_call_output` items.
//! - `conversation` (an ID or `{"id"}`, never with `previous_response_id`)
//!   needs the durable conversation store. The conversation's items, read as
//!   one snapshot, precede this request's input; the Response echoes
//!   `conversation: {"id"}`. A completed or incomplete Response appends this
//!   request's new input and its output to the conversation exactly once,
//!   keyed by the response ID and only if the conversation is still at the
//!   version the prompt was built from, before its terminal state is
//!   reported; if that append fails the Response fails instead. Failed and
//!   cancelled Responses append nothing.
//! - Function tools only; built-in, MCP and custom tools need hosted
//!   services this server lacks. `tool_choice` (`auto`, `none`, `required` or
//!   `{"type":"function","name"}`), `parallel_tool_calls` and a tool's
//!   `strict` map onto the engine's native tool grammar, which enforces them
//!   on every sampled token, and are echoed back as accepted (`strict` as
//!   `false` when omitted: this server enforces arguments only on request).
//! - `text.format` is `text`, `json_object`, or the flat `json_schema`
//!   (`name`, `schema`, optional `description` and `strict`), parsed strictly
//!   and echoed back as accepted. The engine compiles the schema before
//!   queueing (an invalid or unsupported one is a 400) and enforces it on
//!   every answer token whatever `strict` says; reasoning stays unconstrained.
//!   With tools, the answer is either a call or (unless a call is
//!   required) the final answer in that format, enforced by one grammar.
//!   An answer the format did not complete is never `completed`: a token
//!   limit is `incomplete` (`max_output_tokens`), and an end of turn during
//!   reasoning, before any answer, is `failed`.
//! - Text only: image, file and audio parts fail.
//! - `reasoning.effort` is `none` or `xhigh`, the checkpoint's only two modes,
//!   and reasoning is returned as `reasoning_text` content, never a summary.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use local_engine::bonsai_model::StopReason;
use local_engine::{ChatMessage, ChatRequest, Event, Sampling, Stats, ToolCall, ToolDefinition};
use local_services::conversations::{MAX_APPEND_ITEMS, MAX_ID_CHARS};
use local_services::{Append, Error as ServiceError, Store as ConversationStore};

use super::background::Temporary;
use super::reasoning_crypto::ReasoningCipher;
use super::request::{
    effective_thinking, function_definition, responses_text_format, responses_tool_choice,
    service_tier, session, tool_call, tool_policy,
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
    /// Opaque client telemetry (Codex extension), never model input or stored.
    #[allow(dead_code)]
    client_metadata: Option<HashMap<String, String>>,
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
    /// The accepted `text.format`, as the Response reports it.
    text_format: Value,
    /// The accepted `tool_choice`: a mode string or the named function.
    tool_choice: Value,
    parallel_tool_calls: bool,
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
    /// `background: true`: generated detached from the request, and persisted
    /// by [`super::background`] rather than by [`ResponsesState`].
    background: bool,
    /// The durable conversation this Response continues and is appended to.
    conversation: Option<Conversing>,
}

/// A durable conversation a Response continues, and what it will append.
#[derive(Debug)]
struct Conversing {
    store: ConversationStore,
    id: String,
    /// The history version the prompt was built from. The append expects it,
    /// so generation that raced another writer is refused, never appended.
    version: u64,
    /// This request's new input items as the client sent them (normalized,
    /// before any reasoning was decrypted), appended before the output. The
    /// replayed history is never among them.
    input: Vec<Value>,
}

impl Conversing {
    /// Append this request's input and `response`'s output, exactly once:
    /// keyed by the response ID, so a retry of the same append is a no-op.
    /// `Err` says why nothing was appended.
    fn append(&self, response: &Value) -> Result<(), String> {
        let request_id = response
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "the response has no id".to_owned())?
            .to_owned();
        let mut items = self.input.clone();
        if let Some(output) = response.get("output").and_then(Value::as_array) {
            items.extend_from_slice(output);
        }
        if items.is_empty() {
            // Nothing new to record; the history is unchanged either way.
            return Ok(());
        }
        let append = Append {
            request_id,
            expected_version: self.version,
            items,
        };
        // A conflict (another writer moved the conversation on) says so in
        // its own message: "conversation ... is at version N, not the expected M".
        self.store
            .append_items(&self.id, append)
            .map(drop)
            .map_err(|error| {
                format!(
                    "the response could not be added to conversation {}: {error}",
                    self.id
                )
            })
    }
}

impl Echo {
    pub(super) const fn background(&self) -> bool {
        self.background
    }
}

/// A Response that is to be stored, and the resolved input it answers.
#[derive(Debug)]
struct Persist {
    store: Arc<ResponseStore>,
    input_items: Vec<Value>,
    /// A `background` `store: false` Response: `store` is this server's
    /// private temporary store, which expires it after it ends.
    temporary: Option<Arc<Temporary>>,
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
    /// Whether this is a background request kept only temporarily: `store`
    /// is `false`, explicitly or by default on a server without a store.
    fn temporary_background(&self, stored: bool) -> bool {
        self.background == Some(true) && !self.store.unwrap_or(stored)
    }

    /// Fail on any option whose effect this server cannot produce. `stored`
    /// says whether a response store was configured, `temporary` whether the
    /// private temporary store is available, `conversations` whether the
    /// durable conversation store is.
    #[allow(clippy::too_many_lines, clippy::fn_params_excessive_bools)]
    fn reject_unsupported(
        &self,
        stored: bool,
        temporary: bool,
        encrypted: bool,
        conversations: bool,
    ) -> crate::Result<()> {
        if !stored && self.store == Some(true) {
            return Err(unsupported(
                "store=true",
                "this server was started without --response-store, so it keeps no responses; \
                 send store=false and carry history in input",
            ));
        }
        if self.temporary_background(stored) && !temporary {
            return Err(unsupported(
                "background",
                if stored {
                    "with store=false a background response is kept briefly in a private \
                     temporary store, which could not be created on this server; send store=true"
                } else {
                    "a background response is kept until it is retrieved, durably in \
                     --response-store or, with store=false, briefly in a private temporary \
                     store, and this server has neither"
                },
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
        if !conversations && present(self.conversation.as_ref()) {
            return Err(unsupported(
                "conversation",
                "this server has no durable conversation store; use previous_response_id or send \
                 the earlier items in input",
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
            // `text.format` is parsed, strictly, in `prepare_responses_with`.
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

/// [`prepare_input_tokens_conversing`] on a server with no conversation store.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn prepare_input_tokens(
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<ChatRequest> {
    prepare_input_tokens_conversing(body, thinking, store, None, cipher, model)
}

/// Count requests use the generation input contract, but never persist a
/// response or admit generation. Reject generation-only options rather than
/// silently accepting a misspelled count request.
///
/// A `conversation` is read, as one snapshot, exactly as generation would
/// read it, and never written. Blocks on the stores.
pub(super) fn prepare_input_tokens_conversing(
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    conversations: Option<&ConversationStore>,
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
    let prepared =
        prepare_responses_conversing(&body, thinking, store, None, conversations, cipher, model)?;
    Ok(prepared.request)
}

/// [`prepare_responses_retaining`] on a server with no temporary store.
#[cfg(test)]
pub(super) fn prepare_responses_with(
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<PreparedResponses> {
    prepare_responses_retaining(body, thinking, store, None, cipher, model)
}

/// [`prepare_responses_conversing`] on a server with no conversation store.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn prepare_responses_retaining(
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    temporary: Option<&Arc<Temporary>>,
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<PreparedResponses> {
    prepare_responses_conversing(body, thinking, store, temporary, None, cipher, model)
}

/// Parse a `POST /v1/responses` body, resolving `previous_response_id`
/// against `store` and deciding whether the result will be stored there, or,
/// for `background` with `store: false`, kept in `temporary`.
///
/// A `conversation` is read from `conversations` as one ordered snapshot
/// whose items precede this request's input; its version and the request's
/// new items are kept for the append when the Response ends. Blocks on the
/// stores, so it runs on the blocking pool.
#[allow(clippy::too_many_lines)]
pub(super) fn prepare_responses_conversing(
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    temporary: Option<&Arc<Temporary>>,
    conversations: Option<&ConversationStore>,
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<PreparedResponses> {
    let request: ResponsesRequest = serde_json::from_slice(body)
        .map_err(|error| invalid(format!("invalid JSON request: {error}")))?;
    reject_temporary_previous(request.previous_response_id.as_deref(), temporary)?;
    request.reject_unsupported(
        store.is_some(),
        temporary.is_some(),
        cipher.is_some(),
        conversations.is_some(),
    )?;
    let conversation_id = conversation_id(request.conversation.as_ref())?;
    let thinking = effective_thinking(reasoning_effort(request.reasoning.as_ref())?, thinking)?;
    let (choice, echoed_choice) = responses_tool_choice(request.tool_choice.as_ref())?;
    let mut echoed_tools = Vec::new();
    let mut tools = Vec::new();
    for tool in request.tools.as_deref().unwrap_or_default() {
        let definition = function_tool(tool)?;
        echoed_tools.push(json!({"type":"function","name":definition.name,"description":definition.description,"parameters":definition.parameters,"strict":definition.strict}));
        tools.push(definition);
    }
    tool_policy(&tools, &choice)?;
    let parallel_tool_calls = request.parallel_tool_calls.unwrap_or(true);
    let (response_format, text_format) =
        responses_text_format(request.text.as_ref().and_then(|text| text.format.as_ref()))?;
    let empty_input = json!([]);
    let continues = request.previous_response_id.is_some() || conversation_id.is_some();
    let input = request
        .input
        .as_ref()
        .or_else(|| continues.then_some(&empty_input))
        .ok_or_else(|| invalid("input is required"))?;
    // The resolved input: the earlier response's stored history, which never
    // includes its instructions, then this request's items. Each record keeps
    // its full history, so this is one read however long the chain is. A
    // conversation (never combined with `previous_response_id`) is likewise
    // one consistent snapshot, oldest item first, at a known version.
    let mut items = match (&request.previous_response_id, store) {
        (Some(id), Some(store)) => previous_history(store, id)?,
        _ => Vec::new(),
    };
    let snapshot = if let (Some(id), Some(conversations)) = (conversation_id, conversations) {
        let (version, history) = conversation_history(conversations, &id)?;
        items = history;
        Some((conversations.clone(), id, version))
    } else {
        None
    };
    let replayed = items.len();
    normalize_input(input, &mut items)?;
    // This request's own items, before any reasoning is decrypted below: only
    // these, never the replayed history, are appended to a conversation.
    let new_input = if snapshot.is_some() {
        items[replayed..].to_vec()
    } else {
        Vec::new()
    };
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
    // Refused now, before generation, rather than when the append fails.
    let conversation = snapshot
        .map(|(conversations, id, version)| conversing(conversations, id, version, new_input))
        .transpose()?;
    let persist = persist(&request, store, temporary, items);
    let params = request.params();
    let echo = Echo {
        instructions: request.instructions.clone(),
        max_output_tokens: request.max_output_tokens,
        metadata: request.metadata.clone().unwrap_or_else(|| json!({})),
        text_format,
        tool_choice: echoed_choice,
        parallel_tool_calls,
        tools: echoed_tools,
        temperature: params.temperature,
        top_p: params.top_p,
        effort: if thinking { "xhigh" } else { "none" },
        user: request.user.clone(),
        prompt_cache_key: request.prompt_cache_key.clone(),
        safety_identifier: request.safety_identifier.clone(),
        previous_response_id: request.previous_response_id.clone(),
        cipher: cipher.cloned(),
        // `reject_unsupported` has ensured a background response is stored,
        // durably or temporarily.
        background: request.background == Some(true) && persist.is_some(),
        persist,
        conversation,
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
            tool_choice: choice,
            parallel_tool_calls,
            response_format,
        },
        echo,
    })
}

/// Where the Response to `request` is to be kept, with its resolved input
/// `items`, if anywhere.
///
/// Storing follows `OpenAI`'s default of `true` once a store exists; without
/// one, `reject_unsupported` has already refused an explicit `true`. A
/// temporary background response goes only to the temporary store.
fn persist(
    request: &ResponsesRequest,
    store: Option<&Arc<ResponseStore>>,
    temporary: Option<&Arc<Temporary>>,
    items: Vec<Value>,
) -> Option<Persist> {
    match temporary.filter(|_| request.temporary_background(store.is_some())) {
        Some(temporary) => Some(Persist {
            store: Arc::clone(temporary.store()),
            input_items: items,
            temporary: Some(Arc::clone(temporary)),
        }),
        None => store
            .filter(|_| request.store.unwrap_or(true))
            .map(|store| Persist {
                store: Arc::clone(store),
                input_items: items,
                temporary: None,
            }),
    }
}

/// Refuse to continue a temporary response. It is never looked up in the
/// response store, so it is not reported as merely missing there.
fn reject_temporary_previous(
    previous: Option<&str>,
    temporary: Option<&Arc<Temporary>>,
) -> crate::Result<()> {
    match (previous, temporary) {
        (Some(id), Some(temporary)) if temporary.knows(id) => Err(unsupported(
            "previous_response_id",
            &format!(
                "response {id} was created with background=true and store=false, so it is kept \
                 only briefly for retrieval and cannot be continued; send its items in input"
            ),
        )),
        _ => Ok(()),
    }
}

/// The history stored response `id` continues with.
fn previous_history(store: &ResponseStore, id: &str) -> crate::Result<Vec<Value>> {
    let stored = store
        .load(id)
        .map_err(|error| {
            crate::Error::Generation(format!("could not read stored response {id}: {error}"))
        })?
        .ok_or_else(|| invalid(format!("previous response with id {id:?} not found")))?;
    // A background response has no output to continue until it ends.
    match stored.response.get("status").and_then(Value::as_str) {
        Some(status @ ("queued" | "in_progress")) => Err(invalid(format!(
            "previous response {id} is still {status}; wait until it finishes"
        ))),
        _ => Ok(stored.history()),
    }
}

/// The conversation ID a request names: a string, or `{"id": string}`.
fn conversation_id(conversation: Option<&Value>) -> crate::Result<Option<String>> {
    let id = match conversation {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(id)) => Some(id.clone()),
        Some(Value::Object(object)) if object.len() == 1 => {
            object.get("id").and_then(Value::as_str).map(str::to_owned)
        }
        Some(_) => None,
    };
    id.filter(|id| !id.is_empty()).map(Some).ok_or_else(|| {
        invalid("conversation must be a conversation ID or an object with only an id string")
    })
}

/// Conversation `id`'s version and items, oldest first, from one snapshot,
/// each item as one flat object with its `id`.
fn conversation_history(
    conversations: &ConversationStore,
    id: &str,
) -> crate::Result<(u64, Vec<Value>)> {
    let history = conversations
        .conversation_history(id)
        .map_err(|error| match error {
            ServiceError::NotFound(_) => invalid(format!("conversation with id {id:?} not found")),
            ServiceError::InvalidArgument(message) => invalid(message),
            other => crate::Error::Generation(format!("could not read conversation {id}: {other}")),
        })?;
    let items = history
        .items
        .into_iter()
        .map(|item| {
            let mut body = item.body;
            body.insert("id".into(), json!(item.id));
            Value::Object(body)
        })
        .collect();
    Ok((history.conversation.version, items))
}

/// What a Response continuing conversation `id`, read at `version`, will
/// append: this request's `input` items, which must fit the conversation
/// item contract (checked here, so a misfit is refused before generation
/// rather than failing the append after it).
fn conversing(
    store: ConversationStore,
    id: String,
    version: u64,
    input: Vec<Value>,
) -> crate::Result<Conversing> {
    // Room is kept for at least one output item.
    if input.len() >= MAX_APPEND_ITEMS {
        return Err(invalid(format!(
            "a request continuing a conversation may add at most {} input items",
            MAX_APPEND_ITEMS - 1
        )));
    }
    for item in &input {
        conversation_input(item)?;
    }
    Ok(Conversing {
        store,
        id,
        version,
        input,
    })
}

/// Refuse a new input item the conversation store would not keep as sent.
///
/// `input_messages` has already validated the item for generation; this
/// only refuses fields the stored item forms do not carry, an item ID the
/// store cannot key by, and non-string call arguments, instead of dropping
/// or rewriting them. The store validates everything else on append.
fn conversation_input(item: &Value) -> crate::Result<()> {
    let Some(object) = item.as_object() else {
        return Err(invalid("each input item must be an object"));
    };
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message");
    let allowed: &[&str] = match kind {
        "message" => &["id", "type", "role", "content", "status"],
        "function_call" => &["id", "type", "call_id", "name", "arguments", "status"],
        "function_call_output" => &["id", "type", "call_id", "output", "status"],
        "reasoning" => &[
            "id",
            "type",
            "summary",
            "content",
            "encrypted_content",
            "status",
        ],
        other => {
            return Err(unsupported(
                &format!("input item type {other:?}"),
                "only messages, reasoning, function_call and function_call_output are accepted",
            ));
        }
    };
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(unsupported(
            &format!("{kind} field {key:?} with conversation"),
            "a conversation keeps only the fields of the items local generation implements",
        ));
    }
    let id = object.get("id").and_then(Value::as_str).unwrap_or_default();
    let keyable = !id.is_empty()
        && id.len() <= MAX_ID_CHARS
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'));
    if !keyable {
        return Err(invalid(format!(
            "with conversation, an input item id must be 1 to {MAX_ID_CHARS} ASCII letters, \
             digits, '_' or '-', not {id:?}"
        )));
    }
    if kind == "function_call" && !object.get("arguments").is_some_and(Value::is_string) {
        return Err(invalid(
            "with conversation, function_call arguments must be a JSON string",
        ));
    }
    Ok(())
}

fn function_tool(tool: &Value) -> crate::Result<ToolDefinition> {
    if !tool.is_object() {
        return Err(invalid(format!("each tool must be an object, not {tool}")));
    }
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
    let strict = match tool.get("strict") {
        None | Some(Value::Null) => None,
        Some(Value::Bool(strict)) => Some(*strict),
        Some(_) => return Err(invalid(format!("tool {name:?} strict must be a boolean"))),
    };
    function_definition(
        name.to_owned(),
        description,
        tool.get("parameters").cloned(),
        strict,
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

    /// Whether this Response materialises streaming events.
    pub(super) const fn streaming(&self) -> bool {
        self.streaming
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

    /// A background stream's opening: `response.created` and
    /// `response.queued`, both carrying the `queued` Response, since nothing
    /// has run yet.
    pub(super) fn start_queued(&mut self) {
        let response = self.snapshot("queued");
        self.emit("response.created", json!({"response":response}));
        self.emit("response.queued", json!({"response":response}));
    }

    /// `response.in_progress`, once the engine has started the job.
    pub(super) fn started(&mut self) {
        let response = self.snapshot("in_progress");
        self.emit("response.in_progress", json!({"response":response}));
    }

    /// Withdraw the events announcing the end that was just folded in —
    /// `response.completed`, `response.incomplete`, or `response.failed` and
    /// the `error` before it — leaving the output events before them.
    ///
    /// A background stream journals its end from the stored record instead,
    /// once that record is durable (see [`super::journal::terminal_events`]).
    pub(super) fn retract_end(&mut self) {
        let kind = |event: Option<&Value>| {
            event
                .and_then(|event| event.get("type"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        };
        let last = kind(self.events.last());
        if matches!(
            last.as_deref(),
            Some("response.completed" | "response.incomplete" | "response.failed")
        ) {
            self.events.pop();
            if last.as_deref() == Some("response.failed")
                && kind(self.events.last()).as_deref() == Some("error")
            {
                self.events.pop();
            }
        }
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
        // An end of turn the format did not complete (during reasoning, so
        // there is no answer at all) is no completed structured answer, and
        // `incomplete_details` has no reason for it: the Response fails.
        if let Some(message) = super::response::unfinished_format(stats) {
            return self.fail(message);
        }
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
        // A conversation's new items are appended, and a stored response is
        // on disk, before any client is told it finished, so what a client
        // has seen complete can always be continued. A response whose items
        // could not be kept fails rather than claim them. The conversation
        // goes first: its append is the step that can conflict, and when it
        // fails nothing at all has been written.
        // A background response is persisted by its pump, under the lock that
        // also serialises cancellation and deletion, never from here (see
        // [`Self::save`]).
        if error.is_none() && !self.echo.background {
            if let Some(conversing) = &self.echo.conversation
                && let Err(message) = conversing.append(&response)
            {
                eprintln!("response {}: {message}", self.id);
                return self.not_kept(&message, stats);
            }
            if let Some(persist) = &self.echo.persist
                && let Err(failure) = persist.store.save(&response, &persist.input_items)
            {
                eprintln!("response {} could not be stored: {failure}", self.id);
                let mut message = format!("the response could not be stored: {failure}");
                if let Some(conversing) = &self.echo.conversation {
                    message.push_str(&already_appended(&conversing.id));
                }
                return self.not_kept(&message, stats);
            }
        }
        self.emit(kind, json!({"response":response}));
        response
    }

    /// End a Response whose items could not be kept as `failed`, with `message`.
    fn not_kept(&mut self, message: &str, stats: &Stats) -> Value {
        self.emit(
            "error",
            json!({"code":"server_error","message":message,"param":null}),
        );
        let error = json!({"code":"server_error","message":message});
        let failed = self.response("failed", None, Some(&error), Some(stats));
        self.emit("response.failed", json!({"response":failed}));
        failed
    }

    /// The Response as it stands, before any terminal state: `queued` or
    /// `in_progress`. Usage is only known at the end, so it is `null`.
    pub(super) fn snapshot(&self, status: &str) -> Value {
        self.response(status, None, None, None)
    }

    /// End the Response as `cancelled`, keeping the output produced so far
    /// with any unfinished item marked `incomplete`.
    pub(super) fn cancelled(&mut self) -> Value {
        self.close("incomplete");
        self.response("cancelled", None, None, None)
    }

    /// Persist `response` with this request's resolved input, durably. Fails
    /// when the response is not to be stored at all.
    ///
    /// A completed or incomplete `response` continuing a conversation first
    /// appends its items there, exactly once: saving the same response again
    /// replays that append rather than repeating it. When the append fails
    /// nothing is saved and the error says why.
    pub(super) fn save(&self, response: &Value) -> std::io::Result<()> {
        let persist = self.echo.persist.as_ref().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "response is not stored")
        })?;
        let ended = matches!(
            response.get("status").and_then(Value::as_str),
            Some("completed" | "incomplete")
        );
        let Some(conversing) = self.echo.conversation.as_ref().filter(|_| ended) else {
            return persist.store.save(response, &persist.input_items);
        };
        conversing.append(response).map_err(std::io::Error::other)?;
        persist
            .store
            .save(response, &persist.input_items)
            .map_err(|error| {
                std::io::Error::new(
                    error.kind(),
                    format!("{error}{}", already_appended(&conversing.id)),
                )
            })
    }

    /// The store this Response is persisted in, when it is stored.
    pub(super) fn store(&self) -> Option<&Arc<ResponseStore>> {
        self.echo.persist.as_ref().map(|persist| &persist.store)
    }

    /// The temporary store that expires this Response, when it is kept only
    /// temporarily (`background` with `store: false`).
    pub(super) fn temporary(&self) -> Option<&Arc<Temporary>> {
        self.echo
            .persist
            .as_ref()
            .and_then(|persist| persist.temporary.as_ref())
    }

    pub(super) fn id(&self) -> &str {
        &self.id
    }

    pub(super) fn fail(&mut self, message: &str) -> Value {
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
        let mut response = json!({
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
            "parallel_tool_calls":echo.parallel_tool_calls,
            "previous_response_id":echo.previous_response_id,
            "reasoning":{"effort":echo.effort,"summary":null,"context":"all_turns"},
            // Only a response that reached the store says so. A failed
            // foreground one, including one whose write failed, was never
            // stored; a background one is stored in every state it reports.
            // A temporary background one is kept only to be retrieved, so it
            // says `false`, as the client asked.
            "store":echo.persist.as_ref().is_some_and(|persist| persist.temporary.is_none())
                && (echo.background || status != "failed"),
            "background":echo.background,
            "access_programs":null,
            "service_tier":"default",
            "temperature":echo.temperature,
            "text":{"format":echo.text_format},
            "tool_choice":echo.tool_choice,
            "tools":echo.tools,
            "top_p":echo.top_p,
            "truncation":"disabled",
            "usage":usage,
            "user":echo.user,
            "prompt_cache_key":echo.prompt_cache_key,
            "safety_identifier":echo.safety_identifier,
            "metadata":echo.metadata,
        });
        // Only a Response that continues a conversation names one, so every
        // other Response keeps exactly its existing shape.
        if let Some(conversing) = &echo.conversation {
            response["conversation"] = json!({"id":conversing.id});
        }
        response
    }
}

/// Said of a Response that failed to be stored after its items were appended
/// to `conversation`, which keeps them: the append is not undone.
fn already_appended(conversation: &str) -> String {
    format!("; its items were already added to conversation {conversation}")
}

/// One Responses SSE frame: the event type is repeated in the `event:` line,
/// as `OpenAI` sends it.
pub(super) fn sse_frame(event: &Value) -> String {
    let kind = event.get("type").and_then(Value::as_str).unwrap_or("error");
    format!("event: {kind}\ndata: {event}\n\n")
}

#[cfg(test)]
#[allow(clippy::expect_used)]
#[path = "conversation_response_tests.rs"]
mod conversation_response_tests;
