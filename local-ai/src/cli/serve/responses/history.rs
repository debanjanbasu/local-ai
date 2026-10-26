//! A Response's history: input items, compaction, and their chat messages.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use local_engine::bonsai_model::StopReason;
use local_engine::compaction::{compaction_request, summary_message};
use local_engine::{ChatMessage, ChatRequest, Stats, ToolCall};
use local_services::Store as ConversationStore;
use local_services::compaction::{COMPACTION, Compacted, latest, plan};

use super::super::reasoning_crypto::ReasoningCipher;
use super::super::request::{add_verbosity, service_tier, tool_call};
use super::super::response::{new_id, unix_now};
use super::super::store::ResponseStore;
use super::parse::{PreparedResponses, invalid, present, previous_history, unsupported};

/// The `type` of the input item that asks for a compaction turn: Codex's
/// remote compaction (provider capability `remote_compaction = "v2"`)
/// appends one, payload-free, to an otherwise ordinary request and expects
/// exactly one `compaction` output item back.
const COMPACTION_TRIGGER: &str = "compaction_trigger";

/// Decrypt every encrypted reasoning item in `items` into its `content`.
pub(super) fn open_reasoning(
    items: &mut [Value],
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<()> {
    for item in items {
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
    Ok(())
}

/// What compaction item `item` stands for, opened with `cipher`.
fn open_compaction(
    item: &Value,
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<Compacted> {
    let cipher =
        cipher.ok_or_else(|| unsupported("a compaction item", "requires --reasoning-key"))?;
    let id = item
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("a compaction item requires its original item id"))?;
    let envelope = item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("a compaction item requires encrypted_content"))?;
    let payload = cipher
        .unseal_compaction(model, id, envelope)
        .map_err(|error| invalid(format!("compaction item {id}: {error}")))?;
    serde_json::from_str(&payload)
        .map_err(|error| invalid(format!("compaction item {id} is malformed: {error}")))
}

/// A prompt's history from its latest compaction item onward, as it is
/// rendered.
#[derive(Debug)]
pub(in crate::cli::serve) struct History {
    /// The latest compaction item's summary, rendered before `items`.
    pub(super) summary: Option<String>,
    /// The items that compaction item kept, then every item after it; with
    /// no compaction item, every item.
    pub(super) items: Vec<Value>,
    pub(super) instructions: Option<String>,
    /// Where `input_image` parts naming a `file_id` read their bytes.
    pub(super) files: Option<ConversationStore>,
}

impl History {
    /// Resolve `items` (reasoning already decrypted) from their latest
    /// compaction item onward; items before it are not rendered.
    pub(super) fn resolve(
        items: &[Value],
        instructions: Option<String>,
        cipher: Option<&Arc<ReasoningCipher>>,
        model: &str,
        files: Option<&ConversationStore>,
    ) -> crate::Result<Self> {
        let files = files.cloned();
        let Some(at) = latest(items) else {
            return Ok(Self {
                summary: None,
                items: items.to_vec(),
                instructions,
                files,
            });
        };
        let compacted = open_compaction(&items[at], cipher, model)?;
        let mut kept = compacted.items;
        open_reasoning(&mut kept, cipher, model)?;
        kept.extend_from_slice(items.get(at + 1..).unwrap_or_default());
        Ok(Self {
            // An item that summarized nothing ([`Self::sealed`]) has none.
            summary: Some(compacted.summary).filter(|summary| !summary.is_empty()),
            items: kept,
            instructions,
            files,
        })
    }

    /// The whole history as chat messages.
    pub(in crate::cli::serve) fn messages(&self) -> crate::Result<Vec<ChatMessage>> {
        self.messages_of(&self.items)
    }

    /// The summary, if any, and then `items`, as chat messages, after the
    /// instructions. The summary follows the leading system and developer
    /// messages, which the template only takes first.
    fn messages_of(&self, items: &[Value]) -> crate::Result<Vec<ChatMessage>> {
        let mut messages = input_messages(
            &Value::Array(items.to_vec()),
            self.instructions.as_deref(),
            self.files.as_ref(),
        )?;
        if let Some(summary) = &self.summary {
            let at = messages
                .iter()
                .take_while(|message| matches!(message.role.as_str(), "system" | "developer"))
                .count();
            messages.insert(at, summary_message(summary));
        }
        Ok(messages)
    }

    /// Plan a compaction pass that keeps about `keep_tokens` of the newest
    /// items verbatim, measuring with `count`; `None` when nothing can be
    /// summarized. `conversation` is the request this history renders as
    /// (its messages are this history's); the summary request extends it,
    /// so the prefix its turns already prefilled is reused (see
    /// [`compaction_request`]).
    pub(in crate::cli::serve) fn compaction(
        &self,
        keep_tokens: usize,
        conversation: &ChatRequest,
        count: impl Fn(Vec<ChatMessage>) -> crate::Result<usize>,
    ) -> crate::Result<Option<CompactionPass>> {
        let planned = plan(&self.items, keep_tokens, |slice| {
            count(input_messages(
                &Value::Array(slice.to_vec()),
                None,
                self.files.as_ref(),
            )?)
        })?;
        let Some(planned) = planned else {
            return Ok(None);
        };
        Ok(Some(CompactionPass {
            request: compaction_request(conversation),
            kept: planned.kept,
        }))
    }
}

/// One planned compaction: the summary generation to run and the items the
/// compaction item will keep verbatim.
pub(in crate::cli::serve) struct CompactionPass {
    pub(in crate::cli::serve) request: ChatRequest,
    kept: Vec<Value>,
}

impl CompactionPass {
    /// Seal the generated `summary` and the kept items into a compaction
    /// item, returning it and the history it renders as in place of `from`.
    pub(in crate::cli::serve) fn finish(
        self,
        summary: &str,
        from: &History,
        cipher: &ReasoningCipher,
        model: &str,
    ) -> crate::Result<(Value, History)> {
        let summary = summary.trim();
        if summary.is_empty() {
            return Err(crate::Error::Generation(
                "the compaction summary came back empty, so nothing was compacted".into(),
            ));
        }
        let compacted = Compacted {
            summary: summary.to_owned(),
            items: self.kept,
        };
        let item = seal(&compacted, cipher, model)?;
        let history = History {
            summary: Some(compacted.summary),
            items: compacted.items,
            instructions: from.instructions.clone(),
            files: from.files.clone(),
        };
        Ok((item, history))
    }

    /// The items the compaction item keeps, as a client sees them.
    pub(in crate::cli::serve) fn kept(&self) -> &[Value] {
        &self.kept
    }
}

/// `compacted`, sealed into a new compaction item.
fn seal(compacted: &Compacted, cipher: &ReasoningCipher, model: &str) -> crate::Result<Value> {
    let id = new_id("cmp_");
    let payload = serde_json::to_string(compacted)
        .map_err(|error| crate::Error::Generation(error.to_string()))?;
    let sealed = cipher
        .seal_compaction(model, &id, &payload)
        .map_err(|error| crate::Error::Generation(error.to_string()))?;
    Ok(json!({"type":COMPACTION,"id":id,"encrypted_content":sealed}))
}

impl History {
    /// A compaction item that renders exactly as this history does: its
    /// summary, if any, and every item kept. A `compaction_trigger` turn
    /// answers with it when no summary would shorten the history.
    pub(in crate::cli::serve) fn sealed(
        &self,
        cipher: &ReasoningCipher,
        model: &str,
    ) -> crate::Result<Value> {
        let compacted = Compacted {
            summary: self.summary.clone().unwrap_or_default(),
            items: self.items.clone(),
        };
        seal(&compacted, cipher, model)
    }
}

impl PreparedResponses {
    /// Answer this `compaction_trigger` turn with compaction `item` alone,
    /// finishing with `stats` (the summary generation's, if one ran) instead
    /// of generating. A summary cut short by its token budget still
    /// compacted, so the Response is `completed`.
    pub(in crate::cli::serve) fn answer_with_compaction(&mut self, item: Value, mut stats: Stats) {
        stats.stop_reason = StopReason::Eos;
        self.echo.compaction = Some(item);
        self.echo.compaction_turn = Some(stats);
    }

    /// Generate from compacted `history` instead, reporting compaction
    /// `item` as the Response's first output item.
    pub(in crate::cli::serve) fn compacted(
        &mut self,
        item: Value,
        history: History,
    ) -> crate::Result<()> {
        self.request.messages = history.messages()?;
        add_verbosity(&mut self.request.messages, self.echo.verbosity);
        self.history = history;
        self.echo.compaction = Some(item);
        Ok(())
    }
}

#[derive(Deserialize)]
struct CompactRequest {
    /// Required by the contract; checked by
    /// [`super::super::models::check_generation_value`], the loaded model answers.
    model: Option<Value>,
    input: Option<Value>,
    previous_response_id: Option<String>,
    instructions: Option<String>,
    prompt_cache_key: Option<String>,
    prompt_cache_options: Option<Value>,
    service_tier: Option<String>,
}

/// A parsed `POST /v1/responses/compact`.
pub(in crate::cli::serve) struct PreparedCompact {
    pub(in crate::cli::serve) history: History,
    /// The resolved input, returned as it is when nothing can be compacted.
    pub(in crate::cli::serve) items: Vec<Value>,
    pub(in crate::cli::serve) session: Option<String>,
    pub(in crate::cli::serve) cipher: Arc<ReasoningCipher>,
}

/// Parse a `POST /v1/responses/compact` body, resolving
/// `previous_response_id` against `store`. Blocks on the store.
pub(in crate::cli::serve) fn prepare_compact(
    body: &[u8],
    store: Option<&Arc<ResponseStore>>,
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<PreparedCompact> {
    let request: CompactRequest = serde_json::from_slice(body)
        .map_err(|error| invalid(format!("invalid JSON request: {error}")))?;
    super::super::models::check_generation_value(request.model.as_ref())?;
    let cipher = cipher.ok_or_else(|| {
        unsupported(
            "compaction",
            "compaction items are sealed with --reasoning-key, which this server was started \
             without",
        )
    })?;
    if present(request.prompt_cache_options.as_ref()) {
        return Err(unsupported(
            "prompt_cache_options",
            "it needs a hosted OpenAI service",
        ));
    }
    service_tier(request.service_tier.as_deref())?;
    let mut items = match (&request.previous_response_id, store) {
        (Some(id), Some(store)) => previous_history(store, id)?,
        (Some(_), None) => {
            return Err(unsupported(
                "previous_response_id",
                "this server was started without --response-store, so it keeps no responses; \
                 send the earlier items in input instead",
            ));
        }
        (None, _) => Vec::new(),
    };
    match (&request.input, request.previous_response_id.is_some()) {
        // Compacting is what this route does anyway, so a trigger adds nothing.
        (Some(input), _) if !input.is_null() => {
            normalize_input(input, &mut items)?;
        }
        (_, true) => {}
        _ => return Err(invalid("input is required")),
    }
    open_reasoning(&mut items, Some(cipher), model)?;
    let history = History::resolve(&items, request.instructions, Some(cipher), model, None)?;
    // Invalid items fail here, before any generation.
    history.messages()?;
    Ok(PreparedCompact {
        history,
        items,
        session: request.prompt_cache_key,
        cipher: Arc::clone(cipher),
    })
}

/// `item`, a normalized input item, in the output form a `response.compaction`
/// lists it in: messages carry a `status` and output text its `annotations`
/// and `logprobs`. Both forms are accepted back as input.
fn output_item(item: &Value) -> Value {
    let mut item = item.clone();
    if item.get("type").and_then(Value::as_str) == Some("message")
        && let Some(fields) = item.as_object_mut()
    {
        fields.entry("status").or_insert_with(|| json!("completed"));
        for part in fields
            .get_mut("content")
            .and_then(Value::as_array_mut)
            .into_iter()
            .flatten()
            .filter_map(Value::as_object_mut)
        {
            if part.get("type").and_then(Value::as_str) == Some("output_text") {
                part.entry("annotations").or_insert_with(|| json!([]));
                part.entry("logprobs").or_insert_with(|| json!([]));
            }
        }
    }
    item
}

/// The `response.compaction` object: the compacted window `output`, and the
/// usage of the summary generation, if one ran.
pub(in crate::cli::serve) fn compaction_resource(output: &[Value], stats: Option<&Stats>) -> Value {
    let output: Vec<Value> = output.iter().map(output_item).collect();
    let (input, cached, generated) = stats.map_or((0, 0, 0), |stats| {
        let generation = &stats.generation;
        (
            generation.prompt_tokens,
            generation.reused_prompt_tokens,
            generation.generated_tokens,
        )
    });
    json!({
        "id":new_id("resp_"),
        "object":"response.compaction",
        "output":output,
        "created_at":unix_now(),
        "usage":{"input_tokens":input,"input_tokens_details":{"cached_tokens":cached,"cache_write_tokens":0},"output_tokens":generated,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":input+generated},
    })
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
    /// What calls that carry their own result (`file_search_call`)
    /// returned, as tool messages after the turn's.
    answers: Vec<ChatMessage>,
}

impl Turn {
    const fn is_empty(&self) -> bool {
        self.reasoning.is_none() && self.content.is_none() && self.calls.is_empty()
    }
}

fn flush(turn: &mut Turn, messages: &mut Vec<ChatMessage>) {
    if !turn.is_empty() {
        let Turn {
            reasoning,
            content,
            calls,
            answers,
        } = std::mem::take(turn);
        messages.push(ChatMessage {
            role: "assistant".into(),
            content: content.unwrap_or_default(),
            reasoning_content: reasoning,
            tool_calls: calls,
            ..ChatMessage::default()
        });
        messages.extend(answers);
    }
}

fn plain(role: &str, content: String, tool_call_id: Option<String>) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content,
        tool_call_id,
        ..ChatMessage::default()
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
///
/// `compaction_trigger` items are requests, not history, so none is kept;
/// `true` says the input ended with one (see [`COMPACTION_TRIGGER`]).
pub(super) fn normalize_input(input: &Value, items: &mut Vec<Value>) -> crate::Result<bool> {
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
    let triggered = new.last().is_some_and(is_trigger);
    for item in new.into_iter().filter(|item| !is_trigger(item)) {
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
            COMPACTION => "cmp_",
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
        let encrypted = kind == "reasoning" && present(item.get("encrypted_content"));
        let id = match item.get("id").and_then(Value::as_str) {
            Some(id) if !seen.contains(id) => Some(id.to_owned()),
            Some(_) => None,
            // A sealed item replayed without its `id` (the AI SDK sends none
            // with `store: false`) takes the one its envelope carries, which
            // opening the envelope authenticates.
            None if encrypted || kind == COMPACTION => item
                .get("encrypted_content")
                .and_then(Value::as_str)
                .and_then(ReasoningCipher::sealed_item_id)
                .filter(|id| !seen.contains(id)),
            None => Some(new_id(prefix)),
        };
        let id = match id {
            Some(id) => id,
            None if encrypted => {
                return Err(invalid(
                    "encrypted reasoning requires its original unique item id",
                ));
            }
            None if kind == COMPACTION => {
                return Err(invalid(
                    "a compaction item requires its original unique item id",
                ));
            }
            None => new_id(prefix),
        };
        seen.insert(id.clone());
        item.insert("id".into(), json!(id));
        items.push(Value::Object(item));
    }
    Ok(triggered)
}

/// Whether `item` is a `compaction_trigger` input item.
fn is_trigger(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some(COMPACTION_TRIGGER)
}

/// Convert already-authenticated `input` into the chat history the engine renders.
fn input_messages(
    input: &Value,
    instructions: Option<&str>,
    files: Option<&ConversationStore>,
) -> crate::Result<Vec<ChatMessage>> {
    let mut messages = Vec::new();
    match input {
        Value::String(text) => messages.push(plain("user", text.clone(), None)),
        Value::Array(items) => {
            let mut turn = Turn::default();
            super::super::images::request_scope(|| {
                items
                    .iter()
                    .try_for_each(|item| input_item(item, &mut turn, &mut messages, files))
            })?;
            flush(&mut turn, &mut messages);
        }
        _ => return Err(invalid("input must be a string or an array of items")),
    }
    if let Some(instructions) = instructions.filter(|text| !text.is_empty()) {
        // The template takes one system message and only first, so the
        // instructions join a leading system message instead of preceding it.
        match messages.first_mut() {
            Some(first) if matches!(first.role.as_str(), "system" | "developer") => {
                first.content = format!("{instructions}\n\n{}", first.content);
            }
            _ => messages.insert(0, plain("system", instructions.to_owned(), None)),
        }
    }
    Ok(messages)
}

fn input_item(
    item: &Value,
    turn: &mut Turn,
    messages: &mut Vec<ChatMessage>,
    files: Option<&ConversationStore>,
) -> crate::Result<()> {
    let kind = item.get("type").and_then(Value::as_str);
    match kind {
        None | Some("message") => {
            let role = item
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("a message input item requires a role"))?;
            let parts = item_parts(item.get("content").unwrap_or(&Value::Null), role, files)?;
            match role {
                "assistant" => {
                    if !parts.images.is_empty() {
                        return Err(invalid("assistant messages cannot contain images"));
                    }
                    if turn.content.is_some() || !turn.calls.is_empty() {
                        flush(turn, messages);
                    }
                    turn.content = Some(parts.text);
                }
                "user" | "system" | "developer" => {
                    flush(turn, messages);
                    messages.push(ChatMessage {
                        images: parts.images,
                        ..plain(role, parts.text, None)
                    });
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
            let name = match item.get("namespace").and_then(Value::as_str) {
                Some(namespace) => super::super::hosted::qualified(namespace, field("name")?),
                None => field("name")?.to_owned(),
            };
            let call = tool_call(
                field("call_id")?.to_owned(),
                name,
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
                Some(output @ (Value::String(_) | Value::Array(_))) => {
                    item_parts(output, "tool", files)?
                }
                _ => return Err(invalid("function_call_output output must be text")),
            };
            messages.push(ChatMessage {
                images: output.images,
                ..plain("tool", output.text, Some(call_id.to_owned()))
            });
        }
        Some("item_reference") => {
            return Err(unsupported(
                "item_reference",
                "item lookup is not implemented; send the item itself",
            ));
        }
        // `History::resolve` renders the latest compaction item from its
        // sealed content and skips everything before it, so none is left.
        Some(COMPACTION) => {
            return Err(invalid(
                "a compaction item is only rendered from its sealed content",
            ));
        }
        Some(other) => match super::super::hosted::input_item(other, item) {
            Some(converted) => match converted? {
                super::super::hosted::Input::Call(call) => turn.calls.push(call),
                super::super::hosted::Input::Output { call_id, content } => {
                    flush(turn, messages);
                    messages.push(plain("tool", content, Some(call_id)));
                }
                super::super::hosted::Input::Answered { call, content } => {
                    turn.answers
                        .push(plain("tool", content, Some(call.id.clone())));
                    turn.calls.push(call);
                }
            },
            None => {
                return Err(unsupported(
                    &format!("input item type {other:?}"),
                    "only messages, reasoning, file_search calls, and function, custom, shell, \
                     local_shell and apply_patch calls and their outputs are accepted",
                ));
            }
        },
    }
    Ok(())
}

/// Text, images and videos of a message or tool output: a string, or text
/// parts concatenated with each `input_image` (a data or `http(s)` URL
/// `image_url`, or an uploaded `file_id`, with `detail`) and `input_video`
/// (a `video_url`, likewise, or `file_id`; this server's extension) in its
/// place.
fn item_parts(
    content: &Value,
    role: &str,
    files: Option<&ConversationStore>,
) -> crate::Result<super::super::images::Parts> {
    use super::super::images::Kind;
    let mut out = super::super::images::Parts::default();
    match content {
        Value::String(text) => {
            out.push_text(text);
            Ok(out)
        }
        Value::Array(parts) => {
            for part in parts {
                let kind = part.get("type").and_then(Value::as_str).unwrap_or_default();
                let field = match kind {
                    "input_text" | "output_text" => "text",
                    "refusal" if role == "assistant" => "refusal",
                    "input_image" => {
                        let data = match (
                            part.get("image_url").and_then(Value::as_str),
                            part.get("file_id").and_then(Value::as_str),
                        ) {
                            (Some(url), None) => super::super::images::url(url, Kind::Image)?,
                            (None, Some(id)) => super::super::images::file(files, id)?,
                            _ => {
                                return Err(invalid(
                                    "an input_image part needs exactly one of image_url and \
                                     file_id",
                                ));
                            }
                        };
                        out.push_image(data, super::super::images::detail(part.get("detail"))?);
                        continue;
                    }
                    "input_video" => {
                        let url = match part.get("video_url") {
                            Some(Value::Object(video)) => video.get("url").and_then(Value::as_str),
                            other => other.and_then(Value::as_str),
                        };
                        let data = match (url, part.get("file_id").and_then(Value::as_str)) {
                            (Some(url), None) => super::super::images::url(url, Kind::Video)?,
                            (None, Some(id)) => super::super::images::file(files, id)?,
                            _ => {
                                return Err(invalid(
                                    "an input_video part needs exactly one of video_url and \
                                     file_id",
                                ));
                            }
                        };
                        out.push_video(data);
                        continue;
                    }
                    "input_file" | "input_audio" => {
                        return Err(unsupported(
                            &format!("content part {kind:?}"),
                            "Bonsai 2 reads text, images (input_image) and video (input_video)",
                        ));
                    }
                    other => {
                        return Err(invalid(format!("unsupported content part type {other:?}")));
                    }
                };
                out.push_text(part.get(field).and_then(Value::as_str).ok_or_else(|| {
                    invalid(format!("a {kind} content part requires a {field} string"))
                })?);
            }
            Ok(out)
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
