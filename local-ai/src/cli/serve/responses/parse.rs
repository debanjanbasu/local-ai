//! Parsing a `POST /v1/responses` body into the engine request, and the
//! request members its Response echoes.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use local_engine::compaction::SummaryDetail;
use local_engine::{ChatRequest, ReasoningEffort, Sampling, Stats, ToolDefinition};
use local_services::compaction::COMPACTION;
use local_services::conversations::{MAX_APPEND_ITEMS, MAX_ID_CHARS};
use local_services::{Append, Error as ServiceError, Store as ConversationStore};

use super::super::background::Temporary;
use super::super::reasoning_crypto::ReasoningCipher;
use super::super::request::{
    SamplingOptions, add_verbosity, effective_reasoning, function_definition,
    responses_text_format, responses_tool_choice, service_tier, tool_policy, top_logprobs,
    unsupported, verbosity_level,
};
use super::super::response::Reply;
use super::super::store::{ResponseStore, StoredResponse};
use super::history::{History, normalize_input, open_reasoning};
use crate::GenerateParams;

#[derive(Deserialize)]
struct ResponsesRequest {
    /// Checked by [`super::super::models::check_generation`]; the loaded model answers.
    model: Option<String>,
    input: Option<Value>,
    instructions: Option<String>,
    max_output_tokens: Option<usize>,
    /// Bounds the hosted tool calls this server runs for the Response.
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
    moderation: Option<Value>,
    access_programs: Option<Value>,
    // This server's sampling extensions, the same as on Chat Completions.
    top_k: Option<usize>,
    min_p: Option<f32>,
    seed: Option<u64>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    /// Checked by `reject_unsupported`. Codex's `client_metadata` telemetry
    /// is ignored like any other member this server does not know.
    truncation: Option<String>,
}

#[derive(Deserialize)]
struct Reasoning {
    effort: Option<String>,
    summary: Option<String>,
    context: Option<String>,
    mode: Option<String>,
}

#[derive(Deserialize)]
struct StreamOptions {
    include_obfuscation: Option<bool>,
}

#[derive(Deserialize)]
struct TextConfig {
    format: Option<Value>,
    verbosity: Option<String>,
}

/// The request members a Response object reports back.
#[derive(Debug)]
pub(in crate::cli::serve) struct Echo {
    pub(super) instructions: Option<String>,
    pub(super) max_output_tokens: Option<usize>,
    pub(super) metadata: Value,
    /// The accepted `text.format`, as the Response reports it.
    pub(super) text_format: Value,
    /// The accepted `text.verbosity`, `medium` by default.
    pub(super) verbosity: &'static str,
    /// The accepted `tool_choice`: a mode string or the named function.
    pub(super) tool_choice: Value,
    pub(super) parallel_tool_calls: bool,
    pub(super) tools: Vec<Value>,
    pub(super) temperature: f32,
    pub(super) top_p: f32,
    /// The reasoning level that ran, as `reasoning.effort` reports it.
    pub(super) effort: &'static str,
    /// The effective `reasoning.context`.
    pub(super) context: &'static str,
    pub(super) prompt_cache_key: Option<String>,
    pub(super) safety_identifier: Option<String>,
    pub(super) previous_response_id: Option<String>,
    /// Seals each reasoning item's `encrypted_content`; set only when the
    /// request's `include` asks for `reasoning.encrypted_content`.
    pub(super) cipher: Option<Arc<ReasoningCipher>>,
    /// Where the final Response is persisted, when it is to be stored.
    pub(super) persist: Option<Persist>,
    /// `background: true`: generated detached from the request, and persisted
    /// by [`super::super::background`] rather than by [`ResponsesState`](super::ResponsesState).
    pub(super) background: bool,
    /// The durable conversation this Response continues and is appended to.
    pub(super) conversation: Option<Conversing>,
    /// `reasoning.summary`: the detail the raw reasoning is summarized at.
    pub(super) summary: Option<SummaryDetail>,
    /// What runs the summary generation; attached by the server.
    pub(super) summarizer: Option<Summarizer>,
    /// The compaction item made before generation (`context_management`),
    /// the Response's first output item.
    pub(super) compaction: Option<Value>,
    /// A `compaction_trigger` turn's summary generation: the Response is
    /// `compaction` alone, finished with these measurements, and nothing
    /// else is generated.
    pub(super) compaction_turn: Option<Stats>,
    /// What the compaction generation run before this Response's own cost,
    /// whether or not its item was kept; added to the Response's `usage`.
    pub(super) spent: Spent,
    /// Who executes each declared tool that is not a plain function.
    pub(super) kinds: super::super::hosted::Kinds,
    /// `max_tool_calls`: hosted calls this Response may run.
    pub(super) max_tool_calls: Option<u64>,
    /// `include` lists `file_search_call.results`.
    pub(super) file_search_results: bool,
}

/// Runs one generation of the loaded model to its end, handing each piece of
/// visible content to the callback, and returns its measurements. The
/// callback answers `false` when nobody is listening any more, which cancels
/// the generation.
///
/// The server's runs it on the engine; tests substitute their own.
pub(in crate::cli::serve) type Generate =
    Arc<dyn Fn(ChatRequest, &mut dyn FnMut(&str) -> bool) -> crate::Result<Stats> + Send + Sync>;

/// The tokens of the extra generations a Response runs with the same model
/// besides its own (a `context_management` compaction, a reasoning summary).
///
/// The pinned `ResponseUsage` has no field of its own for them, so they are
/// added to `input_tokens` (with their reused prefix in `cached_tokens`) and
/// `output_tokens`, and so to `total_tokens`; `reasoning_tokens` stays the
/// Response's own, as neither generation reasons.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::cli::serve) struct Spent {
    pub(super) prompt: usize,
    pub(super) reused: usize,
    pub(super) generated: usize,
}

impl Spent {
    pub(in crate::cli::serve) const fn add(&mut self, stats: &Stats) {
        self.prompt += stats.generation.prompt_tokens;
        self.reused += stats.generation.reused_prompt_tokens;
        self.generated += stats.generation.generated_tokens;
    }
}

/// The generation a Response summarizes its reasoning with.
#[derive(Clone)]
pub(in crate::cli::serve) struct Summarizer(pub(in crate::cli::serve) Generate);

impl std::fmt::Debug for Summarizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Summarizer")
    }
}

/// A durable conversation a Response continues, and what it will append.
#[derive(Debug)]
pub(super) struct Conversing {
    store: ConversationStore,
    pub(super) id: String,
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
    pub(super) fn append(&self, response: &Value) -> Result<(), String> {
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
    pub(in crate::cli::serve) const fn background(&self) -> bool {
        self.background
    }

    /// Whether `reasoning.summary` asked for a summary generation.
    pub(in crate::cli::serve) const fn summarizes(&self) -> bool {
        self.summary.is_some()
    }

    /// Attach what runs the summary generation.
    pub(in crate::cli::serve) fn set_summarizer(&mut self, summarizer: Summarizer) {
        self.summarizer = Some(summarizer);
    }

    /// Who executes each declared tool.
    pub(in crate::cli::serve) const fn kinds(&self) -> &super::super::hosted::Kinds {
        &self.kinds
    }

    pub(in crate::cli::serve) const fn max_tool_calls(&self) -> Option<u64> {
        self.max_tool_calls
    }

    /// Count a compaction generation run for this Response in its `usage`.
    pub(in crate::cli::serve) const fn spend(&mut self, stats: &Stats) {
        self.spent.add(stats);
    }

    /// For a `compaction_trigger` turn, the measurements its Response
    /// finishes with in place of a generation of its own.
    pub(in crate::cli::serve) const fn compaction_turn(&self) -> Option<&Stats> {
        self.compaction_turn.as_ref()
    }

    /// Record who creates this Response (see [`StoredResponse::creator`]),
    /// for a durable background one that a restarted server may resume.
    pub(in crate::cli::serve) fn set_creator(&mut self, creator: Option<Value>) {
        if let Some(persist) = self
            .persist
            .as_mut()
            .filter(|persist| persist.request.is_some())
        {
            persist.creator = creator;
        }
    }

    /// The durable store of a background Response that a restarted server
    /// may regenerate: one kept in `--response-store`, not temporarily.
    pub(in crate::cli::serve) fn resumable_store(&self) -> Option<&Arc<ResponseStore>> {
        self.persist
            .as_ref()
            .filter(|persist| self.background && persist.request.is_some())
            .map(|persist| &persist.store)
    }
}

/// A Response that is to be stored, and the resolved input it answers.
#[derive(Debug)]
pub(super) struct Persist {
    pub(super) store: Arc<ResponseStore>,
    pub(super) input_items: Vec<Value>,
    /// A `background` `store: false` Response: `store` is this server's
    /// private temporary store, which expires it after it ends.
    pub(super) temporary: Option<Arc<Temporary>>,
    /// The request body of a durable background Response, kept in its record
    /// while it is pending so a restarted server can regenerate it (see
    /// [`super::super::background::Background::resume`]). `None` for everything
    /// else, including one continuing a conversation, which is never
    /// regenerated because its append cannot be repeated with new output.
    pub(super) request: Option<Value>,
    /// Who created `request` (see [`StoredResponse::creator`]), kept with it.
    pub(super) creator: Option<Value>,
    /// The project that owns the Response (see [`StoredResponse::project`]).
    pub(super) project: Option<String>,
}

/// A `context_management` compaction entry.
#[derive(Clone, Copy, Debug)]
pub(in crate::cli::serve) struct ContextCompaction {
    /// `compact_threshold`: the prompt tokens that make compaction due;
    /// `None` (null or absent) is the server's default,
    /// [`local_services::compaction::DEFAULT_COMPACT_THRESHOLD`].
    pub(in crate::cli::serve) threshold: Option<usize>,
}

pub(in crate::cli::serve) struct PreparedResponses {
    pub(in crate::cli::serve) stream: bool,
    pub(in crate::cli::serve) request: ChatRequest,
    pub(in crate::cli::serve) echo: Echo,
    /// `context_management` compaction, when requested.
    pub(in crate::cli::serve) context_compaction: Option<ContextCompaction>,
    /// The input ended with a `compaction_trigger`: this Response compacts
    /// the history and answers with the compaction item alone.
    pub(in crate::cli::serve) compaction_trigger: bool,
    /// The prompt's history, as rendered: what a compaction pass summarizes.
    pub(in crate::cli::serve) history: History,
}

pub(super) fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::InvalidArgument(message.into())
}

pub(super) fn present(value: Option<&Value>) -> bool {
    value.is_some_and(|value| !value.is_null())
}

impl ResponsesRequest {
    /// Whether `include` asks for `reasoning.encrypted_content`.
    fn includes_encrypted_reasoning(&self) -> bool {
        self.include
            .iter()
            .flatten()
            .any(|value| value == "reasoning.encrypted_content")
    }

    /// Whether `include` asks for `file_search_call.results`.
    fn includes_file_search_results(&self) -> bool {
        self.include
            .iter()
            .flatten()
            .any(|value| value == "file_search_call.results")
    }

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
        if !matches!(self.truncation.as_deref(), None | Some("disabled" | "auto")) {
            return Err(invalid(
                "truncation must be \"auto\" or \"disabled\"; either way the engine never drops \
                 input, and a request over the context fails instead",
            ));
        }
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
        if present(self.context_management.as_ref()) && !encrypted {
            return Err(unsupported(
                "context_management",
                "compaction items are sealed with --reasoning-key, which this server was started \
                 without",
            ));
        }
        for (name, value) in [
            ("prompt", &self.prompt),
            ("prompt_cache_options", &self.prompt_cache_options),
            ("moderation", &self.moderation),
            ("access_programs", &self.access_programs),
        ] {
            if present(value.as_ref()) {
                return Err(unsupported(name, "it needs a hosted OpenAI service"));
            }
        }
        if self.include.as_ref().is_some_and(|include| {
            include.iter().any(|value| match value.as_str() {
                "reasoning.encrypted_content" => !encrypted,
                "file_search_call.results" | LOGPROBS_INCLUDE => false,
                _ => true,
            })
        }) {
            return Err(unsupported(
                "include",
                "only reasoning.encrypted_content (which requires --reasoning-key), \
                 file_search_call.results and message.output_text.logprobs are implemented",
            ));
        }
        top_logprobs(self.top_logprobs)?;
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
        service_tier(self.service_tier.as_deref())
    }

    fn params(&self) -> GenerateParams {
        SamplingOptions {
            max_tokens: self.max_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            seed: self.seed,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repetition_penalty: self.repetition_penalty,
            logprobs: self.logprobs(),
        }
        .params()
    }

    /// The alternatives each answer token reports, when `include` asks for
    /// `message.output_text.logprobs` or `top_logprobs` asks for any.
    fn logprobs(&self) -> Option<usize> {
        let included = self
            .include
            .iter()
            .flatten()
            .any(|value| value == LOGPROBS_INCLUDE);
        let top = self.top_logprobs.unwrap_or(0);
        (included || top > 0).then_some(top as usize)
    }
}

/// The `include` entry that reports `output_text` log-probabilities.
const LOGPROBS_INCLUDE: &str = "message.output_text.logprobs";

/// The detail `reasoning.summary` asks the raw reasoning to be summarized at,
/// if any.
fn summary_detail(reasoning: Option<&Reasoning>) -> crate::Result<Option<SummaryDetail>> {
    let Some(reasoning) = reasoning else {
        return Ok(None);
    };
    let Some(name) = &reasoning.summary else {
        return Ok(None);
    };
    SummaryDetail::parse(name).map(Some).ok_or_else(|| {
        invalid(format!(
            "reasoning.summary must be auto, concise or detailed, not {name:?}"
        ))
    })
}

/// The `context_management` entries: `None` when absent, else the one
/// `compaction` entry.
fn context_compaction(value: Option<&Value>) -> crate::Result<Option<ContextCompaction>> {
    let entries = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(entries)) if !entries.is_empty() => entries,
        Some(_) => {
            return Err(invalid(
                "context_management must be a non-empty array of entries",
            ));
        }
    };
    let [entry] = entries.as_slice() else {
        return Err(invalid(
            "context_management may hold one compaction entry, the only type there is",
        ));
    };
    let entry = entry
        .as_object()
        .ok_or_else(|| invalid("a context_management entry must be an object"))?;
    if let Some(key) = entry
        .keys()
        .find(|key| !matches!(key.as_str(), "type" | "compact_threshold"))
    {
        return Err(invalid(format!(
            "unknown field context_management[0].{key}"
        )));
    }
    match entry.get("type").and_then(Value::as_str) {
        Some("compaction") => {}
        Some(other) => {
            return Err(unsupported(
                &format!("context_management type {other:?}"),
                "only compaction is implemented",
            ));
        }
        None => return Err(invalid("context_management[0].type must be a string")),
    }
    match entry.get("compact_threshold") {
        None | Some(Value::Null) => Ok(Some(ContextCompaction { threshold: None })),
        Some(value) => match value.as_u64() {
            Some(threshold) if threshold >= 1000 => Ok(Some(ContextCompaction {
                threshold: Some(usize::try_from(threshold).unwrap_or(usize::MAX)),
            })),
            _ => Err(invalid(
                "context_management[0].compact_threshold must be an integer of at least 1000",
            )),
        },
    }
}

/// The requested effort, and whether earlier turns keep their reasoning
/// (`reasoning.context`: `current_turn` drops it; `auto`, the default, keeps
/// it as the checkpoint template does), refusing options this server cannot
/// honour.
fn reasoning_options(reasoning: Option<&Reasoning>) -> crate::Result<(Option<&str>, bool)> {
    let Some(reasoning) = reasoning else {
        return Ok((None, true));
    };
    let preserve = match reasoning.context.as_deref() {
        None | Some("auto" | "all_turns") => true,
        Some("current_turn") => false,
        Some(other) => {
            return Err(invalid(format!(
                "reasoning.context {other:?} is not one of auto, current_turn or all_turns"
            )));
        }
    };
    if !matches!(reasoning.mode.as_deref(), None | Some("standard")) {
        return Err(unsupported(
            "reasoning.mode",
            "only standard execution exists",
        ));
    }
    Ok((reasoning.effort.as_deref(), preserve))
}

/// [`prepare_responses_with`] on a server with no response store.
#[cfg(test)]
pub(in crate::cli::serve) fn prepare_responses(
    body: &[u8],
    thinking: bool,
) -> crate::Result<PreparedResponses> {
    prepare_responses_with(body, thinking, None, None, "m")
}

/// [`prepare_input_tokens_conversing`] on a server with no conversation store.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::cli::serve) fn prepare_input_tokens(
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<ChatRequest> {
    prepare_input_tokens_conversing(body, thinking, store, (None, None), cipher, model)
}

/// Count requests use the generation input contract, but never persist a
/// response or admit generation. Reject generation-only options rather than
/// silently accepting a misspelled count request.
///
/// A `conversation` is read, as one snapshot, exactly as generation would
/// read it, and never written. Stored objects are those `project` may see
/// (see [`prepare_responses_conversing`]). Blocks on the stores.
pub(in crate::cli::serve) fn prepare_input_tokens_conversing(
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    (conversations, project): (Option<&ConversationStore>, Option<&str>),
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
                | "truncation"
                | "tools"
                | "text"
                | "reasoning"
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
    let prepared = prepare_responses_conversing(
        &body,
        thinking,
        store,
        None,
        (conversations, project),
        cipher,
        model,
    )?;
    Ok(prepared.request)
}

/// [`prepare_responses_retaining`] on a server with no temporary store.
#[cfg(test)]
pub(in crate::cli::serve) fn prepare_responses_with(
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
pub(in crate::cli::serve) fn prepare_responses_retaining(
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    temporary: Option<&Arc<Temporary>>,
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<PreparedResponses> {
    prepare_responses_conversing(
        body,
        thinking,
        store,
        temporary,
        (None, None),
        cipher,
        model,
    )
}

/// Parse a `POST /v1/responses` body, resolving `previous_response_id`
/// against `store` and deciding whether the result will be stored there, or,
/// for `background` with `store: false`, kept in `temporary`.
///
/// A `conversation` is read from `conversations` as one ordered snapshot
/// whose items precede this request's input; its version and the request's
/// new items are kept for the append when the Response ends. Blocks on the
/// stores, so it runs on the blocking pool.
///
/// The caller acts for `project` (see [`super::super::admin::Caller::project`]):
/// `previous_response_id` must name a Response it may see, the Response is
/// stored as its own, and `conversations` (which also resolves `file_id`
/// parts) is scoped to it.
#[allow(clippy::too_many_lines)]
pub(in crate::cli::serve) fn prepare_responses_conversing(
    body: &[u8],
    thinking: bool,
    store: Option<&Arc<ResponseStore>>,
    temporary: Option<&Arc<Temporary>>,
    (conversations, project): (Option<&ConversationStore>, Option<&str>),
    cipher: Option<&Arc<ReasoningCipher>>,
    model: &str,
) -> crate::Result<PreparedResponses> {
    let request: ResponsesRequest = serde_json::from_slice(body)
        .map_err(|error| invalid(format!("invalid JSON request: {error}")))?;
    super::super::models::check_generation(request.model.as_deref())?;
    reject_temporary_previous(request.previous_response_id.as_deref(), temporary, project)?;
    request.reject_unsupported(
        store.is_some(),
        temporary.is_some(),
        cipher.is_some(),
        conversations.is_some(),
    )?;
    let conversation_id = conversation_id(request.conversation.as_ref())?;
    let (effort, preserve_thinking) = reasoning_options(request.reasoning.as_ref())?;
    let reasoning = effective_reasoning(effort, thinking)?;
    let thinking = reasoning.is_some();
    let summary = summary_detail(request.reasoning.as_ref())?;
    let context_compaction = context_compaction(request.context_management.as_ref())?;
    let (choice, echoed_choice) =
        match super::super::hosted::tool_choice(request.tool_choice.as_ref()) {
            Some(choice) => choice?,
            None => responses_tool_choice(request.tool_choice.as_ref())?,
        };
    let mut echoed_tools = Vec::new();
    let mut tools = Vec::new();
    let mut kinds = super::super::hosted::Kinds::default();
    for tool in request.tools.as_deref().unwrap_or_default() {
        echoed_tools.push(super::super::hosted::declare(
            tool,
            function_tool,
            &mut tools,
            &mut kinds,
        )?);
    }
    tool_policy(&tools, &choice)?;
    if conversation_id.is_some() && kinds.reports_other_items() {
        return Err(unsupported(
            "namespace, custom, shell, local_shell and apply_patch tools with conversation",
            "the conversation store keeps only message, reasoning and function call items",
        ));
    }
    let parallel_tool_calls = request.parallel_tool_calls.unwrap_or(true);
    let (response_format, text_format) =
        responses_text_format(request.text.as_ref().and_then(|text| text.format.as_ref()))?;
    let verbosity = verbosity_level(
        request
            .text
            .as_ref()
            .and_then(|text| text.verbosity.as_deref()),
        "text.verbosity",
    )?;
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
        (Some(id), Some(store)) => previous_history(store, id, project)?,
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
    let compaction_trigger = normalize_input(input, &mut items)?;
    if compaction_trigger {
        if cipher.is_none() {
            return Err(unsupported(
                "compaction_trigger",
                "compaction items are sealed with --reasoning-key, which this server was started \
                 without",
            ));
        }
        if request.background == Some(true) {
            return Err(unsupported(
                "compaction_trigger with background",
                "a compaction turn is answered while the request waits",
            ));
        }
    }
    // This request's own items, before any reasoning is decrypted below: only
    // these, never the replayed history, are appended to a conversation.
    let new_input = if snapshot.is_some() {
        items[replayed..].to_vec()
    } else {
        Vec::new()
    };
    open_reasoning(&mut items, cipher, model)?;
    let history = History::resolve(
        &items,
        request.instructions.clone(),
        cipher,
        model,
        conversations,
    )?;
    let mut messages = history.messages()?;
    add_verbosity(&mut messages, verbosity);
    // Refused now, before generation, rather than when the append fails.
    let conversation = snapshot
        .map(|(conversations, id, version)| conversing(conversations, id, version, new_input))
        .transpose()?;
    let mut persist = persist(&request, (store, temporary), items, project);
    if let Some(persist) = persist.as_mut().filter(|persist| {
        request.background == Some(true) && persist.temporary.is_none() && conversation.is_none()
    }) {
        persist.request = serde_json::from_slice(body).ok();
    }
    let params = request.params();
    let echo = Echo {
        instructions: request.instructions.clone(),
        max_output_tokens: request.max_output_tokens,
        metadata: request.metadata.clone().unwrap_or_else(|| json!({})),
        text_format,
        verbosity,
        tool_choice: echoed_choice,
        parallel_tool_calls,
        tools: echoed_tools,
        temperature: params.temperature,
        top_p: params.top_p,
        effort: reasoning.map_or("none", ReasoningEffort::name),
        context: if preserve_thinking {
            "all_turns"
        } else {
            "current_turn"
        },
        prompt_cache_key: request.prompt_cache_key.clone(),
        safety_identifier: request.safety_identifier.clone(),
        previous_response_id: request.previous_response_id.clone(),
        // Reasoning is sealed only on request, as `OpenAI` does: a key on
        // the server is not a request, and a client replaying an unrequested
        // envelope without its item id would be refused. Stored and
        // conversation history keep the plaintext, so nothing else needs it.
        cipher: cipher
            .filter(|_| request.includes_encrypted_reasoning())
            .cloned(),
        // `reject_unsupported` has ensured a background response is stored,
        // durably or temporarily.
        background: request.background == Some(true) && persist.is_some(),
        persist,
        conversation,
        summary,
        summarizer: None,
        compaction: None,
        compaction_turn: None,
        spent: Spent::default(),
        kinds,
        max_tool_calls: request.max_tool_calls,
        file_search_results: request.includes_file_search_results(),
    };
    Ok(PreparedResponses {
        stream: request.stream.unwrap_or(false),
        request: ChatRequest {
            messages,
            max_tokens: params.max_tokens,
            sampling: Sampling(params),
            thinking,
            effort: reasoning.unwrap_or_default(),
            reasoning_instruction: false,
            preserve_thinking,
            session: request.prompt_cache_key,
            tools,
            tool_choice: choice,
            parallel_tool_calls,
            response_format,
            stop: Vec::new(),
        },
        echo,
        context_compaction,
        compaction_trigger,
        history,
    })
}

/// Rebuild the generation of the pending background Response `stored` from
/// the request body its record keeps, for a server resuming it after a
/// restart.
///
/// The same request is prepared again, except that its input is the
/// resolved input already in the record (so a `previous_response_id` deleted
/// since is not read again) and its effort is the one the Response reports,
/// whatever this server's default (`thinking`), which must still allow it.
/// The Response keeps its ID, creation time
/// and model; the record keeps `request`, so a later restart can do the same.
/// A compaction item the record already reports is reused rather than run
/// again; `summarizer` runs a requested reasoning summary. A Response that
/// declares hosted tools is regenerated only when no hosted call of it had
/// started executing (see [`ResponseStore::tools_started`]). Blocks on the
/// store and on decryption.
pub(in crate::cli::serve) fn prepare_resumed(
    stored: &StoredResponse,
    request: &Value,
    (thinking, summarizer): (bool, Option<&Summarizer>),
    store: &Arc<ResponseStore>,
    cipher: Option<&Arc<ReasoningCipher>>,
) -> crate::Result<(ChatRequest, Reply)> {
    let response = &stored.response;
    let model = response
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("the stored response names no model"))?;
    let mut body = request.clone();
    let object = body
        .as_object_mut()
        .ok_or_else(|| invalid("the stored request is not an object"))?;
    if object
        .get("conversation")
        .is_some_and(|value| !value.is_null())
    {
        return Err(unsupported(
            "resuming",
            "a response continuing a conversation is not regenerated",
        ));
    }
    object.remove("previous_response_id");
    object.insert("input".into(), Value::Array(stored.input_items.clone()));
    let effort = response
        .pointer("/reasoning/effort")
        .cloned()
        .unwrap_or(Value::Null);
    match object.get_mut("reasoning").and_then(Value::as_object_mut) {
        Some(reasoning) => {
            reasoning.insert("effort".into(), effort);
        }
        None => {
            object.insert("reasoning".into(), json!({"effort":effort}));
        }
    }
    let bytes = serde_json::to_vec(&body).map_err(|error| invalid(error.to_string()))?;
    // Stored again as the project's that owns it.
    let owner = (None, stored.project.as_deref());
    let mut prepared =
        prepare_responses_conversing(&bytes, thinking, Some(store), None, owner, cipher, model)?;
    let id = response
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if prepared.echo.kinds().all_hosted().next().is_some() {
        // Regenerating would run its hosted calls again, and a shell command
        // or file write may already have taken effect before the restart.
        // The marker is durable before any hosted call starts, so without it
        // none ever did.
        let started = store.tools_started(&id).map_err(|error| {
            crate::Error::Generation(format!("its hosted tool record is unreadable: {error}"))
        })?;
        if started {
            return Err(unsupported(
                "resuming",
                "a hosted tool call of this response had started executing before the \
                 restart, so regenerating it could run that call twice",
            ));
        }
    }
    // A compaction made before the restart is reused, not run again: it is
    // already the first output item the record (and any stream) reported.
    if let Some(item) = response
        .pointer("/output/0")
        .filter(|item| item.get("type").and_then(Value::as_str) == Some(COMPACTION))
    {
        let history = History::resolve(
            std::slice::from_ref(item),
            prepared.history.instructions.clone(),
            cipher,
            model,
            prepared.history.files.as_ref(),
        )?;
        prepared.compacted(item.clone(), history)?;
    }
    if let Some(summarizer) = summarizer.filter(|_| prepared.echo.summarizes()) {
        prepared.echo.set_summarizer(summarizer.clone());
    }
    prepared.echo.previous_response_id = response
        .get("previous_response_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    match prepared.echo.persist.as_mut() {
        Some(persist) if prepared.echo.background && persist.temporary.is_none() => {
            persist.request = Some(request.clone());
            persist.creator.clone_from(&stored.creator);
        }
        _ => {
            return Err(invalid(
                "the stored request is not a stored background request",
            ));
        }
    }
    let reply = Reply {
        protocol: super::super::response::Protocol::Responses(Arc::new(prepared.echo)),
        id,
        created: response
            .get("created_at")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        model: model.into(),
    };
    Ok((prepared.request, reply))
}

/// Where the Response to `request` is to be kept, with its resolved input
/// `items`, if anywhere, as `project`'s.
///
/// Storing follows `OpenAI`'s default of `true` once a store exists; without
/// one, `reject_unsupported` has already refused an explicit `true`. A
/// temporary background response goes only to the temporary store.
fn persist(
    request: &ResponsesRequest,
    (store, temporary): (Option<&Arc<ResponseStore>>, Option<&Arc<Temporary>>),
    items: Vec<Value>,
    project: Option<&str>,
) -> Option<Persist> {
    let project = project.map(str::to_owned);
    match temporary.filter(|_| request.temporary_background(store.is_some())) {
        Some(temporary) => Some(Persist {
            store: Arc::clone(temporary.store()),
            input_items: items,
            temporary: Some(Arc::clone(temporary)),
            request: None,
            creator: None,
            project,
        }),
        None => store
            .filter(|_| request.store.unwrap_or(true))
            .map(|store| Persist {
                store: Arc::clone(store),
                input_items: items,
                temporary: None,
                request: None,
                creator: None,
                project,
            }),
    }
}

/// Refuse to continue a temporary response. It is never looked up in the
/// response store, so it is not reported as merely missing there.
///
/// One `project` may not see is not mentioned: it is then looked up in the
/// response store, which does not have it either.
fn reject_temporary_previous(
    previous: Option<&str>,
    temporary: Option<&Arc<Temporary>>,
    project: Option<&str>,
) -> crate::Result<()> {
    match (previous, temporary) {
        (Some(id), Some(temporary))
            if temporary.knows(id) && visible(temporary.store(), id, project).unwrap_or(false) =>
        {
            Err(unsupported(
                "previous_response_id",
                &format!(
                    "response {id} was created with background=true and store=false, so it is \
                     kept only briefly for retrieval and cannot be continued; send its items in \
                     input"
                ),
            ))
        }
        _ => Ok(()),
    }
}

/// Whether a caller acting for `project` may see response `id` in `store`:
/// any stored one without a project, otherwise only one that `project`
/// owns (see [`StoredResponse::project`]). Reads the record only for a
/// project. Blocks on the store.
pub(in crate::cli::serve) fn visible(
    store: &ResponseStore,
    id: &str,
    project: Option<&str>,
) -> std::io::Result<bool> {
    let Some(project) = project else {
        return Ok(true);
    };
    Ok(store
        .load(id)?
        .is_some_and(|stored| stored.project.as_deref() == Some(project)))
}

/// The history stored response `id` continues with, when `project` may see
/// it (see [`visible`]).
pub(super) fn previous_history(
    store: &ResponseStore,
    id: &str,
    project: Option<&str>,
) -> crate::Result<Vec<Value>> {
    let stored = store
        .load(id)
        .map_err(|error| {
            crate::Error::Generation(format!("could not read stored response {id}: {error}"))
        })?
        .filter(|stored| project.is_none_or(|project| stored.project.as_deref() == Some(project)))
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
        COMPACTION => &["id", "type", "encrypted_content", "created_by"],
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
            "only function, custom, namespace, local_shell, apply_patch and (with server \
             permission) shell tools run here, and web_search is accepted but never offered; \
             other built-in and MCP tools need hosted services",
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
