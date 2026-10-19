//! Stable, synchronous library API.

use std::cell::Cell;
use std::collections::VecDeque;
use std::future::Future;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

use futures_core::Stream;
use tokio::sync::{mpsc as async_mpsc, oneshot};

use crate::bonsai_model::{BonsaiEngine, BonsaiInfo, CancelToken, PromptCacheSource, StopReason};
use crate::bonsai_native::KvOptions;
use crate::bonsai_ngram::NgramSettings;
use crate::bonsai_tokenizer::{BonsaiTokenizer, ReasoningEffort};
pub use crate::bonsai_vision::ImageDetail;
use crate::bonsai_vision::{PromptImages, VisionEncoder};
use crate::judgment::{Decision, DecisionCapture, DecisionRequest, JudgmentHead, PreparedDecision};
use crate::resources::{PREFILL_CHUNK, Resources, SERVE_QUEUE};
use crate::runtime::{EVENT_BUFFER, PrefillProgress};
use crate::sampler::TokenLogprob;
use crate::structured::{Grammar, ResponseFormat};
use crate::tools::{
    ToolCall, ToolCallDelta, ToolCallParser, ToolChoice, ToolDefinition, ToolSet, Turn,
};
use crate::{GenerateParams, GenerationStats};

mod worker;

#[cfg(test)]
use worker::Delivery;
use worker::{DecisionJob, Worker};

const THINK_END: &str = "</think>";

/// One message rendered through the checkpoint chat template.
#[derive(Clone, Debug, Default)]
pub struct ChatMessage {
    /// OpenAI-compatible role: `system` or `developer` (first message only;
    /// both use the template's single system slot), `user`, `assistant`, or
    /// `tool`.
    pub role: String,
    /// Visible message content, or the result text of a `tool` message.
    pub content: String,
    /// Optional prior assistant reasoning content.
    pub reasoning_content: Option<String>,
    /// Calls a prior `assistant` message made, replayed verbatim. Each must be
    /// answered by exactly one following `tool` message.
    pub tool_calls: Vec<ToolCall>,
    /// For a `tool` message, the [`ToolCall::id`] this result answers.
    pub tool_call_id: Option<String>,
    /// Images and videos in `content`, each at a byte offset of it, in
    /// order. The template renders each image as
    /// `<|vision_start|><|image_pad|><|vision_end|>` there (a video with
    /// `<|video_pad|>`); system and developer messages may not hold them.
    /// Needs the vision projector installed.
    pub images: Vec<ChatImage>,
}

/// An image or video part of a [`ChatMessage`].
#[derive(Clone, Debug)]
pub struct ChatImage {
    /// Byte offset in the message content (a character boundary) where the
    /// part sits; offsets never decrease along a message's parts.
    pub offset: usize,
    /// The encoded file: JPEG, PNG, WebP, GIF (first frame) or BMP; for a
    /// video, MP4 or MOV.
    pub data: Arc<[u8]>,
    /// An image's pixel budget (a video's is fixed).
    pub detail: ImageDetail,
    /// Whether `data` is a video.
    pub video: bool,
}

/// Sampling controls for one generation.
#[derive(Clone, Debug, Default)]
pub struct Sampling(pub GenerateParams);

/// A templated chat generation request.
#[derive(Clone, Debug)]
pub struct ChatRequest {
    /// Conversation in chronological order.
    pub messages: Vec<ChatMessage>,
    /// Maximum number of generated tokens.
    pub max_tokens: usize,
    /// Sampling controls.
    pub sampling: Sampling,
    /// Whether the model reasons before answering.
    pub thinking: bool,
    /// How hard it reasons when `thinking`; ignored otherwise.
    pub effort: ReasoningEffort,
    /// Whether assistant turns up to the last user query keep their
    /// reasoning when rendered (the template's `preserve_thinking`, on by
    /// default). `false` drops it, as `OpenAI`'s `reasoning.context:
    /// "current_turn"` does; turns after that query always keep theirs.
    pub preserve_thinking: bool,
    /// Optional prompt-cache affinity hint.
    pub session: Option<String>,
    /// Functions the model may call. Empty disables tool-call parsing, so
    /// `<tool_call>` text stays ordinary content. With the defaults
    /// ([`ToolChoice::Auto`], `parallel_tool_calls`, no
    /// [`ToolDefinition::strict`] tool, text answer) the model decides
    /// whether to call and nothing constrains generation.
    pub tools: Vec<ToolDefinition>,
    /// Whether and which tool must be called. Anything but
    /// [`ToolChoice::Auto`] is enforced by the native tool grammar; a
    /// required or named call needs `tools` (and a configured name).
    pub tool_choice: ToolChoice,
    /// Whether one answer may hold several calls. `false` enforces at most
    /// one call through the native tool grammar: nothing but whitespace may
    /// follow it, and free text can never contain `<tool_call>`.
    pub parallel_tool_calls: bool,
    /// The shape the final answer must take. Anything but
    /// [`ResponseFormat::Text`] is compiled before the request is queued
    /// and enforced on every generated answer token; reasoning (with
    /// `thinking`) stays unconstrained up to `</think>`. With `tools`, the
    /// answer is either a call or (unless a call is required) the final
    /// answer in this format, both enforced by one grammar.
    pub response_format: ResponseFormat,
    /// Sequences that end the visible answer, as `OpenAI`'s Chat `stop`
    /// does. Only answer text is matched: reasoning and tool-call markup and
    /// arguments never are, though the text between calls is. Generation
    /// stops at the first match, the content ends before it (the sequence is
    /// never emitted, and no text that might still begin one is emitted
    /// until it cannot), and [`Stats::stop_reason`] is
    /// [`StopReason::Stopped`]. A call already made stands; a match before a
    /// call means it is never made. Empty sequences never match.
    pub stop: Vec<String>,
}

/// Typed automatic startup decisions.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Selected model path.
    pub model: std::path::PathBuf,
    /// Why this model was selected.
    pub model_reason: String,
    /// Whether an MTP head was selected.
    pub speculation: bool,
    /// Why speculation was enabled or disabled.
    pub speculation_reason: String,
    /// Selected disk prompt-cache directory.
    pub prompt_cache_dir: Option<std::path::PathBuf>,
    /// Disk prompt-cache budget in bytes.
    pub prompt_cache_budget: u64,
    /// Whether image input is available (the vision projector loaded).
    pub vision: bool,
    /// Why image input is or is not available.
    pub vision_reason: String,
}

/// Loaded engine information and startup policy.
#[derive(Clone, Debug)]
pub struct EngineInfo {
    /// Resource-discovery decisions.
    pub plan: Plan,
    /// Model and memory decisions.
    pub model: BonsaiInfo,
    /// Existing machine-readable startup record.
    pub json: serde_json::Value,
}

/// Final generation measurements.
#[derive(Clone, Debug)]
pub struct Stats {
    /// Why generation stopped.
    pub stop_reason: StopReason,
    /// Prompt-cache tier used by the request.
    pub cache_source: PromptCacheSource,
    /// Generated tokens through the first `</think>` delimiter (inclusive),
    /// or all generated tokens if thinking ended before that delimiter.
    /// Zero for requests with thinking disabled.
    pub reasoning_tokens: usize,
    /// Token and speculation measurements.
    pub generation: GenerationStats,
}

/// Collected response from the convenient synchronous chat API.
#[derive(Clone, Debug)]
pub struct ChatOutput {
    /// Final answer, excluding model reasoning.
    pub content: String,
    /// Model reasoning emitted before the closing thinking delimiter.
    pub reasoning: String,
    /// Validated tool calls, identical to the streamed [`Event::ToolCall`]s.
    pub tool_calls: Vec<ToolCall>,
    /// Generated token IDs.
    pub token_ids: Vec<u32>,
    /// Generation measurements.
    pub stats: Stats,
}

/// A generation stream item.
#[derive(Clone, Debug)]
pub enum Event {
    /// Visible assistant text.
    Content(String),
    /// Hidden reasoning text when the template exposes it separately.
    Reasoning(String),
    /// Part of a call's arguments while the call is generated (see
    /// [`ToolCallDelta`]). Emitted only for requests with tools; a consumer
    /// that only wants complete calls can ignore it.
    ToolCallDelta(ToolCallDelta),
    /// A complete call whose name and arguments passed the tool's schema.
    ///
    /// Emitted only for requests with tools, once `</tool_call>` has been
    /// generated and parsed, with the ID of the call's deltas. The engine does not execute it. A
    /// malformed, invalid or truncated call ends the generation with an error
    /// instead ([`Event::Error`] on a stream, `Err` from the synchronous API)
    /// and no [`Event::Finished`].
    ToolCall(ToolCall),
    /// The result of a call the caller executed itself, fed back by a
    /// caller-driven tool loop through an [`EventRelay`]. The engine never
    /// produces one.
    ToolOutput(ToolOutput),
    /// The raw log-probabilities of the tokens whose text the next
    /// [`Event::Content`] holds, sent just before it, for a request whose
    /// [`GenerateParams::logprobs`] asks for them. Only answer tokens are
    /// reported: reasoning, `</think>`, tool-call markup and arguments, the
    /// whitespace the answer drops around them and text from a stop sequence
    /// on never are. A token whose text two content events share goes with
    /// the first.
    Logprobs(Vec<TokenLogprob>),
    /// Generated token IDs, emitted as a batch before [`Event::Finished`].
    TokenIds(Vec<u32>),
    /// Successful terminal event.
    Finished(Box<Stats>),
    /// Terminal worker error.
    Error(String),
}

/// A tool result produced outside the model, by the caller, for the call
/// with [`ToolCall::id`] `call_id`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolOutput {
    /// The call this answers.
    pub call_id: String,
    /// The called tool's name.
    pub name: String,
    /// What the model reads back.
    pub content: String,
    /// The caller's own structured record of the result.
    pub detail: serde_json::Value,
}

/// A next-item wait that a prefill boundary can interrupt instead of an
/// [`Event`].
///
/// Prefill emits no events at all, so a consumer watching only for generated text
/// cannot tell a working engine from an abandoned socket until the first token,
/// and a prompt longer than one `PREFILL_CHUNK` is silent for whole chunks of it.
/// A boundary carries that liveness on a second variant rather than a variant of
/// [`Event`], so `Event` keeps the shape every consumer already matches on.
#[derive(Clone, Debug)]
pub enum Signal {
    /// A generation event, byte-for-byte what [`EventStream::next`] returns.
    Event(Event),
    /// A prefill chunk is about to be submitted.
    Progress(PrefillProgress),
}

/// What the engine worker puts on one request's channel.
///
/// Progress rides the same channel as the events rather than a second one, so a
/// single wait can return whichever arrives first. That is what lets a parked
/// consumer be woken by a boundary instead of polling a second channel for it.
#[derive(Clone, Debug)]
enum Delivered {
    Event(Event),
    Progress(PrefillProgress),
}

/// Synchronous Metal inference engine.
///
/// It is `Send` but deliberately not `Sync`; use [`EngineHandle`] to share it.
pub struct Engine {
    inner: BonsaiEngine,
    info: EngineInfo,
    vision: Option<Arc<VisionEncoder>>,
    not_sync: std::marker::PhantomData<Cell<()>>,
}

// SAFETY: Metal command queues and resource objects may be transferred between
// threads. `Engine` has exclusive ownership of every object and is deliberately
// `!Sync`, so no command encoder or mutable model state is used concurrently.
#[allow(unsafe_code)]
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for Engine {}

impl Engine {
    /// Discover all resources and load the installed pinned model.
    pub fn open() -> crate::Result<Self> {
        Self::open_inner(None)
    }

    /// Load an explicit model while auto-discovering every other resource.
    pub fn open_model(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::open_inner(Some(path.as_ref()))
    }

    fn open_inner(path: Option<&Path>) -> crate::Result<Self> {
        Self::from_resources(&Resources::discover(path, true)?)
    }

    /// Load the model and policy an earlier [`Resources::discover`] chose, for a
    /// caller that also needs the other discovered resources (TLS, paths) and
    /// should not pay for discovery twice.
    pub fn from_resources(resources: &Resources) -> crate::Result<Self> {
        let mut inner = BonsaiEngine::open_with_options(
            &resources.model,
            None,
            PREFILL_CHUNK,
            None,
            &resources.mtp,
            NgramSettings::default(),
            KvOptions::default(),
        )?;
        inner.configure_prompt_cache(
            resources.prompt_cache_dir.clone(),
            resources.disk_budget_bytes,
            resources.prompt_cache_write_bytes_per_second,
        )?;
        let vision = resources
            .vision
            .as_deref()
            .map(|path| VisionEncoder::open(path, inner.input_signs()?).map(Arc::new))
            .transpose()?;
        let plan = Plan {
            model: resources.model.clone(),
            model_reason: resources.model_reason.clone(),
            speculation: inner.mtp_enabled(),
            speculation_reason: resources.mtp_reason.clone(),
            prompt_cache_dir: resources.prompt_cache_dir.clone(),
            prompt_cache_budget: resources.disk_budget_bytes,
            vision: vision.is_some(),
            vision_reason: resources.vision_reason.clone(),
        };
        let model = inner.info().clone();
        let json =
            serde_json::json!({"experimental_bonsai":{"engine":model,"policy":resources.policy()}});
        Ok(Self {
            inner,
            info: EngineInfo { plan, model, json },
            vision,
            not_sync: std::marker::PhantomData,
        })
    }

    /// Return model and resource-discovery decisions.
    #[must_use]
    pub const fn info(&self) -> &EngineInfo {
        &self.info
    }

    /// Tokenize raw text.
    pub fn tokenize(&self, text: &str) -> crate::Result<Vec<u32>> {
        self.inner.encode_prompt(text, true, false)
    }

    /// Decode token IDs, omitting checkpoint special tokens.
    pub fn detokenize(&self, ids: &[u32]) -> crate::Result<String> {
        self.inner.decode_tokens(ids)
    }

    /// Render messages through the checkpoint's chat template.
    pub fn render_chat(messages: &[ChatMessage], thinking: bool) -> crate::Result<String> {
        Self::render_chat_with_tools(messages, thinking, &[])
    }

    /// Render messages and tool definitions through the checkpoint's chat
    /// template, validating the tools and any replayed calls and results.
    pub fn render_chat_with_tools(
        messages: &[ChatMessage],
        thinking: bool,
        tools: &[ToolDefinition],
    ) -> crate::Result<String> {
        render_messages(
            messages,
            thinking.then_some(ReasoningEffort::XHigh),
            true,
            &ToolSet::new(tools)?,
        )
    }

    /// Count the prompt tokens `request` would prefill, without generating.
    ///
    /// The count is exact: the request is validated, rendered (messages,
    /// replayed tool calls and results, tool definitions and `thinking`) and
    /// tokenized, special tokens included, through the same code as
    /// [`Engine::chat_with`]. It equals the `prompt_tokens` that generation
    /// reports in [`Stats::generation`], cached prefix included. Invalid
    /// prompt content fails with the error generation would return. `max_tokens`,
    /// `sampling`, `session` and `response_format` are ignored (no schema is
    /// compiled), and the count is not checked against the context window.
    /// CPU-only; no GPU work is submitted.
    pub fn count_chat_tokens(&self, request: &ChatRequest) -> crate::Result<usize> {
        count_chat(self.inner.tokenizer(), self.vision.as_deref(), request)
    }

    /// Chat with one user prompt and collect the final answer and statistics.
    ///
    /// ```no_run
    /// let mut engine = local_engine::Engine::open()?;
    /// let answer = engine.chat("What is 17 * 23?")?;
    /// println!("{}", answer.content);
    /// # Ok::<(), local_engine::Error>(())
    /// ```
    pub fn chat(&mut self, prompt: impl Into<String>) -> crate::Result<ChatOutput> {
        let request = ChatRequest {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: prompt.into(),
                ..ChatMessage::default()
            }],
            max_tokens: crate::DEFAULT_MAX_OUTPUT_TOKENS,
            sampling: Sampling::default(),
            thinking: true,
            effort: ReasoningEffort::default(),
            preserve_thinking: true,
            session: None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: true,
            response_format: ResponseFormat::Text,
            stop: Vec::new(),
        };
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = Vec::new();
        let mut token_ids = Vec::new();
        let stats = self.chat_with(&request, |event| {
            match event {
                Event::Content(piece) => content.push_str(&piece),
                Event::Reasoning(piece) => reasoning.push_str(&piece),
                Event::ToolCall(call) => tool_calls.push(call),
                Event::TokenIds(ids) => token_ids = ids,
                Event::ToolCallDelta(_)
                | Event::ToolOutput(_)
                | Event::Logprobs(_)
                | Event::Finished(_)
                | Event::Error(_) => {}
            }
            ControlFlow::Continue(())
        })?;
        Ok(ChatOutput {
            content,
            reasoning,
            tool_calls,
            token_ids,
            stats,
        })
    }

    /// Generate a chat response, stopping when the callback returns `Break`.
    pub fn chat_with(
        &mut self,
        request: &ChatRequest,
        callback: impl FnMut(Event) -> ControlFlow<()>,
    ) -> crate::Result<Stats> {
        let cancel = CancelToken::default();
        let mut progress = ignore_progress;
        self.chat_with_cancel(request, &cancel, &mut progress, callback)
    }

    fn chat_with_cancel(
        &mut self,
        request: &ChatRequest,
        cancel: &CancelToken,
        progress: &mut dyn FnMut(PrefillProgress),
        callback: impl FnMut(Event) -> ControlFlow<()>,
    ) -> crate::Result<Stats> {
        let grammar = chat_grammar(self.inner.tokenizer(), request)?;
        let prompt = chat_prompt(
            self.inner.tokenizer(),
            self.vision.as_deref(),
            None,
            request,
        )?;
        self.generate(
            &prompt.ids,
            prompt.images,
            request.max_tokens,
            &request.sampling,
            request.session.as_deref(),
            request.thinking,
            prompt.tools,
            &request.stop,
            grammar,
            cancel,
            progress,
            callback,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate(
        &mut self,
        ids: &[u32],
        images: Option<Arc<PromptImages>>,
        max_tokens: usize,
        sampling: &Sampling,
        session: Option<&str>,
        thinking: bool,
        tools: Option<Arc<ToolSet>>,
        stop: &[String],
        grammar: Option<Grammar>,
        cancel: &CancelToken,
        progress: &mut dyn FnMut(PrefillProgress),
        mut callback: impl FnMut(Event) -> ControlFlow<()>,
    ) -> crate::Result<Stats> {
        let mut params = sampling.0.clone();
        params.max_tokens = max_tokens;
        let mut splitter = EventSplitter::new(thinking, tools, stop.to_vec());
        let output = self.inner.generate_constrained(
            ids,
            &params,
            session,
            |piece, logprobs| splitter.emit(piece, logprobs, &mut callback).is_continue(),
            cancel,
            progress,
            grammar,
            images,
        )?;
        let reasoning_tokens = crate::bonsai_tokenizer::answer_start(&output.token_ids, thinking)
            .unwrap_or(output.token_ids.len());
        let (finished, stop_reason) = splitter.close(output.stop_reason, &mut callback);
        if let Some(failure) = splitter.take_failure() {
            return Err(crate::Error::Generation(failure));
        }
        if finished.is_break() {
            return Ok(Stats {
                stop_reason: StopReason::Cancelled,
                cache_source: output.cache_source,
                reasoning_tokens,
                generation: output.stats,
            });
        }
        if callback(Event::TokenIds(output.token_ids.clone())).is_break() {
            return Ok(Stats {
                stop_reason: StopReason::Cancelled,
                cache_source: output.cache_source,
                reasoning_tokens,
                generation: output.stats,
            });
        }
        let stats = Stats {
            stop_reason,
            cache_source: output.cache_source,
            reasoning_tokens,
            generation: output.stats,
        };
        let _ = callback(Event::Finished(Box::new(stats.clone())));
        Ok(stats)
    }

    /// Generate and collect only visible content.
    pub fn generate_to_string(&mut self, request: &ChatRequest) -> crate::Result<String> {
        let mut text = String::new();
        self.chat_with(request, |event| {
            if let Event::Content(piece) = event {
                text.push_str(&piece);
            }
            ControlFlow::Continue(())
        })?;
        Ok(text)
    }

    /// EXPERIMENTAL: answer a typed text question with an explicitly
    /// supplied judgment `head`; see [`crate::judgment`].
    ///
    /// The request is rendered exactly as the head's training rows were,
    /// tokenized, and its option and decision rows are captured on this
    /// engine's loaded model (no second model) in
    /// [`DECISION_CAPTURE_ROWS`](crate::judgment::DECISION_CAPTURE_ROWS)-row
    /// blocks, FP16-rounded and scored. The result carries the head's
    /// probabilities per caller option, the argmax and, for scores, the
    /// probability-weighted level index — no confidence, threshold or
    /// refusal ([`CalibratedHead`](crate::judgment::CalibratedHead) adds
    /// those).
    ///
    /// Invalid requests (including text spelling a special token) and a head
    /// whose width is not the model's, or whose declared renderer is not
    /// [`DECISION_RENDERER`](crate::judgment::DECISION_RENDERER), fail with
    /// [`crate::Error::InvalidArgument`] before any GPU work. A single-level
    /// score is answered without GPU work. Capture overwrites one sequence
    /// buffer set; the GPU prompt-cache tier is preserved when a free set
    /// exists and cleared otherwise, so a later request may prefill more.
    pub fn decide(
        &mut self,
        head: &JudgmentHead,
        request: &DecisionRequest,
    ) -> crate::Result<Decision> {
        let prepared = PreparedDecision::new(self.inner.tokenizer(), head, request)?;
        prepared
            .run(&mut self.inner, head, &CancelToken::new())?
            .ok_or_else(|| crate::Error::Generation("decision was cancelled".into()))
    }

    /// EXPERIMENTAL: the head-independent features of a decision, for
    /// building evaluation and calibration data on the serving path.
    ///
    /// Renders, tokenizes and captures `request` exactly as
    /// [`decide`](Self::decide) does before scoring (same renderer, endpoint
    /// resolution, [`DECISION_CAPTURE_ROWS`](crate::judgment::DECISION_CAPTURE_ROWS)
    /// blocks and FP16 rounding), so scoring
    /// [`DecisionCapture::features`] with a head reproduces that head's
    /// decision. Requests with fewer than two options are refused. Capture
    /// overwrites one sequence buffer set, as [`decide`](Self::decide) does.
    pub fn capture_decision(
        &mut self,
        request: &DecisionRequest,
    ) -> crate::Result<DecisionCapture> {
        DecisionCapture::prepare(self.inner.tokenizer(), request)?.run(&mut self.inner)
    }

    /// Move this engine to a dedicated worker thread.
    #[must_use]
    pub fn into_handle(self) -> EngineHandle {
        EngineHandle::new(self)
    }
}

/// A queued chat request, its constraint already compiled.
struct ChatJob {
    request: ChatRequest,
    grammar: Option<Grammar>,
    /// The tokenized prompt of a request with images, encoded on the
    /// submitting thread; the worker tokenizes every other request itself.
    prompt: Option<ChatPrompt>,
    events: async_mpsc::Sender<Delivered>,
    cancel: CancelToken,
}

/// A queued request, its response format already compiled.
enum Job {
    Chat(Box<ChatJob>),
    /// An experimental decision, validated and tokenized.
    Decision(Box<DecisionJob>),
}

/// A progress reporter that records nothing.
///
/// The synchronous public API has no channel to report a boundary on, and the
/// prefill loop cannot skip the report: it shares the cancel poll, which is the
/// only place a cancelled prefill is noticed.
const fn ignore_progress(_: PrefillProgress) {}

/// Cloneable, thread-safe interface to a dedicated engine worker.
///
/// The job queue is bounded. Request methods are plain `&self` calls that work
/// without an async runtime and hand back a blocking [`EventStream`], which is
/// also a [`Stream`]; [`EngineHandle::chat_stream`] names that intent at call
/// sites.
#[derive(Clone)]
pub struct EngineHandle {
    sender: async_mpsc::Sender<Job>,
    /// The worker engine's tokenizer, shared so counting never queues a job.
    tokenizer: BonsaiTokenizer,
    /// Decode activity the worker publishes for low-priority GPU users.
    activity: Arc<crate::activity::EngineActivity>,
    /// Closes the worker engine's prompt-cache disk writer at shutdown.
    prompt_cache: crate::prompt_cache::Closer,
    /// The vision tower, run on the submitting thread.
    vision: Option<Arc<VisionEncoder>>,
}

impl EngineHandle {
    /// Load a model off the runtime and move it to a dedicated worker thread.
    ///
    /// [`Engine::open_model`] blocks for the whole mmap plus GPU upload, so it
    /// runs on the blocking pool and keeps a runtime worker free.
    pub async fn open(path: impl AsRef<Path>) -> crate::Result<Self> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let engine = tokio::task::spawn_blocking(move || Engine::open_model(path))
            .await
            .map_err(|error| crate::Error::Generation(error.to_string()))?;
        Ok(engine?.into_handle())
    }

    fn new(engine: Engine) -> Self {
        let tokenizer = engine.inner.tokenizer().clone();
        let prompt_cache = engine.inner.prompt_cache_closer();
        let vision = engine.vision.clone();
        let (sender, receiver) = async_mpsc::channel::<Job>(SERVE_QUEUE);
        let activity = Arc::new(crate::activity::EngineActivity::default());
        let published = Arc::clone(&activity);
        std::thread::spawn(move || Worker::new(engine, receiver, published).run());
        Self {
            sender,
            tokenizer,
            activity,
            prompt_cache,
            vision,
        }
    }

    /// Finish the prompt-cache disk writes the engine has accepted, within
    /// `deadline`, for a process about to exit.
    ///
    /// The worker thread owns the engine and is never joined, so exiting
    /// would otherwise kill a snapshot write partway and leave its temporary
    /// behind. From this call on no new snapshot is accepted; accepted ones
    /// get `deadline` to commit, and past it the write in flight is aborted
    /// and its temporary removed. Returns whether every accepted write
    /// finished in time. Blocks: call it off the async runtime.
    #[must_use]
    pub fn close_prompt_cache(&self, deadline: std::time::Duration) -> bool {
        self.prompt_cache.close(deadline) == crate::prompt_cache::Closed::Flushed
    }

    /// Decode activity of this handle's worker, for scheduling low-priority
    /// GPU work (embeddings) around running generations.
    #[must_use]
    pub fn activity(&self) -> Arc<crate::activity::EngineActivity> {
        Arc::clone(&self.activity)
    }

    /// Count the prompt tokens `request` would prefill, exactly as
    /// [`Engine::count_chat_tokens`] does.
    ///
    /// Runs on the calling thread against the worker's shared tokenizer: it
    /// queues no job, never waits for running generations and submits no GPU
    /// work, so it succeeds even when the queue is full. Tokenizing a long
    /// prompt is CPU-bound; async callers should run it on a blocking pool.
    pub fn count_chat_tokens(&self, request: &ChatRequest) -> crate::Result<usize> {
        count_chat(&self.tokenizer, self.vision.as_deref(), request)
    }

    /// Queue a chat request and return a blocking event iterator.
    ///
    /// A [`ChatRequest::response_format`] other than text, and any native
    /// tool constraint, is compiled here, on the calling thread, so an
    /// invalid or unsupported schema fails this call with
    /// [`crate::Error::InvalidArgument`] and nothing is queued. The
    /// first constrained request also builds the tokenizer's grammar tables
    /// once; async callers should submit constrained requests from a
    /// blocking pool.
    ///
    /// Dropping the iterator cancels queued or running work. Use
    /// [`EventStream::cancel_handle`] when cancellation must be explicit.
    ///
    /// ```no_run
    /// # use local_engine::{ChatMessage, ChatRequest, Engine, Event, Sampling};
    /// let handle = Engine::open()?.into_handle();
    /// let request = ChatRequest { messages: vec![ChatMessage { role: "user".into(), content: "Hello".into(), ..ChatMessage::default() }], max_tokens: 32, sampling: Sampling::default(), thinking: true, effort: local_engine::ReasoningEffort::default(), preserve_thinking: true, session: None, tools: Vec::new(), tool_choice: local_engine::ToolChoice::Auto, parallel_tool_calls: true, response_format: local_engine::ResponseFormat::Text, stop: Vec::new() };
    /// for event in handle.chat(request)? {
    ///     if let Event::Content(text) = event { print!("{text}"); }
    /// }
    /// # Ok::<(), local_engine::Error>(())
    /// ```
    ///
    /// A request with images is rendered, tokenized and run through the
    /// vision tower here as well (each image's embedding is cached for
    /// reuse), yielding to running decode steps between GPU submissions;
    /// an undecodable image or a missing projector fails this call.
    pub fn chat(&self, request: ChatRequest) -> crate::Result<EventStream> {
        let grammar = chat_grammar(&self.tokenizer, &request)?;
        let prompt = has_images(&request)
            .then(|| {
                chat_prompt(
                    &self.tokenizer,
                    self.vision.as_deref(),
                    Some(&self.activity),
                    &request,
                )
            })
            .transpose()?;
        self.submit(|events, cancel| {
            Job::Chat(Box::new(ChatJob {
                request,
                grammar,
                prompt,
                events,
                cancel,
            }))
        })
    }

    /// Queue a chat request for `Stream` consumption.
    ///
    /// Thin alias for [`EngineHandle::chat`]; nothing happens before the stream
    /// exists, so this needs no runtime.
    pub fn chat_stream(&self, request: ChatRequest) -> crate::Result<EventStream> {
        self.chat(request)
    }

    /// EXPERIMENTAL: queue a typed text question for an explicitly supplied
    /// judgment `head`, as [`Engine::decide`] answers it.
    ///
    /// The request is validated, rendered and tokenized here, on the calling
    /// thread against the shared tokenizer, so an invalid request or a head
    /// of the wrong width fails this call and nothing is queued; async
    /// callers should call it from a blocking pool for long state text. A
    /// single-level score resolves at once without queueing.
    ///
    /// The decision shares the bounded job queue (a full queue is
    /// [`crate::Error::QueueFull`]) and its FIFO order: once it reaches the
    /// worker, nothing queued behind it is admitted until it has run, and it
    /// runs only once every running generation has finished, so neither side
    /// starves. Dropping the returned [`PendingDecision`] or cancelling its
    /// [`CancelHandle`] skips a queued decision, or stops a running capture
    /// before its next block, resolving to `Ok(None)`.
    pub fn decide(
        &self,
        head: Arc<JudgmentHead>,
        request: &DecisionRequest,
    ) -> crate::Result<PendingDecision> {
        let prepared = PreparedDecision::new(&self.tokenizer, &head, request)?;
        let (reply, receiver) = oneshot::channel();
        let cancel = CancelToken::new();
        let pending = PendingDecision {
            receiver,
            cancel: cancel.clone(),
        };
        if !prepared.needs_capture() {
            let _ = reply.send(prepared.ready().map(Some));
            return Ok(pending);
        }
        self.send(Job::Decision(Box::new(DecisionJob {
            prepared,
            head,
            reply,
            cancel,
        })))?;
        Ok(pending)
    }

    fn send(&self, job: Job) -> crate::Result<()> {
        self.sender.try_send(job).map_err(|error| match error {
            async_mpsc::error::TrySendError::Full(_) => crate::Error::QueueFull,
            async_mpsc::error::TrySendError::Closed(_) => {
                crate::Error::Generation("engine worker stopped".into())
            }
        })
    }

    fn submit(
        &self,
        job: impl FnOnce(async_mpsc::Sender<Delivered>, CancelToken) -> Job,
    ) -> crate::Result<EventStream> {
        let (sender, receiver) = async_mpsc::channel::<Delivered>(EVENT_BUFFER);
        let cancel = CancelToken::new();
        self.send(job(sender, cancel.clone()))?;
        Ok(EventStream {
            receiver,
            progress: VecDeque::new(),
            cancel,
            not_sync: std::marker::PhantomData,
        })
    }
}

/// An experimental decision queued on an [`EngineHandle`]; dropping it
/// cancels the decision.
///
/// Resolves to `Ok(Some(decision))`, `Ok(None)` when cancelled before it
/// finished, or the decision's error. Await it as a [`Future`], or block
/// with [`Self::wait`].
pub struct PendingDecision {
    receiver: oneshot::Receiver<crate::Result<Option<Decision>>>,
    cancel: CancelToken,
}

impl PendingDecision {
    /// Obtain an independently owned cancellation control.
    #[must_use]
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle {
            cancel: self.cancel.clone(),
        }
    }

    /// Block the calling thread until the decision resolves. Safe inside a
    /// runtime task (it parks rather than panicking), though it blocks the
    /// runtime worker.
    pub fn wait(mut self) -> crate::Result<Option<Decision>> {
        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        let mut context = Context::from_waker(&waker);
        loop {
            if let Poll::Ready(result) = Pin::new(&mut self).poll(&mut context) {
                return result;
            }
            std::thread::park();
        }
    }
}

impl Future for PendingDecision {
    type Output = crate::Result<Option<Decision>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.get_mut().receiver)
            .poll(cx)
            .map(|result| {
                result.unwrap_or_else(|_| {
                    Err(crate::Error::Generation("engine worker stopped".into()))
                })
            })
    }
}

impl Drop for PendingDecision {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Explicit cancellation control for a queued or running request.
#[derive(Clone)]
pub struct CancelHandle {
    cancel: CancelToken,
}

impl CancelHandle {
    /// Request cooperative cancellation.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

/// Generation event stream; dropping it cancels the request.
///
/// Implements both [`Iterator`] (blocking) and [`Stream`] (async) over one
/// shared receiver, so a request can be consumed either way. Mixing the two in
/// a single session interleaves events; use one per request.
///
/// [`Iterator::next`] blocks the calling thread and panics inside a runtime
/// task; [`Stream::poll_next`] is the non-blocking path. Dropping either form
/// mid-flight cancels the running generation.
///
/// Like [`Engine`] it is `Send` but deliberately not `Sync`: the underlying
/// `tokio` receiver is `Sync`, so the marker keeps the original surface rather
/// than silently widening it.
pub struct EventStream {
    receiver: async_mpsc::Receiver<Delivered>,
    /// Boundaries this stream has taken and not yet handed to a caller that
    /// asked for them, in arrival order.
    ///
    /// A caller that only wants [`Event`]s drops them here instead of the
    /// channel, so they cost one entry per prefill chunk of a request that can
    /// never exceed the context. A prompt prefilling beside decoding requests
    /// reports every 48 tokens, so the smallest useful context of 32,768
    /// tokens is at most 683 entries, about 11 KiB, and it lasts only as long
    /// as the stream.
    progress: VecDeque<PrefillProgress>,
    cancel: CancelToken,
    not_sync: std::marker::PhantomData<Cell<()>>,
}

/// The producing end of an [`EventStream`] fed by the caller rather than
/// the engine worker, from [`EventStream::relay`].
///
/// A tool loop uses it to present several engine generations, and the tool
/// results between them, as one stream: the consumer's cancellation (an
/// explicit [`CancelHandle::cancel`] or dropping the stream) is observable
/// here through [`Self::cancelled`], so the loop can stop its tools and the
/// generation it is forwarding.
pub struct EventRelay {
    sender: async_mpsc::Sender<Delivered>,
    cancel: CancelToken,
}

impl EventRelay {
    /// Deliver `signal`, parking until the consumer has room. `false` when
    /// the consumer has gone. Call from a thread that may block, never from
    /// inside an async task.
    #[must_use]
    pub fn blocking_send(&self, signal: Signal) -> bool {
        let delivered = match signal {
            Signal::Event(event) => Delivered::Event(event),
            Signal::Progress(progress) => Delivered::Progress(progress),
        };
        self.sender.blocking_send(delivered).is_ok()
    }

    /// [`Self::blocking_send`] for async callers.
    #[must_use]
    pub async fn send(&self, signal: Signal) -> bool {
        let delivered = match signal {
            Signal::Event(event) => Delivered::Event(event),
            Signal::Progress(progress) => Delivered::Progress(progress),
        };
        self.sender.send(delivered).await.is_ok()
    }

    /// Whether the consumer cancelled or dropped the stream.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled() || self.sender.is_closed()
    }

    /// Resolve once the consumer cancels or drops the stream (dropping it
    /// cancels).
    pub async fn cancelled(&self) {
        self.cancel.cancelled().await;
    }
}

impl EventStream {
    /// A stream whose items come from the returned [`EventRelay`] instead of
    /// the engine. It ends when the relay is dropped.
    #[must_use]
    pub fn relay() -> (EventRelay, Self) {
        let (sender, receiver) = async_mpsc::channel::<Delivered>(EVENT_BUFFER);
        let cancel = CancelToken::new();
        (
            EventRelay {
                sender,
                cancel: cancel.clone(),
            },
            Self {
                receiver,
                progress: VecDeque::new(),
                cancel,
                not_sync: std::marker::PhantomData,
            },
        )
    }

    /// Obtain an independently owned cancellation control.
    #[must_use]
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle {
            cancel: self.cancel.clone(),
        }
    }

    /// Wait for the next event, giving up after `timeout`.
    ///
    /// `Ok(Some(event))` is an event, `Ok(None)` is end-of-stream, and
    /// `Err(RecvTimeoutError::Timeout)` is a deadline that passed while the
    /// generation was still open. `Err(RecvTimeoutError::Disconnected)` is
    /// never produced: a closed channel is reported as `Ok(None)`.
    ///
    /// The three states cannot be folded into one, because end-of-stream and a
    /// silent-but-running generation both arrive as `None` from
    /// [`Iterator::next`], and a consumer that must give up on a wedged reader
    /// needs to tell them apart. A caller tracking that with a side flag has to
    /// remember to clear it; [`RecvTimeoutError`](std::sync::mpsc::RecvTimeoutError) is std's own name for the
    /// same distinction, so the whole answer stays in the return type.
    ///
    /// Unlike [`Iterator::next`] this is safe to call from inside a runtime
    /// task: it parks the calling thread instead of panicking, and reports
    /// `Timeout` once the deadline passes. It holds `&mut self`, so it cannot
    /// race a concurrent [`Stream`](futures_core::Stream) poll on the same
    /// stream.
    ///
    /// A prefill boundary does not end this wait and does not consume the
    /// deadline: it is kept for [`Self::next_signal`] instead, and it also
    /// releases the wait early. The deadline therefore still measures silence
    /// from generated text, which is the only clock a caller of this method can
    /// interpret. Use [`Self::next_signal`] to see prefill at all.
    pub fn next_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<Event>, std::sync::mpsc::RecvTimeoutError> {
        // Written on top of the one parking loop rather than beside it. Two
        // loops that both park on this channel and both have to cope with a
        // boundary are two chances to get the wakeup wrong, and there is only one
        // that has to be right.
        //
        // A boundary is kept for `next_signal` rather than returned, so the
        // deadline keeps measuring silence from generated text alone. It is also
        // why this cannot read the stash back out: doing so would return the
        // boundary it just put there, forever.
        let deadline = Instant::now().checked_add(timeout);
        loop {
            match self.wait(deadline)? {
                Some(Signal::Event(event)) => return Ok(Some(event)),
                Some(Signal::Progress(progress)) => self.progress.push_back(progress),
                None => return Ok(None),
            }
        }
    }

    /// Wait for the next event *or* prefill boundary, giving up after `timeout`.
    ///
    /// Same three outcomes as [`Self::next_timeout`], with [`Signal`] in place of
    /// [`Event`]. Prefill is silent per token, so a caller that has to notice a
    /// client that stopped reading during it needs the boundary to end the wait;
    /// a caller that does not should use [`Self::next_timeout`], which hides
    /// boundaries, keeps the deadline measuring generated text alone, and is
    /// written on top of this.
    ///
    /// Boundaries this stream took for [`Self::next_timeout`] come back first, in
    /// the order they arrived, so a request consumed both ways still sees all of
    /// them and sees them once.
    ///
    /// `timeout` of `None` parks until something arrives. That is the honest
    /// budget for the first item of a request: `PREFILL_CHUNK` is 128 tokens and
    /// this M2 prefills at 3.6-4.2 tok/s, so one chunk is about 35 s of work and a
    /// prompt over 128 tokens is ordinary. It is unbounded here for the same
    /// reason it is unbounded in [`Self::next_timeout`]: a deadline under one
    /// chunk would cancel long prompts that were never stalled.
    ///
    /// ```no_run
    /// # use local_engine::{Engine, Signal};
    /// # use std::time::Duration;
    /// let handle = Engine::open()?.into_handle();
    /// let mut events = handle.chat_stream(local_engine::ChatRequest {
    ///     messages: vec![local_engine::ChatMessage {
    ///         role: "user".into(),
    ///         content: "Hello".into(),
    ///         ..local_engine::ChatMessage::default()
    ///     }],
    ///     max_tokens: 8,
    ///     sampling: local_engine::Sampling::default(),
    ///     thinking: false,
    ///     effort: local_engine::ReasoningEffort::default(),
    ///     preserve_thinking: true,
    ///     session: None,
    ///     tools: Vec::new(),
    ///     tool_choice: local_engine::ToolChoice::Auto,
    ///     parallel_tool_calls: true,
    ///     response_format: local_engine::ResponseFormat::Text,
    ///     stop: Vec::new(),
    /// })?;
    /// // Prefill has started, and the model has produced no text yet.
    /// assert!(matches!(
    ///     events.next_signal(None)?,
    ///     Some(Signal::Progress(_)) | Some(Signal::Event(_))
    /// ));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn next_signal(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<Option<Signal>, std::sync::mpsc::RecvTimeoutError> {
        if let Some(progress) = self.progress.pop_front() {
            return Ok(Some(Signal::Progress(progress)));
        }
        self.wait(timeout.and_then(|timeout| Instant::now().checked_add(timeout)))
    }

    /// Park until an item arrives or `deadline` passes, `None` meaning forever.
    ///
    /// The single place this stream waits. Both waits above are this plus a
    /// policy: which items count, and what to do with a boundary.
    fn wait(
        &mut self,
        deadline: Option<Instant>,
    ) -> Result<Option<Signal>, std::sync::mpsc::RecvTimeoutError> {
        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        let mut context = Context::from_waker(&waker);
        loop {
            match self.receiver.poll_recv(&mut context) {
                Poll::Ready(Some(Delivered::Event(event))) => {
                    return Ok(Some(Signal::Event(event)));
                }
                Poll::Ready(Some(Delivered::Progress(progress))) => {
                    return Ok(Some(Signal::Progress(progress)));
                }
                Poll::Ready(None) => return Ok(None),
                Poll::Pending => {}
            }
            let Some(deadline) = deadline else {
                std::thread::park();
                continue;
            };
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(std::sync::mpsc::RecvTimeoutError::Timeout);
            }
            std::thread::park_timeout(left);
        }
    }
}

/// Unparks the thread that is waiting in [`EventStream::next_timeout`].
///
/// The receiver is woken from whatever thread produces the next event, so the
/// wait ends at that moment instead of at the next poll. Sleeping for a fixed
/// slice and retrying instead would put that slice in front of every slow
/// generation step.
struct ThreadWaker(std::thread::Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

impl Iterator for EventStream {
    type Item = Event;
    fn next(&mut self) -> Option<Self::Item> {
        // A boundary is never an item of this iterator: it is kept for
        // `next_signal`, so a caller of the iterator sees exactly the events it
        // saw before a boundary existed.
        loop {
            match self.receiver.blocking_recv()? {
                Delivered::Event(event) => return Some(event),
                Delivered::Progress(progress) => self.progress.push_back(progress),
            }
        }
    }
}

impl Stream for EventStream {
    type Item = Event;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let stream = self.get_mut();
        loop {
            match stream.receiver.poll_recv(cx) {
                Poll::Ready(Some(Delivered::Event(event))) => return Poll::Ready(Some(event)),
                Poll::Ready(Some(Delivered::Progress(progress))) => {
                    stream.progress.push_back(progress);
                }
                Poll::Ready(None) => return Poll::Ready(None),
                // Looping re-polls with the same waker, so a boundary does not
                // leave the stream parked with nothing registered to wake it.
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl Drop for EventStream {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Render a request and validate its tools; the one path shared by the
/// synchronous [`Engine`] and the worker.
fn prepare_chat(request: &ChatRequest) -> crate::Result<(String, Option<Arc<ToolSet>>)> {
    let tools = ToolSet::new(&request.tools)?;
    let prompt = render_messages(
        &request.messages,
        request.thinking.then_some(request.effort),
        request.preserve_thinking,
        &tools,
    )?;
    Ok((prompt, (!tools.is_empty()).then(|| Arc::new(tools))))
}

/// Tokenize a rendered prompt; every generation and count goes through
/// here, so a count is the prompt length generation prefills.
fn encode_prompt(tokenizer: &BonsaiTokenizer, prompt: &str) -> crate::Result<Vec<u32>> {
    tokenizer.encode(prompt)
}

/// A validated, rendered and tokenized chat request.
struct ChatPrompt {
    ids: Vec<u32>,
    images: Option<Arc<PromptImages>>,
    tools: Option<Arc<ToolSet>>,
}

fn has_images(request: &ChatRequest) -> bool {
    request
        .messages
        .iter()
        .any(|message| !message.images.is_empty())
}

fn vision_required(vision: Option<&VisionEncoder>) -> crate::Result<&VisionEncoder> {
    vision.ok_or_else(|| {
        crate::Error::InvalidArgument(format!(
            "image and video input need the Bonsai vision projector {}, which is not installed beside \
             the model",
            crate::bonsai_vision::VISION_PROJECTOR_FILE
        ))
    })
}

/// One image or video of a request, decoded and resized.
enum Prepared {
    Image(crate::bonsai_vision::PreparedImage),
    Video(Arc<crate::bonsai_vision::PreparedVideo>),
}

/// Decode and resize every image and video of `request`, in prompt order.
fn prepare_images(request: &ChatRequest) -> crate::Result<Vec<Prepared>> {
    request
        .messages
        .iter()
        .flat_map(|message| &message.images)
        .map(|part| {
            if part.video {
                crate::bonsai_vision::prepare_video(&part.data).map(Prepared::Video)
            } else {
                crate::bonsai_vision::prepare_image(&part.data, part.detail).map(Prepared::Image)
            }
        })
        .collect()
}

/// `prepared` laid out for the prompt, each grid's rows from `rows`.
fn visuals<'a>(
    tokenizer: &BonsaiTokenizer,
    prepared: &'a [Prepared],
    mut rows: impl FnMut(&crate::bonsai_vision::PreparedImage) -> crate::Result<Arc<[f32]>>,
) -> crate::Result<Vec<crate::bonsai_vision::Visual<'a>>> {
    use crate::bonsai_vision::{Visual, video::timestamp_text};
    prepared
        .iter()
        .map(|part| match part {
            Prepared::Image(image) => Ok(Visual::Image(image, rows(image)?)),
            Prepared::Video(video) => video
                .groups
                .iter()
                .zip(&video.timestamps)
                .map(|(group, &seconds)| {
                    Ok((
                        tokenizer.encode(&timestamp_text(seconds))?,
                        group,
                        rows(group)?,
                    ))
                })
                .collect::<crate::Result<Vec<_>>>()
                .map(Visual::Video),
        })
        .collect()
}

/// Validate, render and tokenize a chat request, encoding its images (with
/// `activity`, yielding to its decode steps between GPU submissions).
fn chat_prompt(
    tokenizer: &BonsaiTokenizer,
    vision: Option<&VisionEncoder>,
    activity: Option<&crate::activity::EngineActivity>,
    request: &ChatRequest,
) -> crate::Result<ChatPrompt> {
    let (prompt, tools) = prepare_chat(request)?;
    let ids = encode_prompt(tokenizer, &prompt)?;
    if !has_images(request) {
        return Ok(ChatPrompt {
            ids,
            images: None,
            tools,
        });
    }
    let vision = vision_required(vision)?;
    let prepared = prepare_images(request)?;
    let encoded = visuals(tokenizer, &prepared, |image| vision.encode(image, activity))?;
    let (ids, images) = crate::bonsai_vision::expand_prompt(&ids, vision.digest(), &encoded)?;
    Ok(ChatPrompt {
        ids,
        images: Some(Arc::new(images)),
        tools,
    })
}

/// Compile a chat request's constraint: from `</think>` on with thinking,
/// else from the first token.
///
/// A request whose tool settings need it (see
/// [`crate::tools::grammar::needs_grammar`]) gets the native tool grammar,
/// which also carries its response format; otherwise only a response format
/// is compiled, as before tools could be constrained.
fn chat_grammar(
    tokenizer: &BonsaiTokenizer,
    request: &ChatRequest,
) -> crate::Result<Option<Grammar>> {
    crate::tools::check_choice(&request.tools, &request.tool_choice)?;
    if request.tools.is_empty() {
        return tokenizer.compile_format(&request.response_format, request.thinking);
    }
    let tools = ToolSet::new(&request.tools)?;
    if !crate::tools::grammar::needs_grammar(
        &tools,
        &request.tool_choice,
        request.parallel_tool_calls,
        request.response_format.is_text(),
    ) {
        return Ok(None);
    }
    tokenizer
        .compile_tools(
            &tools,
            &request.tool_choice,
            request.parallel_tool_calls,
            &request.response_format,
            request.thinking,
        )
        .map(Some)
}

fn count_chat(
    tokenizer: &BonsaiTokenizer,
    vision: Option<&VisionEncoder>,
    request: &ChatRequest,
) -> crate::Result<usize> {
    let (prompt, _) = prepare_chat(request)?;
    let ids = encode_prompt(tokenizer, &prompt)?;
    if !has_images(request) {
        return Ok(ids.len());
    }
    vision_required(vision)?;
    let prepared = prepare_images(request)?;
    let empty: Arc<[f32]> = Arc::new([]);
    let layout = visuals(tokenizer, &prepared, |_| Ok(Arc::clone(&empty)))?;
    crate::bonsai_vision::expanded_len(&ids, &layout)
}

fn render_messages(
    messages: &[ChatMessage],
    thinking: Option<ReasoningEffort>,
    preserve_thinking: bool,
    tools: &ToolSet,
) -> crate::Result<String> {
    let contents = messages
        .iter()
        .map(content_with_images)
        .collect::<crate::Result<Vec<_>>>()?;
    let turns = messages
        .iter()
        .zip(&contents)
        .map(|(message, content)| Turn {
            role: &message.role,
            content,
            reasoning_content: message.reasoning_content.as_deref(),
            tool_calls: &message.tool_calls,
            tool_call_id: message.tool_call_id.as_deref(),
        })
        .collect::<Vec<_>>();
    crate::tools::render(&turns, thinking, preserve_thinking, tools)
}

/// A message's content with the template's placeholder at each image, as
/// the template's `render_content` writes list content.
fn content_with_images(message: &ChatMessage) -> crate::Result<std::borrow::Cow<'_, str>> {
    if message.images.is_empty() {
        return Ok(std::borrow::Cow::Borrowed(&message.content));
    }
    if matches!(message.role.as_str(), "system" | "developer") {
        return Err(crate::Error::InvalidArgument(format!(
            "{} messages cannot contain images or videos",
            message.role
        )));
    }
    let content = &message.content;
    let mut rendered = String::with_capacity(
        content.len() + message.images.len() * crate::bonsai_vision::IMAGE_PLACEHOLDER.len(),
    );
    let mut done = 0;
    for image in &message.images {
        let text = content.get(done..image.offset).ok_or_else(|| {
            crate::Error::InvalidArgument(
                "image offsets must be increasing character boundaries of the content".into(),
            )
        })?;
        rendered.push_str(text);
        rendered.push_str(if image.video {
            crate::bonsai_vision::VIDEO_PLACEHOLDER
        } else {
            crate::bonsai_vision::IMAGE_PLACEHOLDER
        });
        done = image.offset;
    }
    rendered.push_str(content.get(done..).unwrap_or_default());
    Ok(std::borrow::Cow::Owned(rendered))
}

/// Splits generated text into reasoning, content and (when the request has
/// tools) validated tool calls. Reasoning is split off first, so tool-call
/// markup is only recognised in the visible answer, and stop sequences are
/// matched last, in the content that remains.
///
/// Every content event keeps where its text starts in the generated text,
/// so the log-probabilities of the tokens that produced it can go with it
/// (see [`Event::Logprobs`]).
struct EventSplitter {
    /// Original template mode, retained after the reasoning delimiter for usage.
    thinking: bool,
    reasoning: bool,
    pending: String,
    /// The chat template puts `\n\n` after `</think>`; drop it from the answer.
    answer_started: bool,
    tools: Option<ToolCallParser>,
    stop: StopSequences,
    /// Bytes of generated text so far.
    generated: usize,
    /// Where the tool parser's input starts in the generated text.
    parsed_from: Option<usize>,
    /// Log-probabilities not yet sent, oldest first.
    logprobs: VecDeque<Spanned>,
}

/// Log-probabilities of the tokens that produced generated text
/// `start..end`.
struct Spanned {
    start: usize,
    end: usize,
    tokens: Vec<TokenLogprob>,
}

impl EventSplitter {
    fn new(reasoning: bool, tools: Option<Arc<ToolSet>>, stop: Vec<String>) -> Self {
        Self {
            thinking: reasoning,
            reasoning,
            pending: String::new(),
            answer_started: !reasoning,
            tools: tools.map(ToolCallParser::new),
            stop: StopSequences::new(stop),
            generated: 0,
            parsed_from: None,
            logprobs: VecDeque::new(),
        }
    }

    /// Whether a tool-call parse failure has stopped this stream.
    const fn failed(&self) -> bool {
        matches!(&self.tools, Some(parser) if parser.failed())
    }

    /// The tool-call parse failure that stopped this stream, if any.
    fn take_failure(&mut self) -> Option<String> {
        self.tools.as_mut().and_then(ToolCallParser::take_failure)
    }

    /// Split `piece`, which ends the generated text so far.
    fn content(
        &mut self,
        piece: &str,
        callback: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        let text = if self.answer_started {
            piece
        } else {
            piece.trim_start_matches('\n')
        };
        if text.is_empty() {
            return ControlFlow::Continue(());
        }
        self.answer_started = true;
        let offset = self.generated - text.len();
        let base = *self.parsed_from.get_or_insert(offset);
        let stop = &mut self.stop;
        let logprobs = &mut self.logprobs;
        let mut deliver = |event, at| deliver(logprobs, event, at, callback);
        match &mut self.tools {
            Some(parser) => parser.feed(text, &mut |event, at| {
                stop.event(event, base + at, &mut deliver)
            }),
            None => stop.content(text, offset, &mut deliver),
        }
    }

    /// Split the next generated `piece`, which `logprobs` (empty unless the
    /// request asked for them) are the tokens of.
    fn emit(
        &mut self,
        piece: &str,
        logprobs: Vec<TokenLogprob>,
        callback: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        let start = self.generated;
        self.generated += piece.len();
        if !logprobs.is_empty() {
            self.logprobs.push_back(Spanned {
                start,
                end: self.generated,
                tokens: logprobs,
            });
        }
        if !self.reasoning {
            return self.content(piece, callback);
        }
        self.pending.push_str(piece);
        if let Some(boundary) = self.pending.find(THINK_END) {
            let mut content = self.pending.split_off(boundary);
            content.drain(..THINK_END.len());
            let reasoning = std::mem::take(&mut self.pending);
            self.reasoning = false;
            if !reasoning.is_empty() && callback(Event::Reasoning(reasoning)).is_break() {
                return ControlFlow::Break(());
            }
            return self.content(&content, callback);
        }
        let mut safe = self.pending.len().saturating_sub(THINK_END.len() - 1);
        while !self.pending.is_char_boundary(safe) {
            safe = safe.saturating_sub(1);
        }
        if safe != 0 {
            return self.flush_prefix(safe, callback);
        }
        ControlFlow::Continue(())
    }

    fn flush_prefix(
        &mut self,
        length: usize,
        callback: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        let remainder = self.pending.split_off(length);
        let text = std::mem::replace(&mut self.pending, remainder);
        // Tokens wholly in sent reasoning are never reported.
        let sent = self.generated - self.pending.len();
        while self.logprobs.front().is_some_and(|span| span.end <= sent) {
            self.logprobs.pop_front();
        }
        callback(Event::Reasoning(text))
    }

    /// Flush held text for generation that stopped for `reason`; see
    /// [`ToolCallParser::finish`] for what an unfinished call becomes.
    fn finish(
        &mut self,
        reason: StopReason,
        callback: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        if !self.pending.is_empty() {
            let text = std::mem::take(&mut self.pending);
            let offset = self.generated - text.len();
            let event = if self.reasoning {
                Event::Reasoning(text)
            } else {
                Event::Content(text)
            };
            if deliver(&mut self.logprobs, event, offset, callback).is_break() {
                return ControlFlow::Break(());
            }
        }
        let base = self.parsed_from.unwrap_or(self.generated);
        let stop = &mut self.stop;
        let logprobs = &mut self.logprobs;
        let mut deliver = |event, at| deliver(logprobs, event, at, callback);
        let flow = self
            .tools
            .as_mut()
            .map_or(ControlFlow::Continue(()), |parser| {
                parser.finish(reason, &mut |event, at| {
                    stop.event(event, base + at, &mut deliver)
                })
            });
        if flow.is_break() {
            return flow;
        }
        stop.flush(&mut deliver)
    }

    /// Close the stream of a generation the engine stopped for `reason`:
    /// whether the consumer still listens, and the reason the engine API
    /// reports. A stop sequence that matched, before or in the held text
    /// flushed now, is [`StopReason::Stopped`], and what followed it is
    /// dropped; any other refusal is the consumer stopping, [`StopReason::Cancelled`].
    fn close(
        &mut self,
        reason: StopReason,
        callback: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> (ControlFlow<()>, StopReason) {
        if self.failed() {
            return (ControlFlow::Break(()), reason);
        }
        if !self.stop.matched {
            let reason = if reason == StopReason::Stopped {
                StopReason::Cancelled
            } else {
                reason
            };
            let flow = self.finish(reason, callback);
            if !self.stop.matched {
                return (flow, reason);
            }
        }
        (ControlFlow::Continue(()), StopReason::Stopped)
    }
}

/// Send `event`, which starts at `offset` of the generated text; content
/// goes after the log-probabilities of the tokens whose text it holds.
/// Tokens wholly before it were dropped from the answer and are discarded.
fn deliver(
    logprobs: &mut VecDeque<Spanned>,
    event: Event,
    offset: usize,
    callback: &mut impl FnMut(Event) -> ControlFlow<()>,
) -> ControlFlow<()> {
    if let Event::Content(text) = &event {
        let end = offset + text.len();
        let mut tokens = Vec::new();
        while let Some(span) = logprobs.pop_front_if(|span| span.start < end) {
            if span.end > offset {
                tokens.extend(span.tokens);
            }
        }
        if !tokens.is_empty() && callback(Event::Logprobs(tokens)).is_break() {
            return ControlFlow::Break(());
        }
    }
    callback(event)
}

/// Ends the visible answer at the first of [`ChatRequest::stop`].
///
/// Content is held back while its end could still begin a sequence, so no
/// part of a sequence is ever emitted; any other event releases what is held,
/// since a call or the end of generation ends the text a sequence could span.
/// Each event goes on with where its text starts in the generated text.
struct StopSequences {
    sequences: Vec<String>,
    held: String,
    /// Where `held` starts in the generated text. Held content is one
    /// contiguous run: any other event flushes it.
    held_from: usize,
    /// A sequence matched: the answer is over and nothing more is emitted.
    matched: bool,
}

impl StopSequences {
    fn new(mut sequences: Vec<String>) -> Self {
        sequences.retain(|sequence| !sequence.is_empty());
        Self {
            sequences,
            held: String::new(),
            held_from: 0,
            matched: false,
        }
    }

    fn event(
        &mut self,
        event: Event,
        offset: usize,
        callback: &mut impl FnMut(Event, usize) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        match event {
            Event::Content(text) => self.content(&text, offset, callback),
            _ if self.matched => ControlFlow::Break(()),
            other => {
                if self.flush(callback).is_break() {
                    return ControlFlow::Break(());
                }
                callback(other, offset)
            }
        }
    }

    /// Emit `text`, which starts at `offset`, holding back a possible start
    /// of a sequence; at a match, emit only what precedes it and break.
    fn content(
        &mut self,
        text: &str,
        offset: usize,
        callback: &mut impl FnMut(Event, usize) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        if self.matched {
            return ControlFlow::Break(());
        }
        if self.sequences.is_empty() {
            return callback(Event::Content(text.to_owned()), offset);
        }
        if self.held.is_empty() {
            self.held_from = offset;
        }
        self.held.push_str(text);
        if let Some(at) = self
            .sequences
            .iter()
            .filter_map(|sequence| self.held.find(sequence.as_str()))
            .min()
        {
            self.held.truncate(at);
            self.matched = true;
            // The consumer's answer to the text before the match does not
            // matter: generation is over either way.
            let _ = self.flush(callback);
            return ControlFlow::Break(());
        }
        // The longest end of `held` that begins a sequence. It starts where
        // a sequence's first character does, so on a character boundary.
        let partial = self
            .sequences
            .iter()
            .filter_map(|sequence| {
                (1..sequence.len().min(self.held.len() + 1))
                    .rev()
                    .find(|&length| {
                        sequence
                            .get(..length)
                            .is_some_and(|prefix| self.held.ends_with(prefix))
                    })
            })
            .max()
            .unwrap_or(0);
        let ready = self.held.len() - partial;
        if ready == 0 {
            return ControlFlow::Continue(());
        }
        let rest = self.held.split_off(ready);
        let text = std::mem::replace(&mut self.held, rest);
        let from = self.held_from;
        self.held_from += text.len();
        callback(Event::Content(text), from)
    }

    /// Emit the held text, which no sequence can complete any more.
    fn flush(
        &mut self,
        callback: &mut impl FnMut(Event, usize) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        if self.held.is_empty() {
            return ControlFlow::Continue(());
        }
        let text = std::mem::take(&mut self.held);
        let from = self.held_from;
        self.held_from += text.len();
        callback(Event::Content(text), from)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
