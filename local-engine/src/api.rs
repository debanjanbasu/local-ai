//! Stable, synchronous library API.

use std::cell::Cell;
use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

use futures_core::Stream;
use tokio::sync::mpsc as async_mpsc;

use crate::bonsai_model::{BonsaiEngine, BonsaiInfo, CancelToken, PromptCacheSource, StopReason};
use crate::bonsai_native::KvOptions;
use crate::bonsai_ngram::NgramSettings;
use crate::bonsai_tokenizer::BonsaiTokenizer;
use crate::resources::{PREFILL_CHUNK, Resources, SERVE_QUEUE};
use crate::runtime::{EVENT_BUFFER, PrefillProgress};
use crate::tools::{ToolCall, ToolCallParser, ToolDefinition, ToolSet, Turn};
use crate::{GenerateParams, GenerationStats};

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
    /// Whether to ask the model for xhigh reasoning.
    pub thinking: bool,
    /// Optional prompt-cache affinity hint.
    pub session: Option<String>,
    /// Functions the model may call. Empty disables tool-call parsing, so
    /// `<tool_call>` text stays ordinary content. The model decides whether to
    /// call; nothing forces a call.
    pub tools: Vec<ToolDefinition>,
}

/// A raw-text generation request.
#[derive(Clone, Debug)]
pub struct CompletionRequest {
    /// Prompt passed directly to the tokenizer.
    pub prompt: String,
    /// Maximum number of generated tokens.
    pub max_tokens: usize,
    /// Sampling controls.
    pub sampling: Sampling,
    /// Optional prompt-cache affinity hint.
    pub session: Option<String>,
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
    /// Zero for requests with thinking disabled and for raw completions.
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
    /// A complete call whose name and arguments passed the tool's schema.
    ///
    /// Emitted only for requests with tools, once `</tool_call>` has been
    /// generated and parsed; never partial. The engine does not execute it. A
    /// malformed, invalid or truncated call ends the generation with an error
    /// instead ([`Event::Error`] on a stream, `Err` from the synchronous API)
    /// and no [`Event::Finished`].
    ToolCall(ToolCall),
    /// Generated token IDs, emitted as a batch before [`Event::Finished`].
    TokenIds(Vec<u32>),
    /// Successful terminal event.
    Finished(Box<Stats>),
    /// Terminal worker error.
    Error(String),
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
        let plan = Plan {
            model: resources.model.clone(),
            model_reason: resources.model_reason.clone(),
            speculation: inner.mtp_enabled(),
            speculation_reason: resources.mtp_reason.clone(),
            prompt_cache_dir: resources.prompt_cache_dir.clone(),
            prompt_cache_budget: resources.disk_budget_bytes,
        };
        let model = inner.info().clone();
        let json =
            serde_json::json!({"experimental_bonsai":{"engine":model,"policy":resources.policy()}});
        Ok(Self {
            inner,
            info: EngineInfo { plan, model, json },
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
        render_messages(messages, thinking, &ToolSet::new(tools)?)
    }

    /// Count the prompt tokens `request` would prefill, without generating.
    ///
    /// The count is exact: the request is validated, rendered (messages,
    /// replayed tool calls and results, tool definitions and `thinking`) and
    /// tokenized, special tokens included, through the same code as
    /// [`Engine::chat_with`]. It equals the `prompt_tokens` that generation
    /// reports in [`Stats::generation`], cached prefix included. An invalid
    /// request fails with the error generation would return. `max_tokens`,
    /// `sampling` and `session` are ignored, and the count is not checked
    /// against the context window. CPU-only; no GPU work is submitted.
    pub fn count_chat_tokens(&self, request: &ChatRequest) -> crate::Result<usize> {
        count_chat(self.inner.tokenizer(), request)
    }

    /// Count the prompt tokens a raw completion would prefill, without
    /// generating.
    ///
    /// The prompt is tokenized as [`Engine::complete`] tokenizes it: no
    /// template, special-token text parsed as special tokens. An empty prompt
    /// counts zero, although generation rejects it. `max_tokens`, `sampling`
    /// and `session` are ignored. CPU-only; no GPU work is submitted.
    pub fn count_completion_tokens(&self, request: &CompletionRequest) -> crate::Result<usize> {
        count_completion(self.inner.tokenizer(), request)
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
            session: None,
            tools: Vec::new(),
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
                Event::Finished(_) | Event::Error(_) => {}
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
        let (ids, tools) = chat_prompt_ids(self.inner.tokenizer(), request)?;
        self.generate(
            &ids,
            request.max_tokens,
            &request.sampling,
            request.session.as_deref(),
            request.thinking,
            tools,
            cancel,
            progress,
            callback,
        )
    }

    /// Generate a raw completion, stopping when the callback returns `Break`.
    pub fn complete(
        &mut self,
        request: &CompletionRequest,
        callback: impl FnMut(Event) -> ControlFlow<()>,
    ) -> crate::Result<Stats> {
        let cancel = CancelToken::default();
        let mut progress = ignore_progress;
        self.complete_with_cancel(request, &cancel, &mut progress, callback)
    }

    fn complete_with_cancel(
        &mut self,
        request: &CompletionRequest,
        cancel: &CancelToken,
        progress: &mut dyn FnMut(PrefillProgress),
        callback: impl FnMut(Event) -> ControlFlow<()>,
    ) -> crate::Result<Stats> {
        let ids = encode_prompt(self.inner.tokenizer(), &request.prompt)?;
        self.generate(
            &ids,
            request.max_tokens,
            &request.sampling,
            request.session.as_deref(),
            false,
            None,
            cancel,
            progress,
            callback,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate(
        &mut self,
        ids: &[u32],
        max_tokens: usize,
        sampling: &Sampling,
        session: Option<&str>,
        thinking: bool,
        tools: Option<Arc<ToolSet>>,
        cancel: &CancelToken,
        progress: &mut dyn FnMut(PrefillProgress),
        mut callback: impl FnMut(Event) -> ControlFlow<()>,
    ) -> crate::Result<Stats> {
        let mut params = sampling.0.clone();
        params.max_tokens = max_tokens;
        let mut splitter = EventSplitter::new(thinking, tools);
        let output = self.inner.generate_session_progress(
            ids,
            &params,
            session,
            |piece| splitter.emit(piece, &mut callback).is_continue(),
            cancel,
            progress,
        )?;
        let reasoning_tokens = crate::bonsai_tokenizer::answer_start(&output.token_ids, thinking)
            .unwrap_or(output.token_ids.len());
        let finished = if splitter.failed() {
            ControlFlow::Break(())
        } else {
            splitter.finish(&mut callback)
        };
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
            stop_reason: output.stop_reason,
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

    /// Move this engine to a dedicated worker thread.
    #[must_use]
    pub fn into_handle(self) -> EngineHandle {
        EngineHandle::new(self)
    }
}

enum Job {
    Chat(ChatRequest, async_mpsc::Sender<Delivered>, CancelToken),
    Completion(
        CompletionRequest,
        async_mpsc::Sender<Delivered>,
        CancelToken,
    ),
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
/// also a [`Stream`]; [`EngineHandle::chat_stream`] and
/// [`EngineHandle::complete_stream`] name that intent at call sites.
#[derive(Clone)]
pub struct EngineHandle {
    sender: async_mpsc::Sender<Job>,
    /// The worker engine's tokenizer, shared so counting never queues a job.
    tokenizer: BonsaiTokenizer,
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
        let (sender, receiver) = async_mpsc::channel::<Job>(SERVE_QUEUE);
        std::thread::spawn(move || Worker::new(engine, receiver).run());
        Self { sender, tokenizer }
    }

    /// Count the prompt tokens `request` would prefill, exactly as
    /// [`Engine::count_chat_tokens`] does.
    ///
    /// Runs on the calling thread against the worker's shared tokenizer: it
    /// queues no job, never waits for running generations and submits no GPU
    /// work, so it succeeds even when the queue is full. Tokenizing a long
    /// prompt is CPU-bound; async callers should run it on a blocking pool.
    pub fn count_chat_tokens(&self, request: &ChatRequest) -> crate::Result<usize> {
        count_chat(&self.tokenizer, request)
    }

    /// Count the prompt tokens a raw completion would prefill, exactly as
    /// [`Engine::count_completion_tokens`] does, without queueing a job.
    pub fn count_completion_tokens(&self, request: &CompletionRequest) -> crate::Result<usize> {
        count_completion(&self.tokenizer, request)
    }

    /// Queue a chat request and return a blocking event iterator.
    ///
    /// Dropping the iterator cancels queued or running work. Use
    /// [`EventStream::cancel_handle`] when cancellation must be explicit.
    ///
    /// ```no_run
    /// # use local_engine::{ChatMessage, ChatRequest, Engine, Event, Sampling};
    /// let handle = Engine::open()?.into_handle();
    /// let request = ChatRequest { messages: vec![ChatMessage { role: "user".into(), content: "Hello".into(), ..ChatMessage::default() }], max_tokens: 32, sampling: Sampling::default(), thinking: true, session: None, tools: Vec::new() };
    /// for event in handle.chat(request)? {
    ///     if let Event::Content(text) = event { print!("{text}"); }
    /// }
    /// # Ok::<(), local_engine::Error>(())
    /// ```
    pub fn chat(&self, request: ChatRequest) -> crate::Result<EventStream> {
        self.submit(|events, cancel| Job::Chat(request, events, cancel))
    }

    /// Queue a raw completion and return its event stream and cancellation handle.
    pub fn complete(&self, request: CompletionRequest) -> crate::Result<EventStream> {
        self.submit(|events, cancel| Job::Completion(request, events, cancel))
    }

    /// Queue a chat request for `Stream` consumption.
    ///
    /// Thin alias for [`EngineHandle::chat`]; nothing happens before the stream
    /// exists, so this needs no runtime.
    pub fn chat_stream(&self, request: ChatRequest) -> crate::Result<EventStream> {
        self.chat(request)
    }

    /// Queue a raw completion for `Stream` consumption.
    ///
    /// Thin alias for [`EngineHandle::complete`]; nothing happens before the
    /// stream exists, so this needs no runtime.
    pub fn complete_stream(&self, request: CompletionRequest) -> crate::Result<EventStream> {
        self.complete(request)
    }

    fn submit(
        &self,
        job: impl FnOnce(async_mpsc::Sender<Delivered>, CancelToken) -> Job,
    ) -> crate::Result<EventStream> {
        let (sender, receiver) = async_mpsc::channel::<Delivered>(EVENT_BUFFER);
        let cancel = CancelToken::new();
        self.sender
            .try_send(job(sender, cancel.clone()))
            .map_err(|error| match error {
                async_mpsc::error::TrySendError::Full(_) => crate::Error::QueueFull,
                async_mpsc::error::TrySendError::Closed(_) => {
                    crate::Error::Generation("engine worker stopped".into())
                }
            })?;
        Ok(EventStream {
            receiver,
            progress: VecDeque::new(),
            cancel,
            not_sync: std::marker::PhantomData,
        })
    }
}

/// Forward prefill boundaries onto a request's channel.
///
/// The report never blocks and never cancels. A blocking send would put a slow
/// consumer on the prefill critical path of every running request, and a full
/// channel can only mean that the consumer is behind on events: a dropped
/// boundary costs one heartbeat, because the next chunk reports again. The client
/// going away is detected by the event send failing or the stream being dropped,
/// both of which do cancel.
fn reporter(sender: &async_mpsc::Sender<Delivered>) -> impl FnMut(PrefillProgress) {
    move |progress| {
        let _ = sender.try_send(Delivered::Progress(progress));
    }
}

/// A request the worker has rendered and tokenized but not yet admitted.
struct Prepared {
    ids: Vec<u32>,
    params: GenerateParams,
    session: Option<String>,
    thinking: bool,
    tools: Option<Arc<ToolSet>>,
    events: async_mpsc::Sender<Delivered>,
    cancel: CancelToken,
}

/// One admitted request's delivery side.
///
/// Events go to an outbox first and reach the channel without blocking, so a
/// client that reads slowly does not hold up the requests decoding beside it.
/// A client that stops reading is dropped by the server's stall timer, which
/// closes the channel and cancels the generation.
struct Delivery {
    events: async_mpsc::Sender<Delivered>,
    cancel: CancelToken,
    splitter: EventSplitter,
    outbox: VecDeque<Delivered>,
}

impl Delivery {
    fn push(&mut self, event: Event) -> ControlFlow<()> {
        if self.cancel.is_cancelled() || self.events.is_closed() {
            return ControlFlow::Break(());
        }
        self.outbox.push_back(Delivered::Event(event));
        ControlFlow::Continue(())
    }

    fn emit(&mut self, piece: &str) -> bool {
        let mut splitter = std::mem::replace(&mut self.splitter, EventSplitter::new(false, None));
        let flow = splitter.emit(piece, &mut |event| self.push(event));
        self.splitter = splitter;
        flow.is_continue()
    }

    /// The events that close a generation, as [`Engine::generate`] sends them.
    fn complete(&mut self, result: crate::Result<crate::bonsai_model::BonsaiGeneration>) {
        let output = match result {
            Ok(output) => output,
            Err(error) => {
                self.outbox
                    .push_back(Delivered::Event(Event::Error(error.to_string())));
                return;
            }
        };
        let mut splitter = std::mem::replace(&mut self.splitter, EventSplitter::new(false, None));
        let finished = if splitter.failed() {
            ControlFlow::Break(())
        } else {
            splitter.finish(&mut |event| self.push(event))
        };
        if let Some(failure) = splitter.take_failure() {
            // A parse failure is terminal: report it, never `Finished`.
            self.outbox
                .push_back(Delivered::Event(Event::Error(failure)));
            return;
        }
        let reasoning_tokens =
            crate::bonsai_tokenizer::answer_start(&output.token_ids, splitter.thinking)
                .unwrap_or(output.token_ids.len());
        if finished.is_break() || self.push(Event::TokenIds(output.token_ids)).is_break() {
            return;
        }
        let stats = Stats {
            stop_reason: output.stop_reason,
            cache_source: output.cache_source,
            reasoning_tokens,
            generation: output.stats,
        };
        let _ = self.push(Event::Finished(Box::new(stats)));
    }

    /// Hand queued events to the channel; `wait` blocks for room. Returns
    /// whether anything is left to deliver.
    fn flush(&mut self, wait: bool) -> bool {
        while let Some(item) = self.outbox.pop_front() {
            let sent = if wait {
                self.events
                    .blocking_send(item)
                    .map_err(|_| async_mpsc::error::TrySendError::Closed(()))
            } else {
                self.events.try_send(item).map_err(|error| match error {
                    async_mpsc::error::TrySendError::Full(item) => {
                        self.outbox.push_front(item);
                        async_mpsc::error::TrySendError::Full(())
                    }
                    async_mpsc::error::TrySendError::Closed(_) => {
                        async_mpsc::error::TrySendError::Closed(())
                    }
                })
            };
            match sent {
                Ok(()) => {}
                Err(async_mpsc::error::TrySendError::Full(())) => return true,
                Err(async_mpsc::error::TrySendError::Closed(())) => {
                    self.cancel.cancel();
                    self.outbox.clear();
                    return false;
                }
            }
        }
        false
    }
}

/// The engine worker: admits queued requests while they fit and advances
/// every admitted one together (see [`BonsaiEngine::step`]).
struct Worker {
    engine: Engine,
    receiver: async_mpsc::Receiver<Job>,
    /// A request taken off the queue that did not fit yet.
    waiting: Option<Prepared>,
    running: std::collections::HashMap<u64, Delivery>,
    /// Finished requests whose last events have not reached their channel.
    draining: Vec<Delivery>,
}

impl Worker {
    fn new(engine: Engine, receiver: async_mpsc::Receiver<Job>) -> Self {
        Self {
            engine,
            receiver,
            waiting: None,
            running: std::collections::HashMap::new(),
            draining: Vec::new(),
        }
    }

    fn run(mut self) {
        loop {
            if self.running.is_empty() && self.waiting.is_none() {
                for mut delivery in std::mem::take(&mut self.draining) {
                    delivery.flush(true);
                    delivery.cancel.cancel();
                }
                let Some(job) = self.receiver.blocking_recv() else {
                    return;
                };
                self.waiting = self.prepare(job);
            }
            self.admit_waiting();
            if !self.running.is_empty() {
                // `step` calls one of the two at a time, never both at once.
                let running = std::cell::RefCell::new(&mut self.running);
                let finished = self.engine.inner.step(
                    &mut |id, piece| {
                        running
                            .borrow_mut()
                            .get_mut(&id)
                            .is_some_and(|delivery| delivery.emit(piece))
                    },
                    &mut |id, progress| {
                        if let Some(delivery) = running.borrow().get(&id) {
                            reporter(&delivery.events)(progress);
                        }
                    },
                );
                for (id, result) in finished {
                    if let Some(mut delivery) = self.running.remove(&id) {
                        delivery.complete(result);
                        self.draining.push(delivery);
                    }
                }
            }
            self.flush();
        }
    }

    /// Admit the waiting request and any queued behind it while they fit.
    fn admit_waiting(&mut self) {
        loop {
            let prepared = match self.waiting.take() {
                Some(prepared) => prepared,
                None => match self.receiver.try_recv() {
                    Ok(job) => match self.prepare(job) {
                        Some(prepared) => prepared,
                        None => continue,
                    },
                    Err(_) => return,
                },
            };
            if prepared.cancel.is_cancelled() || prepared.events.is_closed() {
                prepared.cancel.cancel();
                continue;
            }
            if !self
                .engine
                .inner
                .can_admit(prepared.ids.len(), prepared.params.max_tokens)
            {
                self.waiting = Some(prepared);
                return;
            }
            let mut delivery = Delivery {
                events: prepared.events,
                cancel: prepared.cancel,
                splitter: EventSplitter::new(prepared.thinking, prepared.tools),
                outbox: VecDeque::new(),
            };
            let cancel = delivery.cancel.clone();
            let admitted = self.engine.inner.admit(
                &prepared.ids,
                &prepared.params,
                prepared.session.as_deref(),
                cancel,
            );
            match admitted {
                Ok(id) => {
                    self.running.insert(id, delivery);
                }
                Err(error) => {
                    delivery.complete(Err(error));
                    self.draining.push(delivery);
                }
            }
        }
    }

    /// Render and tokenize a job, answering it at once if that fails.
    fn prepare(&self, job: Job) -> Option<Prepared> {
        let (prompt, max_tokens, sampling, session, thinking, events, cancel) = match job {
            Job::Chat(request, events, cancel) => (
                prepare_chat(&request),
                request.max_tokens,
                request.sampling,
                request.session,
                request.thinking,
                events,
                cancel,
            ),
            Job::Completion(request, events, cancel) => (
                Ok((request.prompt, None)),
                request.max_tokens,
                request.sampling,
                request.session,
                false,
                events,
                cancel,
            ),
        };
        let tokenizer = self.engine.inner.tokenizer();
        let ids =
            prompt.and_then(|(prompt, tools)| Ok((encode_prompt(tokenizer, &prompt)?, tools)));
        match ids {
            Ok((ids, tools)) => {
                let mut params = sampling.0;
                params.max_tokens = max_tokens;
                Some(Prepared {
                    ids,
                    params,
                    session,
                    thinking,
                    tools,
                    events,
                    cancel,
                })
            }
            Err(error) => {
                let _ = events.blocking_send(Delivered::Event(Event::Error(error.to_string())));
                cancel.cancel();
                None
            }
        }
    }

    fn flush(&mut self) {
        // A request alone keeps the old backpressure: its generation waits for
        // its reader. Beside others, nobody waits for a slow reader.
        let wait = self.running.len() <= 1 && self.waiting.is_none();
        for delivery in self.running.values_mut() {
            delivery.flush(wait);
        }
        self.draining.retain_mut(|delivery| {
            let left = delivery.flush(false);
            if !left {
                delivery.cancel.cancel();
            }
            left
        });
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

impl EventStream {
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
    /// remember to clear it; [`RecvTimeoutError`] is std's own name for the
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
    /// let mut events = handle.complete_stream(local_engine::CompletionRequest {
    ///     prompt: "Hello".into(),
    ///     max_tokens: 8,
    ///     sampling: local_engine::Sampling::default(),
    ///     session: None,
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
    let prompt = render_messages(&request.messages, request.thinking, &tools)?;
    Ok((prompt, (!tools.is_empty()).then(|| Arc::new(tools))))
}

/// Tokenize a rendered or raw prompt; every generation and count goes through
/// here, so a count is the prompt length generation prefills.
fn encode_prompt(tokenizer: &BonsaiTokenizer, prompt: &str) -> crate::Result<Vec<u32>> {
    tokenizer.encode(prompt)
}

/// Validate, render and tokenize a chat request.
fn chat_prompt_ids(
    tokenizer: &BonsaiTokenizer,
    request: &ChatRequest,
) -> crate::Result<(Vec<u32>, Option<Arc<ToolSet>>)> {
    let (prompt, tools) = prepare_chat(request)?;
    Ok((encode_prompt(tokenizer, &prompt)?, tools))
}

fn count_chat(tokenizer: &BonsaiTokenizer, request: &ChatRequest) -> crate::Result<usize> {
    chat_prompt_ids(tokenizer, request).map(|(ids, _)| ids.len())
}

fn count_completion(
    tokenizer: &BonsaiTokenizer,
    request: &CompletionRequest,
) -> crate::Result<usize> {
    encode_prompt(tokenizer, &request.prompt).map(|ids| ids.len())
}

fn render_messages(
    messages: &[ChatMessage],
    thinking: bool,
    tools: &ToolSet,
) -> crate::Result<String> {
    let turns = messages
        .iter()
        .map(|message| Turn {
            role: &message.role,
            content: &message.content,
            reasoning_content: message.reasoning_content.as_deref(),
            tool_calls: &message.tool_calls,
            tool_call_id: message.tool_call_id.as_deref(),
        })
        .collect::<Vec<_>>();
    crate::tools::render(&turns, thinking, tools)
}

/// Splits generated text into reasoning, content and (when the request has
/// tools) validated tool calls. Reasoning is split off first, so tool-call
/// markup is only recognised in the visible answer.
struct EventSplitter {
    /// Original template mode, retained after the reasoning delimiter for usage.
    thinking: bool,
    reasoning: bool,
    pending: String,
    /// The chat template puts `\n\n` after `</think>`; drop it from the answer.
    answer_started: bool,
    tools: Option<ToolCallParser>,
}

impl EventSplitter {
    fn new(reasoning: bool, tools: Option<Arc<ToolSet>>) -> Self {
        Self {
            thinking: reasoning,
            reasoning,
            pending: String::new(),
            answer_started: !reasoning,
            tools: tools.map(ToolCallParser::new),
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
        match &mut self.tools {
            Some(parser) => parser.feed(text, callback),
            None => callback(Event::Content(text.to_owned())),
        }
    }

    fn emit(
        &mut self,
        piece: &str,
        callback: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
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
        callback(Event::Reasoning(text))
    }

    fn finish(&mut self, callback: &mut impl FnMut(Event) -> ControlFlow<()>) -> ControlFlow<()> {
        if !self.pending.is_empty() {
            let text = std::mem::take(&mut self.pending);
            let event = if self.reasoning {
                Event::Reasoning(text)
            } else {
                Event::Content(text)
            };
            if callback(event).is_break() {
                return ControlFlow::Break(());
            }
        }
        self.tools
            .as_mut()
            .map_or(ControlFlow::Continue(()), |parser| parser.finish(callback))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
