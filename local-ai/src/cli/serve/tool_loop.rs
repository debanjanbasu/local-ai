//! The server-side loop that runs hosted tools for one Response.
//!
//! The model chooses; Rust executes, outside model execution. Each round
//! forwards one engine generation unchanged. When it ends its turn with
//! calls to hosted tools, those calls are executed here, their results are
//! reported as [`Event::ToolOutput`] and appended to the conversation, and a
//! new generation is queued; the consumer sees one stream throughout (see
//! [`EventStream::relay`]).
//!
//! Batch rules:
//! - Every hosted call of a turn runs, in order, independently. A refusal,
//!   malformed arguments, a timeout or a failure becomes that call's output
//!   (marked as an error, `incomplete`) and is fed back to the model; it
//!   never fails the Response or skips the other calls.
//! - A turn that also calls a client-executed tool runs its hosted calls and
//!   then ends the Response, so the client can answer its own calls; the
//!   hosted outputs are already part of the output it continues from.
//! - A turn that ended otherwise than at end of turn (token limit,
//!   cancellation) runs nothing and keeps its own status.
//!
//! Bounds: at most [`Bounds::max_iterations`] rounds run tools, at most
//! `max_tool_calls` hosted calls run, their combined wall time is limited by
//! [`Bounds::budget`] and each call by the per-call deadline. A call past a
//! bound is not run (its output says why) and the next round is generated
//! with `tool_choice: none`, so the model must answer. The output token
//! budget (`max_output_tokens`) covers all rounds together, and reported
//! usage is their sum.
//!
//! `file_search` calls run here too, on the vector stores (see
//! [`super::file_search`]); they need no permission, and their results are
//! numbered across the Response so citations name one result each.
//!
//! Cancellation — the client going away, `POST .../cancel`, shutdown —
//! cancels the generation being forwarded and drops the running tool, which
//! kills a shell command's process group.

use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use local_engine::bonsai_model::StopReason;
use local_engine::{
    CancelHandle, ChatMessage, ChatRequest, Event, EventRelay, EventStream, PrefillProgress,
    Signal, Stats, ToolCall, ToolChoice, ToolOutput,
};
use local_services::tools::{Output, ToolError, Toolbox, denial};

use super::file_search;
use super::hosted::{self, Hosted, Kinds};
use super::vector_stores::VectorStores;

/// How often a running tool reports liveness to the consumer, so a
/// non-streaming body (whose stall budget is at least 10 s) keeps writing
/// while no generation is producing events. A timer, not a poll: nothing is
/// checked on it.
const HEARTBEAT: Duration = Duration::from_secs(5);

/// The server's limits on one Response's tool loop.
#[derive(Debug, Clone, Copy)]
pub(super) struct Bounds {
    /// Rounds that may execute hosted calls.
    pub(super) max_iterations: usize,
    /// Combined wall time of every hosted call of one Response.
    pub(super) budget: Duration,
    /// Deadline of one hosted call.
    pub(super) call_timeout: Duration,
}

/// The hosted tools a server was started with: the toolbox's grants (none
/// by default) and the loop's bounds, which `file_search` also runs within.
pub(super) struct Tools {
    pub(super) toolbox: Toolbox,
    pub(super) bounds: Bounds,
}

/// Refuse hosted tools in `kinds` that `tools` does not permit, and a
/// `file_search` this server cannot run (`vector_stores` is `None` when it
/// has none).
pub(super) fn check(
    tools: &Tools,
    kinds: &Kinds,
    vector_stores: Option<&VectorStores>,
) -> Result<(), String> {
    for hosted in kinds.all_hosted() {
        match hosted.capability() {
            Some(capability) if !tools.toolbox.allows(capability) => {
                return Err(format!(
                    "a declared tool is not permitted: {}",
                    denial(capability)
                ));
            }
            Some(_) => {}
            None => {
                let Some(vector_stores) = vector_stores else {
                    return Err(
                        "file_search is unavailable: this server has no vector stores \
                                (no services store, or another local-ai process owns this data \
                                directory's index)"
                            .into(),
                    );
                };
                if kinds
                    .file_search()
                    .is_some_and(file_search::Config::needs_embeddings)
                    && !vector_stores.embeds()
                {
                    return Err(
                        "file_search ranking_options.hybrid_search needs the embedding \
                                model, which is not installed on this server"
                            .into(),
                    );
                }
            }
        }
    }
    Ok(())
}

/// Queues one generation: the engine's `chat`, or a fake in tests.
pub(super) type Submit = Box<dyn FnMut(ChatRequest) -> crate::Result<EventStream> + Send>;

/// What one Response's loop runs with.
pub(super) struct Loop {
    pub(super) submit: Submit,
    pub(super) request: ChatRequest,
    pub(super) toolbox: Toolbox,
    /// What `file_search` calls search; `None` refuses them.
    pub(super) vector_stores: Option<Arc<VectorStores>>,
    pub(super) kinds: Kinds,
    pub(super) bounds: Bounds,
    /// The request's `max_tool_calls`.
    pub(super) max_calls: Option<u64>,
    /// Runs once, before the first hosted call starts executing; an `Err`
    /// ends the Response with that error instead, running nothing. A
    /// resumable background Response records there that a call may have
    /// run (see [`super::store::ResponseStore::mark_tools_started`]).
    pub(super) before_tools: Option<BeforeTools>,
}

/// See [`Loop::before_tools`].
pub(super) type BeforeTools = Box<dyn FnOnce() -> Result<(), String> + Send>;

/// Run the loop over `first`, the request's first generation, returning the
/// one stream its consumer reads.
pub(super) fn start(tool_loop: Loop, first: EventStream) -> EventStream {
    let (relay, stream) = EventStream::relay();
    let relay = Arc::new(relay);
    let current = Arc::new(Mutex::new(first.cancel_handle()));
    let runtime = tokio::runtime::Handle::current();
    // The consumer's cancellation reaches whichever generation is running.
    // Aborted when the driver ends, so it never keeps the stream open.
    let canceller = {
        let (relay, current) = (Arc::clone(&relay), Arc::clone(&current));
        Canceller(runtime.spawn(async move {
            relay.cancelled().await;
            current
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .cancel();
        }))
    };
    let driver = Driver {
        tool_loop,
        relay,
        current,
        runtime,
        _canceller: canceller,
    };
    tokio::task::spawn_blocking(move || driver.run(first));
    stream
}

struct Driver {
    tool_loop: Loop,
    relay: Arc<EventRelay>,
    current: Arc<Mutex<CancelHandle>>,
    runtime: tokio::runtime::Handle,
    _canceller: Canceller,
}

/// The task forwarding cancellation, aborted (dropping its share of the
/// relay) when the driver ends.
struct Canceller(tokio::task::JoinHandle<()>);

impl Drop for Canceller {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// The assistant turn a round generated.
#[derive(Default)]
struct Turn {
    reasoning: String,
    content: String,
    calls: Vec<ToolCall>,
}

/// What the loop has spent so far.
#[derive(Default)]
struct Spent {
    rounds: usize,
    calls: u64,
    tool_time: Duration,
    usage: Option<Stats>,
}

impl Spent {
    /// Fold one round's measurements into the reported usage.
    fn add(&mut self, stats: &Stats) {
        match &mut self.usage {
            None => self.usage = Some(stats.clone()),
            Some(total) => {
                let (sum, round) = (&mut total.generation, &stats.generation);
                sum.prompt_tokens += round.prompt_tokens;
                sum.reused_prompt_tokens += round.reused_prompt_tokens;
                sum.generated_tokens += round.generated_tokens;
                total.reasoning_tokens += stats.reasoning_tokens;
                total.stop_reason = stats.stop_reason;
                total.cache_source = stats.cache_source;
                total.generation.response_format_complete = round.response_format_complete;
                total.generation.tool_constraints_complete = round.tool_constraints_complete;
            }
        }
    }

    fn generated(&self) -> usize {
        self.usage
            .as_ref()
            .map_or(0, |usage| usage.generation.generated_tokens)
    }
}

impl Driver {
    fn send(&self, event: Event) -> bool {
        self.relay.blocking_send(Signal::Event(event))
    }

    fn run(mut self, mut inner: EventStream) {
        let budget = self.tool_loop.request.max_tokens;
        let mut spent = Spent::default();
        // The citation marker number of the next file search result.
        let mut marker = 1;
        loop {
            let Some((turn, stats)) = self.forward(&mut inner) else {
                return;
            };
            spent.add(&stats);
            let hosted: Vec<(ToolCall, Hosted)> = turn
                .calls
                .iter()
                .filter_map(|call| {
                    self.tool_loop
                        .kinds
                        .hosted(&call.name)
                        .map(|hosted| (call.clone(), hosted))
                })
                .collect();
            if hosted.is_empty() || stats.stop_reason != StopReason::Eos {
                self.finish(spent);
                return;
            }
            let client_pending = turn.calls.len() > hosted.len();
            if let Some(before) = self.tool_loop.before_tools.take()
                && let Err(error) = before()
            {
                self.send(Event::Error(error));
                return;
            }
            let mut outputs = Vec::with_capacity(hosted.len());
            for (call, operation) in hosted {
                let Some(output) = self.run_call(&call, operation, &mut spent, &mut marker) else {
                    return;
                };
                if !self.send(Event::ToolOutput(output.clone())) {
                    return;
                }
                outputs.push(output);
            }
            if client_pending {
                self.finish(spent);
                return;
            }
            spent.rounds += 1;
            let left = budget.saturating_sub(spent.generated());
            if left == 0 {
                if let Some(usage) = &mut spent.usage {
                    usage.stop_reason = StopReason::TokenLimit;
                }
                self.finish(spent);
                return;
            }
            let choice = if self.exhausted(&spent) {
                ToolChoice::None
            } else {
                ToolChoice::Auto
            };
            let request = &mut self.tool_loop.request;
            request.messages.push(ChatMessage {
                role: "assistant".into(),
                content: turn.content,
                reasoning_content: (!turn.reasoning.is_empty()).then_some(turn.reasoning),
                tool_calls: turn.calls,
                ..ChatMessage::default()
            });
            for output in outputs {
                request.messages.push(ChatMessage {
                    role: "tool".into(),
                    content: output.content,
                    tool_call_id: Some(output.call_id),
                    ..ChatMessage::default()
                });
            }
            request.max_tokens = left;
            request.tool_choice = choice;
            if self.relay.is_cancelled() {
                return;
            }
            let next = request.clone();
            match (self.tool_loop.submit)(next) {
                Ok(next) => {
                    *self.current.lock().unwrap_or_else(PoisonError::into_inner) =
                        next.cancel_handle();
                    // A cancellation that landed before the handle was in
                    // place cancelled the previous generation, not this one.
                    if self.relay.is_cancelled() {
                        next.cancel_handle().cancel();
                    }
                    inner = next;
                }
                Err(error) => {
                    self.send(Event::Error(error.to_string()));
                    return;
                }
            }
        }
    }

    /// Forward one generation, returning its turn and measurements, or
    /// `None` once the stream has ended (an error was forwarded, the engine
    /// stopped, or the consumer went away).
    fn forward(&self, inner: &mut EventStream) -> Option<(Turn, Box<Stats>)> {
        let mut turn = Turn::default();
        loop {
            let signal = inner.next_signal(None).ok().flatten()?;
            match signal {
                Signal::Event(Event::Finished(stats)) => return Some((turn, stats)),
                Signal::Event(Event::Error(error)) => {
                    self.send(Event::Error(error));
                    return None;
                }
                Signal::Event(event) => {
                    match &event {
                        Event::Reasoning(piece) => turn.reasoning.push_str(piece),
                        Event::Content(piece) => turn.content.push_str(piece),
                        Event::ToolCall(call) => turn.calls.push(call.clone()),
                        _ => {}
                    }
                    if !self.send(event) {
                        inner.cancel_handle().cancel();
                        return None;
                    }
                }
                progress @ Signal::Progress(_) => {
                    if !self.relay.blocking_send(progress) {
                        inner.cancel_handle().cancel();
                        return None;
                    }
                }
            }
        }
    }

    /// Report the summed usage as the Response's end.
    fn finish(&self, spent: Spent) {
        if let Some(usage) = spent.usage {
            self.send(Event::Finished(Box::new(usage)));
        }
    }

    /// Whether another hosted call may run.
    fn exhausted(&self, spent: &Spent) -> bool {
        spent.rounds >= self.tool_loop.bounds.max_iterations
            || self
                .tool_loop
                .max_calls
                .is_some_and(|max| spent.calls >= max)
            || spent.tool_time >= self.tool_loop.bounds.budget
    }

    /// Run one hosted call and report its output; `None` when the consumer
    /// cancelled while it ran. `marker` numbers file search results.
    fn run_call(
        &self,
        call: &ToolCall,
        operation: Hosted,
        spent: &mut Spent,
        marker: &mut usize,
    ) -> Option<ToolOutput> {
        if operation == Hosted::FileSearch {
            let result = self.file_search(call, spent)?;
            let output = file_search::output(call, &result, *marker);
            *marker += file_search::numbered(&output);
            Some(output)
        } else {
            let result = self.execute(call, operation, spent)?;
            Some(hosted::tool_output(call, operation, &result))
        }
    }

    /// Why no further call may run, once the bounds are used up.
    fn denied(&self, spent: &Spent) -> Option<ToolError> {
        self.exhausted(spent).then(|| {
            ToolError::Denied(format!(
                "not run: this response's tool budget is used up ({} rounds, {} calls, {:.1}s \
                 of {}s); answer with what you have",
                spent.rounds,
                spent.calls,
                spent.tool_time.as_secs_f64(),
                self.tool_loop.bounds.budget.as_secs()
            ))
        })
    }

    /// Run one toolbox call within the bounds; `None` when the consumer
    /// cancelled while it ran.
    fn execute(
        &self,
        call: &ToolCall,
        operation: Hosted,
        spent: &mut Spent,
    ) -> Option<Result<Output, ToolError>> {
        if let Some(denied) = self.denied(spent) {
            return Some(Err(denied));
        }
        let request = match hosted::request(operation, &call.arguments) {
            Ok(request) => request,
            Err(message) => return Some(Err(ToolError::Invalid(message))),
        };
        let toolbox = &self.tool_loop.toolbox;
        self.bounded(spent, |timeout| toolbox.run(request, timeout))
    }

    /// Run one `file_search` call within the bounds; `None` when the
    /// consumer cancelled while it ran.
    fn file_search(
        &self,
        call: &ToolCall,
        spent: &mut Spent,
    ) -> Option<Result<local_services::vector_stores::SearchResults, ToolError>> {
        if let Some(denied) = self.denied(spent) {
            return Some(Err(denied));
        }
        let (Some(config), Some(vector_stores)) = (
            self.tool_loop.kinds.file_search(),
            self.tool_loop.vector_stores.as_ref(),
        ) else {
            return Some(Err(ToolError::Denied(
                "file_search is unavailable on this server".into(),
            )));
        };
        let request = match config.request(&call.arguments) {
            Ok(request) => request,
            Err(message) => return Some(Err(ToolError::Invalid(message))),
        };
        let stores = config.vector_store_ids.clone();
        let ranking = config.ranking();
        self.bounded(spent, |timeout| async move {
            tokio::time::timeout(timeout, vector_stores.file_search(stores, request, ranking))
                .await
                .unwrap_or(Err(ToolError::TimedOut(timeout)))
        })
    }

    /// Run `work` (given its deadline) as one counted call, reporting
    /// liveness while it runs; `None` when the consumer cancelled meanwhile.
    fn bounded<T, F>(
        &self,
        spent: &mut Spent,
        work: impl FnOnce(Duration) -> F,
    ) -> Option<Result<T, ToolError>>
    where
        F: Future<Output = Result<T, ToolError>>,
    {
        let timeout = self
            .tool_loop
            .bounds
            .call_timeout
            .min(self.tool_loop.bounds.budget.saturating_sub(spent.tool_time));
        spent.calls += 1;
        let started = Instant::now();
        let result = self.runtime.block_on(async {
            let mut run = pin!(work(timeout));
            let mut heartbeat = tokio::time::interval(HEARTBEAT);
            heartbeat.tick().await;
            loop {
                tokio::select! {
                    result = &mut run => return Some(result),
                    () = self.relay.cancelled() => return None,
                    _ = heartbeat.tick() => {
                        let alive = Signal::Progress(PrefillProgress { tokens: 0, chunks: 0 });
                        if !self.relay.send(alive).await {
                            return None;
                        }
                    }
                }
            }
        });
        spent.tool_time += started.elapsed();
        result
    }
}

#[cfg(test)]
#[path = "tool_tests.rs"]
mod tests;
