//! The engine worker thread: admission, batched stepping and per-request
//! delivery.

use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::task::Poll;

use tokio::sync::{mpsc as async_mpsc, oneshot};

use super::{ChatJob, Delivered, Engine, Event, EventSplitter, Job};
use crate::GenerateParams;
use crate::bonsai_model::CancelToken;
use crate::judgment::{Decision, JudgmentHead, PreparedDecision};
use crate::runtime::PrefillProgress;
use crate::sampler::TokenLogprob;
use crate::structured::Grammar;
use crate::tools::ToolSet;

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

/// Run `future` to completion on this thread, parked until its waker fires.
/// The worker is a plain thread without a runtime; tokio's channel futures
/// need none.
fn block_on<T>(future: impl Future<Output = T>) -> T {
    struct Unpark(std::thread::Thread);
    impl std::task::Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::park();
    }
}

/// A request the worker has rendered and tokenized but not yet admitted.
struct Prepared {
    ids: Vec<u32>,
    images: Option<Arc<crate::bonsai_vision::PromptImages>>,
    params: GenerateParams,
    session: Option<String>,
    thinking: bool,
    tools: Option<Arc<ToolSet>>,
    /// The request's stop sequences, matched in its visible answer.
    stop: Vec<String>,
    grammar: Option<Grammar>,
    events: async_mpsc::Sender<Delivered>,
    cancel: CancelToken,
}

/// An experimental decision waiting for the engine.
pub(in crate::api) struct DecisionJob {
    pub(in crate::api) prepared: PreparedDecision,
    pub(in crate::api) head: Arc<JudgmentHead>,
    pub(in crate::api) reply: oneshot::Sender<crate::Result<Option<Decision>>>,
    pub(in crate::api) cancel: CancelToken,
}

impl DecisionJob {
    /// Whether nobody wants the answer any more.
    fn abandoned(&self) -> bool {
        self.cancel.is_cancelled() || self.reply.is_closed()
    }
}

/// What a waiting decision does at an admission point.
#[derive(Debug, PartialEq, Eq)]
enum Gate {
    /// Answer `Ok(None)` without GPU work.
    Drop,
    /// Hold the queue until running generations finish.
    Wait,
    /// Capture now.
    Run,
}

/// Decisions capture on the resident buffers, so they run only with no
/// active generation; a cancelled one is dropped wherever it waits.
const fn decision_gate(abandoned: bool, active: usize) -> Gate {
    if abandoned {
        Gate::Drop
    } else if active == 0 {
        Gate::Run
    } else {
        Gate::Wait
    }
}

/// A job taken off the queue and not yet admitted. There is at most one, so
/// its size does not matter.
#[expect(clippy::large_enum_variant)]
enum Pending {
    Generation(Prepared),
    Decision(Box<DecisionJob>),
}

/// One admitted request's delivery side.
///
/// Events go to an outbox first and reach the channel without blocking, so a
/// client that reads slowly does not hold up the requests decoding beside it.
/// A client that stops reading is dropped by the server's stall timer, which
/// closes the channel and cancels the generation.
pub(super) struct Delivery {
    pub(super) events: async_mpsc::Sender<Delivered>,
    pub(super) cancel: CancelToken,
    pub(super) splitter: EventSplitter,
    pub(super) outbox: VecDeque<Delivered>,
}

impl Delivery {
    fn push(&mut self, event: Event) -> ControlFlow<()> {
        if self.cancel.is_cancelled() || self.events.is_closed() {
            return ControlFlow::Break(());
        }
        self.outbox.push_back(Delivered::Event(event));
        ControlFlow::Continue(())
    }

    pub(super) fn emit(&mut self, piece: &str, logprobs: Vec<TokenLogprob>) -> bool {
        let mut splitter = std::mem::replace(
            &mut self.splitter,
            EventSplitter::new(false, None, Vec::new()),
        );
        let flow = splitter.emit(piece, logprobs, &mut |event| self.push(event));
        self.splitter = splitter;
        flow.is_continue()
    }

    /// The events that close a generation, as [`Engine::generate`] sends them.
    pub(super) fn complete(
        &mut self,
        result: crate::Result<crate::bonsai_model::BonsaiGeneration>,
    ) {
        let output = match result {
            Ok(output) => output,
            Err(error) => {
                self.outbox
                    .push_back(Delivered::Event(Event::Error(error.to_string())));
                return;
            }
        };
        let mut splitter = std::mem::replace(
            &mut self.splitter,
            EventSplitter::new(false, None, Vec::new()),
        );
        let closed = Engine::close_generation(&mut splitter, output, &mut |event| self.push(event));
        self.splitter = splitter;
        if let Err(error) = closed {
            // A parse failure is terminal: report it, never `Finished`.
            let failure = match error {
                crate::Error::Generation(failure) => failure,
                other => other.to_string(),
            };
            self.outbox
                .push_back(Delivered::Event(Event::Error(failure)));
        }
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
/// every admitted one together (see [`BonsaiEngine::step`](crate::bonsai_model::BonsaiEngine::step)).
pub(super) struct Worker {
    engine: Engine,
    receiver: async_mpsc::Receiver<Job>,
    /// A job taken off the queue that cannot start yet. While it waits no
    /// later job is taken, so admission stays FIFO.
    waiting: Option<Pending>,
    running: std::collections::HashMap<u64, Delivery>,
    /// Finished requests whose last events have not reached their channel.
    draining: Vec<Delivery>,
    /// Published decode activity (see [`crate::activity`]).
    activity: Arc<crate::activity::EngineActivity>,
}

impl Worker {
    pub(super) fn new(
        engine: Engine,
        receiver: async_mpsc::Receiver<Job>,
        activity: Arc<crate::activity::EngineActivity>,
    ) -> Self {
        Self {
            engine,
            receiver,
            waiting: None,
            running: std::collections::HashMap::new(),
            draining: Vec::new(),
            activity,
        }
    }

    pub(super) fn run(mut self) {
        loop {
            if self.running.is_empty() && self.waiting.is_none() {
                let Some(job) = self.next_job() else {
                    return;
                };
                self.waiting = Self::prepare(job);
            }
            self.admit_waiting();
            if !self.running.is_empty() {
                self.activity.stepping();
                // `step` calls one of the two at a time, never both at once.
                let running = std::cell::RefCell::new(&mut self.running);
                let finished = self.engine.inner.step(
                    &mut |id, piece, logprobs| {
                        running
                            .borrow_mut()
                            .get_mut(&id)
                            .is_some_and(|delivery| delivery.emit(piece, logprobs))
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
                self.activity.stepped(!self.running.is_empty());
            }
            self.flush();
        }
    }

    /// Wait, idle, for the next job, handing finished requests' last events
    /// to their channels as their readers make room. A reader that stopped
    /// never holds up the next job: the server's stall timer closes its
    /// channel eventually, which drops what is left. `None` once the queue
    /// is closed.
    fn next_job(&mut self) -> Option<Job> {
        loop {
            self.flush_draining();
            if self.draining.is_empty() {
                return self.receiver.blocking_recv();
            }
            let receiver = &mut self.receiver;
            let mut room = self
                .draining
                .iter()
                .map(|delivery| Box::pin(delivery.events.reserve()))
                .collect::<Vec<_>>();
            let woken = block_on(std::future::poll_fn(|context| {
                if let Poll::Ready(job) = receiver.poll_recv(context) {
                    return Poll::Ready(Some(job));
                }
                // A permit (dropped at once, returning its slot) or a
                // closed channel: either way the next flush settles it.
                if room
                    .iter_mut()
                    .any(|reserve| reserve.as_mut().poll(context).is_ready())
                {
                    return Poll::Ready(None);
                }
                Poll::Pending
            }));
            drop(room);
            if let Some(job) = woken {
                return job;
            }
        }
    }

    /// Admit the waiting request and any queued behind it while they fit.
    /// A decision runs here, synchronously, once no generation is active.
    fn admit_waiting(&mut self) {
        loop {
            let pending = match self.waiting.take() {
                Some(pending) => pending,
                None => match self.receiver.try_recv() {
                    Ok(job) => match Self::prepare(job) {
                        Some(pending) => pending,
                        None => continue,
                    },
                    Err(_) => return,
                },
            };
            let prepared = match pending {
                Pending::Generation(prepared) => prepared,
                Pending::Decision(job) => {
                    match decision_gate(job.abandoned(), self.engine.inner.active_generations()) {
                        Gate::Drop => {
                            let _ = job.reply.send(Ok(None));
                        }
                        Gate::Wait => {
                            self.waiting = Some(Pending::Decision(job));
                            return;
                        }
                        Gate::Run => {
                            let DecisionJob {
                                prepared,
                                head,
                                reply,
                                cancel,
                            } = *job;
                            let result = prepared.run(&mut self.engine.inner, &head, &cancel);
                            let _ = reply.send(result);
                        }
                    }
                    continue;
                }
            };
            if prepared.cancel.is_cancelled() || prepared.events.is_closed() {
                prepared.cancel.cancel();
                continue;
            }
            // A request that can never fit fails now rather than holding the
            // queue until the engine is idle.
            if let Err(error) = self.engine.inner.validate_with_images(
                &prepared.ids,
                &prepared.params,
                prepared.images.as_deref(),
            ) {
                let _ = prepared
                    .events
                    .try_send(Delivered::Event(Event::Error(error.to_string())));
                prepared.cancel.cancel();
                continue;
            }
            if !self
                .engine
                .inner
                .can_admit(prepared.ids.len(), prepared.params.max_tokens)
            {
                self.waiting = Some(Pending::Generation(prepared));
                return;
            }
            let mut delivery = Delivery {
                events: prepared.events,
                cancel: prepared.cancel,
                splitter: EventSplitter::new(prepared.thinking, prepared.tools, prepared.stop),
                outbox: VecDeque::new(),
            };
            let cancel = delivery.cancel.clone();
            let admitted = self.engine.inner.admit_constrained(
                &prepared.ids,
                &prepared.params,
                prepared.session.as_deref(),
                cancel,
                prepared.grammar,
                prepared.images,
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

    /// Unpack a job for admission. Chat prompts and decisions were prepared
    /// when they were queued.
    fn prepare(job: Job) -> Option<Pending> {
        let (prompt, max_tokens, sampling, session, thinking, stop, grammar, events, cancel) =
            match job {
                Job::Decision(job) => return Some(Pending::Decision(job)),
                Job::Chat(job) => {
                    let ChatJob {
                        request,
                        grammar,
                        prompt,
                        events,
                        cancel,
                    } = *job;
                    // Every request arrives rendered and tokenized (and its
                    // images encoded) by the submitting thread.
                    let prompt: crate::Result<_> = Ok((prompt.ids, prompt.images, prompt.tools));
                    (
                        prompt,
                        request.max_tokens,
                        request.sampling,
                        request.session,
                        request.thinking,
                        request.stop,
                        grammar,
                        events,
                        cancel,
                    )
                }
            };
        match prompt {
            Ok((ids, images, tools)) => {
                let mut params = sampling.0;
                params.max_tokens = max_tokens;
                Some(Pending::Generation(Prepared {
                    ids,
                    images,
                    params,
                    session,
                    thinking,
                    tools,
                    stop,
                    grammar,
                    events,
                    cancel,
                }))
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
        self.flush_draining();
    }

    /// Hand finished requests' queued events to their channels without
    /// waiting, releasing each once nothing is left.
    fn flush_draining(&mut self) {
        self.draining.retain_mut(|delivery| {
            let left = delivery.flush(false);
            if !left {
                delivery.cancel.cancel();
            }
            left
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::mpsc as async_mpsc;

    use super::{Gate, Job, decision_gate};
    use crate::api::EngineHandle;
    use crate::bonsai_tokenizer::BonsaiTokenizer;
    use crate::judgment::{DecisionKind, DecisionRequest, DecisionValue, JudgmentHead};

    /// A head of the model's width; its weights are never used here.
    fn head() -> Arc<JudgmentHead> {
        let width = crate::bonsai::WIDTH;
        Arc::new(JudgmentHead::for_tests(
            width,
            1,
            vec![0.0; width],
            vec![0.0],
            vec![0.0; width],
            vec![0.0],
            1.0,
        ))
    }

    fn handle(capacity: usize) -> (EngineHandle, async_mpsc::Receiver<Job>) {
        let (sender, receiver) = async_mpsc::channel(capacity);
        let handle = EngineHandle {
            sender,
            tokenizer: BonsaiTokenizer::tiny_for_tests(),
            activity: Arc::default(),
            prompt_cache: crate::prompt_cache::Closer::new(),
            vision: None,
            context: 32_768,
        };
        (handle, receiver)
    }

    fn request(kind: DecisionKind) -> DecisionRequest {
        DecisionRequest {
            state: "Diff:\n+x".into(),
            question: "Is it right?".into(),
            kind,
        }
    }

    fn decision(job: Job) -> super::DecisionJob {
        match job {
            Job::Decision(job) => *job,
            Job::Chat(_) => panic!("not a decision job"),
        }
    }

    #[test]
    fn decisions_wait_for_running_generations_and_drop_when_abandoned() {
        assert_eq!(decision_gate(false, 0), Gate::Run);
        assert_eq!(decision_gate(false, 3), Gate::Wait);
        assert_eq!(decision_gate(true, 3), Gate::Drop);
        assert_eq!(decision_gate(true, 0), Gate::Drop);
    }

    /// Invalid requests and foreign heads fail on the caller's thread and
    /// queue nothing; valid ones share the bounded queue.
    #[test]
    fn decisions_validate_before_queueing_and_respect_the_bound() {
        let (handle, mut jobs) = handle(1);
        let bad = DecisionRequest {
            state: "<|x|>".into(),
            ..request(DecisionKind::Predicate)
        };
        assert!(handle.decide(head(), &bad).is_err());
        let narrow = Arc::new(JudgmentHead::for_tests(
            3,
            1,
            vec![0.0; 3],
            vec![0.0],
            vec![0.0; 3],
            vec![0.0],
            1.0,
        ));
        assert!(
            handle
                .decide(narrow, &request(DecisionKind::Predicate))
                .is_err()
        );
        assert!(jobs.try_recv().is_err(), "nothing was queued");

        let first = handle
            .decide(head(), &request(DecisionKind::Predicate))
            .expect("queued");
        assert!(matches!(
            handle.decide(head(), &request(DecisionKind::Predicate)),
            Err(crate::Error::QueueFull)
        ));
        let job = decision(jobs.try_recv().expect("one job"));
        assert!(!job.abandoned());
        drop(first);
        assert!(job.abandoned(), "dropping the pending decision cancels it");
    }

    #[test]
    fn cancelled_decisions_resolve_to_none() {
        let (handle, mut jobs) = handle(2);
        let pending = handle
            .decide(
                head(),
                &request(DecisionKind::Choice(vec![
                    DecisionValue::Text("left".into()),
                    DecisionValue::Bool(false),
                ])),
            )
            .expect("queued");
        let job = decision(jobs.try_recv().expect("job"));
        pending.cancel_handle().cancel();
        assert!(job.abandoned());
        assert_eq!(decision_gate(job.abandoned(), 1), Gate::Drop);
        // What the worker answers for a dropped decision.
        let _ = job.reply.send(Ok(None));
        assert!(matches!(pending.wait(), Ok(None)));

        // A worker that goes away without answering is an error, not a hang.
        let pending = handle
            .decide(head(), &request(DecisionKind::Predicate))
            .expect("queued");
        drop(jobs);
        assert!(pending.wait().is_err());
    }

    #[test]
    fn a_single_level_score_resolves_without_the_worker() {
        let (handle, mut jobs) = handle(1);
        let pending = handle
            .decide(head(), &request(DecisionKind::Score(vec!["only".into()])))
            .expect("ready");
        assert!(jobs.try_recv().is_err(), "nothing was queued");
        let decision = pending.wait().expect("ok").expect("decided");
        assert_eq!((decision.argmax, decision.score), (0, Some(0.0)));
        assert!(!decision.captured);
    }
}
