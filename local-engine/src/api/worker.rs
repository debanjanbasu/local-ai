//! The engine worker thread: admission, batched stepping and per-request
//! delivery.

use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::sync::Arc;

use tokio::sync::mpsc as async_mpsc;

use super::{Delivered, Engine, Event, EventSplitter, Job, Stats, encode_prompt, prepare_chat};
use crate::GenerateParams;
use crate::bonsai_model::CancelToken;
use crate::runtime::PrefillProgress;
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

    pub(super) fn emit(&mut self, piece: &str) -> bool {
        let mut splitter = std::mem::replace(&mut self.splitter, EventSplitter::new(false, None));
        let flow = splitter.emit(piece, &mut |event| self.push(event));
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
/// every admitted one together (see [`BonsaiEngine::step`](crate::bonsai_model::BonsaiEngine::step)).
pub(super) struct Worker {
    engine: Engine,
    receiver: async_mpsc::Receiver<Job>,
    /// A request taken off the queue that did not fit yet.
    waiting: Option<Prepared>,
    running: std::collections::HashMap<u64, Delivery>,
    /// Finished requests whose last events have not reached their channel.
    draining: Vec<Delivery>,
}

impl Worker {
    pub(super) fn new(engine: Engine, receiver: async_mpsc::Receiver<Job>) -> Self {
        Self {
            engine,
            receiver,
            waiting: None,
            running: std::collections::HashMap::new(),
            draining: Vec::new(),
        }
    }

    pub(super) fn run(mut self) {
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
