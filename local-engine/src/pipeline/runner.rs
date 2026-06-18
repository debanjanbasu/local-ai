//! Continuous-batching scheduler: drive many independent generation requests to
//! completion over a fixed set of decode lanes.
//!
//! This is the orchestration layer on top of the verified batched primitives
//! ([`Pipeline::prefill_lane`], [`Pipeline::decode_batch_step`]). It admits
//! queued requests into free lanes, prefills each lane at its own prompt length,
//! decodes all active lanes together in one batched forward per step, samples a
//! token per lane, applies per-lane stop conditions (EOS / max tokens /
//! capacity), and recycles finished lanes so queued requests pack in
//! continuously — the throughput payoff for many-agent / sub-agent serving.
//!
//! The decode step always runs at the full lane width: idle lanes carry a
//! filler token whose logits are discarded. On this GPU the MMA kernels pad
//! small `M` toward their sweet spot anyway, so a partially-full batch costs the
//! same as a full one — there is no benefit to packing active lanes into a
//! shorter prefix, and full-width keeps each lane pinned to its own KV pool
//! region (no KV shuffling on eviction).

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{Receiver, TryRecvError};

use crate::Error;
use crate::continuous_batching::ContinuousBatcher;
use crate::multimodal::{MediaInput, MultimodalPrompt};
use crate::request_queue::{InferenceRequest, RequestPriority};
use crate::sampler::{SamplingParams, sample};

use super::{BatchedDecodeState, Pipeline};

/// One generation request for the continuous-batching runner.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Caller-supplied identifier, echoed back on the matching [`BatchOutput`].
    pub id: String,
    /// Prompt token ids (already encoded / templated by the caller).
    pub prompt: Vec<u32>,
    /// Per-request sampling parameters (temperature, top-p, EOS set, …).
    pub params: SamplingParams,
    /// Maximum number of tokens to generate for this request.
    pub max_new_tokens: usize,
}

/// Why a request stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// An end-of-sequence token was produced.
    Eos,
    /// The request hit its `max_new_tokens` budget.
    MaxTokens,
    /// The lane reached its KV capacity (effective context window).
    Capacity,
}

/// Completed generation for one [`BatchRequest`].
#[derive(Debug, Clone)]
pub struct BatchOutput {
    /// Echoes [`BatchRequest::id`].
    pub id: String,
    /// Generated token ids (excludes the prompt).
    pub tokens: Vec<u32>,
    /// Why generation stopped.
    pub stop: StopReason,
}

/// In-flight state for a single decode lane.
struct LaneState {
    id: String,
    params: SamplingParams,
    /// Prompt + generated tokens, fed to the sampler for repetition penalty.
    context: Vec<u32>,
    /// Generated tokens only (the request's output).
    output: Vec<u32>,
    /// Absolute position where the next fed token's K/V is written.
    pos: usize,
    /// Token to feed on the next decode step (the latest sampled token).
    last_token: u32,
    max_new_tokens: usize,
    stop: Option<StopReason>,
    /// Server mode: where to send this request's events (and the request id to
    /// release from the [`ContinuousBatcher`]). `None` for the synchronous
    /// [`Pipeline::run_batched_to_completion`] batch API.
    reply: Option<EventSink>,
    /// Whether to emit incremental [`ServeEvent::Token`] deltas.
    stream: bool,
    /// Full decoded text already emitted as token deltas (for computing the next
    /// delta without re-sending text).
    streamed_text: String,
}

impl LaneState {
    fn into_output(self) -> BatchOutput {
        BatchOutput {
            id: self.id,
            tokens: self.output,
            stop: self.stop.unwrap_or(StopReason::MaxTokens),
        }
    }

    /// Apply per-step stop conditions after a token was appended.
    const fn check_stop(&mut self, is_eos: bool, lane_capacity: usize) {
        if is_eos {
            self.stop = Some(StopReason::Eos);
        } else if self.output.len() >= self.max_new_tokens {
            self.stop = Some(StopReason::MaxTokens);
        } else if self.pos >= lane_capacity {
            self.stop = Some(StopReason::Capacity);
        }
    }
}

impl Pipeline {
    /// Run every request in `requests` to completion over `state`'s lanes,
    /// returning one [`BatchOutput`] per request **in completion order**.
    ///
    /// More requests than lanes is the normal case: as lanes finish they are
    /// recycled and the next queued request is admitted, so a small lane pool
    /// serves an unbounded request stream at the batched-decode throughput.
    ///
    /// # Errors
    ///
    /// Returns an error on any prefill / decode / sampling failure. A prompt
    /// longer than the lane capacity fails that request's prefill.
    pub fn run_batched_to_completion(
        &mut self,
        state: &mut BatchedDecodeState,
        requests: Vec<BatchRequest>,
    ) -> crate::Result<Vec<BatchOutput>> {
        let n_lanes = state.n_lanes();
        let capacity = state.lane_capacity();
        let mut queue: VecDeque<BatchRequest> = requests.into();
        let mut lanes: Vec<Option<LaneState>> = (0..n_lanes).map(|_| None).collect();
        let mut results: Vec<BatchOutput> = Vec::new();

        loop {
            // 1. Admit queued requests into free lanes: prefill + first token.
            for (lane, lane_state) in lanes.iter_mut().enumerate() {
                if lane_state.is_some() {
                    continue;
                }
                let Some(req) = queue.pop_front() else { break };
                state.reset_lane(lane);
                let mut logits = self.prefill_lane(state, lane, &req.prompt)?;
                let pos = req.prompt.len();
                let mut context = req.prompt;
                let res = sample(&mut logits, &req.params, &context, &mut fastrand::f32)
                    .map_err(|e| crate::Error::Sampling(e.to_string()))?;
                context.push(res.token_id);
                let mut ls = LaneState {
                    id: req.id,
                    params: req.params,
                    context,
                    output: vec![res.token_id],
                    pos,
                    last_token: res.token_id,
                    max_new_tokens: req.max_new_tokens.max(1),
                    stop: None,
                    reply: None,
                    stream: false,
                    streamed_text: String::new(),
                };
                // `pos + 1` is the position the *next* token would occupy.
                ls.check_stop(res.is_eos, capacity.saturating_sub(1));
                *lane_state = Some(ls);
            }

            // 2. Finalize any lanes that have already stopped (incl. first-token
            //    stops from admission), freeing them for the next admit pass.
            for (lane, lane_state) in lanes.iter_mut().enumerate() {
                if lane_state.as_ref().is_some_and(|s| s.stop.is_some())
                    && let Some(s) = lane_state.take()
                {
                    state.reset_lane(lane);
                    results.push(s.into_output());
                }
            }

            // 3. Collect still-active lanes.
            let active: Vec<usize> = (0..n_lanes).filter(|&l| lanes[l].is_some()).collect();
            if active.is_empty() {
                if queue.is_empty() {
                    break;
                }
                // Lanes were just freed; loop back to admit the remaining queue.
                continue;
            }

            // 4. One full-width batched decode step. Idle lanes carry a filler
            //    token at position 0; their logits are discarded.
            let mut tokens = vec![0u32; n_lanes];
            let mut positions = vec![0u32; n_lanes];
            for &lane in &active {
                if let Some(s) = lanes[lane].as_ref() {
                    tokens[lane] = s.last_token;
                    positions[lane] = s.pos as u32;
                }
            }
            let mut logits = self.decode_batch_step(state, &tokens, &tokens, &positions)?;

            // 5. Sample per active lane, advance, and apply stop conditions.
            for &lane in &active {
                if let Some(s) = lanes[lane].as_mut() {
                    let res = sample(&mut logits[lane], &s.params, &s.context, &mut fastrand::f32)
                        .map_err(|e| crate::Error::Sampling(e.to_string()))?;
                    s.context.push(res.token_id);
                    s.output.push(res.token_id);
                    s.last_token = res.token_id;
                    s.pos += 1;
                    s.check_stop(res.is_eos, capacity);
                }
            }
        }

        Ok(results)
    }
}

/// A streaming server request submitted to [`Pipeline::serve_batched`] from a
/// connection thread.
///
/// All fields are `Send`, so the raw prompt is handed to the single GPU-owning
/// worker thread (which does the tokenization, chat templating, and
/// detokenization) while the connection thread just parses HTTP and blocks on
/// `reply`.
pub struct ServeRequest {
    /// Unique request id (used for [`ContinuousBatcher`] admission/release).
    pub id: String,
    /// Raw user prompt text; the worker applies the chat template + encodes it.
    pub prompt: String,
    /// Optional media inputs (images/audio/video). When non-empty the worker
    /// runs the multimodal prefill path; when empty this is a plain text request
    /// served by the fast token-only path. The HTTP layer decodes data URIs into
    /// these owned, `Send` payloads off the GPU thread.
    pub media: Vec<MediaInput>,
    /// Sampling temperature (`0.0` = greedy).
    pub temperature: f32,
    /// Nucleus sampling top-p.
    pub top_p: f32,
    /// Maximum number of tokens to generate.
    pub max_new_tokens: usize,
    /// Scheduling priority: interactive requests are admitted before background.
    pub priority: RequestPriority,
    /// Stream incremental [`ServeEvent::Token`] deltas as they are generated
    /// (time-to-first-token instead of waiting for the whole completion). The
    /// terminal [`ServeEvent::Done`] is always sent regardless.
    pub stream: bool,
    /// Sink the worker invokes for each [`ServeEvent`]. A boxed callback (rather
    /// than a concrete channel) keeps the engine transport-agnostic — the HTTP
    /// layer can forward events into a `tokio` channel, an `mpsc`, an SSE stream,
    /// etc., without the engine depending on any async runtime.
    pub reply: EventSink,
}

/// A `Send` callback the worker drives once per generated event for a request.
pub type EventSink = Box<dyn FnMut(ServeEvent) + Send>;

/// Worker-side prompt payload. Text requests carry their already-encoded tokens
/// so admission is cheap; multimodal requests carry the raw text + media and
/// defer the GPU media encode to lane admission (never at enqueue, which would
/// stall active decode lanes).
enum PendingPrompt {
    Text(Vec<u32>),
    Multimodal {
        text: String,
        media: Vec<MediaInput>,
    },
}

/// Worker-side payload: the prompt + resolved sampling params, parked in the
/// pending map until a lane frees up.
struct PendingServe {
    prompt: PendingPrompt,
    params: SamplingParams,
    max_new_tokens: usize,
    stream: bool,
    reply: EventSink,
}

/// An event streamed back for one [`ServeRequest`]. Streaming clients consume
/// [`ServeEvent::Token`] deltas; every request (streaming or not) ends with a
/// single authoritative [`ServeEvent::Done`].
pub enum ServeEvent {
    /// Newly decoded text since the previous event (only sent when
    /// [`ServeRequest::stream`] is set).
    Token(String),
    /// Terminal event carrying the full completion.
    Done(ServeResponse),
}

/// The completion the worker sends back for one [`ServeRequest`].
pub struct ServeResponse {
    /// Generated text (detokenized output, excludes the prompt).
    pub text: String,
    /// Generated token ids.
    pub tokens: Vec<u32>,
    /// Why generation stopped.
    pub stop: StopReason,
}

impl Pipeline {
    /// Run the continuous-batching **server loop**: pull [`ServeRequest`]s from
    /// `rx`, pack them into free lanes (priority-ordered via
    /// [`ContinuousBatcher`]), decode all active lanes together each step, and
    /// send each request's [`ServeResponse`] back the moment it finishes — so a
    /// small lane pool serves a continuous request stream at batched throughput.
    ///
    /// Blocks waiting for the first request when idle; returns when `rx` is
    /// disconnected (all senders dropped) and every active lane has drained.
    ///
    /// This is meant to run on the single thread that owns the GPU `Pipeline`;
    /// connection threads submit `ServeRequest`s and block on their `reply`.
    ///
    /// # Errors
    ///
    /// Returns an error on any prefill / decode / sampling failure. A request
    /// whose prompt exceeds the lane capacity fails *that request only* (its
    /// `reply` is dropped, surfacing as a recv error to the connection thread)
    /// and the loop continues serving the rest.
    #[allow(clippy::too_many_lines)]
    pub fn serve_batched(
        &mut self,
        state: &mut BatchedDecodeState,
        rx: &Receiver<ServeRequest>,
    ) -> crate::Result<()> {
        let n_lanes = state.n_lanes();
        let capacity = state.lane_capacity();
        let mut lanes: Vec<Option<LaneState>> = (0..n_lanes).map(|_| None).collect();
        // ContinuousBatcher provides priority ordering + the running-lane cap;
        // the actual prompt payloads ride alongside in `pending`, keyed by id.
        let mut batcher = ContinuousBatcher::new(n_lanes, 0);
        let mut pending: HashMap<String, PendingServe> = HashMap::new();
        let mut closed = false;

        loop {
            let active = lanes.iter().filter(|l| l.is_some()).count();

            // When nothing is in flight and nothing is queued, block for work
            // (or exit if every sender has hung up).
            if active == 0 && batcher.pending_count() == 0 {
                if closed {
                    break;
                }
                match rx.recv() {
                    Ok(req) => self.enqueue_serve(&mut batcher, &mut pending, req),
                    Err(_) => break,
                }
            }

            // Drain any further ready requests without blocking.
            if !closed {
                loop {
                    match rx.try_recv() {
                        Ok(req) => self.enqueue_serve(&mut batcher, &mut pending, req),
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            closed = true;
                            break;
                        }
                    }
                }
            }

            // 1. Admit queued requests into free lanes (priority-ordered).
            for (lane, lane_state) in lanes.iter_mut().enumerate() {
                if lane_state.is_some() {
                    continue;
                }
                let Some(admission) = batcher.admit_next() else {
                    break;
                };
                let id = admission.request_id.clone();
                let Some(req) = pending.remove(&id) else {
                    // Should not happen: every admitted id has a payload.
                    continue;
                };
                state.reset_lane(lane);
                // Text uses the fast token-only prefill; multimodal builds the
                // soft-token prompt (GPU media encode happens here, at admission,
                // not at enqueue) and stages it via the soft-token prefill.
                let prefilled = match req.prompt {
                    PendingPrompt::Text(tokens) => self
                        .prefill_lane(state, lane, &tokens)
                        .map(|logits| (logits, tokens)),
                    PendingPrompt::Multimodal { text, media } => {
                        let mm = MultimodalPrompt { text, media };
                        self.prepare_multimodal_prompt(&mm).and_then(|prepared| {
                            self.prefill_lane_prepared(state, lane, &prepared)
                                .map(|logits| (logits, prepared.tokens))
                        })
                    }
                };
                let Ok((mut logits, context_tokens)) = prefilled else {
                    // Prompt too long / prefill failure: drop the reply so
                    // the connection thread sees an error, keep serving.
                    batcher.release(&id);
                    continue;
                };
                let pos = context_tokens.len();
                let mut context = context_tokens;
                let res = sample(&mut logits, &req.params, &context, &mut fastrand::f32)
                    .map_err(|e| crate::Error::Sampling(e.to_string()))?;
                context.push(res.token_id);
                let mut ls = LaneState {
                    id,
                    params: req.params,
                    context,
                    output: vec![res.token_id],
                    pos,
                    last_token: res.token_id,
                    max_new_tokens: req.max_new_tokens.max(1),
                    stop: None,
                    reply: Some(req.reply),
                    stream: req.stream,
                    streamed_text: String::new(),
                };
                ls.check_stop(res.is_eos, capacity.saturating_sub(1));
                self.stream_lane_delta(&mut ls);
                *lane_state = Some(ls);
            }

            // 2. Finalize stopped lanes: detokenize, reply, release, recycle.
            for (lane, lane_state) in lanes.iter_mut().enumerate() {
                if lane_state.as_ref().is_some_and(|s| s.stop.is_some())
                    && let Some(s) = lane_state.take()
                {
                    state.reset_lane(lane);
                    self.finish_serve_lane(&mut batcher, s);
                }
            }

            // 3. Active lanes after admission/finalize.
            let active_lanes: Vec<usize> = (0..n_lanes).filter(|&l| lanes[l].is_some()).collect();
            if active_lanes.is_empty() {
                continue;
            }

            // 4. One full-width batched decode step (idle lanes carry filler).
            let mut tokens = vec![0u32; n_lanes];
            let mut positions = vec![0u32; n_lanes];
            for &lane in &active_lanes {
                if let Some(s) = lanes[lane].as_ref() {
                    tokens[lane] = s.last_token;
                    positions[lane] = s.pos as u32;
                }
            }
            let mut logits = self.decode_batch_step(state, &tokens, &tokens, &positions)?;

            // 5. Sample per active lane, advance, apply stop conditions.
            for &lane in &active_lanes {
                if let Some(s) = lanes[lane].as_mut() {
                    let res = sample(&mut logits[lane], &s.params, &s.context, &mut fastrand::f32)
                        .map_err(|e| crate::Error::Sampling(e.to_string()))?;
                    s.context.push(res.token_id);
                    s.output.push(res.token_id);
                    s.last_token = res.token_id;
                    s.pos += 1;
                    s.check_stop(res.is_eos, capacity);
                }
            }
            // Emit streaming deltas after the borrow of `logits` is done.
            for &lane in &active_lanes {
                if let Some(mut s) = lanes[lane].take() {
                    self.stream_lane_delta(&mut s);
                    lanes[lane] = Some(s);
                }
            }
        }

        Ok(())
    }

    /// Emit the newly decoded text for a streaming lane as a [`ServeEvent::Token`].
    /// No-op for non-streaming lanes. Decodes the full output and sends only the
    /// suffix beyond what was already streamed; if a re-tokenization shifts the
    /// prefix (rare with `SentencePiece`), it resyncs silently rather than emit a
    /// broken delta — the terminal [`ServeEvent::Done`] is always authoritative.
    fn stream_lane_delta(&self, lane: &mut LaneState) {
        if !lane.stream {
            return;
        }
        let cur = self.detokenize(&lane.output).unwrap_or_default();
        if cur.len() > lane.streamed_text.len()
            && cur.starts_with(&lane.streamed_text)
            && let Some(sink) = lane.reply.as_mut()
        {
            let delta = cur
                .strip_prefix(&lane.streamed_text)
                .unwrap_or_default()
                .to_string();
            sink(ServeEvent::Token(delta));
        }
        if cur != lane.streamed_text {
            lane.streamed_text = cur;
        }
    }

    /// Apply the model's chat template (if any) and encode to token ids — the
    /// same path `generate_chat` takes, exposed so the batched server worker and
    /// its tests tokenize identically.
    ///
    /// # Errors
    ///
    /// Returns an error if no tokenizer is loaded or encoding fails.
    pub fn encode_chat(&self, prompt: &str) -> crate::Result<Vec<u32>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::Tokenizer("no tokenizer loaded".into()))?;
        let wrapped = tokenizer.chat_prompt(prompt);
        let text = wrapped.as_deref().unwrap_or(prompt);
        tokenizer.encode(text, true)
    }

    /// The model's end-of-sequence / end-of-turn token ids (empty if no
    /// tokenizer is loaded). Exposed for building matching sampling params.
    #[must_use]
    pub fn eos_token_ids(&self) -> Vec<u32> {
        self.tokenizer
            .as_ref()
            .map(|t| t.eos_ids().to_vec())
            .unwrap_or_default()
    }

    /// Build the sampling params the batched server uses: on-device softcap is
    /// already applied by `decode_batch_step` / `write_logits`, so the CPU pass
    /// is disabled (`logit_softcap = 0.0`), matching the single-sequence path.
    fn serve_sampling_params(&self, temperature: f32, top_p: f32) -> SamplingParams {
        let eos_tokens = self
            .tokenizer
            .as_ref()
            .map(|t| t.eos_ids().to_vec())
            .unwrap_or_default();
        SamplingParams {
            temperature,
            top_p,
            eos_tokens,
            logit_softcap: 0.0,
            ..SamplingParams::default()
        }
    }

    fn enqueue_serve(
        &self,
        batcher: &mut ContinuousBatcher,
        pending: &mut HashMap<String, PendingServe>,
        req: ServeRequest,
    ) {
        // Resolve the prompt payload + a token count for scheduling. Text uses
        // the fast encode; multimodal expands placeholder tokens on the CPU only
        // (the GPU media encode is deferred to admission) so the batcher gets an
        // accurate context length without stalling decode lanes.
        let (prompt, prompt_tokens, prompt_hash) = if req.media.is_empty() {
            let tokens = match self.encode_chat(&req.prompt) {
                Ok(t) if !t.is_empty() => t,
                // Encode failure / empty prompt: drop the reply so the client
                // sees an error; never enqueue an unservable request.
                _ => return,
            };
            let hash = fxhash(&tokens);
            let len = tokens.len();
            (PendingPrompt::Text(tokens), len, hash)
        } else {
            let mm = MultimodalPrompt {
                text: req.prompt.clone(),
                media: req.media,
            };
            // Cheap CPU-only placeholder expansion for the scheduling estimate.
            let len = self
                .prepare_multimodal_prompt_tokens(&mm)
                .map_or(0, |p| p.tokens.len());
            let hash = fxhash_str(&mm.text);
            (
                PendingPrompt::Multimodal {
                    text: mm.text,
                    media: mm.media,
                },
                len,
                hash,
            )
        };
        let params = self.serve_sampling_params(req.temperature, req.top_p);
        let inference =
            InferenceRequest::new(req.id.clone(), "local", "local", prompt_hash, req.priority)
                .with_token_limits(prompt_tokens, req.max_new_tokens);
        batcher.enqueue(inference);
        pending.insert(
            req.id,
            PendingServe {
                prompt,
                params,
                max_new_tokens: req.max_new_tokens,
                stream: req.stream,
                reply: req.reply,
            },
        );
    }

    fn finish_serve_lane(&self, batcher: &mut ContinuousBatcher, mut lane: LaneState) {
        batcher.release(&lane.id);
        let Some(mut sink) = lane.reply.take() else {
            return;
        };
        let text = self.detokenize(&lane.output).unwrap_or_default();
        sink(ServeEvent::Done(ServeResponse {
            text,
            tokens: lane.output,
            stop: lane.stop.unwrap_or(StopReason::MaxTokens),
        }));
    }
}

/// Small non-cryptographic hash for the shared-prefix cache key (FNV-1a).
fn fxhash(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in tokens {
        h ^= u64::from(t);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// FNV-1a over a string's bytes — used for the prefix-cache hash of multimodal
/// requests, whose token expansion is media-dependent.
fn fxhash_str(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in text.as_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::explicit_counter_loop,
        clippy::redundant_clone
    )]

    use super::{BatchRequest, Pipeline, StopReason};
    use crate::sampler::SamplingParams;
    use std::path::{Path, PathBuf};

    fn argmax(v: &[f32]) -> usize {
        let mut bi = 0;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i;
            }
        }
        bi
    }

    fn find_model_dir() -> Option<PathBuf> {
        for cand in [
            "../models/gemma-4-e2b-it",
            "models/gemma-4-e2b-it",
            "../models/gemma-4-e2b-q4",
            "models/gemma-4-e2b-q4",
        ] {
            let p = Path::new(cand);
            if p.join("model.lma").exists() || p.join("config.json").exists() {
                return Some(p.to_path_buf());
            }
        }
        None
    }

    fn greedy_params() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            logit_softcap: 0.0,
            // No EOS so the run is a fixed length we can compare exactly.
            eos_tokens: vec![],
        }
    }

    /// Single-sequence greedy reference for one prompt: exactly `steps` tokens.
    fn ref_greedy(pipe: &mut Pipeline, prompt: &[u32], steps: usize) -> Vec<u32> {
        let vocab = pipe.config.vocab_size;
        pipe.reset();
        let fa = pipe.prefill_prompt(prompt).expect("prefill");
        pipe.write_logits(fa);
        let mut logits = pipe.logits_buf.as_slice::<f32>()[..vocab].to_vec();
        let mut seq = Vec::with_capacity(steps);
        let mut pos = prompt.len();
        for _ in 0..steps {
            let tok = argmax(&logits) as u32;
            seq.push(tok);
            let fa = pipe.forward_token(tok, pos).expect("decode");
            pipe.write_logits(fa);
            logits = pipe.logits_buf.as_slice::<f32>()[..vocab].to_vec();
            pos += 1;
        }
        seq
    }

    /// End-to-end scheduler correctness: 5 requests with distinct prompt lengths
    /// served over only 2 lanes (forcing admission, eviction, and lane recycle)
    /// must each produce the *same* greedy continuation as the proven
    /// single-sequence path. Run with:
    /// `cargo test -p local-engine --release runner_matches_single -- --ignored --nocapture`
    #[test]
    #[ignore = "requires the local model bundle; run manually"]
    fn runner_matches_single_sequence_with_recycling() {
        const STEPS: usize = 8;
        let Some(model_dir) = find_model_dir() else {
            eprintln!("model bundle not found; skipping");
            return;
        };
        let mut pipe = Pipeline::new_qat(&model_dir, 4096).expect("load model");
        let vocab = pipe.config.vocab_size;

        let prompts: Vec<Vec<u32>> = vec![
            vec![2u32, 100, 1000],
            vec![2u32, 100, 1000, 5000, 7],
            vec![2u32, 42],
            vec![2u32, 100, 9, 9, 9, 13],
            vec![2u32, 500, 12, 80],
        ]
        .into_iter()
        .map(|p| p.into_iter().filter(|&t| (t as usize) < vocab).collect())
        .collect();

        // Reference: single-sequence greedy per prompt.
        let refs: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| ref_greedy(&mut pipe, p, STEPS))
            .collect();

        // Runner: only 2 lanes for 5 requests => forced recycling.
        let mut state = pipe.new_batched_decode_state(2, 4096).expect("state");
        let requests: Vec<BatchRequest> = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| BatchRequest {
                id: format!("r{i}"),
                prompt: p.clone(),
                params: greedy_params(),
                max_new_tokens: STEPS,
            })
            .collect();
        let outputs = pipe
            .run_batched_to_completion(&mut state, requests)
            .expect("run");

        assert_eq!(outputs.len(), prompts.len());
        for (i, expected) in refs.iter().enumerate() {
            let got = outputs
                .iter()
                .find(|o| o.id == format!("r{i}"))
                .unwrap_or_else(|| panic!("missing output r{i}"));
            assert_eq!(
                &got.tokens,
                expected,
                "request r{i} (prompt len {}) diverged:\n runner={:?}\n single={:?}",
                prompts[i].len(),
                got.tokens,
                expected
            );
            assert_eq!(got.stop, StopReason::MaxTokens, "r{i} stop reason");
        }
        eprintln!(
            "runner matches single-sequence: {} requests over 2 lanes, {STEPS} tokens each",
            prompts.len()
        );
    }

    /// End-to-end **server** correctness: drive `serve_batched` on this thread
    /// while a producer thread streams `ServeRequest`s (mixed priorities) over a
    /// channel, then verify each request's generated tokens match the proven
    /// `run_batched_to_completion` output for the same chat-encoded prompt. This
    /// exercises the channel wiring, `ContinuousBatcher` priority admission,
    /// per-request reply routing, lane recycling, and detokenization together.
    /// Run with:
    /// `cargo test -p local-engine --release serve_batched_matches -- --ignored --nocapture`
    #[test]
    #[ignore = "requires the local model bundle; run manually"]
    #[allow(clippy::too_many_lines)]
    fn serve_batched_matches_batch_runner() {
        use super::{ServeEvent, ServeRequest};
        use crate::request_queue::RequestPriority;
        use std::sync::mpsc;

        const MAX_NEW: usize = 8;
        let Some(model_dir) = find_model_dir() else {
            eprintln!("model bundle not found; skipping");
            return;
        };
        let mut pipe = Pipeline::new_qat(&model_dir, 4096).expect("load model");

        let prompts = [
            "What is the capital of France?",
            "Name a primary color.",
            "Say hello.",
            "Two plus two is",
            "The opposite of hot is",
        ];

        // Reference: encode each prompt the same way the worker does, then run
        // the verified batch runner with greedy params (same EOS set).
        let eos = pipe.eos_token_ids();
        let greedy = SamplingParams {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            logit_softcap: 0.0,
            eos_tokens: eos.clone(),
        };
        let mut state_ref = pipe.new_batched_decode_state(2, 4096).expect("state");
        let ref_requests: Vec<BatchRequest> = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| BatchRequest {
                id: format!("r{i}"),
                prompt: pipe.encode_chat(p).expect("encode"),
                params: greedy.clone(),
                max_new_tokens: MAX_NEW,
            })
            .collect();
        let ref_out = pipe
            .run_batched_to_completion(&mut state_ref, ref_requests)
            .expect("batch runner");

        // Server: enqueue the same prompts (greedy => temperature 0) onto a
        // channel with mixed priorities, then drop the sender so `serve_batched`
        // drains the buffered stream and returns. Each request carries its own
        // reply receiver, mirroring per-connection threads.
        let (tx, rx) = mpsc::channel::<ServeRequest>();
        let mut reply_rxs: Vec<(String, bool, mpsc::Receiver<ServeEvent>)> = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            let (rtx, rrx) = mpsc::channel::<ServeEvent>();
            // Stream odd requests, non-stream even ones, to cover both paths.
            let stream = i % 2 == 1;
            reply_rxs.push((format!("r{i}"), stream, rrx));
            let sink: super::EventSink = Box::new(move |ev| {
                let _ = rtx.send(ev);
            });
            tx.send(ServeRequest {
                id: format!("r{i}"),
                prompt: (*p).to_string(),
                media: Vec::new(),
                temperature: 0.0,
                top_p: 1.0,
                max_new_tokens: MAX_NEW,
                priority: if i % 2 == 0 {
                    RequestPriority::Interactive
                } else {
                    RequestPriority::Background
                },
                stream,
                reply: sink,
            })
            .expect("send");
        }
        drop(tx); // end-of-stream

        // Serve over only 2 lanes => forced admission/eviction/recycling.
        let mut state = pipe.new_batched_decode_state(2, 4096).expect("state");
        pipe.serve_batched(&mut state, &rx).expect("serve");

        for (id, stream, rrx) in &reply_rxs {
            // Drain events: accumulate streamed token deltas, capture the Done.
            let mut streamed = String::new();
            let mut done = None;
            while let Ok(ev) = rrx.recv() {
                match ev {
                    ServeEvent::Token(t) => streamed.push_str(&t),
                    ServeEvent::Done(resp) => {
                        done = Some(resp);
                        break;
                    }
                }
            }
            let resp = done.unwrap_or_else(|| panic!("no Done for {id}"));
            let expected = ref_out
                .iter()
                .find(|o| &o.id == id)
                .unwrap_or_else(|| panic!("missing ref {id}"));
            assert_eq!(
                resp.tokens, expected.tokens,
                "server tokens for {id} diverged from batch runner"
            );
            assert!(!resp.text.is_empty(), "server text for {id} empty");
            if *stream {
                // Streamed deltas must reconstruct the final text exactly.
                assert_eq!(
                    streamed, resp.text,
                    "streamed deltas for {id} != final text"
                );
            }
        }
        eprintln!(
            "serve_batched matches batch runner: {} requests over 2 lanes (stream + non-stream)",
            prompts.len()
        );
    }
}
