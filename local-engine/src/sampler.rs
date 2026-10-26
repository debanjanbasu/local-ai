use std::collections::VecDeque;

use llguidance::toktrie::SimpleVob;
use local_metal::batch::CommandBatch;
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::sampling::{GpuTopK, TopKCandidate};

use crate::structured::Grammar;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub eos_tokens: Vec<u32>,
    pub seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingResult {
    pub token_id: u32,
    pub is_eos: bool,
}

/// One generated token's log-probability under the target model's raw
/// distribution: its logits before temperature, penalties, top-k, top-p,
/// min-p and any response-format mask (vLLM's `raw_logprobs`).
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprob {
    pub token: u32,
    /// Natural log of the token's probability.
    pub logprob: f32,
    /// The token's bytes, which need not be whole UTF-8 characters.
    pub bytes: Vec<u8>,
    /// The most likely tokens at this position, most likely first: as many
    /// as the request's `top_logprobs` asked for.
    pub top: Vec<TopLogprob>,
}

/// One of a position's most likely tokens (see [`TokenLogprob::top`]).
#[derive(Clone, Debug, PartialEq)]
pub struct TopLogprob {
    pub token: u32,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

/// A request's log-probability reporting.
#[derive(Clone)]
struct Logprobs {
    /// Alternatives reported per token.
    top: usize,
    /// One record per target selection not yet taken, in selection order.
    records: VecDeque<TokenLogprob>,
    /// The raw logit row, kept while host selection overwrites the buffer.
    raw: Vec<f32>,
}

#[derive(Clone)]
pub struct Sampler {
    params: SamplingParams,
    seen_tokens: Vec<usize>,
    observed: Vec<bool>,
    candidates: Vec<usize>,
    rng: SplitMix64,
    /// The request's response format, when it has one. Only target
    /// selections consult it; [`Self::greedy_draft`] drops it.
    grammar: Option<Box<Grammar>>,
    /// Set when the request asked for log-probabilities. Only target
    /// selections record them; [`Self::greedy_draft`] drops it.
    logprobs: Option<Box<Logprobs>>,
}

impl Sampler {
    #[must_use]
    pub fn new(vocab_size: usize, params: SamplingParams) -> Self {
        Self {
            rng: SplitMix64::new(params.seed),
            params,
            seen_tokens: Vec::new(),
            observed: vec![false; vocab_size],
            candidates: Vec::with_capacity(vocab_size),
            grammar: None,
            logprobs: None,
        }
    }

    /// Record the raw log-probability of every later target selection, with
    /// its `top` most likely alternatives.
    pub(crate) fn report_logprobs(&mut self, top: usize) {
        self.logprobs = Some(Box::new(Logprobs {
            top,
            records: VecDeque::new(),
            raw: Vec::new(),
        }));
    }

    /// The alternatives each record reports, when the request asked for
    /// log-probabilities.
    #[must_use]
    pub fn logprobs(&self) -> Option<usize> {
        self.logprobs.as_ref().map(|logprobs| logprobs.top)
    }

    /// The record of the oldest target selection not yet taken.
    pub(crate) fn take_logprob(&mut self) -> Option<TokenLogprob> {
        self.logprobs.as_mut()?.records.pop_front()
    }

    /// Constrain every later selection to `grammar` (see [`crate::structured`]).
    pub(crate) fn constrain(&mut self, grammar: Grammar) {
        self.grammar = Some(Box::new(grammar));
    }

    /// Why the response-format grammar stopped this request, once it has.
    /// A failed selection is reported as end-of-sequence so verification
    /// stops there; the caller must check this before treating it as one.
    pub(crate) fn grammar_failure(&self) -> Option<&str> {
        self.grammar.as_deref().and_then(Grammar::failure)
    }

    /// For a constrained request that stopped at end-of-sequence (`eos`) or
    /// otherwise: whether its answer is a complete document. End-of-sequence
    /// is only selectable once the grammar accepts, so a complete answer is
    /// one that ended there after its constraint began. `None` unconstrained.
    pub(crate) fn response_format_complete(&self, eos: bool) -> Option<bool> {
        self.grammar
            .as_deref()
            .filter(|grammar| grammar.response_format)
            .map(|grammar| eos && grammar.answering() && grammar.failure().is_none())
    }

    pub(crate) fn tool_constraints_complete(&self, eos: bool) -> Option<bool> {
        self.grammar
            .as_deref()
            .filter(|grammar| grammar.tool_constraints)
            .map(|grammar| eos && grammar.tools_complete() && grammar.failure().is_none())
    }

    pub fn observe(&mut self, tokens: &[u32]) {
        for &token in tokens {
            let index = token as usize;
            if let Some(seen) = self.observed.get_mut(index)
                && !*seen
            {
                *seen = true;
                self.seen_tokens.push(index);
            }
        }
    }

    /// The request's greedy proposal sampler for drafting. It never carries
    /// the response-format grammar: drafts are unconstrained proposals that
    /// the target's own masked selections verify, and drafting must not
    /// advance the request's matcher.
    pub fn greedy_draft(&self) -> Self {
        Self {
            params: SamplingParams {
                temperature: 0.0,
                top_k: 1,
                ..self.params.clone()
            },
            seen_tokens: self.seen_tokens.clone(),
            observed: self.observed.clone(),
            candidates: self.candidates.clone(),
            rng: self.rng.clone(),
            grammar: None,
            logprobs: None,
        }
    }

    /// The target selection from the logit row at `byte_offset` of `buffer`:
    /// `argmax`, the row's argmax a block selected on the device, when the
    /// selection is the plain argmax, else [`Self::sample_buffer`], drawing
    /// from `device_top`, the row's best logits a block selected on the device,
    /// when they are the candidates it would select (see
    /// [`Self::device_top_k`]).
    ///
    /// When the request asked for log-probabilities, also records the row's
    /// raw distribution: from `distribution` (its log-sum-exp and best logits,
    /// at least [`Self::logprobs`] of them) when the GPU computed it, else on
    /// the host. The row is read before selection overwrites it.
    #[allow(clippy::too_many_arguments)]
    pub fn select_row(
        &mut self,
        buffer: &mut MetalBuffer,
        byte_offset: usize,
        context: &MetalContext,
        topk: &GpuTopK,
        argmax: Option<u32>,
        device_top: Option<&[TopKCandidate]>,
        distribution: Option<(f32, Vec<TopKCandidate>)>,
    ) -> crate::Result<SamplingResult> {
        let device = (argmax, device_top);
        let Some(mut logprobs) = self.logprobs.take() else {
            return self.select_plain(buffer, byte_offset, context, topk, device);
        };
        let result = self.select_recorded(
            &mut logprobs,
            buffer,
            byte_offset,
            context,
            topk,
            device,
            distribution,
        );
        self.logprobs = Some(logprobs);
        result
    }

    fn select_plain(
        &mut self,
        buffer: &mut MetalBuffer,
        byte_offset: usize,
        context: &MetalContext,
        topk: &GpuTopK,
        (argmax, device_top): (Option<u32>, Option<&[TopKCandidate]>),
    ) -> crate::Result<SamplingResult> {
        match argmax {
            Some(token_id) if self.selects_argmax() => Ok(self.greedy_result(token_id)),
            _ => self.sample_from(buffer, byte_offset, context, topk, device_top),
        }
    }

    #[allow(unsafe_code, clippy::too_many_arguments)]
    fn select_recorded(
        &mut self,
        logprobs: &mut Logprobs,
        buffer: &mut MetalBuffer,
        byte_offset: usize,
        context: &MetalContext,
        topk: &GpuTopK,
        device: (Option<u32>, Option<&[TopKCandidate]>),
        distribution: Option<(f32, Vec<TopKCandidate>)>,
    ) -> crate::Result<SamplingResult> {
        let argmax = device.0;
        let start = byte_offset / size_of::<f32>();
        // SAFETY: the caller finished inference before exposing the buffer, and
        // `row` is last read before selection below writes the buffer.
        let Some(row) = unsafe { buffer.as_slice::<f32>() }.get(start..start + self.observed.len())
        else {
            return Err(crate::Error::Sampling(
                "logit row offset exceeds the buffer".into(),
            ));
        };
        // Device argmax leaves the row untouched; host selection does not.
        let untouched = argmax.is_some() && self.selects_argmax();
        if !untouched {
            logprobs.raw.clear();
            logprobs.raw.extend_from_slice(row);
        }
        let (log_sum_exp, top) = match distribution {
            Some((log_sum_exp, top)) if top.len() >= logprobs.top => (
                log_sum_exp,
                top[..logprobs.top]
                    .iter()
                    .map(|candidate| (candidate.token_id, candidate.logit))
                    .collect(),
            ),
            _ => raw_distribution(if untouched { row } else { &logprobs.raw }, logprobs.top),
        };
        let result = self.select_plain(buffer, byte_offset, context, topk, device)?;
        let token = result.token_id as usize;
        let logit = if untouched {
            // SAFETY: selection has finished with the buffer; this is a copy.
            unsafe { buffer.as_slice::<f32>()[start + token] }
        } else {
            logprobs.raw[token]
        };
        logprobs.records.push_back(TokenLogprob {
            token: result.token_id,
            logprob: logit - log_sum_exp,
            bytes: Vec::new(),
            top: top
                .into_iter()
                .map(|(token, logit)| TopLogprob {
                    token,
                    logprob: logit - log_sum_exp,
                    bytes: Vec::new(),
                })
                .collect(),
        });
        Ok(result)
    }

    /// Select on the GPU, but keep penalty arithmetic, filtering and RNG on the CPU.
    /// The caller must finish inference before exposing the shared logit buffer here.
    #[allow(unsafe_code)]
    pub fn sample_buffer(
        &mut self,
        buffer: &mut MetalBuffer,
        byte_offset: usize,
        context: &MetalContext,
        topk: &GpuTopK,
    ) -> crate::Result<SamplingResult> {
        self.sample_from(buffer, byte_offset, context, topk, None)
    }

    /// [`Self::sample_buffer`], taking its candidates from `precomputed`, the
    /// row's best logits selected on the device, when nothing (no mask, no
    /// penalty) could change them and they hold at least `top_k`.
    #[allow(unsafe_code)]
    fn sample_from(
        &mut self,
        buffer: &mut MetalBuffer,
        byte_offset: usize,
        context: &MetalContext,
        topk: &GpuTopK,
        precomputed: Option<&[TopKCandidate]>,
    ) -> crate::Result<SamplingResult> {
        let vocab = self.observed.len();
        let float_offset = byte_offset / size_of::<f32>();
        let Some(end) = float_offset.checked_add(vocab) else {
            return Err(crate::Error::Sampling("logit row offset overflow".into()));
        };
        if !byte_offset.is_multiple_of(size_of::<f32>()) || end > buffer.length() / size_of::<f32>()
        {
            return Err(crate::Error::Sampling(
                "logit row offset is unaligned or exceeds the buffer".into(),
            ));
        }
        let Ok(mask) = self.grammar_mask() else {
            return Ok(self.failed());
        };
        // SAFETY (every view of `buffer` below): the caller finished inference
        // before exposing it, each view ends before `topk.encode`, and that
        // batch is waited for before the next view.
        if let Some(mask) = &mask {
            forbid(mask, unsafe {
                &mut buffer.as_mut_slice::<f32>()[float_offset..end]
            });
        }
        let keep = if self.greedy() {
            1
        } else if self.params.top_k == 0 {
            vocab
        } else {
            self.params.top_k.min(vocab)
        };
        let precomputed = precomputed
            .filter(|top| mask.is_none() && !self.applies_penalties() && top.len() >= keep);
        let token = if vocab <= topk.maximum_supported_k() || keep > topk.maximum_supported_k() {
            self.select(
                unsafe { &mut buffer.as_mut_slice::<f32>()[float_offset..end] },
                mask.as_ref(),
            )?
        } else {
            self.candidates.clear();
            if let Some(top) = precomputed {
                // Exactly what the selection below would return: the same
                // kernels, `total_cmp` order, ties to the lower id.
                self.candidates
                    .extend(top[..keep].iter().map(|item| item.token_id as usize));
            } else {
                self.apply_penalties(unsafe {
                    &mut buffer.as_mut_slice::<f32>()[float_offset..end]
                });
                let mut batch = CommandBatch::new(context)?;
                topk.encode(&mut batch, buffer, byte_offset, vocab, keep)?;
                batch.commit_and_wait()?;
                self.candidates.extend(
                    topk.candidates(keep)
                        .iter()
                        .map(|item| item.token_id as usize),
                );
            }
            if let Some(mask) = &mask {
                // Forbidden tokens sit at -inf, behind every allowed one.
                self.candidates
                    .retain(|&index| mask.is_allowed(index as u32));
            }
            if self.candidates.is_empty() {
                return Err(crate::Error::Sampling(
                    "the response format allows no candidate token".into(),
                ));
            }
            if self.greedy() {
                self.candidates[0] as u32
            } else {
                self.sample_selected(unsafe {
                    &mut buffer.as_mut_slice::<f32>()[float_offset..end]
                })?
            }
        };
        Ok(self.finish(token, mask.as_ref()))
    }

    /// Greedy selection that also reports the winner's logit lead over the
    /// runner-up after penalties: a draft model's own confidence in its
    /// proposal. Only meaningful for a greedy sampler ([`Self::greedy_draft`]).
    #[allow(unsafe_code)]
    pub fn sample_buffer_with_margin(
        &self,
        buffer: &mut MetalBuffer,
        byte_offset: usize,
        context: &MetalContext,
        topk: &GpuTopK,
    ) -> crate::Result<(SamplingResult, f32)> {
        let vocab = self.observed.len();
        let float_offset = byte_offset / size_of::<f32>();
        let Some(end) = float_offset.checked_add(vocab) else {
            return Err(crate::Error::Sampling("logit row offset overflow".into()));
        };
        if !byte_offset.is_multiple_of(size_of::<f32>()) || end > buffer.length() / size_of::<f32>()
        {
            return Err(crate::Error::Sampling(
                "logit row offset is unaligned or exceeds the buffer".into(),
            ));
        }
        if !self.greedy() || self.grammar.is_some() {
            return Err(crate::Error::Sampling(
                "draft margins need an unconstrained greedy sampler".into(),
            ));
        }
        // SAFETY: the caller finished inference before exposing the buffer, and
        // `logits` is last used before `topk.encode` below.
        let logits = unsafe { &mut buffer.as_mut_slice::<f32>()[float_offset..end] };
        self.apply_penalties(logits);
        if vocab < 2 {
            return Ok((self.result(0), f32::INFINITY));
        }
        let (best, second) = if vocab <= topk.maximum_supported_k() {
            top_two(logits)
        } else {
            let mut batch = CommandBatch::new(context)?;
            topk.encode(&mut batch, buffer, byte_offset, vocab, 2)?;
            batch.commit_and_wait()?;
            let candidates = topk.candidates(2);
            (
                (candidates[0].token_id, candidates[0].logit),
                (candidates[1].token_id, candidates[1].logit),
            )
        };
        Ok((self.result(best.0), best.1 - second.1))
    }

    /// Whether presence or repetition penalties can change a logit. Without
    /// them a greedy draft is the plain top two of the logits, which the GPU
    /// can select in place (see [`Self::greedy_from_top_two`]).
    #[must_use]
    pub fn applies_penalties(&self) -> bool {
        !(self.params.presence_penalty == 0.0
            && (self.params.repetition_penalty - 1.0).abs() <= f32::EPSILON)
    }

    /// What [`Self::sample_buffer_with_margin`] returns, from an exact top two
    /// (`f32::total_cmp` order, ties to the lower id) selected elsewhere.
    /// Only valid for a greedy sampler that [`Self::applies_penalties`] not.
    #[must_use]
    pub fn greedy_from_top_two(
        &self,
        best_id: u32,
        best: f32,
        second: f32,
    ) -> (SamplingResult, f32) {
        (self.result(best_id), best - second)
    }

    /// Whether selection is the plain argmax of the logits (`f32::total_cmp`
    /// order, ties to the lower id): greedy with no penalty to apply, which
    /// the GPU can select in place (see [`Self::greedy_result`]).
    /// Never while a response-format grammar masks the selection: a
    /// precomputed argmax of unmasked logits could be a forbidden token. The
    /// answer can start mid-round (at `</think>`), so ask again per row.
    #[must_use]
    pub fn selects_argmax(&self) -> bool {
        self.greedy()
            && !self.applies_penalties()
            && !self.grammar.as_deref().is_some_and(Grammar::masking)
    }

    /// The top k a block may select on the device for this sampler's next
    /// target selection: `Some(top_k)` when it samples from the plain top k of
    /// the logits (not greedy, no penalty, no masking grammar, a top-k the GPU
    /// selects). A grammar can start masking mid-round, so the selection
    /// itself asks again.
    #[must_use]
    pub fn device_top_k(&self) -> Option<usize> {
        let vocab = self.observed.len();
        (!self.greedy()
            && !self.applies_penalties()
            && !self.grammar.as_deref().is_some_and(Grammar::masking)
            && vocab > local_metal::sampling::MAX_TOP_K
            && (1..=local_metal::sampling::MAX_TOP_K).contains(&self.params.top_k))
        .then_some(self.params.top_k)
    }

    /// What [`Self::sample_buffer`] returns for the argmax `token_id` selected
    /// elsewhere, recording the selection. Only valid when
    /// [`Self::selects_argmax`]; a masking grammar fails the request instead.
    pub fn greedy_result(&mut self, token_id: u32) -> SamplingResult {
        self.finish(token_id, None)
    }

    const fn greedy(&self) -> bool {
        self.params.temperature <= f32::EPSILON || self.params.top_k == 1
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> crate::Result<SamplingResult> {
        if logits.len() != self.observed.len() || logits.is_empty() {
            return Err(crate::Error::Sampling(format!(
                "logit width {} does not match vocabulary {}",
                logits.len(),
                self.observed.len()
            )));
        }
        let Ok(mask) = self.grammar_mask() else {
            return Ok(self.failed());
        };
        if let Some(mask) = &mask {
            forbid(mask, logits);
        }
        let token = self.select(logits, mask.as_ref())?;
        Ok(self.finish(token, mask.as_ref()))
    }

    /// Select from `logits`, already masked to `mask` when there is one.
    fn select(&mut self, logits: &mut [f32], mask: Option<&SimpleVob>) -> crate::Result<u32> {
        self.apply_penalties(logits);
        if self.greedy() {
            return Ok(argmax(logits));
        }

        self.candidates.clear();
        match mask {
            Some(mask) => {
                let candidates = &mut self.candidates;
                mask.iter_set_entries(|index| candidates.push(index));
                if candidates.is_empty() {
                    return Err(crate::Error::Sampling(
                        "the response format allows no token".into(),
                    ));
                }
            }
            None => self.candidates.extend(0..logits.len()),
        }
        let keep = if self.params.top_k == 0 {
            logits.len()
        } else {
            self.params.top_k.min(logits.len())
        };
        if keep < self.candidates.len() {
            self.candidates.select_nth_unstable_by(keep, |&a, &b| {
                logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b))
            });
            self.candidates.truncate(keep);
        }
        self.candidates
            .sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
        self.sample_selected(logits)
    }

    fn sample_selected(&mut self, logits: &mut [f32]) -> crate::Result<u32> {
        let inverse_temperature = self.params.temperature.recip();
        let maximum = logits[self.candidates[0]] * inverse_temperature;
        let mut denominator = 0.0_f32;
        for &index in &self.candidates {
            logits[index] = logits[index].mul_add(inverse_temperature, -maximum).exp();
            denominator += logits[index];
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(crate::Error::Sampling(
                "non-finite sampling distribution".into(),
            ));
        }
        for &index in &self.candidates {
            logits[index] /= denominator;
        }

        if self.params.min_p > 0.0 {
            let cutoff = logits[self.candidates[0]] * self.params.min_p;
            self.candidates.retain(|&index| logits[index] >= cutoff);
        }
        if self.params.top_p < 1.0 {
            let mut cumulative = 0.0_f32;
            let mut length = self.candidates.len();
            for (rank, &index) in self.candidates.iter().enumerate() {
                cumulative += logits[index];
                if cumulative >= self.params.top_p {
                    length = rank + 1;
                    break;
                }
            }
            self.candidates.truncate(length.max(1));
        }

        let active_sum: f32 = self.candidates.iter().map(|&index| logits[index]).sum();
        let target = self.rng.next_f32() * active_sum;
        let mut cumulative = 0.0_f32;
        let mut selected = *self.candidates.last().unwrap_or(&0);
        for &index in &self.candidates {
            cumulative += logits[index];
            if cumulative >= target {
                selected = index;
                break;
            }
        }
        Ok(selected as u32)
    }

    /// The mask for this target selection: `Ok(None)` when nothing masks it,
    /// `Err(())` once the grammar has failed (the failure is recorded on it).
    fn grammar_mask(&mut self) -> Result<Option<SimpleVob>, ()> {
        let Some(grammar) = self.grammar.as_deref_mut() else {
            return Ok(None);
        };
        if grammar.failure().is_some() {
            return Err(());
        }
        if !grammar.masking() {
            return Ok(None);
        }
        grammar
            .mask()
            .map(Some)
            .map_err(|error| grammar.fail(error))
    }

    /// The result of selecting `token_id` under `mask`, recorded on the
    /// grammar exactly once.
    fn finish(&mut self, token_id: u32, mask: Option<&SimpleVob>) -> SamplingResult {
        let result = self.result(token_id);
        let accepted = self
            .grammar
            .as_deref_mut()
            .is_none_or(|grammar| grammar.select(token_id, result.is_eos, mask));
        if accepted { result } else { self.failed() }
    }

    /// A selection the grammar refused: an end-of-sequence that stops the
    /// round, which [`Self::grammar_failure`] turns into the request's error.
    fn failed(&self) -> SamplingResult {
        SamplingResult {
            token_id: self.params.eos_tokens.first().copied().unwrap_or(0),
            is_eos: true,
        }
    }

    fn apply_penalties(&self, logits: &mut [f32]) {
        if !self.applies_penalties() {
            return;
        }
        for &index in &self.seen_tokens {
            logits[index] -= self.params.presence_penalty;
            if self.params.repetition_penalty > 0.0
                && (self.params.repetition_penalty - 1.0).abs() > f32::EPSILON
            {
                if logits[index] > 0.0 {
                    logits[index] /= self.params.repetition_penalty;
                } else {
                    logits[index] *= self.params.repetition_penalty;
                }
            }
        }
    }

    fn result(&self, token_id: u32) -> SamplingResult {
        SamplingResult {
            token_id,
            is_eos: self.params.eos_tokens.contains(&token_id),
        }
    }
}

pub struct Verification {
    pub samples: Vec<SamplingResult>,
    pub accepted: usize,
}

/// Greedy drafts have point-mass proposal distributions. Sample the target once
/// per row and reuse subsequent verified rows only while those samples match.
/// This preserves target filtering/penalties/RNG without an extra acceptance draw.
pub fn verify_greedy_drafts<F>(
    sampler: &mut Sampler,
    drafts: &[u32],
    mut sample: F,
) -> crate::Result<Verification>
where
    F: FnMut(usize, &mut Sampler) -> crate::Result<SamplingResult>,
{
    let mut results = Vec::with_capacity(drafts.len() + 1);
    let mut accepted = 0;
    for row in 0..=drafts.len() {
        let next = sample(row, sampler)?;
        results.push(next);
        if next.is_eos || drafts.get(row) != Some(&next.token_id) {
            break;
        }
        accepted += 1;
        sampler.observe(&[next.token_id]);
    }
    Ok(Verification {
        samples: results,
        accepted,
    })
}

/// Push every token `mask` forbids to -inf, so no selection can reach it.
/// Ids past either the mask or the logits are left alone. Works a mask word
/// (32 ids) at a time: allowed words are skipped, forbidden words filled.
fn forbid(mask: &SimpleVob, logits: &mut [f32]) {
    const BITS: usize = u32::BITS as usize;
    let covered = mask.len().min(logits.len());
    let (chunks, remainder) = logits[..covered].as_chunks_mut::<BITS>();
    for (chunk, &word) in chunks.iter_mut().zip(mask.as_slice()) {
        match word {
            u32::MAX => {}
            0 => chunk.fill(f32::NEG_INFINITY),
            _ => {
                for (bit, logit) in chunk.iter_mut().enumerate() {
                    if word & (1 << bit) == 0 {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }
    let start = covered - covered % BITS;
    for (index, logit) in (start..).zip(remainder) {
        if !mask.is_allowed(index as u32) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// The log-sum-exp of `row` and its `k` best logits, best first with ties to
/// the lower id: what the GPU's log-probability kernels compute.
pub fn raw_distribution(row: &[f32], k: usize) -> (f32, Vec<(u32, f32)>) {
    let top_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = row
        .iter()
        .map(|&logit| f64::from(logit - top_logit).exp())
        .sum::<f64>();
    let log_sum_exp = top_logit + sum.ln() as f32;
    let order = |a: &u32, b: &u32| {
        row[*b as usize]
            .total_cmp(&row[*a as usize])
            .then_with(|| a.cmp(b))
    };
    let mut ids = (0..row.len() as u32).collect::<Vec<_>>();
    let k = k.min(ids.len());
    if k < ids.len() {
        ids.select_nth_unstable_by(k, order);
        ids.truncate(k);
    }
    ids.sort_unstable_by(order);
    let top = ids.into_iter().map(|id| (id, row[id as usize])).collect();
    (log_sum_exp, top)
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(a_index, a), (b_index, b)| a.total_cmp(b).then_with(|| b_index.cmp(a_index)))
        .map_or(0, |(index, _)| index as u32)
}

/// `((best_id, best), (second_id, second))` with the GPU top-k's tie order
/// (higher logit first, then lower token ID). Needs at least two values.
fn top_two(values: &[f32]) -> ((u32, f32), (u32, f32)) {
    let mut best = (0_u32, f32::NEG_INFINITY);
    let mut second = (0_u32, f32::NEG_INFINITY);
    for (index, &value) in values.iter().enumerate() {
        let candidate = (index as u32, value);
        if value.total_cmp(&best.1).is_gt() {
            second = best;
            best = candidate;
        } else if value.total_cmp(&second.1).is_gt() {
            second = candidate;
        }
    }
    (best, second)
}

#[derive(Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        ((value >> 40) as f32) * (1.0 / 16_777_216.0)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
