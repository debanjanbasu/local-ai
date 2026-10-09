//! Batched decode across independent sequences.
//!
//! Decode is bound by the ALU work of expanding ternary weights, so one
//! projection pass over several activation rows costs little more than over
//! one. A batched step stacks one row from each active sequence and runs every
//! projection, normalization, residual and the output head over all of them at
//! once; only what reads a sequence's own state runs per row: the short
//! convolution and gated-delta recurrence of the 48 recurrent layers (each
//! sequence's history and F16 state), and the K/V append and attention of the
//! 16 full-attention layers (each sequence's caches and position).
//!
//! A sequence's state is either *resident* (in the model's layers, where every
//! single-sequence path reads it) or *parked* in a [`SequenceState`]. Swapping
//! exchanges buffer handles, never bytes, so moving a sequence in for a solo
//! speculative round costs nothing on the GPU.

use std::sync::atomic::{AtomicU64, Ordering};

use local_metal::bonsai_ops::AttentionWorkspace;
use local_metal::draft::GreedyRows;
use local_metal::shaders::ShaderLibrary;

use super::{
    AttentionLayer, BonsaiModel, BufferCopyRequest, CONV_STATE_BYTES, CommandBatch,
    FullAttentionLayer, HadamardDirection, MetalBuffer, RecurrentLayer, Selection, VOCAB, WIDTH,
    decode_embeddings, grow_cache, layers::decay_projection,
};
use crate::bonsai_mtp::HeadSequence;
use crate::sampler::{Sampler, SamplingResult};

/// Most sequences one batched step decodes together.
pub const MAX_BATCH_SEQUENCES: usize = 8;

/// Most decoded rows a sequence may owe its MTP head before it gives the head
/// up for the rest of its generation (20 KiB of hidden per row).
const MAX_HEAD_LAG: usize = 4096;

static NEXT_STATE_ID: AtomicU64 = AtomicU64::new(1);

fn next_state_id() -> u64 {
    NEXT_STATE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Everything one sequence owns.
///
/// Recurrent state and convolution history per recurrent layer, K/V caches
/// per full-attention layer, its position, and the MTP head's caches and
/// hidden handoff when speculating.
pub struct SequenceState {
    id: u64,
    recurrent: Vec<(MetalBuffer, MetalBuffer)>,
    kv: Vec<(MetalBuffer, MetalBuffer)>,
    position: usize,
    kv_allocated: usize,
    head: Option<HeadSequence>,
}

impl SequenceState {
    /// Identity of this buffer set; it moves with the buffers on a swap.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Bytes of GPU memory this parked state holds.
    pub fn bytes(&self) -> usize {
        self.recurrent
            .iter()
            .chain(&self.kv)
            .map(|(a, b)| a.length() + b.length())
            .sum::<usize>()
            + self.head.as_ref().map_or(0, HeadSequence::bytes)
    }
}

/// Decoded rows a sequence's MTP head has not ingested yet.
///
/// The tokens and the target's output-normalized hidden for each, replayed
/// through the head in one block when the sequence next speculates alone.
#[derive(Default)]
pub struct HeadLag {
    tokens: Vec<u32>,
    hidden: Option<MetalBuffer>,
    /// The lag outgrew [`MAX_HEAD_LAG`]: the head is stale for good.
    abandoned: bool,
}

impl HeadLag {
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Whether the head can still be caught up.
    #[must_use]
    pub const fn usable(&self) -> bool {
        !self.abandoned
    }

    /// Reserve room for one more row, returning its byte offset.
    fn reserve_row(&mut self, context: &local_metal::context::MetalContext) -> Option<usize> {
        if self.abandoned {
            return None;
        }
        if self.tokens.len() >= MAX_HEAD_LAG {
            self.abandoned = true;
            self.tokens = Vec::new();
            self.hidden = None;
            return None;
        }
        let row_bytes = WIDTH * size_of::<f32>();
        let needed = (self.tokens.len() + 1) * row_bytes;
        let capacity = self.hidden.as_ref().map_or(0, MetalBuffer::length);
        if needed > capacity {
            let rows = (self.tokens.len() + 1).next_power_of_two().max(64);
            let grown = MetalBuffer::empty(context.device(), rows * row_bytes).ok()?;
            if let Some(old) = &self.hidden {
                // Host-visible and idle: every batch that wrote it has finished.
                grown.copy_from_bytes(&old.as_slice::<u8>()[..self.tokens.len() * row_bytes], 0);
            }
            self.hidden = Some(grown);
        }
        Some(self.tokens.len() * row_bytes)
    }
}

/// One sequence's row in a batched step.
pub struct BatchRow<'a> {
    /// The token to decode at the sequence's position.
    pub token: u32,
    /// `None` for the resident sequence.
    pub state: Option<&'a mut SequenceState>,
    /// Where the head's missed rows go, for a speculating model.
    pub lag: Option<&'a mut HeadLag>,
}

/// Output buffers of a batched step, made on first use.
pub(super) struct BatchScratch {
    logits: MetalBuffer,
    greedy: GreedyRows,
    /// Rows the last batched step produced, and whether `greedy` selected them.
    rows: usize,
    selected: bool,
}

impl BatchScratch {
    pub(super) fn new(
        context: &local_metal::context::MetalContext,
        shaders: &ShaderLibrary,
    ) -> crate::Result<Self> {
        Ok(Self {
            logits: MetalBuffer::empty(
                context.device(),
                MAX_BATCH_SEQUENCES * VOCAB * size_of::<f32>(),
            )?,
            greedy: GreedyRows::new(context, shaders, VOCAB, MAX_BATCH_SEQUENCES)?,
            rows: 0,
            selected: false,
        })
    }
}

/// Per-row view of the sequence-specific buffers of one layer.
struct RowBuffers<'a> {
    recurrent: Vec<(&'a MetalBuffer, &'a MetalBuffer)>,
    kv: Vec<(&'a MetalBuffer, &'a MetalBuffer)>,
    positions: Vec<usize>,
    capacities: Vec<usize>,
}

impl BonsaiModel {
    /// Identity of the resident buffer set.
    pub(crate) const fn state_id(&self) -> u64 {
        self.state_id
    }

    /// A fresh, empty sequence: zero recurrent state and history, K/V caches
    /// at the initial allocation, position zero.
    pub(crate) fn new_sequence(&self) -> crate::Result<SequenceState> {
        let empty = |bytes| MetalBuffer::empty(self.context.device(), bytes);
        let mut recurrent = Vec::new();
        let mut kv = Vec::new();
        for layer in &self.layers {
            match &layer.attention {
                AttentionLayer::Recurrent(_) => {
                    let state = empty(self.state_format.state_bytes())?;
                    let history = empty(CONV_STATE_BYTES)?;
                    state.clear();
                    history.clear();
                    recurrent.push((state, history));
                }
                AttentionLayer::Full(_) => kv.push((
                    empty(self.kv_initial * self.kv_layout.key.token_bytes())?,
                    empty(self.kv_initial * self.kv_layout.value.token_bytes())?,
                )),
            }
        }
        let head = self
            .speculation
            .as_ref()
            .map(|_| HeadSequence::new(&self.context, self.kv_initial))
            .transpose()?;
        Ok(SequenceState {
            id: next_state_id(),
            recurrent,
            kv,
            position: 0,
            kv_allocated: self.kv_initial,
            head,
        })
    }

    /// Exchange the resident sequence with `parked`: afterwards the model
    /// runs `parked`'s sequence and `parked` holds the one that was resident.
    pub(crate) fn swap_sequence(&mut self, parked: &mut SequenceState) -> crate::Result<()> {
        let mut recurrent = parked.recurrent.iter_mut();
        let mut kv = parked.kv.iter_mut();
        for layer in &mut self.layers {
            match &mut layer.attention {
                AttentionLayer::Recurrent(layer) => {
                    let (state, history) = recurrent.next().ok_or_else(mismatch)?;
                    std::mem::swap(&mut layer.state, state);
                    std::mem::swap(&mut layer.history, history);
                }
                AttentionLayer::Full(layer) => {
                    let (key, value) = kv.next().ok_or_else(mismatch)?;
                    std::mem::swap(&mut layer.key_cache, key);
                    std::mem::swap(&mut layer.value_cache, value);
                }
            }
        }
        std::mem::swap(&mut self.position, &mut parked.position);
        std::mem::swap(&mut self.kv_allocated, &mut parked.kv_allocated);
        std::mem::swap(&mut self.state_id, &mut parked.id);
        match (&mut self.speculation, &mut parked.head) {
            (Some(speculation), Some(head)) => {
                speculation
                    .mtp
                    .swap_sequence(&self.context, head, self.kv_allocated)?;
            }
            (None, None) => {}
            _ => return Err(mismatch()),
        }
        if self.scratch.attention.max_context() < self.kv_allocated as u32 {
            self.scratch.attention =
                AttentionWorkspace::new(&self.context, self.kv_allocated as u32)?;
        }
        self.selection = Selection::None;
        Ok(())
    }

    /// [`Self::reserve_kv`] for a parked sequence.
    fn reserve_parked(&self, state: &mut SequenceState, end: usize) -> crate::Result<()> {
        if end <= state.kv_allocated {
            return Ok(());
        }
        if end > self.info.context {
            return Err(crate::Error::ContextOverflow(
                "Bonsai request exceeds the configured context".into(),
            ));
        }
        let step = state.kv_allocated.min(super::checkpoint::KV_GROWTH_STEP);
        let target = (state.kv_allocated + step).max(end).min(self.info.context);
        let used = state.position;
        let layout = self.kv_layout;
        for (key, value) in &mut state.kv {
            *key = grow_cache(
                &self.context,
                key,
                used * layout.key.token_bytes(),
                target * layout.key.token_bytes(),
            )?;
            *value = grow_cache(
                &self.context,
                value,
                used * layout.value.token_bytes(),
                target * layout.value.token_bytes(),
            )?;
        }
        if let Some(head) = &mut state.head {
            head.grow(&self.context, used, target)?;
        }
        state.kv_allocated = target;
        Ok(())
    }

    /// Decode one token for each row's sequence in a single pass, leaving
    /// each row's logits for [`Self::sample_batch_row`].
    ///
    /// Every row must belong to a different sequence, and at most one row may
    /// be the resident sequence.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn decode_batch(&mut self, rows: &mut [BatchRow<'_>]) -> crate::Result<()> {
        let count = rows.len();
        if count == 0
            || count > MAX_BATCH_SEQUENCES
            || count > self.block_rows
            || rows.iter().filter(|row| row.state.is_none()).count() > 1
            || rows.iter().any(|row| row.token as usize >= VOCAB)
        {
            return Err(crate::Error::InvalidArgument(
                "invalid batched decode rows".into(),
            ));
        }
        if self.batch.is_none() {
            let shaders = ShaderLibrary::new(self.context.device())?;
            self.batch = Some(BatchScratch::new(&self.context, &shaders)?);
        }
        // Grow every sequence's caches for its new row, then make the shared
        // attention workspace cover the longest prefix.
        let mut longest = 0;
        for row in rows.iter_mut() {
            let position = row.state.as_ref().map_or(self.position, |s| s.position);
            if position >= self.info.context {
                return Err(crate::Error::ContextOverflow(
                    "invalid Bonsai token block or position".into(),
                ));
            }
            match row.state.as_deref_mut() {
                Some(state) => self.reserve_parked(state, position + 1)?,
                None => self.reserve_kv(position + 1, None)?,
            }
            longest = longest.max(position + 1);
        }
        if self.scratch.attention.max_context() < longest as u32 {
            self.scratch.attention = AttentionWorkspace::new(&self.context, longest as u32)?;
        }
        let lag_offsets = rows
            .iter_mut()
            .map(|row| {
                row.lag
                    .as_deref_mut()
                    .and_then(|lag| lag.reserve_row(&self.context))
            })
            .collect::<Vec<_>>();
        self.selection = Selection::None;
        let tokens = rows.iter().map(|row| row.token).collect::<Vec<_>>();
        decode_embeddings(
            &self.package,
            &tokens,
            self.scratch.embedding.as_mut_slice::<f32>(),
        )?;
        let buffers = self.row_buffers(rows);
        let rows_u32 = count as u32;
        let scratch = &self.scratch;
        let mut batch = CommandBatch::new_concurrent(&self.context)?;
        self.kernels.transform(
            &mut batch,
            &self.input_rotation,
            &scratch.embedding,
            &scratch.hidden,
            rows_u32,
            HadamardDirection::Inverse,
        )?;
        let mut recurrent_index = 0;
        let mut full_index = 0;
        for (index, layer) in self.layers.iter().enumerate() {
            if index == 1 {
                batch.submit_and_renew(&self.context)?;
            }
            self.normalize_input(&mut batch, &scratch.hidden, &layer.attention_norm, rows_u32)?;
            match &layer.attention {
                AttentionLayer::Recurrent(recurrent) => {
                    self.recurrent_rows(&mut batch, recurrent, &buffers, recurrent_index)?;
                    recurrent_index += 1;
                }
                AttentionLayer::Full(full) => {
                    self.attention_rows(&mut batch, full, &buffers, full_index)?;
                    full_index += 1;
                }
            }
            self.ops.residual_add(
                &mut batch,
                &scratch.branch,
                &scratch.hidden,
                &scratch.hidden,
                WIDTH as u32 * rows_u32,
            )?;
            self.feed_forward(&mut batch, layer, rows_u32)?;
        }
        self.normalize_input(&mut batch, &scratch.hidden, &self.output_norm, rows_u32)?;
        let Some(output) = self.batch.as_ref() else {
            return Err(crate::Error::InvalidArgument(
                "batch scratch missing".into(),
            ));
        };
        self.project(
            &mut batch,
            &self.output,
            &scratch.rotated_hidden,
            &output.logits,
            rows_u32,
        )?;
        let selected = self.device_greedy;
        if selected {
            output.greedy.encode(&mut batch, &output.logits, count)?;
        }
        let row_bytes = WIDTH * size_of::<f32>();
        let copies = rows
            .iter()
            .zip(&lag_offsets)
            .enumerate()
            .filter_map(|(row, (entry, offset))| {
                let lag = entry.lag.as_deref()?;
                Some(BufferCopyRequest {
                    source: &scratch.normalized,
                    source_offset: row * row_bytes,
                    destination: lag.hidden.as_ref()?,
                    destination_offset: (*offset)?,
                    size: row_bytes,
                })
            })
            .collect::<Vec<_>>();
        if !copies.is_empty() {
            batch.blit_buffer_copies(copies)?;
        }
        drop(buffers);
        self.finish(batch)?;
        for (row, offset) in rows.iter_mut().zip(lag_offsets) {
            match row.state.as_deref_mut() {
                Some(state) => state.position += 1,
                None => self.position += 1,
            }
            if let (Some(lag), Some(_)) = (row.lag.as_deref_mut(), offset) {
                lag.tokens.push(row.token);
            }
        }
        let output = self
            .batch
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidArgument("batch scratch missing".into()))?;
        output.rows = count;
        output.selected = selected;
        let nonfinite = if selected {
            output
                .greedy
                .results(count)
                .iter()
                .any(|result| result.nonfinite != 0)
        } else {
            output.logits.as_slice::<f32>()[..count * VOCAB]
                .iter()
                .any(|value| !value.is_finite())
        };
        if nonfinite {
            return Err(crate::Error::Generation("non-finite Bonsai logits".into()));
        }
        Ok(())
    }

    /// Sample row `row` of the last [`Self::decode_batch`].
    pub(crate) fn sample_batch_row(
        &mut self,
        row: usize,
        sampler: &mut Sampler,
    ) -> crate::Result<SamplingResult> {
        let Some(output) = self.batch.as_mut() else {
            return Err(crate::Error::InvalidArgument(
                "no batched step to sample".into(),
            ));
        };
        if row >= output.rows {
            return Err(crate::Error::InvalidArgument(
                "batched row out of range".into(),
            ));
        }
        if output.selected && sampler.selects_argmax() {
            return Ok(sampler.greedy_result(output.greedy.results(output.rows)[row].best_id));
        }
        sampler.sample_buffer(
            &mut output.logits,
            row * VOCAB * size_of::<f32>(),
            &self.context,
            &self.sampling,
        )
    }

    /// Feed the resident sequence's MTP head the rows it missed while decoding
    /// in a batch, so its next speculative round drafts from exact state.
    pub(crate) fn catch_up_head(&mut self, lag: &mut HeadLag) -> crate::Result<()> {
        if lag.tokens.is_empty() {
            return Ok(());
        }
        let Some(mut speculation) = self.speculation.take() else {
            *lag = HeadLag::default();
            return Ok(());
        };
        let result = (|| {
            let hidden = lag
                .hidden
                .as_ref()
                .ok_or_else(|| crate::Error::InvalidArgument("head lag has no rows".into()))?;
            let total = lag.tokens.len();
            let first = self.position.checked_sub(total).ok_or_else(|| {
                crate::Error::InvalidArgument("head lag is ahead of the sequence".into())
            })?;
            let row_bytes = WIDTH * size_of::<f32>();
            let mut done = 0;
            while done < total {
                let rows = (total - done).min(self.block_rows);
                decode_embeddings(
                    &self.package,
                    &lag.tokens[done..done + rows],
                    speculation.mtp.embedding_rows(rows)?,
                )?;
                let mut batch = CommandBatch::new(&self.context)?;
                batch.blit_buffer_copies([BufferCopyRequest {
                    source: hidden,
                    source_offset: done * row_bytes,
                    destination: &self.scratch.normalized,
                    destination_offset: 0,
                    size: rows * row_bytes,
                }])?;
                self.encode_head_rows(&mut batch, &speculation, rows, first + done)?;
                self.finish(batch)?;
                done += rows;
            }
            Ok(())
        })();
        self.speculation = Some(speculation);
        lag.tokens.clear();
        result
    }

    fn row_buffers<'a>(&'a self, rows: &'a [BatchRow<'_>]) -> RowBuffers<'a> {
        let mut buffers = RowBuffers {
            recurrent: Vec::new(),
            kv: Vec::new(),
            positions: Vec::new(),
            capacities: Vec::new(),
        };
        // Layer-major: entry `layer * rows + row`.
        let mut recurrent_index = 0;
        let mut full_index = 0;
        for layer in &self.layers {
            match &layer.attention {
                AttentionLayer::Recurrent(resident) => {
                    for row in rows {
                        buffers.recurrent.push(row.state.as_deref().map_or(
                            (&resident.state, &resident.history),
                            |state| {
                                let (s, h) = &state.recurrent[recurrent_index];
                                (s, h)
                            },
                        ));
                    }
                    recurrent_index += 1;
                }
                AttentionLayer::Full(resident) => {
                    for row in rows {
                        buffers.kv.push(row.state.as_deref().map_or(
                            (&resident.key_cache, &resident.value_cache),
                            |state| {
                                let (k, v) = &state.kv[full_index];
                                (k, v)
                            },
                        ));
                    }
                    full_index += 1;
                }
            }
        }
        for row in rows {
            let (position, capacity) = row
                .state
                .as_deref()
                .map_or((self.position, self.kv_allocated), |state| {
                    (state.position, state.kv_allocated)
                });
            buffers.positions.push(position);
            buffers.capacities.push(capacity);
        }
        buffers
    }

    fn recurrent_rows(
        &self,
        batch: &mut CommandBatch,
        layer: &RecurrentLayer,
        buffers: &RowBuffers<'_>,
        layer_index: usize,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        let rows = buffers.positions.len();
        let count = rows as u32;
        batch.independent(|batch| {
            self.project(
                batch,
                &layer.qkv,
                &scratch.rotated_hidden,
                &scratch.query_gate,
                count,
            )?;
            self.project(
                batch,
                &layer.gate,
                &scratch.rotated_hidden,
                &scratch.gate,
                count,
            )?;
            for (weights, output) in [
                (&layer.alpha, &scratch.alpha),
                (&layer.beta, &scratch.raw_beta),
            ] {
                self.ops.bf16_matmul_rows(
                    batch,
                    decay_projection(weights),
                    &scratch.normalized,
                    output,
                    count,
                )?;
            }
            crate::Result::Ok(())
        })?;
        let sequence = &buffers.recurrent[layer_index * rows..(layer_index + 1) * rows];
        batch.independent(|batch| {
            for (row, &(_, history)) in sequence.iter().enumerate() {
                self.ops.conv_l2_decay_row(
                    batch,
                    &scratch.query_gate,
                    &layer.convolution,
                    history,
                    &scratch.convolved,
                    self.epsilon,
                    [&layer.decay, &scratch.alpha, &layer.dt, &scratch.raw_beta],
                    &scratch.decay,
                    &scratch.beta,
                    row as u32,
                )?;
            }
            crate::Result::Ok(())
        })?;
        batch.independent(|batch| {
            for (row, &(state, _)) in sequence.iter().enumerate() {
                self.ops.gdn_row(
                    batch,
                    self.state_format,
                    &scratch.convolved,
                    &scratch.decay,
                    &scratch.beta,
                    state,
                    &scratch.recurrent_output,
                    row as u32,
                )?;
            }
            crate::Result::Ok(())
        })?;
        self.ops.gdn_postprocess_rows(
            batch,
            &scratch.recurrent_output,
            &scratch.gate,
            &layer.norm,
            &scratch.attention_output,
            self.epsilon,
            count,
        )?;
        self.project_attention(batch, &layer.output, count)
    }

    fn attention_rows(
        &self,
        batch: &mut CommandBatch,
        layer: &FullAttentionLayer,
        buffers: &RowBuffers<'_>,
        layer_index: usize,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        let rows = buffers.positions.len();
        let count = rows as u32;
        batch.independent(|batch| {
            for (weights, output) in [
                (&layer.query_gate, &scratch.query_gate),
                (&layer.key, &scratch.key),
                (&layer.value, &scratch.value),
            ] {
                self.project(batch, weights, &scratch.rotated_hidden, output, count)?;
            }
            crate::Result::Ok(())
        })?;
        let caches = &buffers.kv[layer_index * rows..(layer_index + 1) * rows];
        batch.independent(|batch| {
            for (row, &(key, value)) in caches.iter().enumerate() {
                self.ops.prepare_attention_row_kv(
                    self.kv_layout,
                    batch,
                    [&scratch.query_gate, &scratch.key, &scratch.value],
                    [&layer.query_norm, &layer.key_norm],
                    [&scratch.query, &scratch.gate],
                    (key, value).into(),
                    buffers.positions[row] as u32,
                    buffers.capacities[row] as u32,
                    self.epsilon,
                    self.rope_base,
                    row as u32,
                )?;
            }
            crate::Result::Ok(())
        })?;
        // The rows share one attention workspace, so they stay ordered.
        for (row, &(key, value)) in caches.iter().enumerate() {
            self.ops.attention_row_kv(
                self.kv_layout,
                batch,
                &scratch.query,
                key,
                value,
                Some(&scratch.gate),
                &scratch.attention_output,
                buffers.positions[row] as u32 + 1,
                &scratch.attention,
                row as u32,
            )?;
        }
        self.project_attention(batch, &layer.output, count)
    }
}

fn mismatch() -> crate::Error {
    crate::Error::InvalidArgument("sequence state does not match this model".into())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
