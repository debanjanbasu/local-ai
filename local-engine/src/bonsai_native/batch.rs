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
//! speculative round costs nothing on the GPU. A batched step needs no swap:
//! it binds every sequence's buffers where they are, for its decode rows,
//! for the rows of a prompt chunk prefilled in the same pass, and for its
//! MTP heads drafting together (see `heads`).

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

mod heads;

pub use heads::HeadDraft;

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

    /// Reserve room for `count` more rows, returning the first one's byte
    /// offset.
    fn reserve_rows(
        &mut self,
        context: &local_metal::context::MetalContext,
        count: usize,
    ) -> Option<usize> {
        if self.abandoned {
            return None;
        }
        if self.tokens.len() + count > MAX_HEAD_LAG {
            self.abandoned = true;
            self.tokens = Vec::new();
            self.hidden = None;
            return None;
        }
        let row_bytes = WIDTH * size_of::<f32>();
        let needed = (self.tokens.len() + count) * row_bytes;
        let capacity = self.hidden.as_ref().map_or(0, MetalBuffer::length);
        if needed > capacity {
            let rows = (self.tokens.len() + count).next_power_of_two().max(64);
            let Ok(grown) = MetalBuffer::empty(context.device(), rows * row_bytes) else {
                // Rows that cannot be kept cannot be replayed later either.
                self.abandoned = true;
                self.tokens = Vec::new();
                self.hidden = None;
                return None;
            };
            if let Some(old) = &self.hidden {
                // Host-visible and idle: every batch that wrote it has finished.
                grown.copy_from_bytes(&old.as_slice::<u8>()[..self.tokens.len() * row_bytes], 0);
            }
            self.hidden = Some(grown);
        }
        Some(self.tokens.len() * row_bytes)
    }
}

/// One sequence's rows in a batched step.
pub struct BatchRow<'a> {
    /// The token to decode at the sequence's position.
    pub token: u32,
    /// Draft tokens verified after `token` in the same pass. A sequence with
    /// drafts advances only when [`BonsaiModel::commit_batch`] settles how
    /// many of its rows to keep.
    pub drafts: Vec<u32>,
    /// `None` for the resident sequence.
    pub state: Option<&'a mut SequenceState>,
    /// Where the head's missed rows go, for a speculating model.
    pub lag: Option<&'a mut HeadLag>,
}

impl BatchRow<'_> {
    /// Rows this sequence contributes to the pass.
    const fn rows(&self) -> usize {
        1 + self.drafts.len()
    }
}

/// A prompt chunk prefilled inside a batched step, after the decoding rows.
///
/// Its rows run every layer with the others and update the sequence's state
/// and caches in place, as a prefill block does, but are not projected to
/// logits. The sequence's MTP head ingests them in the same submission.
pub struct PrefillRows<'a> {
    pub tokens: &'a [u32],
    /// `None` for the resident sequence.
    pub state: Option<&'a mut SequenceState>,
}

/// Output buffers of a batched step, made on first use and grown with the
/// rows a step stacks.
pub(super) struct BatchScratch {
    logits: MetalBuffer,
    greedy: GreedyRows,
    /// Rows `logits` and `greedy` hold.
    capacity: usize,
    /// Rows the last batched step produced, and whether `greedy` selected them.
    rows: usize,
    selected: bool,
    /// First row of each sequence in the last batched step.
    starts: Vec<usize>,
    /// What a verifying step keeps for [`BonsaiModel::commit_batch`].
    verify: Option<BatchVerify>,
    /// Buffers of batched MTP drafting.
    heads: Option<heads::HeadDrafts>,
}

impl BatchScratch {
    fn new(context: &local_metal::context::MetalContext, rows: usize) -> crate::Result<Self> {
        let shaders = ShaderLibrary::new(context.device())?;
        Ok(Self {
            logits: MetalBuffer::empty(context.device(), rows * VOCAB * size_of::<f32>())?,
            greedy: GreedyRows::new(context, &shaders, VOCAB, rows)?,
            capacity: rows,
            rows: 0,
            selected: false,
            starts: Vec::new(),
            verify: None,
            heads: None,
        })
    }
}

/// A verifying step's recurrence inputs, kept per recurrent layer for every
/// row so a sequence's committed rows can be replayed from its start state.
///
/// A sequence with drafts reads its state and history and writes the block's
/// final ones to the shared spares, which nobody reads: its own buffers keep
/// the start of the round, and the commit replays exactly the rows it keeps
/// (all of them when every draft is accepted), as a single sequence's partial
/// commit does.
struct BatchVerify {
    layers: Vec<VerifyLayer>,
    spare_state: MetalBuffer,
    spare_history: MetalBuffer,
    rows: usize,
}

struct VerifyLayer {
    /// Raw QKV projection rows.
    inputs: MetalBuffer,
    decay: MetalBuffer,
    beta: MetalBuffer,
}

impl BatchVerify {
    fn new(
        context: &local_metal::context::MetalContext,
        format: local_metal::bonsai_ops::GdnStateFormat,
        layers: usize,
        rows: usize,
    ) -> crate::Result<Self> {
        let empty = |bytes| MetalBuffer::empty(context.device(), bytes);
        let floats = |width: usize| empty(rows * width * size_of::<f32>());
        Ok(Self {
            layers: (0..layers)
                .map(|_| {
                    Ok(VerifyLayer {
                        inputs: floats(QKV_WIDTH)?,
                        decay: floats(GDN_HEADS)?,
                        beta: floats(GDN_HEADS)?,
                    })
                })
                .collect::<crate::Result<Vec<_>>>()?,
            spare_state: empty(format.state_bytes())?,
            spare_history: empty(CONV_STATE_BYTES)?,
            rows,
        })
    }
}

/// Raw QKV projection width of a recurrent layer, and its head count.
const QKV_WIDTH: usize = 10_240;
const GDN_HEADS: usize = 48;

/// Most rows one batched step stacks: every sequence's seed and drafts.
pub const MAX_BATCH_ROWS: usize = 64;

/// Per-sequence view of the sequence-specific buffers of every layer.
struct RowBuffers<'a> {
    recurrent: Vec<(&'a MetalBuffer, &'a MetalBuffer)>,
    kv: Vec<(&'a MetalBuffer, &'a MetalBuffer)>,
    positions: Vec<usize>,
    capacities: Vec<usize>,
    /// First row and row count of each sequence in the stacked block.
    spans: Vec<(usize, usize)>,
    /// Whether each sequence's rows advance its own state in place (a prompt
    /// chunk) rather than leave it for a commit to replay (verify rows).
    in_place: Vec<bool>,
    /// Total stacked rows.
    rows: usize,
}

impl RowBuffers<'_> {
    const fn sequences(&self) -> usize {
        self.positions.len()
    }
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

    /// [`Self::decode_batch_prefilling`] without a prompt chunk.
    #[cfg(test)]
    pub(crate) fn decode_batch(&mut self, rows: &mut [BatchRow<'_>]) -> crate::Result<()> {
        self.decode_batch_prefilling(rows, None)
    }

    /// Decode every row's sequence in a single pass: its token, and after it
    /// any drafts to verify, leaving each row's logits for
    /// [`Self::sample_batch_row`] and [`Self::verify_batch_row`]; and, after
    /// every row, one sequence's prompt chunk (see [`PrefillRows`]), whose
    /// position advances by the chunk.
    ///
    /// A sequence without drafts advances by its token. A sequence with
    /// drafts is left at its position, its recurrent state at the start of
    /// the round, until [`Self::commit_batch`] keeps the rows it accepted.
    ///
    /// Every row must belong to a different sequence, and at most one of
    /// them, the prompt's included, may be the resident sequence.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn decode_batch_prefilling(
        &mut self,
        rows: &mut [BatchRow<'_>],
        mut prefill: Option<PrefillRows<'_>>,
    ) -> crate::Result<()> {
        let count = rows.len();
        let decoded = rows.iter().map(BatchRow::rows).sum::<usize>();
        let prefilled = prefill.as_ref().map_or(0, |chunk| chunk.tokens.len());
        let total = decoded + prefilled;
        let residents = rows.iter().filter(|row| row.state.is_none()).count()
            + usize::from(prefill.as_ref().is_some_and(|chunk| chunk.state.is_none()));
        if count == 0
            || count > MAX_BATCH_SEQUENCES
            || total > MAX_BATCH_ROWS
            || total > self.block_rows
            || residents > 1
            || prefill.as_ref().is_some_and(|chunk| {
                chunk.tokens.is_empty()
                    || chunk.tokens.iter().any(|&token| token as usize >= VOCAB)
                    || self.speculation.as_ref().is_some_and(|speculation| {
                        chunk.tokens.len() > speculation.mtp.scratch_rows()
                    })
            })
            || rows.iter().any(|row| {
                std::iter::once(row.token)
                    .chain(row.drafts.iter().copied())
                    .any(|token| token as usize >= VOCAB)
            })
        {
            return Err(crate::Error::InvalidArgument(
                "invalid batched decode rows".into(),
            ));
        }
        let verifying = rows.iter().any(|row| !row.drafts.is_empty());
        self.reserve_batch(decoded, verifying.then_some(total))?;
        // Grow every sequence's caches for its new rows, then make the shared
        // attention workspace cover the longest prefix.
        let mut longest = 0;
        for row in rows.iter_mut() {
            let position = row.state.as_ref().map_or(self.position, |s| s.position);
            let end = position + row.rows();
            if end > self.info.context {
                return Err(crate::Error::ContextOverflow(
                    "invalid Bonsai token block or position".into(),
                ));
            }
            match row.state.as_deref_mut() {
                Some(state) => self.reserve_parked(state, end)?,
                None => self.reserve_kv(end, None)?,
            }
            longest = longest.max(end);
        }
        if let Some(chunk) = prefill.as_mut() {
            let position = chunk.state.as_ref().map_or(self.position, |s| s.position);
            let end = position + chunk.tokens.len();
            if end > self.info.context {
                return Err(crate::Error::ContextOverflow(
                    "invalid Bonsai token block or position".into(),
                ));
            }
            match chunk.state.as_deref_mut() {
                Some(state) => self.reserve_parked(state, end)?,
                None => self.reserve_kv(end, None)?,
            }
            if let Some(speculation) = self.speculation.as_mut() {
                speculation.mtp.ensure_attention(&self.context, end)?;
            }
            longest = longest.max(end);
        }
        if self.scratch.attention.max_context() < longest as u32 {
            self.scratch.attention = AttentionWorkspace::new(&self.context, longest as u32)?;
        }
        let lag_offsets = rows
            .iter_mut()
            .map(|row| {
                let count = row.rows();
                row.lag
                    .as_deref_mut()
                    .and_then(|lag| lag.reserve_rows(&self.context, count))
            })
            .collect::<Vec<_>>();
        self.selection = Selection::None;
        let tokens = rows
            .iter()
            .flat_map(|row| std::iter::once(row.token).chain(row.drafts.iter().copied()))
            .chain(
                prefill
                    .iter()
                    .flat_map(|chunk| chunk.tokens.iter().copied()),
            )
            .collect::<Vec<_>>();
        decode_embeddings(
            &self.package,
            &tokens,
            self.scratch.embedding.as_mut_slice::<f32>(),
        )?;
        let buffers = self.row_buffers(rows, prefill.as_ref());
        let rows_u32 = total as u32;
        let scratch = &self.scratch;
        let Some(output) = self.batch.as_ref() else {
            return Err(crate::Error::InvalidArgument(
                "batch scratch missing".into(),
            ));
        };
        let verify = output.verify.as_ref().filter(|_| verifying);
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
                    let keep = verify.map(|verify| (verify, &verify.layers[recurrent_index]));
                    self.recurrent_rows(&mut batch, recurrent, &buffers, recurrent_index, keep)?;
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
        // Only the decoding rows, which come first, are projected to logits.
        self.project(
            &mut batch,
            &self.output,
            &scratch.rotated_hidden,
            &output.logits,
            decoded as u32,
        )?;
        let selected = self.device_greedy;
        if selected {
            output.greedy.encode(&mut batch, &output.logits, decoded)?;
        }
        let row_bytes = WIDTH * size_of::<f32>();
        let copies = rows
            .iter()
            .zip(&lag_offsets)
            .zip(&buffers.spans)
            .filter_map(|((entry, offset), &(start, length))| {
                let lag = entry.lag.as_deref()?;
                Some(BufferCopyRequest {
                    source: &scratch.normalized,
                    source_offset: start * row_bytes,
                    destination: lag.hidden.as_ref()?,
                    destination_offset: (*offset)?,
                    size: length * row_bytes,
                })
            })
            .collect::<Vec<_>>();
        if !copies.is_empty() {
            batch.blit_buffer_copies(copies)?;
        }
        if let Some(chunk) = prefill.as_ref() {
            self.ingest_prefill_rows(&mut batch, chunk, decoded)?;
        }
        let spans = buffers.spans.clone();
        drop(buffers);
        self.finish(batch)?;
        if let Some(chunk) = prefill.as_mut() {
            match chunk.state.as_deref_mut() {
                Some(state) => state.position += chunk.tokens.len(),
                None => self.position += chunk.tokens.len(),
            }
        }
        for (row, offset) in rows.iter_mut().zip(lag_offsets) {
            if !row.drafts.is_empty() {
                continue;
            }
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
        output.rows = decoded;
        output.selected = selected;
        output.starts = spans[..count].iter().map(|&(start, _)| start).collect();
        let nonfinite = if selected {
            output
                .greedy
                .results(decoded)
                .iter()
                .any(|result| result.nonfinite != 0)
        } else {
            output.logits.as_slice::<f32>()[..decoded * VOCAB]
                .iter()
                .any(|value| !value.is_finite())
        };
        if nonfinite {
            return Err(crate::Error::Generation("non-finite Bonsai logits".into()));
        }
        Ok(())
    }

    /// Most prompt rows one batched pass may prefill beside its decoding
    /// rows: what the MTP head ingests in one pass, when speculating.
    pub(crate) fn max_prefill_fold(&self) -> usize {
        self.speculation
            .as_ref()
            .map_or(MAX_BATCH_ROWS, |speculation| speculation.mtp.scratch_rows())
    }

    /// Most rows one [`Self::decode_batch_prefilling`] may stack.
    pub(crate) fn max_batch_rows(&self) -> usize {
        MAX_BATCH_ROWS.min(self.block_rows)
    }

    /// Make the batch scratch hold `rows` rows of logits, and a verifying
    /// step's recurrence inputs for `verifying` stacked rows.
    fn reserve_batch(&mut self, rows: usize, verifying: Option<usize>) -> crate::Result<()> {
        let capacity = self.batch.as_ref().map_or(0, |output| output.capacity);
        if capacity < rows {
            let rows = rows
                .next_power_of_two()
                .clamp(MAX_BATCH_SEQUENCES, MAX_BATCH_ROWS);
            let (verify, heads) = self
                .batch
                .take()
                .map_or((None, None), |output| (output.verify, output.heads));
            let mut output = BatchScratch::new(&self.context, rows)?;
            output.verify = verify;
            output.heads = heads;
            self.batch = Some(output);
        }
        let Some(output) = self.batch.as_mut() else {
            return Err(crate::Error::InvalidArgument(
                "batch scratch missing".into(),
            ));
        };
        if let Some(rows) = verifying
            && output
                .verify
                .as_ref()
                .is_none_or(|verify| verify.rows < rows)
        {
            output.verify = None;
            let layers = self
                .layers
                .iter()
                .filter(|layer| matches!(layer.attention, AttentionLayer::Recurrent(_)))
                .count();
            output.verify = Some(BatchVerify::new(
                &self.context,
                self.state_format,
                layers,
                rows.next_power_of_two().min(MAX_BATCH_ROWS),
            )?);
        }
        Ok(())
    }

    /// Sample row `row` of the last [`Self::decode_batch_prefilling`], counting every
    /// sequence's seed and draft rows.
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

    /// First row of sequence `sequence` in the last [`Self::decode_batch_prefilling`].
    pub(crate) fn batch_row_start(&self, sequence: usize) -> crate::Result<usize> {
        self.batch
            .as_ref()
            .and_then(|output| output.starts.get(sequence).copied())
            .ok_or_else(|| crate::Error::InvalidArgument("batched sequence out of range".into()))
    }

    /// Verify sequence `sequence`'s drafts against its rows of the last
    /// [`Self::decode_batch_prefilling`], exactly as a single sequence's verify block is
    /// verified.
    pub(crate) fn verify_batch_row(
        &mut self,
        sequence: usize,
        drafts: &[u32],
        sampler: &mut Sampler,
    ) -> crate::Result<crate::sampler::Verification> {
        let start = self.batch_row_start(sequence)?;
        crate::sampler::verify_greedy_drafts(sampler, drafts, |row, sampler| {
            self.sample_batch_row(start + row, sampler)
        })
    }

    /// Settle the sequences that verified drafts in the last
    /// [`Self::decode_batch_prefilling`] (with the same `rows`): each keeps its first
    /// `committed[index]` rows. Their recurrent state and history replay
    /// those rows from the start of the round, in one submission for every
    /// sequence; positions and head lags advance by the kept rows. Entries of
    /// sequences without drafts are ignored.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn commit_batch(
        &mut self,
        rows: &mut [BatchRow<'_>],
        committed: &[usize],
    ) -> crate::Result<()> {
        if committed.len() != rows.len()
            || rows
                .iter()
                .zip(committed)
                .any(|(row, &kept)| !row.drafts.is_empty() && !(1..=row.rows()).contains(&kept))
        {
            return Err(crate::Error::InvalidArgument(
                "batched commit needs 1..=rows kept rows per verifying sequence".into(),
            ));
        }
        if rows.iter().all(|row| row.drafts.is_empty()) {
            return Ok(());
        }
        let verify = self
            .batch
            .as_ref()
            .and_then(|output| output.verify.as_ref())
            .ok_or_else(|| crate::Error::InvalidArgument("no verifying batched step".into()))?;
        let starts = self
            .batch
            .as_ref()
            .map(|output| output.starts.clone())
            .unwrap_or_default();
        if starts.len() != rows.len() {
            return Err(crate::Error::InvalidArgument(
                "batched commit does not match the last step".into(),
            ));
        }
        let buffers = self.row_buffers(rows, None);
        let sequences = buffers.sequences();
        let scratch = &self.scratch;
        let mut batch = CommandBatch::new_concurrent(&self.context)?;
        let recurrent = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attention {
                AttentionLayer::Recurrent(layer) => Some(layer),
                AttentionLayer::Full(_) => None,
            });
        for (index, (layer, kept)) in recurrent.zip(&verify.layers).enumerate() {
            let replays = (0..sequences)
                .filter(|&sequence| !rows[sequence].drafts.is_empty())
                .map(|sequence| {
                    let (state, history) = buffers.recurrent[index * sequences + sequence];
                    (
                        starts[sequence] as u32,
                        committed[sequence] as u32,
                        state,
                        history,
                    )
                })
                .collect::<Vec<_>>();
            batch.independent(|batch| {
                for &(start, length, _, history) in &replays {
                    self.ops.conv_rows_at(
                        batch,
                        &kept.inputs,
                        &layer.convolution,
                        history,
                        &scratch.convolved,
                        start,
                        length,
                    )?;
                }
                crate::Result::Ok(())
            })?;
            batch.independent(|batch| {
                for &(start, length, _, _) in &replays {
                    self.ops.l2_normalize_qk_rows_at(
                        batch,
                        &scratch.convolved,
                        self.epsilon,
                        start,
                        length,
                    )?;
                }
                crate::Result::Ok(())
            })?;
            batch.independent(|batch| {
                for &(start, length, state, _) in &replays {
                    self.ops.gdn_rows_at(
                        batch,
                        self.state_format,
                        [&scratch.convolved, &kept.decay, &kept.beta],
                        [state, state],
                        &scratch.recurrent_output,
                        start,
                        length,
                    )?;
                }
                crate::Result::Ok(())
            })?;
        }
        drop(buffers);
        self.finish(batch)?;
        for (row, &kept) in rows.iter_mut().zip(committed) {
            if row.drafts.is_empty() {
                continue;
            }
            match row.state.as_deref_mut() {
                Some(state) => state.position += kept,
                None => self.position += kept,
            }
            if let Some(lag) = row.lag.as_deref_mut()
                && lag.usable()
            {
                lag.tokens.push(row.token);
                lag.tokens.extend_from_slice(&row.drafts[..kept - 1]);
            }
        }
        Ok(())
    }

    /// Draft up to the head's depth after `seed` for the resident sequence,
    /// as its single-sequence speculative round drafts, first feeding its head
    /// the rows it missed in batched steps. The drafts are verified in a
    /// batched step; their rows reach the head through `lag` again.
    pub(crate) fn draft_resident(
        &mut self,
        lag: &mut HeadLag,
        seed: u32,
        sampler: &Sampler,
        remaining: usize,
    ) -> crate::Result<Vec<u32>> {
        if !lag.usable() {
            return Ok(Vec::new());
        }
        self.catch_up_head(lag)?;
        let Some(mut speculation) = self.speculation.take() else {
            return Ok(Vec::new());
        };
        let result = (|| {
            let depth = super::draft_depth(
                speculation.mtp.depth,
                remaining,
                self.info.context,
                self.position,
            );
            if depth == 0 {
                return Ok(Vec::new());
            }
            self.reserve_kv(self.position + depth + 1, Some(&mut speculation))?;
            let mut draft_sampler = sampler.greedy_draft();
            let drafts = if draft_sampler.applies_penalties() {
                self.draft_on_host(
                    &mut speculation,
                    &mut draft_sampler,
                    seed,
                    depth,
                    super::DRAFT_CHAIN_MIN_MARGIN,
                )?
            } else {
                self.draft_on_device(
                    &mut speculation,
                    &draft_sampler,
                    seed,
                    depth,
                    super::DRAFT_CHAIN_MIN_MARGIN,
                )?
            };
            Ok(drafts.iter().map(|draft| draft.token).collect())
        })();
        self.speculation = Some(speculation);
        result
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

    /// The buffers of every sequence of a pass: each row's, then the
    /// prefilling sequence's.
    fn row_buffers<'a>(
        &'a self,
        rows: &'a [BatchRow<'_>],
        prefill: Option<&'a PrefillRows<'_>>,
    ) -> RowBuffers<'a> {
        let mut buffers = RowBuffers {
            recurrent: Vec::new(),
            kv: Vec::new(),
            positions: Vec::new(),
            capacities: Vec::new(),
            spans: Vec::new(),
            in_place: Vec::new(),
            rows: 0,
        };
        let sequences = rows
            .iter()
            .map(|row| (row.state.as_deref(), row.rows(), false))
            .chain(
                prefill
                    .iter()
                    .map(|chunk| (chunk.state.as_deref(), chunk.tokens.len(), true)),
            )
            .collect::<Vec<_>>();
        // Layer-major: entry `layer * sequences + sequence`.
        let mut recurrent_index = 0;
        let mut full_index = 0;
        for layer in &self.layers {
            match &layer.attention {
                AttentionLayer::Recurrent(resident) => {
                    for &(state, _, _) in &sequences {
                        buffers.recurrent.push(state.map_or(
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
                    for &(state, _, _) in &sequences {
                        buffers.kv.push(state.map_or(
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
        for &(state, length, in_place) in &sequences {
            let (position, capacity) = state.map_or((self.position, self.kv_allocated), |state| {
                (state.position, state.kv_allocated)
            });
            buffers.positions.push(position);
            buffers.capacities.push(capacity);
            buffers.spans.push((buffers.rows, length));
            buffers.in_place.push(in_place);
            buffers.rows += length;
        }
        buffers
    }

    /// Feed a prompt chunk's rows of a batched pass (at stacked row `start`,
    /// output-normalized in `scratch.normalized`) to its sequence's MTP head,
    /// as a prefill block's head ingestion does: K/V rows only, then the
    /// newest row's hidden becomes the head's committed hidden.
    fn ingest_prefill_rows(
        &self,
        batch: &mut CommandBatch,
        chunk: &PrefillRows<'_>,
        start: usize,
    ) -> crate::Result<()> {
        let Some(speculation) = self.speculation.as_ref() else {
            return Ok(());
        };
        let mtp = &speculation.mtp;
        let (state, position, capacity) = match chunk.state.as_deref() {
            Some(parked) => (
                parked.head.as_ref().ok_or_else(mismatch)?.state(),
                parked.position,
                parked.kv_allocated,
            ),
            None => (mtp.state(), self.position, self.kv_allocated),
        };
        let rows = chunk.tokens.len();
        let mut values = vec![0.0; rows * WIDTH];
        decode_embeddings(&self.package, chunk.tokens, &mut values)?;
        mtp.stage_embeddings(&values)?;
        let row_bytes = WIDTH * size_of::<f32>();
        let mut copies = vec![BufferCopyRequest {
            source: state.prev_hidden,
            source_offset: 0,
            destination: mtp.hidden_in(),
            destination_offset: 0,
            size: row_bytes,
        }];
        if rows > 1 {
            copies.push(BufferCopyRequest {
                source: &self.scratch.normalized,
                source_offset: start * row_bytes,
                destination: mtp.hidden_in(),
                destination_offset: row_bytes,
                size: (rows - 1) * row_bytes,
            });
        }
        batch.blit_buffer_copies(copies)?;
        let shared = self.mtp_shared();
        mtp.transform_embeddings(batch, &shared, rows)?;
        let span = crate::bonsai_mtp::HeadSpan {
            state,
            capacity,
            row: 0,
            rows,
            position,
        };
        mtp.encode_stacked(
            batch,
            &shared,
            mtp.hidden_in(),
            &[],
            &[span],
            rows,
            mtp.hidden_in(),
        )?;
        batch.blit_buffer_copies([BufferCopyRequest {
            source: &self.scratch.normalized,
            source_offset: (start + rows - 1) * row_bytes,
            destination: state.prev_hidden,
            destination_offset: 0,
            size: row_bytes,
        }])?;
        Ok(())
    }

    /// One recurrent layer over the stacked rows. A sequence's single row
    /// updates its state and history in place. A sequence's verify rows read
    /// them and leave them as they were (the block's final ones go to the
    /// spares); with `keep` the layer's raw QKV rows, decay and beta are kept
    /// there for [`Self::commit_batch`] to replay.
    #[allow(clippy::too_many_lines)]
    fn recurrent_rows(
        &self,
        batch: &mut CommandBatch,
        layer: &RecurrentLayer,
        buffers: &RowBuffers<'_>,
        layer_index: usize,
        keep: Option<(&BatchVerify, &VerifyLayer)>,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        let sequences = buffers.sequences();
        let count = buffers.rows as u32;
        let (raw, decay, beta) = keep.map_or(
            (&scratch.query_gate, &scratch.decay, &scratch.beta),
            |(_, kept)| (&kept.inputs, &kept.decay, &kept.beta),
        );
        batch.independent(|batch| {
            self.project(batch, &layer.qkv, &scratch.rotated_hidden, raw, count)?;
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
        let sequence = &buffers.recurrent[layer_index * sequences..(layer_index + 1) * sequences];
        let inputs = [&layer.decay, &scratch.alpha, &layer.dt, &scratch.raw_beta];
        batch.independent(|batch| {
            for ((&(start, length), &(_, history)), &in_place) in
                buffers.spans.iter().zip(sequence).zip(&buffers.in_place)
            {
                if length == 1 {
                    self.ops.conv_l2_decay_row(
                        batch,
                        raw,
                        &layer.convolution,
                        history,
                        &scratch.convolved,
                        self.epsilon,
                        inputs,
                        decay,
                        beta,
                        start as u32,
                    )?;
                } else {
                    let spare = if in_place {
                        history
                    } else {
                        &keep
                            .ok_or_else(|| {
                                crate::Error::InvalidArgument("verify rows without spares".into())
                            })?
                            .0
                            .spare_history
                    };
                    self.ops.conv_l2_decay_rows_at(
                        batch,
                        raw,
                        &layer.convolution,
                        [history, spare],
                        &scratch.convolved,
                        self.epsilon,
                        inputs,
                        [decay, beta],
                        start as u32,
                        length as u32,
                    )?;
                }
            }
            crate::Result::Ok(())
        })?;
        batch.independent(|batch| {
            for ((&(start, length), &(state, _)), &in_place) in
                buffers.spans.iter().zip(sequence).zip(&buffers.in_place)
            {
                if length == 1 {
                    self.ops.gdn_row(
                        batch,
                        self.state_format,
                        &scratch.convolved,
                        decay,
                        beta,
                        state,
                        &scratch.recurrent_output,
                        start as u32,
                    )?;
                } else {
                    let spare = if in_place {
                        state
                    } else {
                        &keep
                            .ok_or_else(|| {
                                crate::Error::InvalidArgument("verify rows without spares".into())
                            })?
                            .0
                            .spare_state
                    };
                    self.ops.gdn_rows_at(
                        batch,
                        self.state_format,
                        [&scratch.convolved, decay, beta],
                        [state, spare],
                        &scratch.recurrent_output,
                        start as u32,
                        length as u32,
                    )?;
                }
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
        let sequences = buffers.sequences();
        let count = buffers.rows as u32;
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
        let caches = &buffers.kv[layer_index * sequences..(layer_index + 1) * sequences];
        batch.independent(|batch| {
            for (sequence, &(key, value)) in caches.iter().enumerate() {
                let (start, length) = buffers.spans[sequence];
                let position = buffers.positions[sequence] as u32;
                let capacity = buffers.capacities[sequence] as u32;
                if length == 1 {
                    self.ops.prepare_attention_row_kv(
                        self.kv_layout,
                        batch,
                        [&scratch.query_gate, &scratch.key, &scratch.value],
                        [&layer.query_norm, &layer.key_norm],
                        [&scratch.query, &scratch.gate],
                        (key, value).into(),
                        position,
                        capacity,
                        self.epsilon,
                        self.rope_base,
                        start as u32,
                    )?;
                } else {
                    self.ops.prepare_attention_rows_kv_at(
                        self.kv_layout,
                        batch,
                        [&scratch.query_gate, &scratch.key, &scratch.value],
                        [&layer.query_norm, &layer.key_norm],
                        [&scratch.query, &scratch.gate],
                        [key, value],
                        [position, capacity],
                        [self.epsilon, self.rope_base],
                        start as u32,
                        length as u32,
                    )?;
                }
            }
            crate::Result::Ok(())
        })?;
        // The sequences share one attention workspace, so they stay ordered.
        for (sequence, &(key, value)) in caches.iter().enumerate() {
            let (start, length) = buffers.spans[sequence];
            let position = buffers.positions[sequence] as u32;
            if length == 1 {
                self.ops.attention_row_kv(
                    self.kv_layout,
                    batch,
                    &scratch.query,
                    key,
                    value,
                    Some(&scratch.gate),
                    &scratch.attention_output,
                    position + 1,
                    &scratch.attention,
                    start as u32,
                )?;
            } else {
                self.ops.attention_block_kv_at(
                    self.kv_layout,
                    batch,
                    &scratch.query,
                    [key, value],
                    &scratch.gate,
                    &scratch.attention_output,
                    position,
                    length as u32,
                    &scratch.attention,
                    start as u32,
                )?;
            }
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
