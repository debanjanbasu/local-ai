//! MTP drafting for several batched sequences at once.
//!
//! Every sequence of a batch owns its head caches and committed hidden, the
//! resident one in the model's head and the parked ones in their
//! [`SequenceState`]; nothing is swapped. A draft round runs one stacked head
//! pass per draft depth with one row per drafting sequence: the head's
//! matrices read their weights once for all of them, each row attends over
//! its own sequence's head caches, and the next tokens are selected on the
//! GPU, so each depth costs about what one sequence's step costs.
//!
//! The first pass also feeds each head the rows it missed while its sequence
//! decoded in batched steps (its `HeadLag`), as K/V-only rows beside the
//! drafting rows; a lag too long for one pass is fed first, in K/V-only
//! passes.

use local_metal::draft::{DraftRows, DraftTopTwo};

use super::{BonsaiModel, HeadLag, MAX_BATCH_SEQUENCES, SequenceState};
use crate::bonsai::{BonsaiMetalTensor, VOCAB, WIDTH};
use crate::bonsai_mtp::{DRAFT_CHAIN_MIN_MARGIN, HeadSpan, HeadState};
use crate::bonsai_native::{BufferCopyRequest, CommandBatch, MetalBuffer, decode_embeddings};
use crate::sampler::Sampler;

/// One sequence's request for drafts in `BonsaiModel::draft_heads`.
pub struct HeadDraft<'a> {
    /// The sampled token the drafts follow, not yet decoded.
    pub seed: u32,
    /// Most drafts to propose; the chain also stops at EOS or at a draft the
    /// head is unsure of, as a single sequence's does.
    pub depth: usize,
    /// `None` for the resident sequence.
    pub state: Option<&'a mut SequenceState>,
    pub lag: &'a mut HeadLag,
    /// The request's sampler; drafts are its greedy selection, which must
    /// apply no penalties.
    pub sampler: &'a Sampler,
}

/// Buffers of a batched draft round, made on first use.
pub(super) struct HeadDrafts {
    rows: DraftRows,
    /// The target's `PTQ1_0` `token_embd`, bound for the GPU gather.
    embeddings: BonsaiMetalTensor,
    /// Sequence `i`'s seed at `i * stride`, its step-`k` draft at
    /// `i * stride + k + 1`.
    tokens: MetalBuffer,
    /// Sequence `i`'s step-`k` top two at `i * depth + k`.
    results: MetalBuffer,
    /// One row of draft logits per drafting sequence.
    logits: MetalBuffer,
    depth: usize,
}

impl HeadDrafts {
    fn new(model: &BonsaiModel, depth: usize) -> crate::Result<Self> {
        let context = &model.context;
        let shaders = local_metal::shaders::ShaderLibrary::new(context.device())?;
        let depth = depth.max(1);
        let empty = |bytes| MetalBuffer::empty(context.device(), bytes);
        Ok(Self {
            rows: DraftRows::new(context, &shaders, VOCAB, MAX_BATCH_SEQUENCES)?,
            embeddings: model
                .package
                .metal_tensor(context.device(), "token_embd.weight")?,
            tokens: empty(MAX_BATCH_SEQUENCES * (depth + 1) * size_of::<u32>())?,
            results: empty(MAX_BATCH_SEQUENCES * depth * size_of::<DraftTopTwo>())?,
            logits: empty(MAX_BATCH_SEQUENCES * VOCAB * size_of::<f32>())?,
            depth,
        })
    }

    const fn stride(&self) -> usize {
        self.depth + 1
    }
}

/// What a draft round knows of one drafting sequence.
struct Drafting<'a> {
    seed: u32,
    state: HeadState<'a>,
    position: usize,
    capacity: usize,
    lag_tokens: &'a [u32],
    lag_hidden: Option<&'a MetalBuffer>,
    /// Lag rows already fed by an earlier pass of this round.
    fed: usize,
    depth: usize,
}

impl Drafting<'_> {
    /// Position of the first lag row.
    const fn first(&self) -> usize {
        self.position - self.lag_tokens.len()
    }

    /// Copies staging `rows` rows of head hidden input at stacked row `row`
    /// for lag rows `from..from + rows`: lag row `j` reads the target hidden
    /// of the row before it, the committed `prev_hidden` for the first.
    fn stage<'b>(
        &'b self,
        destination: &'b MetalBuffer,
        row: usize,
        from: usize,
        rows: usize,
        copies: &mut Vec<BufferCopyRequest<'b>>,
    ) -> crate::Result<()> {
        let row_bytes = WIDTH * size_of::<f32>();
        let mut row = row;
        let mut from = from;
        let mut rows = rows;
        if from == 0 && rows > 0 {
            copies.push(BufferCopyRequest {
                source: self.state.prev_hidden,
                source_offset: 0,
                destination,
                destination_offset: row * row_bytes,
                size: row_bytes,
            });
            row += 1;
            from += 1;
            rows -= 1;
        }
        if rows > 0 {
            let hidden = self
                .lag_hidden
                .ok_or_else(|| crate::Error::InvalidArgument("head lag has no rows".into()))?;
            copies.push(BufferCopyRequest {
                source: hidden,
                source_offset: (from - 1) * row_bytes,
                destination,
                destination_offset: row * row_bytes,
                size: rows * row_bytes,
            });
        }
        Ok(())
    }
}

impl BonsaiModel {
    /// Draft for several sequences together, each chain exactly as long as
    /// its single-sequence round would make it, and feed each head the rows
    /// it missed. Returns each entry's drafts, empty for a sequence whose
    /// head was given up.
    ///
    /// At most one entry may be the resident sequence. Drafting reads and
    /// writes only head state: positions and the target are untouched.
    pub(crate) fn draft_heads(
        &mut self,
        entries: &mut [HeadDraft<'_>],
    ) -> crate::Result<Vec<Vec<u32>>> {
        if entries.len() > MAX_BATCH_SEQUENCES
            || entries.iter().filter(|entry| entry.state.is_none()).count() > 1
        {
            return Err(crate::Error::InvalidArgument(
                "invalid batched draft rows".into(),
            ));
        }
        let Some(mut speculation) = self.speculation.take() else {
            return Ok(vec![Vec::new(); entries.len()]);
        };
        let result = self
            .prepare_head_drafts(&mut speculation, entries)
            .and_then(|()| self.run_head_drafts(&speculation, entries));
        self.speculation = Some(speculation);
        for entry in entries.iter_mut() {
            if entry.depth > 0 && entry.lag.usable() {
                entry.lag.tokens.clear();
            }
        }
        result
    }

    /// Grow every drafting sequence's caches through its last draft row and
    /// the verify rows after it, as [`Self::decode_batch_prefilling`] would, and make
    /// the round's buffers.
    fn prepare_head_drafts(
        &mut self,
        speculation: &mut super::super::Speculation,
        entries: &mut [HeadDraft<'_>],
    ) -> crate::Result<()> {
        let max_depth = speculation.mtp.depth;
        let mut longest = 0;
        for entry in entries.iter_mut() {
            entry.depth = entry.depth.min(max_depth);
            if entry.depth == 0 || !entry.lag.usable() {
                continue;
            }
            if entry.sampler.greedy_draft().applies_penalties() {
                return Err(crate::Error::InvalidArgument(
                    "batched drafting cannot apply penalties".into(),
                ));
            }
            let position = entry.state.as_ref().map_or(self.position, |s| s.position);
            if entry.lag.tokens.len() > position {
                return Err(crate::Error::InvalidArgument(
                    "head lag is ahead of the sequence".into(),
                ));
            }
            let end = position + entry.depth + 1;
            match entry.state.as_deref_mut() {
                Some(state) => self.reserve_parked(state, end)?,
                None => self.reserve_kv(end, Some(speculation))?,
            }
            longest = longest.max(end);
        }
        speculation.mtp.ensure_attention(&self.context, longest)?;
        if self
            .batch
            .as_ref()
            .is_none_or(|output| output.heads.is_none())
        {
            self.reserve_batch(MAX_BATCH_SEQUENCES, None)?;
            let heads = HeadDrafts::new(self, max_depth)?;
            if let Some(output) = self.batch.as_mut() {
                output.heads = Some(heads);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn run_head_drafts(
        &self,
        speculation: &super::super::Speculation,
        entries: &[HeadDraft<'_>],
    ) -> crate::Result<Vec<Vec<u32>>> {
        let mut drafts = vec![Vec::new(); entries.len()];
        let heads = self
            .batch
            .as_ref()
            .and_then(|output| output.heads.as_ref())
            .ok_or_else(|| crate::Error::InvalidArgument("batch scratch missing".into()))?;
        let mtp = &speculation.mtp;
        let mut slots = Vec::new();
        let mut samplers = Vec::new();
        let mut sequences = Vec::new();
        for (slot, entry) in entries.iter().enumerate() {
            if entry.depth == 0 || !entry.lag.usable() {
                continue;
            }
            let (state, position, capacity) = match entry.state.as_deref() {
                Some(parked) => (
                    parked.head.as_ref().ok_or_else(super::mismatch)?.state(),
                    parked.position,
                    parked.kv_allocated,
                ),
                None => (mtp.state(), self.position, self.kv_allocated),
            };
            slots.push(slot);
            samplers.push(entry.sampler.greedy_draft());
            sequences.push(Drafting {
                seed: entry.seed,
                state,
                position,
                capacity,
                lag_tokens: &entry.lag.tokens,
                lag_hidden: entry.lag.hidden.as_ref(),
                fed: 0,
                depth: entry.depth,
            });
        }
        if sequences.is_empty() {
            return Ok(drafts);
        }
        let shared = self.mtp_shared();
        let limit = mtp.scratch_rows();
        let count = sequences.len();
        let stride = heads.stride();
        let row_bytes = WIDTH * size_of::<f32>();
        let stage_tokens = |tokens: &[u32]| -> crate::Result<()> {
            let mut values = vec![0.0; tokens.len() * WIDTH];
            decode_embeddings(&self.package, tokens, &mut values)?;
            mtp.stage_embeddings(&values)
        };
        // Lags too long to ride beside the drafting rows go first, in
        // K/V-only passes.
        loop {
            let owed = sequences
                .iter()
                .map(|sequence| sequence.lag_tokens.len() - sequence.fed)
                .sum::<usize>();
            if count + owed <= limit {
                break;
            }
            let mut spans = Vec::new();
            let mut tokens = Vec::new();
            let mut copies = Vec::new();
            let mut taken = vec![0; count];
            for (index, sequence) in sequences.iter().enumerate() {
                let left = sequence.lag_tokens.len() - sequence.fed;
                let rows = left.min(limit - tokens.len());
                if rows == 0 {
                    continue;
                }
                sequence.stage(
                    mtp.hidden_in(),
                    tokens.len(),
                    sequence.fed,
                    rows,
                    &mut copies,
                )?;
                spans.push(HeadSpan {
                    state: sequence.state,
                    capacity: sequence.capacity,
                    row: tokens.len(),
                    rows,
                    position: sequence.first() + sequence.fed,
                });
                tokens.extend_from_slice(&sequence.lag_tokens[sequence.fed..sequence.fed + rows]);
                taken[index] = rows;
            }
            let total = tokens.len();
            stage_tokens(&tokens)?;
            let mut batch = CommandBatch::new(&self.context)?;
            batch.blit_buffer_copies(copies)?;
            mtp.transform_embeddings(&mut batch, &shared, total)?;
            mtp.encode_stacked(
                &mut batch,
                &shared,
                mtp.hidden_in(),
                &[],
                &spans,
                total,
                &heads.logits,
            )?;
            self.finish(batch)?;
            for (sequence, rows) in sequences.iter_mut().zip(taken) {
                sequence.fed += rows;
            }
        }
        // The first depth: each sequence's seed row, then whatever lag rows
        // are left to feed.
        let mut tokens = Vec::with_capacity(limit);
        let mut copies = Vec::new();
        let mut drafting = Vec::with_capacity(count);
        let mut catching = Vec::new();
        for (index, sequence) in sequences.iter().enumerate() {
            tokens.push(sequence.seed);
            heads
                .tokens
                .copy_from_bytes(bytemuck::bytes_of(&sequence.seed), index * stride * 4);
            // The seed reads the newest committed row's hidden.
            sequence.stage(
                mtp.hidden_in(),
                index,
                sequence.lag_tokens.len(),
                1,
                &mut copies,
            )?;
            drafting.push(HeadSpan {
                state: sequence.state,
                capacity: sequence.capacity,
                row: index,
                rows: 1,
                position: sequence.position,
            });
        }
        for sequence in &sequences {
            let rows = sequence.lag_tokens.len() - sequence.fed;
            if rows == 0 {
                continue;
            }
            sequence.stage(
                mtp.hidden_in(),
                tokens.len(),
                sequence.fed,
                rows,
                &mut copies,
            )?;
            catching.push(HeadSpan {
                state: sequence.state,
                capacity: sequence.capacity,
                row: tokens.len(),
                rows,
                position: sequence.first() + sequence.fed,
            });
            tokens.extend_from_slice(&sequence.lag_tokens[sequence.fed..]);
        }
        let total = tokens.len();
        stage_tokens(&tokens)?;
        let mut batch = CommandBatch::new(&self.context)?;
        batch.blit_buffer_copies(copies)?;
        mtp.transform_embeddings(&mut batch, &shared, total)?;
        mtp.encode_stacked(
            &mut batch,
            &shared,
            mtp.hidden_in(),
            &drafting,
            &catching,
            total,
            &heads.logits,
        )?;
        let token_slots = (0..count)
            .map(|index| (index * stride + 1) as u32)
            .collect::<Vec<_>>();
        let result_slots = (0..count)
            .map(|index| (index * heads.depth) as u32)
            .collect::<Vec<_>>();
        heads.rows.top_two(
            &mut batch,
            &heads.logits,
            VOCAB,
            &heads.tokens,
            &token_slots,
            &heads.results,
            &result_slots,
        )?;
        // Each head now holds every committed row: its next round's first
        // drafting row reads the newest one's hidden.
        let commits = sequences
            .iter()
            .filter(|sequence| !sequence.lag_tokens.is_empty())
            .map(|sequence| {
                Ok(BufferCopyRequest {
                    source: sequence.lag_hidden.ok_or_else(|| {
                        crate::Error::InvalidArgument("head lag has no rows".into())
                    })?,
                    source_offset: (sequence.lag_tokens.len() - 1) * row_bytes,
                    destination: sequence.state.prev_hidden,
                    destination_offset: 0,
                    size: row_bytes,
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        if !commits.is_empty() {
            batch.blit_buffer_copies(commits)?;
        }
        self.finish(batch)?;
        // Read each depth's selections and run the next one for the chains
        // still going, each row's predicted hidden feeding its next row.
        let mut active = (0..count).collect::<Vec<_>>();
        let mut rows_of = (0..count).collect::<Vec<_>>();
        for step in 0..heads.depth {
            let results = heads.results.as_slice::<DraftTopTwo>();
            let mut going = Vec::with_capacity(active.len());
            for &index in &active {
                let top = results[index * heads.depth + step];
                let (next, margin) = samplers[index].greedy_from_top_two(
                    top.best_id,
                    top.best_logit,
                    top.second_logit,
                );
                let chain = &mut drafts[slots[index]];
                chain.push(next.token_id);
                if !(next.is_eos || margin < DRAFT_CHAIN_MIN_MARGIN)
                    && chain.len() < sequences[index].depth
                {
                    going.push(index);
                }
            }
            active = going;
            if active.is_empty() {
                break;
            }
            let depth = step + 1;
            let mut batch = CommandBatch::new(&self.context)?;
            let copies = active
                .iter()
                .enumerate()
                .map(|(row, &index)| BufferCopyRequest {
                    source: mtp.predicted(),
                    source_offset: rows_of[index] * row_bytes,
                    destination: mtp.hidden_in(),
                    destination_offset: row * row_bytes,
                    size: row_bytes,
                })
                .collect::<Vec<_>>();
            batch.blit_buffer_copies(copies)?;
            let token_slots = active
                .iter()
                .map(|&index| (index * stride + depth) as u32)
                .collect::<Vec<_>>();
            heads.rows.embed_inverse(
                &mut batch,
                heads.embeddings.ptq1_matrix()?,
                shared.input_rotation,
                &heads.tokens,
                &token_slots,
                mtp.embedded(),
            )?;
            let drafting = active
                .iter()
                .enumerate()
                .map(|(row, &index)| {
                    let sequence = &sequences[index];
                    HeadSpan {
                        state: sequence.state,
                        capacity: sequence.capacity,
                        row,
                        rows: 1,
                        position: sequence.position + depth,
                    }
                })
                .collect::<Vec<_>>();
            mtp.encode_stacked(
                &mut batch,
                &shared,
                mtp.hidden_in(),
                &drafting,
                &[],
                active.len(),
                &heads.logits,
            )?;
            let next_slots = active
                .iter()
                .map(|&index| (index * stride + depth + 1) as u32)
                .collect::<Vec<_>>();
            let result_slots = active
                .iter()
                .map(|&index| (index * heads.depth + depth) as u32)
                .collect::<Vec<_>>();
            heads.rows.top_two(
                &mut batch,
                &heads.logits,
                VOCAB,
                &heads.tokens,
                &next_slots,
                &heads.results,
                &result_slots,
            )?;
            self.finish(batch)?;
            for (row, &index) in active.iter().enumerate() {
                rows_of[index] = row;
            }
        }
        Ok(drafts)
    }
}
