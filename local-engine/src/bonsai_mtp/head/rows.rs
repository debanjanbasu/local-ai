//! The head over several sequences in one pass.
//!
//! Batched decode leaves every sequence's head behind by the rows it decoded
//! (see `HeadLag`), and its drafts are verified together, so its drafts are
//! made together too: one stacked head pass per draft depth, one drafting row
//! per sequence. Every matrix of the layer reads its weights once for all the
//! rows; only the K/V append and the attention run per sequence, against that
//! sequence's own head caches, with the target's row-offset wrappers.
//!
//! A pass may also carry rows a sequence's head has yet to ingest. Those only
//! write their K/V rows (as head ingestion of committed rows always has): they
//! are projected through the embedding and hidden FCs, the input norm and the
//! K/V projections with the drafting rows, and stop there.

use local_metal::batch::CommandBatch;
use local_metal::bonsai::HadamardDirection;
use local_metal::bonsai_ops::{AttentionWorkspace, KvLayout};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;

use super::super::weights::MatrixWeight;
use super::super::{FFN, WIDTH};
use super::step::{int8, project, ptq1};
use super::{BonsaiMtp, HeadSequence, Shared};

/// One sequence's head state: its F16 caches and the committed hidden its
/// next drafting row reads.
#[derive(Clone, Copy)]
pub struct HeadState<'a> {
    pub key_cache: &'a MetalBuffer,
    pub value_cache: &'a MetalBuffer,
    pub prev_hidden: &'a MetalBuffer,
}

/// Consecutive rows of one sequence in a stacked head pass.
#[derive(Clone, Copy)]
pub struct HeadSpan<'a> {
    pub state: HeadState<'a>,
    /// Tokens the sequence's caches hold.
    pub capacity: usize,
    /// First stacked row.
    pub row: usize,
    pub rows: usize,
    /// Absolute position of the first row.
    pub position: usize,
}

impl HeadSequence {
    pub const fn state(&self) -> HeadState<'_> {
        HeadState {
            key_cache: &self.key_cache,
            value_cache: &self.value_cache,
            prev_hidden: &self.prev_hidden,
        }
    }
}

impl BonsaiMtp {
    /// The resident sequence's head state.
    pub const fn state(&self) -> HeadState<'_> {
        HeadState {
            key_cache: &self.key_cache,
            value_cache: &self.value_cache,
            prev_hidden: &self.prev_hidden,
        }
    }

    /// Rows one stacked pass may hold.
    pub const fn scratch_rows(&self) -> usize {
        self.scratch.rows
    }

    /// The inverse-rotated embedding rows a pass reads: written by
    /// [`Self::transform_embeddings`] or by a GPU gather.
    pub const fn embedded(&self) -> &MetalBuffer {
        &self.scratch.embedded
    }

    /// Make the shared attention workspace cover `tokens` positions, for
    /// sequences whose caches outgrew the resident one's.
    pub fn ensure_attention(&mut self, context: &MetalContext, tokens: usize) -> crate::Result<()> {
        if self.scratch.attention.max_context() < tokens as u32 {
            self.scratch.attention = AttentionWorkspace::new(context, tokens as u32)?;
        }
        Ok(())
    }

    /// Stage embedding rows decoded on the host, as [`Self::embedding_rows`]
    /// would hold them, without borrowing the head mutably.
    pub fn stage_embeddings(&self, values: &[f32]) -> crate::Result<()> {
        if values.is_empty()
            || !values.len().is_multiple_of(WIDTH)
            || values.len() / WIDTH > self.scratch.rows
        {
            return Err(crate::Error::InvalidArgument(
                "MTP embedding rows exceed scratch".into(),
            ));
        }
        self.scratch
            .embedding
            .copy_from_bytes(bytemuck::cast_slice(values), 0);
        Ok(())
    }

    /// Inverse-rotate `rows` host-staged [`Self::embedding_rows`] into
    /// [`Self::embedded`].
    pub fn transform_embeddings(
        &self,
        batch: &mut CommandBatch,
        shared: &Shared<'_>,
        rows: usize,
    ) -> crate::Result<()> {
        shared.kernels.transform(
            batch,
            shared.input_rotation,
            &self.scratch.embedding,
            &self.scratch.embedded,
            rows as u32,
            HadamardDirection::Inverse,
        )?;
        Ok(())
    }

    /// One stacked pass over `total` rows of [`Self::embedded`] and
    /// `hidden_in`: `drafting[i]` is the single drafting row `i` (rows
    /// `0..drafting.len()`), whose draft logits land in row `i` of `logits`
    /// and whose predicted hidden in row `i` of [`Self::predicted`];
    /// `catching` spans the remaining rows, which only write their K/V.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    pub fn encode_stacked(
        &self,
        batch: &mut CommandBatch,
        shared: &Shared<'_>,
        hidden_in: &MetalBuffer,
        drafting: &[HeadSpan<'_>],
        catching: &[HeadSpan<'_>],
        total: usize,
        logits: &MetalBuffer,
    ) -> crate::Result<()> {
        let drafts = drafting.len();
        let rows_ok = total > 0
            && total <= self.scratch.rows
            && drafting
                .iter()
                .enumerate()
                .all(|(index, span)| span.row == index && span.rows == 1)
            && catching
                .iter()
                .all(|span| span.rows > 0 && span.row >= drafts && span.row + span.rows <= total)
            && drafting.iter().chain(catching).all(|span| {
                span.position
                    .checked_add(span.rows)
                    .is_some_and(|end| end <= span.capacity)
            });
        if !rows_ok || logits.length() < drafts * shared.output.ptq1_matrix()?.rows() as usize * 4 {
            return Err(crate::Error::InvalidArgument(
                "invalid stacked MTP rows".into(),
            ));
        }
        let scratch = &self.scratch;
        let weights = &self.weights;
        let all = total as u32;
        let count = drafts as u32;
        let normalize = |batch: &mut CommandBatch,
                         input: &MetalBuffer,
                         w: &MetalBuffer,
                         normalized: &MetalBuffer,
                         rows: u32|
         -> crate::Result<()> {
            shared.kernels.normalize_transform(
                batch,
                shared.input_rotation,
                input,
                w,
                normalized,
                &scratch.rotated,
                rows,
                shared.epsilon,
            )?;
            Ok(())
        };
        let dense =
            |batch: &mut CommandBatch,
             matrix: &MatrixWeight,
             input: &MetalBuffer,
             output: &MetalBuffer,
             rows: u32| project(batch, shared, matrix, input, output, rows);
        let residual = |batch: &mut CommandBatch, rows: u32| {
            shared.ops.residual_add(
                batch,
                &scratch.branch,
                &scratch.hidden,
                &scratch.hidden,
                WIDTH as u32 * rows,
            )
        };
        let rotate =
            |batch: &mut CommandBatch, rotation, input: &MetalBuffer, output: &MetalBuffer| {
                shared.kernels.transform(
                    batch,
                    rotation,
                    input,
                    output,
                    count,
                    HadamardDirection::Forward,
                )
            };
        normalize(
            batch,
            &scratch.embedded,
            &weights.embedding_norm,
            &scratch.normalized_embedding,
            all,
        )?;
        dense(
            batch,
            &weights.fc_embedding,
            &scratch.rotated,
            &scratch.hidden,
            all,
        )?;
        normalize(
            batch,
            hidden_in,
            &weights.hidden_norm,
            &scratch.normalized_hidden,
            all,
        )?;
        dense(
            batch,
            &weights.fc_hidden,
            &scratch.rotated,
            &scratch.branch,
            all,
        )?;
        residual(batch, all)?;
        normalize(
            batch,
            &scratch.hidden,
            &weights.input_norm,
            &scratch.normalized,
            all,
        )?;
        // Only the drafting rows, which come first, need a query.
        if drafts > 0 {
            dense(
                batch,
                &weights.query_gate,
                &scratch.rotated,
                &scratch.query_gate,
                count,
            )?;
        }
        dense(batch, &weights.key, &scratch.rotated, &scratch.key, all)?;
        dense(batch, &weights.value, &scratch.rotated, &scratch.value, all)?;
        let constants = [shared.epsilon, shared.rope_base];
        for span in catching {
            shared.ops.prepare_kv_rows_kv_at(
                KvLayout::F16,
                batch,
                [&scratch.key, &scratch.value],
                &weights.key_norm,
                [span.state.key_cache, span.state.value_cache],
                [span.position as u32, span.capacity as u32],
                constants,
                span.row as u32,
                span.rows as u32,
            )?;
        }
        if drafts == 0 {
            return Ok(());
        }
        for span in drafting {
            shared.ops.prepare_attention_row_kv(
                KvLayout::F16,
                batch,
                [&scratch.query_gate, &scratch.key, &scratch.value],
                [&weights.query_norm, &weights.key_norm],
                [&scratch.query, &scratch.gate],
                [span.state.key_cache, span.state.value_cache],
                span.position as u32,
                span.capacity as u32,
                shared.epsilon,
                shared.rope_base,
                span.row as u32,
            )?;
        }
        // The sequences share one attention workspace, so they stay ordered.
        for span in drafting {
            shared.ops.attention_row_kv(
                KvLayout::F16,
                batch,
                &scratch.query,
                span.state.key_cache,
                span.state.value_cache,
                Some(&scratch.gate),
                &scratch.attention_output,
                span.position as u32 + 1,
                &scratch.attention,
                span.row as u32,
            )?;
        }
        rotate(
            batch,
            shared.attention_rotation,
            &scratch.attention_output,
            &scratch.query,
        )?;
        dense(
            batch,
            &weights.output,
            &scratch.query,
            &scratch.branch,
            count,
        )?;
        residual(batch, count)?;
        normalize(
            batch,
            &scratch.hidden,
            &weights.post_attention_norm,
            &scratch.normalized,
            count,
        )?;
        match (&weights.gate, &weights.up) {
            (
                MatrixWeight::Ptq1 {
                    packed: gate,
                    rows,
                    columns,
                },
                MatrixWeight::Ptq1 { packed: up, .. },
            ) if count == 1 => {
                shared.kernels.matvec_swiglu(
                    batch,
                    ptq1(gate, *rows, *columns)?,
                    ptq1(up, *rows, *columns)?,
                    &scratch.rotated,
                    &scratch.ffn_product,
                )?;
            }
            (
                MatrixWeight::Int8 {
                    weights: gate,
                    scales: gate_scales,
                    rows,
                    columns,
                },
                MatrixWeight::Int8 {
                    weights: up,
                    scales: up_scales,
                    ..
                },
            ) if count == 1 => {
                shared.kernels.int8_matvec_swiglu(
                    batch,
                    int8(gate, gate_scales, *rows, *columns)?,
                    int8(up, up_scales, *rows, *columns)?,
                    &scratch.rotated,
                    &scratch.ffn_product,
                )?;
            }
            _ => {
                dense(
                    batch,
                    &weights.gate,
                    &scratch.rotated,
                    &scratch.ffn_gate,
                    count,
                )?;
                dense(batch, &weights.up, &scratch.rotated, &scratch.ffn_up, count)?;
                shared.ops.swiglu(
                    batch,
                    &scratch.ffn_gate,
                    &scratch.ffn_up,
                    &scratch.ffn_product,
                    FFN as u32 * count,
                )?;
            }
        }
        rotate(
            batch,
            shared.ffn_rotation,
            &scratch.ffn_product,
            &scratch.ffn_gate,
        )?;
        dense(
            batch,
            &weights.down,
            &scratch.ffn_gate,
            &scratch.branch,
            count,
        )?;
        residual(batch, count)?;
        normalize(
            batch,
            &scratch.hidden,
            &weights.final_norm,
            &scratch.predicted,
            count,
        )?;
        shared.kernels.matmul(
            batch,
            shared.output.ptq1_matrix()?,
            &scratch.rotated,
            logits,
            count,
        )?;
        Ok(())
    }
}
