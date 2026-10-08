use super::{
    Bf16Matrix, BonsaiMetalTensor, BonsaiModel, CommandBatch, FFN, FullAttentionLayer,
    HadamardDirection, Layer, MetalBuffer, RecurrentLayer, RollbackLayer, WIDTH,
};

/// Blocks up to this many rows (decode and speculative verification) take the
/// fused convolution kernel, whose threadgroups normalize each row's heads in
/// turn; prefill chunks keep one threadgroup per row and head.
const SHORT_BLOCK: u32 = 32;

/// A recurrent layer's 48-row BF16 alpha or beta projection. These sensitive
/// projections read the unrotated input: they are NOT Hadamard-folded.
const fn decay_projection(weights: &BonsaiMetalTensor) -> Bf16Matrix<'_> {
    Bf16Matrix {
        buffer: weights.buffer(),
        offset: weights.offset(),
        rows: 48,
        columns: WIDTH as u32,
    }
}

/// Where [`BonsaiModel::recurrent_inputs`] leaves what the recurrence and a
/// later replay read.
#[derive(Clone, Copy)]
struct RecurrentInputs<'a> {
    raw: &'a MetalBuffer,
    final_history: &'a MetalBuffer,
    decay: &'a MetalBuffer,
    beta: &'a MetalBuffer,
}

impl BonsaiModel {
    /// RMS-normalize into `scratch.normalized` and rotate into
    /// `scratch.rotated_hidden`, in one dispatch.
    pub(super) fn normalize_input(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        self.kernels.normalize_transform(
            batch,
            &self.input_rotation,
            input,
            weights,
            &self.scratch.normalized,
            &self.scratch.rotated_hidden,
            tokens,
            self.epsilon,
        )?;
        Ok(())
    }

    pub(super) fn project(
        &self,
        batch: &mut CommandBatch,
        weights: &BonsaiMetalTensor,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        self.kernels
            .matmul(batch, weights.ptq1_matrix()?, input, output, tokens)?;
        Ok(())
    }

    pub(super) fn feed_forward(
        &self,
        batch: &mut CommandBatch,
        layer: &Layer,
        tokens: u32,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        self.normalize_input(batch, &scratch.hidden, &layer.post_attention_norm, tokens)?;
        if tokens == 1 {
            self.kernels.matvec_swiglu(
                batch,
                layer.gate.ptq1_matrix()?,
                layer.up.ptq1_matrix()?,
                &scratch.rotated_hidden,
                &scratch.ffn_product,
            )?;
        } else {
            batch.independent(|batch| {
                self.project(
                    batch,
                    &layer.gate,
                    &scratch.rotated_hidden,
                    &scratch.ffn_gate,
                    tokens,
                )?;
                self.project(
                    batch,
                    &layer.up,
                    &scratch.rotated_hidden,
                    &scratch.ffn_up,
                    tokens,
                )
            })?;
            self.ops.swiglu(
                batch,
                &scratch.ffn_gate,
                &scratch.ffn_up,
                &scratch.ffn_product,
                FFN as u32 * tokens,
            )?;
        }
        self.kernels.transform(
            batch,
            &self.ffn_rotation,
            &scratch.ffn_product,
            &scratch.rotated_ffn,
            tokens,
            HadamardDirection::Forward,
        )?;
        self.project(
            batch,
            &layer.down,
            &scratch.rotated_ffn,
            &scratch.branch,
            tokens,
        )?;
        self.ops.residual_add(
            batch,
            &scratch.branch,
            &scratch.hidden,
            &scratch.hidden,
            WIDTH as u32 * tokens,
        )?;
        Ok(())
    }

    /// During verification the layer keeps its start state and history: the
    /// block writes its final ones, and the compact inputs a partial commit
    /// replays, straight to `rollback` (see [`BonsaiModel::commit_verified`]).
    pub(super) fn recurrent(
        &self,
        batch: &mut CommandBatch,
        layer: &RecurrentLayer,
        tokens: u32,
        rollback: Option<&RollbackLayer>,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        let (inputs, decay, beta, final_state, final_history) = rollback.map_or(
            (
                &scratch.query_gate,
                &scratch.decay,
                &scratch.beta,
                &layer.state,
                &layer.history,
            ),
            |rollback| {
                (
                    &rollback.inputs,
                    &rollback.decay,
                    &rollback.beta,
                    &rollback.state,
                    &rollback.history,
                )
            },
        );
        self.recurrent_inputs(
            batch,
            layer,
            tokens,
            RecurrentInputs {
                raw: inputs,
                final_history,
                decay,
                beta,
            },
        )?;
        self.ops.gdn_sequence_into(
            batch,
            self.state_format,
            &scratch.convolved,
            decay,
            beta,
            &layer.state,
            final_state,
            &scratch.recurrent_output,
            tokens,
        )?;
        self.ops.gdn_postprocess_rows(
            batch,
            &scratch.recurrent_output,
            &scratch.gate,
            &layer.norm,
            &scratch.attention_output,
            self.epsilon,
            tokens,
        )?;
        self.project_attention(batch, &layer.output, tokens)
    }

    /// The recurrence's per-row inputs: the raw QKV projection into
    /// `out.raw`, convolved and normalized Q/K/V, the output gate, and
    /// decay/beta.
    fn recurrent_inputs(
        &self,
        batch: &mut CommandBatch,
        layer: &RecurrentLayer,
        tokens: u32,
        out: RecurrentInputs<'_>,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        if tokens == 1 {
            self.kernels.matvec_concat_bf16(
                batch,
                &[
                    (layer.qkv.ptq1_matrix()?, out.raw),
                    (layer.gate.ptq1_matrix()?, &scratch.gate),
                ],
                &scratch.rotated_hidden,
                [
                    (decay_projection(&layer.alpha), &scratch.alpha),
                    (decay_projection(&layer.beta), &scratch.raw_beta),
                ],
                &scratch.normalized,
            )?;
        } else {
            batch.independent(|batch| {
                self.project(batch, &layer.qkv, &scratch.rotated_hidden, out.raw, tokens)?;
                self.project(
                    batch,
                    &layer.gate,
                    &scratch.rotated_hidden,
                    &scratch.gate,
                    tokens,
                )?;
                for (weights, output) in [
                    (&layer.alpha, &scratch.alpha),
                    (&layer.beta, &scratch.raw_beta),
                ] {
                    self.ops.bf16_matmul(
                        batch,
                        decay_projection(weights),
                        &scratch.normalized,
                        output,
                        tokens,
                    )?;
                }
                crate::Result::Ok(())
            })?;
        }
        if tokens <= SHORT_BLOCK {
            self.ops.conv_l2_decay_into(
                batch,
                out.raw,
                &layer.convolution,
                [&layer.history, out.final_history],
                &scratch.convolved,
                self.epsilon,
                [&layer.decay, &scratch.alpha, &layer.dt, &scratch.raw_beta],
                out.decay,
                out.beta,
                tokens,
            )?;
        } else {
            self.ops.conv_sequence_into(
                batch,
                out.raw,
                &layer.convolution,
                &layer.history,
                out.final_history,
                &scratch.convolved,
                tokens,
            )?;
            self.ops
                .l2_normalize_qk_rows(batch, &scratch.convolved, self.epsilon, tokens)?;
            self.ops.decay_beta_rows(
                batch,
                &layer.decay,
                &scratch.alpha,
                &layer.dt,
                &scratch.raw_beta,
                out.decay,
                out.beta,
                tokens,
            )?;
        }
        Ok(())
    }

    pub(super) fn full_attention(
        &self,
        batch: &mut CommandBatch,
        layer: &FullAttentionLayer,
        tokens: u32,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        let projections = [
            (&layer.query_gate, &scratch.query_gate),
            (&layer.key, &scratch.key),
            (&layer.value, &scratch.value),
        ];
        if tokens == 1 {
            self.kernels.matvec_concat(
                batch,
                &projections
                    .map(|(weights, output)| weights.ptq1_matrix().map(|matrix| (matrix, output)))
                    .into_iter()
                    .collect::<crate::Result<Vec<_>>>()?,
                &scratch.rotated_hidden,
            )?;
        } else {
            batch.independent(|batch| {
                for (weights, output) in projections {
                    self.project(batch, weights, &scratch.rotated_hidden, output, tokens)?;
                }
                crate::Result::Ok(())
            })?;
        }
        self.ops.prepare_attention_rows_kv(
            self.kv_layout,
            batch,
            &scratch.query_gate,
            &scratch.key,
            &scratch.value,
            &layer.query_norm,
            &layer.key_norm,
            &scratch.query,
            &scratch.gate,
            &layer.key_cache,
            &layer.value_cache,
            self.position as u32,
            self.kv_allocated as u32,
            self.epsilon,
            self.rope_base,
            tokens,
        )?;
        if tokens == 1 {
            self.ops.attention_row_kv(
                self.kv_layout,
                batch,
                &scratch.query,
                &layer.key_cache,
                &layer.value_cache,
                Some(&scratch.gate),
                &scratch.attention_output,
                self.position as u32 + 1,
                &scratch.attention,
                0,
            )?;
        } else {
            self.ops.attention_block_kv(
                self.kv_layout,
                batch,
                &scratch.query,
                &layer.key_cache,
                &layer.value_cache,
                Some(&scratch.gate),
                &scratch.attention_output,
                self.position as u32,
                tokens,
                &scratch.attention,
            )?;
        }
        self.project_attention(batch, &layer.output, tokens)
    }

    fn project_attention(
        &self,
        batch: &mut CommandBatch,
        output: &BonsaiMetalTensor,
        tokens: u32,
    ) -> crate::Result<()> {
        self.kernels.transform(
            batch,
            &self.attention_rotation,
            &self.scratch.attention_output,
            &self.scratch.rotated_attention,
            tokens,
            HadamardDirection::Forward,
        )?;
        self.project(
            batch,
            output,
            &self.scratch.rotated_attention,
            &self.scratch.branch,
            tokens,
        )
    }
}
