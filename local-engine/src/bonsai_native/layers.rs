use super::{
    Bf16Matrix, BonsaiMetalTensor, BonsaiModel, BufferCopyRequest, CommandBatch, FFN,
    FullAttentionLayer, HadamardDirection, Layer, MetalBuffer, RECURRENCE_FACTOR_WIDTH,
    RECURRENCE_SCALAR_WIDTH, RecurrentLayer, RmsNormParams, RollbackLayer, WIDTH, seed_rollback,
};

impl BonsaiModel {
    pub(super) fn normalize_input(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        self.ops.rms_norm(
            batch,
            input,
            weights,
            &self.scratch.normalized,
            RmsNormParams {
                dimension: WIDTH as u32,
                rows: tokens,
                stride: WIDTH as u32,
                epsilon: self.epsilon,
                weight_offset: 0,
            },
        )?;
        self.kernels.transform(
            batch,
            &self.input_rotation,
            &self.scratch.normalized,
            &self.scratch.rotated_hidden,
            tokens,
            HadamardDirection::Forward,
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
        )?;
        self.ops.swiglu(
            batch,
            &scratch.ffn_gate,
            &scratch.ffn_up,
            &scratch.ffn_product,
            FFN as u32 * tokens,
        )?;
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

    /// During verification, snapshot the start state and retain only the
    /// compact inputs needed to replay each non-final recurrence row.
    pub(super) fn recurrent(
        &self,
        batch: &mut CommandBatch,
        layer: &RecurrentLayer,
        tokens: u32,
        rollback: Option<&RollbackLayer>,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        seed_rollback(batch, layer, rollback)?;
        self.project(
            batch,
            &layer.qkv,
            &scratch.rotated_hidden,
            &scratch.query_gate,
            tokens,
        )?;
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
            // These sensitive BF16 projections are NOT Hadamard-folded.
            self.ops.bf16_matmul(
                batch,
                Bf16Matrix {
                    buffer: weights.buffer(),
                    offset: weights.offset(),
                    rows: 48,
                    columns: WIDTH as u32,
                },
                &scratch.normalized,
                output,
                tokens,
            )?;
        }
        self.ops.conv_sequence(
            batch,
            &scratch.query_gate,
            &layer.convolution,
            &layer.history,
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
            &scratch.decay,
            &scratch.beta,
            tokens,
        )?;
        if let Some(rollback) = rollback {
            let rows = (tokens as usize).saturating_sub(1);
            batch.blit_buffer_copies([
                BufferCopyRequest {
                    source: &scratch.query_gate,
                    source_offset: 0,
                    destination: &rollback.inputs,
                    destination_offset: 0,
                    size: rows * RECURRENCE_FACTOR_WIDTH * size_of::<f32>(),
                },
                BufferCopyRequest {
                    source: &scratch.decay,
                    source_offset: 0,
                    destination: &rollback.decay,
                    destination_offset: 0,
                    size: rows * RECURRENCE_SCALAR_WIDTH * size_of::<f32>(),
                },
                BufferCopyRequest {
                    source: &scratch.beta,
                    source_offset: 0,
                    destination: &rollback.beta,
                    destination_offset: 0,
                    size: rows * RECURRENCE_SCALAR_WIDTH * size_of::<f32>(),
                },
            ])?;
        }
        self.ops.gdn_sequence(
            batch,
            &scratch.convolved,
            &scratch.decay,
            &scratch.beta,
            &layer.state,
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

    pub(super) fn full_attention(
        &self,
        batch: &mut CommandBatch,
        layer: &FullAttentionLayer,
        tokens: u32,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        for (weights, output) in [
            (&layer.query_gate, &scratch.query_gate),
            (&layer.key, &scratch.key),
            (&layer.value, &scratch.value),
        ] {
            self.project(batch, weights, &scratch.rotated_hidden, output, tokens)?;
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
