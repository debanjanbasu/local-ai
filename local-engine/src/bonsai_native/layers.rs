use super::{
    Bf16Matrix, BonsaiMetalTensor, BonsaiModel, CommandBatch, FullAttentionLayer,
    HadamardDirection, Layer, MetalBuffer, RecurrentLayer, RollbackLayer, Rope, WIDTH,
};

/// Blocks up to this many rows (decode and speculative verification) take the
/// fused convolution kernel, whose threadgroups normalize each row's heads in
/// turn; prefill chunks keep one threadgroup per row and head.
pub(super) const SHORT_BLOCK: u32 = 32;

/// A recurrent layer's 48-row BF16 alpha or beta projection. These sensitive
/// projections read the unrotated input: they are NOT Hadamard-folded.
pub(super) const fn decay_projection(weights: &BonsaiMetalTensor) -> Bf16Matrix<'_> {
    Bf16Matrix {
        buffer: weights.buffer(),
        offset: weights.offset(),
        rows: 48,
        columns: WIDTH as u32,
    }
}

/// Where [`BonsaiModel::recurrent_inputs`] leaves what the recurrence and a
/// later replay read, and the tape rows it folds in first.
#[derive(Clone, Copy)]
struct RecurrentInputs<'a> {
    raw: &'a MetalBuffer,
    final_history: &'a MetalBuffer,
    decay: &'a MetalBuffer,
    beta: &'a MetalBuffer,
    fold: Option<local_metal::bonsai_ops::TapeFold<'a>>,
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

    /// `output = input + scratch.branch`, normalized by `weights` into
    /// `scratch.normalized` and rotated into `scratch.rotated_hidden`, in one
    /// dispatch: bitwise a residual add then [`Self::normalize_input`] of its
    /// sum. `output` must be a different buffer from `input`.
    pub(super) fn residual_normalize(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        output: &MetalBuffer,
        weights: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        self.ops.residual_normalize_transform(
            batch,
            &self.input_rotation,
            input,
            &scratch.branch,
            weights,
            [output, &scratch.normalized, &scratch.rotated_hidden],
            tokens,
            self.epsilon,
        )?;
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
        self.feed_forward_branch(batch, layer, tokens)?;
        self.ops.residual_add(
            batch,
            &scratch.branch,
            &scratch.hidden,
            &scratch.hidden,
            WIDTH as u32 * tokens,
        )?;
        Ok(())
    }

    /// The FFN of rows already normalized and rotated into
    /// `scratch.rotated_hidden`, leaving its output in `scratch.branch`.
    pub(super) fn feed_forward_branch(
        &self,
        batch: &mut CommandBatch,
        layer: &Layer,
        tokens: u32,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        if tokens == 1 {
            self.kernels.matvec_swiglu(
                batch,
                layer.gate.ptq1_matrix()?,
                layer.up.ptq1_matrix()?,
                &scratch.rotated_hidden,
                &scratch.ffn_product,
            )?;
            self.kernels.transform(
                batch,
                &self.ffn_rotation,
                &scratch.ffn_product,
                &scratch.rotated_ffn,
                tokens,
                HadamardDirection::Forward,
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
            // SwiGLU and the forward rotation in one dispatch, bitwise as the
            // separate ones; rotated_ffn shares ffn_gate's buffer, which the
            // fused kernel permits (each block is read before it is written).
            self.kernels.swiglu_transform(
                batch,
                &self.ffn_rotation,
                &scratch.ffn_gate,
                &scratch.ffn_up,
                &scratch.rotated_ffn,
                tokens,
            )?;
        }
        self.project(
            batch,
            &layer.down,
            &scratch.rotated_ffn,
            &scratch.branch,
            tokens,
        )
    }

    /// During verification the layer keeps its start state and history: the
    /// block writes its final ones to `rollback`, and the compact inputs a
    /// partial commit keeps to the tape (up to [`super::tape::TAPE_ROWS`]
    /// rows) or to `rollback` (see [`BonsaiModel::commit_verified`]).
    /// `index` is the layer's place among the recurrent layers.
    pub(super) fn recurrent(
        &self,
        batch: &mut CommandBatch,
        layer: &RecurrentLayer,
        index: usize,
        tokens: u32,
        rollback: Option<&RollbackLayer>,
    ) -> crate::Result<()> {
        self.recurrent_core(batch, layer, index, tokens, rollback)?;
        self.project(
            batch,
            &layer.output,
            &self.scratch.rotated_attention,
            &self.scratch.branch,
            tokens,
        )
    }

    /// [`Self::recurrent`] up to its output projection's rotated input in
    /// `scratch.rotated_attention`. The resident sequence's pending tape rows
    /// fold in first, which only blocks of up to [`SHORT_BLOCK`] rows do: a
    /// longer block's caller settles the tape before it.
    pub(super) fn recurrent_core(
        &self,
        batch: &mut CommandBatch,
        layer: &RecurrentLayer,
        index: usize,
        tokens: u32,
        rollback: Option<&RollbackLayer>,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        let (inputs, decay, beta, final_state, final_history) = match rollback {
            None => (
                &scratch.query_gate,
                &scratch.decay,
                &scratch.beta,
                &layer.state,
                &layer.history,
            ),
            Some(rollback) if tokens as usize <= super::tape::TAPE_ROWS => {
                let step = self.tape.step(index);
                (
                    &step.raw,
                    &step.decay,
                    &step.beta,
                    &rollback.state,
                    &rollback.history,
                )
            }
            Some(rollback) => (
                &rollback.inputs,
                &rollback.decay,
                &rollback.beta,
                &rollback.state,
                &rollback.history,
            ),
        };
        let fold =
            (self.tape.pending() > 0).then(|| self.tape.fold(index, (&self.tape_scratch, 0)));
        if fold.is_some() && tokens > SHORT_BLOCK {
            return Err(crate::Error::InvalidArgument(
                "a long block needs the tape settled first".into(),
            ));
        }
        self.recurrent_inputs(
            batch,
            layer,
            tokens,
            RecurrentInputs {
                raw: inputs,
                final_history,
                decay,
                beta,
                fold,
            },
        )?;
        match fold {
            Some(fold) => self.ops.gdn_fold(
                batch,
                self.state_format,
                [(&scratch.convolved, 0), (decay, 0), (beta, 0)],
                [&layer.state, final_state],
                (&scratch.recurrent_output, 0),
                fold,
                tokens,
            )?,
            None => self.ops.gdn_sequence_into(
                batch,
                self.state_format,
                &scratch.convolved,
                decay,
                beta,
                &layer.state,
                final_state,
                &scratch.recurrent_output,
                tokens,
            )?,
        }
        // The output norm, gate and rotation in one dispatch, bitwise
        // `gdn_postprocess_rows` then `project_attention`'s rotation.
        self.ops.gdn_postprocess_transform(
            batch,
            &scratch.recurrent_output,
            &scratch.gate,
            &layer.norm,
            &self.attention_rotation,
            &scratch.rotated_attention,
            self.epsilon,
            tokens,
        )?;
        Ok(())
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
                self.ops.bf16_matmul_pair(
                    batch,
                    [
                        (decay_projection(&layer.alpha), &scratch.alpha),
                        (decay_projection(&layer.beta), &scratch.raw_beta),
                    ],
                    &scratch.normalized,
                    tokens,
                )?;
                crate::Result::Ok(())
            })?;
        }
        if let Some(fold) = out.fold {
            self.ops.conv_l2_decay_fold(
                batch,
                &layer.convolution,
                [&layer.history, out.final_history],
                local_metal::bonsai_ops::ConvRows {
                    input: (out.raw, 0),
                    out: (&scratch.convolved, 0),
                    decay_inputs: [
                        (&layer.decay, 0),
                        (&scratch.alpha, 0),
                        (&layer.dt, 0),
                        (&scratch.raw_beta, 0),
                    ],
                    decay: (out.decay, 0),
                    beta: (out.beta, 0),
                    raw_copy: None,
                },
                self.epsilon,
                fold,
                tokens,
            )?;
        } else if tokens <= SHORT_BLOCK {
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
        rope: Rope<'_>,
    ) -> crate::Result<()> {
        self.full_attention_at(batch, layer, tokens, rope, self.position)
    }

    /// [`Self::full_attention`] of rows starting at cache position
    /// `position` (a half block of the Neural Engine wavefront).
    pub(super) fn full_attention_at(
        &self,
        batch: &mut CommandBatch,
        layer: &FullAttentionLayer,
        tokens: u32,
        rope: Rope<'_>,
        position: usize,
    ) -> crate::Result<()> {
        self.attention_core_at(batch, layer, tokens, rope, position)?;
        self.project_attention(batch, &layer.output, tokens)
    }

    /// [`Self::full_attention_at`] up to its output projection: the gated
    /// attention rows in `scratch.attention_output`.
    pub(super) fn attention_core_at(
        &self,
        batch: &mut CommandBatch,
        layer: &FullAttentionLayer,
        tokens: u32,
        rope: Rope<'_>,
        position: usize,
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
        self.attention_from_projections_at(batch, layer, tokens, rope, position)
    }

    /// [`Self::attention_core_at`] after its Q, K and V projections: their
    /// normalization, rotary embedding and cache writes, then attention.
    pub(super) fn attention_from_projections_at(
        &self,
        batch: &mut CommandBatch,
        layer: &FullAttentionLayer,
        tokens: u32,
        rope: Rope<'_>,
        position: usize,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
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
            position as u32,
            self.kv_allocated as u32,
            self.epsilon,
            self.rope_base,
            tokens,
            rope,
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
                position as u32 + 1,
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
                position as u32,
                tokens,
                &scratch.attention,
            )?;
        }
        Ok(())
    }

    pub(super) fn project_attention(
        &self,
        batch: &mut CommandBatch,
        output: &BonsaiMetalTensor,
        tokens: u32,
    ) -> crate::Result<()> {
        self.rotate_attention(batch, tokens)?;
        self.project(
            batch,
            output,
            &self.scratch.rotated_attention,
            &self.scratch.branch,
            tokens,
        )
    }

    /// `scratch.attention_output` rotated into `scratch.rotated_attention`,
    /// the output projection's input.
    pub(super) fn rotate_attention(
        &self,
        batch: &mut CommandBatch,
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
        Ok(())
    }
}
