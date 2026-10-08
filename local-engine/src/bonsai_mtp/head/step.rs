use local_metal::batch::{BufferCopyRequest, CommandBatch};
use local_metal::bonsai::{HadamardDirection, Int8Matrix, Ptq1Matrix};
use local_metal::bonsai_ops::RmsNormParams;
use local_metal::buffer::MetalBuffer;
use local_metal::draft::DraftKernels;

use super::super::weights::MatrixWeight;
use super::super::{FFN, WIDTH};
use super::{BonsaiMtp, Shared};

fn ptq1(packed: &MetalBuffer, rows: u32, columns: u32) -> crate::Result<Ptq1Matrix<'_>> {
    Ptq1Matrix::new(packed, 0, rows, columns).map_err(crate::Error::Metal)
}

fn int8<'a>(
    weights: &'a MetalBuffer,
    scales: &'a MetalBuffer,
    rows: u32,
    columns: u32,
) -> crate::Result<Int8Matrix<'a>> {
    Int8Matrix::new(weights, scales, rows, columns).map_err(crate::Error::Metal)
}

/// Multiply `count` rotated rows of `input` by `matrix`, whichever format it
/// is stored in, into `output`.
fn project(
    batch: &mut CommandBatch,
    shared: &Shared<'_>,
    matrix: &MatrixWeight,
    input: &MetalBuffer,
    output: &MetalBuffer,
    count: u32,
) -> crate::Result<()> {
    match matrix {
        MatrixWeight::Ptq1 {
            packed,
            rows,
            columns,
        } => shared
            .kernels
            .matmul(batch, ptq1(packed, *rows, *columns)?, input, output, count)?,
        MatrixWeight::Int8 {
            weights,
            scales,
            rows,
            columns,
        } => shared.kernels.int8_matmul(
            batch,
            int8(weights, scales, *rows, *columns)?,
            input,
            output,
            count,
        )?,
    }
    Ok(())
}

/// Single-row projections of one rotated input by several matrices, as few
/// dispatches as their formats allow: the `PTQ1_0` ones share one concatenated
/// matvec, as the target's own single-token projections do, and the int8 ones
/// another.
fn project_concat(
    batch: &mut CommandBatch,
    shared: &Shared<'_>,
    projections: &[(&MatrixWeight, &MetalBuffer)],
    input: &MetalBuffer,
) -> crate::Result<()> {
    let mut packed = Vec::with_capacity(projections.len());
    let mut wide = Vec::with_capacity(projections.len());
    for &(matrix, output) in projections {
        match matrix {
            MatrixWeight::Ptq1 {
                packed: bytes,
                rows,
                columns,
            } => packed.push((ptq1(bytes, *rows, *columns)?, output)),
            MatrixWeight::Int8 {
                weights,
                scales,
                rows,
                columns,
            } => wide.push((int8(weights, scales, *rows, *columns)?, output)),
        }
    }
    if !packed.is_empty() {
        shared.kernels.matvec_concat(batch, &packed, input)?;
    }
    if !wide.is_empty() {
        shared.kernels.int8_matvec_concat(batch, &wide, input)?;
    }
    Ok(())
}

impl BonsaiMtp {
    /// Stage the hidden inputs for `rows` committed tokens starting at the
    /// previous committed hidden: row 0 reads `prev_hidden`, later rows read
    /// `normalized_rows[0..rows-1]`. `prev_hidden` itself is unchanged; the
    /// caller advances it afterwards with [`Self::commit_hidden`].
    pub fn stage_hidden(
        &self,
        batch: &mut CommandBatch,
        normalized_rows: &MetalBuffer,
        rows: usize,
    ) -> crate::Result<()> {
        if rows == 0 || rows > self.scratch.rows {
            return Err(crate::Error::InvalidArgument(
                "MTP hidden rows exceed scratch".into(),
            ));
        }
        let row_bytes = WIDTH * size_of::<f32>();
        let mut copies = vec![BufferCopyRequest {
            source: &self.prev_hidden,
            source_offset: 0,
            destination: &self.scratch.hidden_in,
            destination_offset: 0,
            size: row_bytes,
        }];
        if rows > 1 {
            copies.push(BufferCopyRequest {
                source: normalized_rows,
                source_offset: 0,
                destination: &self.scratch.hidden_in,
                destination_offset: row_bytes,
                size: (rows - 1) * row_bytes,
            });
        }
        batch.blit_buffer_copies(copies)?;
        Ok(())
    }

    pub fn commit_hidden(
        &self,
        batch: &mut CommandBatch,
        normalized_rows: &MetalBuffer,
        last_row: usize,
    ) -> crate::Result<()> {
        let row_bytes = WIDTH * size_of::<f32>();
        batch.blit_buffer_copies([BufferCopyRequest {
            source: normalized_rows,
            source_offset: last_row * row_bytes,
            destination: &self.prev_hidden,
            destination_offset: 0,
            size: row_bytes,
        }])?;
        Ok(())
    }

    /// Run the head over `rows` staged embedding rows at absolute `position`,
    /// reading hidden inputs from `hidden_in` and writing this head's KV rows.
    ///
    /// With `logits`, the full layer runs and the last row's prediction is
    /// projected to draft logits; without it, only the K/V cache is needed
    /// (committed rows), so work stops after FC, input norm and K/V, leaving
    /// `predicted` untouched.
    ///
    /// The embedding rows are the host-staged [`Self::embedding_rows`].
    pub fn encode(
        &self,
        batch: &mut CommandBatch,
        shared: &Shared<'_>,
        hidden_in: &MetalBuffer,
        position: usize,
        rows: usize,
        logits: bool,
    ) -> crate::Result<()> {
        shared.kernels.transform(
            batch,
            shared.input_rotation,
            &self.scratch.embedding,
            &self.scratch.embedded,
            rows as u32,
            HadamardDirection::Inverse,
        )?;
        self.encode_layer(batch, shared, hidden_in, position, rows, logits)
    }

    /// One draft row whose token the GPU reads from `tokens[token_index]`
    /// (written by the previous chained step or by the host): the `PTQ1_0`
    /// `embeddings` row is decoded and inverse-rotated on the GPU, bit-identical
    /// to staging it through [`Self::embedding_rows`] and [`Self::encode`],
    /// and the full layer projects draft logits into [`Self::logits`].
    #[allow(clippy::too_many_arguments)]
    pub fn encode_draft(
        &self,
        batch: &mut CommandBatch,
        shared: &Shared<'_>,
        draft: &DraftKernels,
        embeddings: Ptq1Matrix<'_>,
        tokens: &MetalBuffer,
        token_index: u32,
        hidden_in: &MetalBuffer,
        position: usize,
    ) -> crate::Result<()> {
        draft.embed_inverse(
            batch,
            embeddings,
            shared.input_rotation,
            tokens,
            token_index,
            &self.scratch.embedded,
        )?;
        self.encode_layer(batch, shared, hidden_in, position, 1, true)
    }

    /// The layer from the inverse-rotated embedding rows in `scratch.embedded`.
    /// The layer from the inverse-rotated embedding rows in `scratch.embedded`.
    ///
    /// Every matrix, `PTQ1_0` or int8, lives in the target's rotated basis, so
    /// every projection input is forward-rotated once and shared by the
    /// projections reading it. Rotated rows land in scratch already dead at
    /// that point of the layer: `rotated` (WIDTH, free until the final
    /// projection), `query` (ATTENTION, after attention) and `ffn_gate` (FFN,
    /// after `SwiGLU`).
    #[allow(clippy::too_many_lines)]
    fn encode_layer(
        &self,
        batch: &mut CommandBatch,
        shared: &Shared<'_>,
        hidden_in: &MetalBuffer,
        position: usize,
        rows: usize,
        logits: bool,
    ) -> crate::Result<()> {
        if rows == 0
            || rows > self.scratch.rows
            || position
                .checked_add(rows)
                .is_none_or(|end| end > shared.capacity)
        {
            return Err(crate::Error::ContextOverflow(
                "invalid MTP row block or position".into(),
            ));
        }
        let count = rows as u32;
        let scratch = &self.scratch;
        let weights = &self.weights;
        // Normalize `input` into `normalized` and its forward rotation into
        // `scratch.rotated`, which the following projections read, in one
        // dispatch.
        let normalize = |batch: &mut CommandBatch,
                         input: &MetalBuffer,
                         w: &MetalBuffer,
                         normalized: &MetalBuffer|
         -> crate::Result<()> {
            shared.kernels.normalize_transform(
                batch,
                shared.input_rotation,
                input,
                w,
                normalized,
                &scratch.rotated,
                count,
                shared.epsilon,
            )?;
            Ok(())
        };
        let dense = |batch: &mut CommandBatch,
                     matrix: &MatrixWeight,
                     input: &MetalBuffer,
                     output: &MetalBuffer| {
            project(batch, shared, matrix, input, output, count)
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
        )?;
        dense(
            batch,
            &weights.fc_embedding,
            &scratch.rotated,
            &scratch.hidden,
        )?;
        normalize(
            batch,
            hidden_in,
            &weights.hidden_norm,
            &scratch.normalized_hidden,
        )?;
        dense(batch, &weights.fc_hidden, &scratch.rotated, &scratch.branch)?;
        let residual = |batch: &mut CommandBatch| {
            shared.ops.residual_add(
                batch,
                &scratch.branch,
                &scratch.hidden,
                &scratch.hidden,
                WIDTH as u32 * count,
            )
        };
        residual(batch)?;

        normalize(
            batch,
            &scratch.hidden,
            &weights.input_norm,
            &scratch.normalized,
        )?;
        // Committed rows only feed later drafts through attention, so without
        // logits the query, attention, FFN and final norm outputs are never
        // read and the query projection is skipped.
        let query = (&weights.query_gate, &scratch.query_gate);
        let key = (&weights.key, &scratch.key);
        let value = (&weights.value, &scratch.value);
        let projections: &[(&MatrixWeight, &MetalBuffer)] = if logits {
            &[query, key, value]
        } else {
            &[key, value]
        };
        if count == 1 {
            project_concat(batch, shared, projections, &scratch.rotated)?;
        } else {
            for &(matrix, output) in projections {
                dense(batch, matrix, &scratch.rotated, output)?;
            }
        }
        if !logits {
            shared.ops.prepare_kv_rows(
                batch,
                &scratch.key,
                &scratch.value,
                &weights.key_norm,
                &self.key_cache,
                &self.value_cache,
                position as u32,
                shared.capacity as u32,
                shared.epsilon,
                shared.rope_base,
                count,
            )?;
            return Ok(());
        }
        shared.ops.prepare_attention_rows(
            batch,
            &scratch.query_gate,
            &scratch.key,
            &scratch.value,
            &weights.query_norm,
            &weights.key_norm,
            &scratch.query,
            &scratch.gate,
            &self.key_cache,
            &self.value_cache,
            position as u32,
            shared.capacity as u32,
            shared.epsilon,
            shared.rope_base,
            count,
        )?;
        if count == 1 {
            shared.ops.attention(
                batch,
                &scratch.query,
                &self.key_cache,
                &self.value_cache,
                Some(&scratch.gate),
                &scratch.attention_output,
                position as u32 + 1,
                &scratch.attention,
            )?;
        } else {
            shared.ops.attention_block(
                batch,
                &scratch.query,
                &self.key_cache,
                &self.value_cache,
                Some(&scratch.gate),
                &scratch.attention_output,
                position as u32,
                count,
                &scratch.attention,
            )?;
        }
        rotate(
            batch,
            shared.attention_rotation,
            &scratch.attention_output,
            &scratch.query,
        )?;
        dense(batch, &weights.output, &scratch.query, &scratch.branch)?;
        residual(batch)?;

        normalize(
            batch,
            &scratch.hidden,
            &weights.post_attention_norm,
            &scratch.normalized,
        )?;
        match (&weights.gate, &weights.up) {
            // One row in one format: a fused single-token SwiGLU, so neither
            // projection is stored.
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
                dense(batch, &weights.gate, &scratch.rotated, &scratch.ffn_gate)?;
                dense(batch, &weights.up, &scratch.rotated, &scratch.ffn_up)?;
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
        dense(batch, &weights.down, &scratch.ffn_gate, &scratch.branch)?;
        residual(batch)?;

        // The shared PTQ1 head consumes rotated rows; project only the newest
        // row, which is the sole draft candidate.
        if count == 1 {
            // Final norm and rotation in one dispatch.
            shared.kernels.normalize_transform(
                batch,
                shared.input_rotation,
                &scratch.hidden,
                &weights.final_norm,
                &scratch.predicted,
                &scratch.rotated,
                1,
                shared.epsilon,
            )?;
        } else {
            shared.ops.rms_norm(
                batch,
                &scratch.hidden,
                &weights.final_norm,
                &scratch.predicted,
                RmsNormParams {
                    dimension: WIDTH as u32,
                    rows: count,
                    stride: WIDTH as u32,
                    epsilon: shared.epsilon,
                    weight_offset: 0,
                },
            )?;
            let row_bytes = WIDTH * size_of::<f32>();
            batch.blit_buffer_copies([BufferCopyRequest {
                source: &scratch.predicted,
                source_offset: (rows - 1) * row_bytes,
                destination: &scratch.embedded,
                destination_offset: 0,
                size: row_bytes,
            }])?;
            shared.kernels.transform(
                batch,
                shared.input_rotation,
                &scratch.embedded,
                &scratch.rotated,
                1,
                HadamardDirection::Forward,
            )?;
        }
        shared.kernels.matmul(
            batch,
            shared.output.ptq1_matrix()?,
            &scratch.rotated,
            &self.logits,
            1,
        )?;
        Ok(())
    }
}
