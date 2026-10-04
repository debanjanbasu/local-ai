use local_metal::batch::{BufferCopyRequest, CommandBatch};
use local_metal::bonsai::HadamardDirection;
use local_metal::bonsai_ops::{Int8Matrix, RmsNormParams};
use local_metal::buffer::MetalBuffer;

use super::super::weights::MatrixWeight;
use super::super::{ATTENTION, FFN, KV, QUERY_GATE, WIDTH};
use super::{BonsaiMtp, Shared};

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
    #[allow(clippy::too_many_lines)]
    pub fn encode(
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
        let norm = |batch: &mut CommandBatch, input: &MetalBuffer, w: &MetalBuffer, output| {
            shared.ops.rms_norm(
                batch,
                input,
                w,
                output,
                RmsNormParams {
                    dimension: WIDTH as u32,
                    rows: count,
                    stride: WIDTH as u32,
                    epsilon: shared.epsilon,
                    weight_offset: 0,
                },
            )
        };
        let dense = |batch: &mut CommandBatch,
                     matrix: &MatrixWeight,
                     out_rows: usize,
                     columns: usize,
                     input: &MetalBuffer,
                     output: &MetalBuffer| {
            shared.ops.int8_matmul(
                batch,
                Int8Matrix {
                    weights: &matrix.weights,
                    scales: &matrix.scales,
                    rows: out_rows as u32,
                    columns: columns as u32,
                },
                input,
                output,
                count,
            )
        };
        shared.kernels.transform(
            batch,
            shared.input_rotation,
            &scratch.embedding,
            &scratch.embedded,
            count,
            HadamardDirection::Inverse,
        )?;
        norm(
            batch,
            &scratch.embedded,
            &weights.embedding_norm,
            &scratch.normalized_embedding,
        )?;
        norm(
            batch,
            hidden_in,
            &weights.hidden_norm,
            &scratch.normalized_hidden,
        )?;
        dense(
            batch,
            &weights.fc_embedding,
            WIDTH,
            WIDTH,
            &scratch.normalized_embedding,
            &scratch.hidden,
        )?;
        dense(
            batch,
            &weights.fc_hidden,
            WIDTH,
            WIDTH,
            &scratch.normalized_hidden,
            &scratch.branch,
        )?;
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

        norm(
            batch,
            &scratch.hidden,
            &weights.input_norm,
            &scratch.normalized,
        )?;
        for (matrix, output) in [
            (&weights.key, &scratch.key),
            (&weights.value, &scratch.value),
        ] {
            dense(batch, matrix, KV, WIDTH, &scratch.normalized, output)?;
        }
        if !logits {
            // Committed rows only feed later drafts through attention, so the
            // query, attention, FFN and final norm outputs are never read.
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
        dense(
            batch,
            &weights.query_gate,
            QUERY_GATE,
            WIDTH,
            &scratch.normalized,
            &scratch.query_gate,
        )?;
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
        dense(
            batch,
            &weights.output,
            WIDTH,
            ATTENTION,
            &scratch.attention_output,
            &scratch.branch,
        )?;
        residual(batch)?;

        norm(
            batch,
            &scratch.hidden,
            &weights.post_attention_norm,
            &scratch.normalized,
        )?;
        dense(
            batch,
            &weights.gate,
            FFN,
            WIDTH,
            &scratch.normalized,
            &scratch.ffn_gate,
        )?;
        dense(
            batch,
            &weights.up,
            FFN,
            WIDTH,
            &scratch.normalized,
            &scratch.ffn_up,
        )?;
        shared.ops.swiglu(
            batch,
            &scratch.ffn_gate,
            &scratch.ffn_up,
            &scratch.ffn_product,
            FFN as u32 * count,
        )?;
        dense(
            batch,
            &weights.down,
            WIDTH,
            FFN,
            &scratch.ffn_product,
            &scratch.branch,
        )?;
        residual(batch)?;

        norm(
            batch,
            &scratch.hidden,
            &weights.final_norm,
            &scratch.predicted,
        )?;
        if logits {
            // The shared PTQ1 head consumes rotated rows; project only the
            // newest row, which is the sole draft candidate.
            let last = if count == 1 {
                &scratch.predicted
            } else {
                let row_bytes = WIDTH * size_of::<f32>();
                batch.blit_buffer_copies([BufferCopyRequest {
                    source: &scratch.predicted,
                    source_offset: (rows - 1) * row_bytes,
                    destination: &scratch.embedded,
                    destination_offset: 0,
                    size: row_bytes,
                }])?;
                &scratch.embedded
            };
            shared.kernels.transform(
                batch,
                shared.input_rotation,
                last,
                &scratch.rotated,
                1,
                HadamardDirection::Forward,
            )?;
            shared.kernels.matmul(
                batch,
                shared.output.ptq1_matrix()?,
                &scratch.rotated,
                &self.logits,
                1,
            )?;
        }
        Ok(())
    }
}
