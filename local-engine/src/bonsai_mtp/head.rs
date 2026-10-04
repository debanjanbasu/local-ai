use local_metal::batch::{BufferCopyRequest, CommandBatch};
use local_metal::bonsai::{BonsaiKernels, HadamardDirection, SignedHadamard};
use local_metal::bonsai_ops::{AttentionWorkspace, BonsaiOps, Int8Matrix, RmsNormParams};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;

use crate::bonsai::BonsaiMetalTensor;
use crate::bonsai_native::grow_cache;

use super::MtpSettings;
use super::cache::HeadCacheStatus;
use super::weights::{MatrixWeight, Weights};
use super::{ATTENTION, FFN, KV, KV_TOKEN_BYTES, QUERY_GATE, WIDTH};

/// Target-owned resources the head shares: kernels, rotations, and the PTQ1
/// output matrix. Borrowed per call so the model keeps single ownership.
pub struct Shared<'a> {
    pub kernels: &'a BonsaiKernels,
    pub ops: &'a BonsaiOps,
    pub input_rotation: &'a SignedHadamard,
    pub output: &'a BonsaiMetalTensor,
    pub epsilon: f32,
    pub rope_base: f32,
    pub capacity: usize,
}

struct Scratch {
    rows: usize,
    embedding: MetalBuffer,
    embedded: MetalBuffer,
    hidden_in: MetalBuffer,
    normalized_embedding: MetalBuffer,
    normalized_hidden: MetalBuffer,
    hidden: MetalBuffer,
    normalized: MetalBuffer,
    branch: MetalBuffer,
    query_gate: MetalBuffer,
    key: MetalBuffer,
    value: MetalBuffer,
    query: MetalBuffer,
    gate: MetalBuffer,
    attention_output: MetalBuffer,
    ffn_gate: MetalBuffer,
    ffn_up: MetalBuffer,
    ffn_product: MetalBuffer,
    predicted: MetalBuffer,
    rotated: MetalBuffer,
    attention: AttentionWorkspace,
}

impl Scratch {
    fn new(context: &MetalContext, rows: usize, kv_allocated: usize) -> crate::Result<Self> {
        let floats = |count| MetalBuffer::empty(context.device(), count * rows * size_of::<f32>());
        Ok(Self {
            rows,
            embedding: floats(WIDTH)?,
            embedded: floats(WIDTH)?,
            hidden_in: floats(WIDTH)?,
            normalized_embedding: floats(WIDTH)?,
            normalized_hidden: floats(WIDTH)?,
            hidden: floats(WIDTH)?,
            normalized: floats(WIDTH)?,
            branch: floats(WIDTH)?,
            query_gate: floats(QUERY_GATE)?,
            key: floats(KV)?,
            value: floats(KV)?,
            query: floats(ATTENTION)?,
            gate: floats(ATTENTION)?,
            attention_output: floats(ATTENTION)?,
            ffn_gate: floats(FFN)?,
            ffn_up: floats(FFN)?,
            ffn_product: floats(FFN)?,
            predicted: floats(WIDTH)?,
            rotated: floats(WIDTH)?,
            attention: AttentionWorkspace::new(context, kv_allocated as u32)?,
        })
    }
}

pub struct BonsaiMtp {
    pub depth: usize,
    weights: Weights,
    scratch: Scratch,
    key_cache: MetalBuffer,
    value_cache: MetalBuffer,
    /// Target output-normalized hidden of the last committed token; zero after reset.
    prev_hidden: MetalBuffer,
    /// Draft logits for the newest row, projected through the shared head.
    pub logits: MetalBuffer,
    pub weight_bytes: u64,
    pub head_cache: HeadCacheStatus,
}

impl BonsaiMtp {
    /// The F16 caches and attention workspace start at `initial_tokens` and
    /// grow through [`Self::grow_caches`] as the target's do.
    pub fn load(
        context: &MetalContext,
        settings: &MtpSettings,
        rows: usize,
        capacity: usize,
        initial_tokens: usize,
        vocab: usize,
    ) -> crate::Result<Self> {
        if rows < settings.depth + 1 {
            return Err(crate::Error::InvalidArgument(
                "MTP scratch rows must cover the seed and every draft".into(),
            ));
        }
        if initial_tokens == 0 || initial_tokens > capacity {
            return Err(crate::Error::InvalidArgument(
                "MTP initial KV allocation must be within the context".into(),
            ));
        }
        let (weights, weight_bytes, head_cache) = Weights::load(context.device(), settings)?;
        Ok(Self {
            depth: settings.depth,
            weights,
            scratch: Scratch::new(context, rows, initial_tokens)?,
            key_cache: MetalBuffer::empty(context.device(), initial_tokens * KV_TOKEN_BYTES)?,
            value_cache: MetalBuffer::empty(context.device(), initial_tokens * KV_TOKEN_BYTES)?,
            prev_hidden: MetalBuffer::empty(context.device(), WIDTH * size_of::<f32>())?,
            logits: MetalBuffer::empty(context.device(), vocab * size_of::<f32>())?,
            weight_bytes,
            head_cache,
        })
    }

    /// F16 cache bytes at `capacity` tokens: the head is one layer, so its
    /// caches stay F16 whatever format the target's sixteen use.
    pub const fn state_bytes(capacity: usize) -> usize {
        2 * capacity * KV_TOKEN_BYTES
    }

    /// Move the first `used` cache rows into caches holding `capacity` tokens.
    pub fn grow_caches(
        &mut self,
        context: &MetalContext,
        used: usize,
        capacity: usize,
    ) -> crate::Result<()> {
        let used = used * KV_TOKEN_BYTES;
        let bytes = capacity * KV_TOKEN_BYTES;
        self.key_cache = grow_cache(context, &self.key_cache, used, bytes)?;
        self.value_cache = grow_cache(context, &self.value_cache, used, bytes)?;
        self.scratch.attention = AttentionWorkspace::new(context, capacity as u32)?;
        Ok(())
    }

    pub fn reset(&self) {
        self.prev_hidden.clear();
    }

    pub const fn predicted(&self) -> &MetalBuffer {
        &self.scratch.predicted
    }

    /// F16 key and value caches, for tests comparing encode paths.
    pub(crate) const fn kv_caches(&self) -> (&MetalBuffer, &MetalBuffer) {
        (&self.key_cache, &self.value_cache)
    }

    pub(crate) fn restore_kv(&self, position: usize, bytes: &[u8]) -> crate::Result<()> {
        let used = position * local_metal::bonsai_ops::KvFormat::F16.token_bytes();
        if bytes.len() != used * 2 || used > self.key_cache.length() {
            return Err(crate::Error::InvalidFormat(
                "invalid prompt snapshot MTP K/V size".into(),
            ));
        }
        self.key_cache.copy_from_bytes(&bytes[..used], 0);
        self.value_cache.copy_from_bytes(&bytes[used..], 0);
        Ok(())
    }

    /// F16 key/value caches and the predicted-hidden scratch, writable so
    /// tests can poison them before comparing encode paths.
    #[cfg(test)]
    pub(crate) const fn cache_buffers_mut(
        &mut self,
    ) -> (&mut MetalBuffer, &mut MetalBuffer, &mut MetalBuffer) {
        (
            &mut self.key_cache,
            &mut self.value_cache,
            &mut self.scratch.predicted,
        )
    }

    pub const fn prev_hidden(&self) -> &MetalBuffer {
        &self.prev_hidden
    }

    /// Hidden rows staged by [`Self::stage_hidden`], for multi-row `encode`.
    pub const fn hidden_in(&self) -> &MetalBuffer {
        &self.scratch.hidden_in
    }

    /// Host-visible embedding rows the caller fills with inverse-rotation inputs.
    pub fn embedding_rows(&mut self, rows: usize) -> crate::Result<&mut [f32]> {
        if rows == 0 || rows > self.scratch.rows {
            return Err(crate::Error::InvalidArgument(
                "MTP embedding rows exceed scratch".into(),
            ));
        }
        Ok(&mut self.scratch.embedding.as_mut_slice::<f32>()[..rows * WIDTH])
    }

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
