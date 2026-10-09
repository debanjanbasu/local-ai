use local_metal::bonsai::{BonsaiKernels, SignedHadamard};
use local_metal::bonsai_ops::{AttentionWorkspace, BonsaiOps};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;

use crate::bonsai::BonsaiMetalTensor;
use crate::bonsai_native::grow_cache;

use super::MtpSettings;
use super::weights::Weights;
use super::{ATTENTION, FFN, KV, KV_TOKEN_BYTES, QUERY_GATE, WIDTH};

mod rows;
mod step;

pub use rows::{HeadSpan, HeadState};

/// Target-owned resources the head shares: kernels, rotations, and the PTQ1
/// output matrix. Borrowed per call so the model keeps single ownership.
pub struct Shared<'a> {
    pub kernels: &'a BonsaiKernels,
    pub ops: &'a BonsaiOps,
    pub input_rotation: &'a SignedHadamard,
    /// Forward rotations for the 6144- and 17408-wide inputs the head's
    /// `o_proj` and `down_proj` consume.
    pub attention_rotation: &'a SignedHadamard,
    pub ffn_rotation: &'a SignedHadamard,
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
        let (weights, weight_bytes) = Weights::load(context.device(), settings)?;
        Ok(Self {
            depth: settings.depth,
            weights,
            scratch: Scratch::new(context, rows, initial_tokens)?,
            key_cache: MetalBuffer::empty(context.device(), initial_tokens * KV_TOKEN_BYTES)?,
            value_cache: MetalBuffer::empty(context.device(), initial_tokens * KV_TOKEN_BYTES)?,
            prev_hidden: MetalBuffer::empty(context.device(), WIDTH * size_of::<f32>())?,
            logits: MetalBuffer::empty(context.device(), vocab * size_of::<f32>())?,
            weight_bytes,
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
}

/// One sequence's head state, parked while another sequence's is resident.
pub struct HeadSequence {
    key_cache: MetalBuffer,
    value_cache: MetalBuffer,
    prev_hidden: MetalBuffer,
}

impl HeadSequence {
    /// Empty caches for `tokens` positions and a zero hidden handoff.
    pub fn new(context: &MetalContext, tokens: usize) -> crate::Result<Self> {
        let state = Self {
            key_cache: MetalBuffer::empty(context.device(), tokens * KV_TOKEN_BYTES)?,
            value_cache: MetalBuffer::empty(context.device(), tokens * KV_TOKEN_BYTES)?,
            prev_hidden: MetalBuffer::empty(context.device(), WIDTH * size_of::<f32>())?,
        };
        state.prev_hidden.clear();
        Ok(state)
    }

    /// Bytes this parked state holds.
    pub fn bytes(&self) -> usize {
        self.key_cache.length() + self.value_cache.length() + self.prev_hidden.length()
    }

    /// Grow a parked state's caches as [`BonsaiMtp::grow_caches`] does.
    pub fn grow(
        &mut self,
        context: &MetalContext,
        used: usize,
        capacity: usize,
    ) -> crate::Result<()> {
        let used = used * KV_TOKEN_BYTES;
        let bytes = capacity * KV_TOKEN_BYTES;
        self.key_cache = grow_cache(context, &self.key_cache, used, bytes)?;
        self.value_cache = grow_cache(context, &self.value_cache, used, bytes)?;
        Ok(())
    }
}

impl BonsaiMtp {
    /// Exchange the resident sequence state with `parked` (no copies), and
    /// make the attention workspace cover `capacity` tokens.
    pub fn swap_sequence(
        &mut self,
        context: &MetalContext,
        parked: &mut HeadSequence,
        capacity: usize,
    ) -> crate::Result<()> {
        std::mem::swap(&mut self.key_cache, &mut parked.key_cache);
        std::mem::swap(&mut self.value_cache, &mut parked.value_cache);
        std::mem::swap(&mut self.prev_hidden, &mut parked.prev_hidden);
        if self.scratch.attention.max_context() < capacity as u32 {
            self.scratch.attention = AttentionWorkspace::new(context, capacity as u32)?;
        }
        Ok(())
    }
}
