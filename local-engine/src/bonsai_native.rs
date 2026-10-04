//! Native Metal execution of the pinned Bonsai 2 27B text graph.
//!
//! PTQ1/BF16 matrices stay in their checkpoint representation, bound straight
//! from the GGUF mapping. Activations and recurrence are F32; full-attention
//! caches are F16. Requests start from empty state, with no context shifting
//! or silent truncation. Optional native MTP speculation verifies every draft
//! with the target before emitting it. [`crate::bonsai_model::BonsaiEngine`]
//! drives this model.

use std::time::Instant;

use local_metal::batch::{BufferCopyRequest, CommandBatch};
use local_metal::bonsai::{
    BonsaiKernels, HadamardDirection, PTQ1_BLOCK_BYTES, PTQ1_BLOCK_ELEMENTS, SignedHadamard,
    decode_ptq1_row,
};
use local_metal::bonsai_ops::{
    AttentionKernel, AttentionWorkspace, Bf16Matrix, BonsaiOps, KvFormat, KvLayout,
    MAX_PREFILL_TOKENS, RmsNormParams,
};
use local_metal::buffer::MetalBuffer;
use local_metal::context::{DeviceCaps, MetalContext};
use local_metal::sampling::GpuTopK;
use local_metal::shaders::ShaderLibrary;

use crate::bonsai::{
    BonsaiMetalTensor, BonsaiPackage, BonsaiTensorType, FFN, FULL_INTERVAL, LAYERS,
    TRAINING_CONTEXT, VOCAB, WIDTH, validate_profile,
};
use crate::bonsai_model::{BonsaiInfo, CancelToken, SpeculativeBatch, draft_depth};
use crate::bonsai_mtp::{BonsaiMtp, DRAFT_CHAIN_MIN_MARGIN, MtpSettings, Shared as MtpShared};
use crate::bonsai_ngram::NgramSettings;
use crate::sampler::{Sampler, SamplingResult, verify_greedy_drafts};
use crate::{MtpStats, NgramStats};

mod block;
mod checkpoint;
mod layers;
mod speculation;

/// What a loaded head costs, for the policy record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculationInfo {
    pub depth: usize,
    pub max_draft_rows: usize,
    pub head_bytes: u64,
    pub head_cache: &'static str,
    pub checkpoint_bytes: usize,
}

const GDN_STATE_BYTES: usize = 48 * 128 * 128 * 4;
const RECURRENT_LAYERS: usize = LAYERS - LAYERS / FULL_INTERVAL;
pub const PROMPT_CHECKPOINT_BYTES: usize =
    RECURRENT_LAYERS * (GDN_STATE_BYTES + CONV_STATE_BYTES) + WIDTH * size_of::<f32>();

/// Recurrent state at one exact token boundary. Full-attention and MTP K/V
/// rows are position indexed and remain in their existing allocations.
pub struct PromptCheckpoint {
    pub position: usize,
    recurrent: Vec<(MetalBuffer, MetalBuffer)>,
    mtp_prev_hidden: Option<MetalBuffer>,
}

impl PromptCheckpoint {
    fn make_volatile(&self) {
        for (state, history) in &self.recurrent {
            state.make_volatile();
            history.make_volatile();
        }
        if let Some(hidden) = &self.mtp_prev_hidden {
            hidden.make_volatile();
        }
    }

    fn make_nonvolatile(&self) -> bool {
        let mut resident = true;
        for (state, history) in &self.recurrent {
            resident &= state.make_nonvolatile();
            resident &= history.make_nonvolatile();
        }
        if let Some(hidden) = &self.mtp_prev_hidden {
            resident &= hidden.make_nonvolatile();
        }
        resident
    }

    #[cfg(test)]
    pub(crate) fn discard(&self) {
        for (state, history) in &self.recurrent {
            state.discard();
            history.discard();
        }
        if let Some(hidden) = &self.mtp_prev_hidden {
            hidden.discard();
        }
    }
}

/// A CPU-visible Metal copy of a complete prompt boundary. The single backing
/// allocation is volatile whenever the snapshot is idle.
pub struct HostPromptSnapshot {
    position: usize,
    layout: String,
    buffer: MetalBuffer,
    section_lengths: [usize; 4],
}

impl HostPromptSnapshot {
    pub(crate) const fn position(&self) -> usize {
        self.position
    }

    fn from_snapshot(model: &BonsaiModel, snapshot: &PromptSnapshot) -> crate::Result<Self> {
        let section_lengths = [
            snapshot.recurrent.len(),
            snapshot.mtp_prev_hidden.len(),
            snapshot.target_kv.len(),
            snapshot.mtp_kv.len(),
        ];
        let mut buffer = MetalBuffer::empty(model.context.device(), section_lengths.iter().sum())?;
        let bytes = buffer.as_mut_slice::<u8>();
        let mut offset = 0;
        for section in [
            &snapshot.recurrent,
            &snapshot.mtp_prev_hidden,
            &snapshot.target_kv,
            &snapshot.mtp_kv,
        ] {
            bytes[offset..offset + section.len()].copy_from_slice(section);
            offset += section.len();
        }
        buffer.make_volatile();
        Ok(Self {
            position: snapshot.position,
            layout: snapshot.layout.clone(),
            buffer,
            section_lengths,
        })
    }

    fn make_nonvolatile(&self) -> bool {
        self.buffer.make_nonvolatile()
    }

    fn snapshot(&self) -> crate::Result<PromptSnapshot> {
        let bytes = self.buffer.as_slice::<u8>();
        let mut offset = 0;
        let mut take = |length: usize| {
            let result = PageBytes::concat([&bytes[offset..offset + length]]);
            offset += length;
            result
        };
        Ok(PromptSnapshot {
            position: self.position,
            layout: self.layout.clone(),
            recurrent: take(self.section_lengths[0])?,
            mtp_prev_hidden: take(self.section_lengths[1])?,
            target_kv: take(self.section_lengths[2])?,
            mtp_kv: take(self.section_lengths[3])?,
        })
    }

    fn make_volatile(&self) {
        self.buffer.make_volatile();
    }

    #[cfg(test)]
    pub(crate) fn discard(&self) {
        self.buffer.discard();
    }
}

/// Page-backed bytes for large transient snapshot sections.
///
/// Anonymous mappings return to the OS on drop; Darwin's allocator otherwise retains
/// freed large blocks, inflating an idle server by the size of the last
/// snapshot (1.2 GB after an 8K-token prompt).
pub struct PageBytes(Option<memmap2::MmapMut>);

impl PageBytes {
    pub fn zeroed(len: usize) -> std::io::Result<Self> {
        Ok(Self(
            (len > 0)
                .then(|| memmap2::MmapMut::map_anon(len))
                .transpose()?,
        ))
    }

    pub fn concat<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> std::io::Result<Self> {
        let parts: Vec<&[u8]> = parts.into_iter().collect();
        let mut bytes = Self::zeroed(parts.iter().map(|part| part.len()).sum())?;
        let mut offset = 0;
        for part in parts {
            bytes[offset..offset + part.len()].copy_from_slice(part);
            offset += part.len();
        }
        Ok(bytes)
    }
}

impl std::ops::Deref for PageBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.0.as_deref().unwrap_or_default()
    }
}

impl AsRef<[u8]> for PageBytes {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl std::ops::DerefMut for PageBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.0.as_deref_mut().unwrap_or_default()
    }
}

/// CPU-owned, persistence-safe copy of one exact recurrent/K/V boundary.
pub struct PromptSnapshot {
    pub position: usize,
    pub layout: String,
    pub recurrent: PageBytes,
    pub mtp_prev_hidden: PageBytes,
    pub target_kv: PageBytes,
    pub mtp_kv: PageBytes,
}
const CONV_STATE_BYTES: usize = 10_240 * 3 * 4;
/// Bytes per token of one F16 cache: the MTP head's format, and the target's
/// default.
#[cfg(test)]
const KV_TOKEN_BYTES: usize = local_metal::bonsai_ops::KvFormat::F16.token_bytes();

/// Tokens each K/V cache holds before its first growth: 1,024 keeps short
/// requests at 2 MiB per F16 cache instead of the context-proportional
/// allocation Metal wires eagerly.
pub const DEFAULT_KV_INITIAL_TOKENS: usize = 1024;

/// How the sixteen full-attention layers store keys and values.
///
/// Attention only ever reads the written prefix, so caches are allocated for
/// `initial_tokens` and doubled on demand up to the context; resident memory
/// follows the request length, not `--context`. The head's own single-layer
/// cache stays F16 and grows alongside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvOptions {
    pub layout: KvLayout,
    pub initial_tokens: usize,
}

impl Default for KvOptions {
    fn default() -> Self {
        Self {
            layout: KvLayout::F16,
            initial_tokens: DEFAULT_KV_INITIAL_TOKENS,
        }
    }
}

/// Allocate a cache of `bytes` and copy the written `used` prefix of `old`
/// into it. Caches never read past the written prefix, so the tail is left
/// uninitialized.
pub fn grow_cache(
    context: &MetalContext,
    old: &MetalBuffer,
    used: usize,
    bytes: usize,
) -> crate::Result<MetalBuffer> {
    if used > old.length() || used > bytes {
        return Err(crate::Error::InvalidArgument(
            "KV cache growth would drop written rows".into(),
        ));
    }
    let new = MetalBuffer::empty(context.device(), bytes)?;
    if used > 0 {
        let mut batch = CommandBatch::new(context)?;
        batch.blit_buffer_copies([BufferCopyRequest {
            source: old,
            source_offset: 0,
            destination: &new,
            destination_offset: 0,
            size: used,
        }])?;
        batch.commit_and_wait()?;
    }
    Ok(new)
}

struct RecurrentLayer {
    qkv: BonsaiMetalTensor,
    gate: BonsaiMetalTensor,
    alpha: BonsaiMetalTensor,
    beta: BonsaiMetalTensor,
    convolution: MetalBuffer,
    decay: MetalBuffer,
    dt: MetalBuffer,
    norm: MetalBuffer,
    output: BonsaiMetalTensor,
    history: MetalBuffer,
    state: MetalBuffer,
}

struct FullAttentionLayer {
    query_gate: BonsaiMetalTensor,
    key: BonsaiMetalTensor,
    value: BonsaiMetalTensor,
    query_norm: MetalBuffer,
    key_norm: MetalBuffer,
    output: BonsaiMetalTensor,
    key_cache: MetalBuffer,
    value_cache: MetalBuffer,
}

enum AttentionLayer {
    Recurrent(RecurrentLayer),
    Full(FullAttentionLayer),
}

struct Layer {
    attention: AttentionLayer,
    attention_norm: MetalBuffer,
    post_attention_norm: MetalBuffer,
    gate: BonsaiMetalTensor,
    up: BonsaiMetalTensor,
    down: BonsaiMetalTensor,
}

struct Scratch {
    embedding: MetalBuffer,
    hidden: MetalBuffer,
    normalized: MetalBuffer,
    rotated_hidden: MetalBuffer,
    branch: MetalBuffer,
    query_gate: MetalBuffer,
    query: MetalBuffer,
    key: MetalBuffer,
    value: MetalBuffer,
    gate: MetalBuffer,
    convolved: MetalBuffer,
    alpha: MetalBuffer,
    raw_beta: MetalBuffer,
    decay: MetalBuffer,
    beta: MetalBuffer,
    recurrent_output: MetalBuffer,
    attention_output: MetalBuffer,
    rotated_attention: MetalBuffer,
    ffn_gate: MetalBuffer,
    ffn_up: MetalBuffer,
    ffn_product: MetalBuffer,
    rotated_ffn: MetalBuffer,
    logits: MetalBuffer,
    attention: AttentionWorkspace,
}

impl Scratch {
    fn new(context: &MetalContext, kv_allocated: usize, tokens: usize) -> crate::Result<Self> {
        let floats =
            |count| MetalBuffer::empty(context.device(), count * tokens * size_of::<f32>());

        // Attention and FFN execute serially within each layer. Reuse three
        // FFN-sized buffers for mutually exclusive intermediates; cloning a
        // MetalBuffer retains the same MTLBuffer and adds no allocation. The
        // full-attention query and gate need separate storage because the
        // prepare kernel writes them while reading query_gate/key/value.
        let work_a = floats(FFN)?;
        let work_b = floats(FFN)?;
        let work_c = floats(FFN)?;
        let query = floats(6144)?;
        let gate = floats(6144)?;
        Ok(Self {
            embedding: floats(WIDTH)?,
            hidden: floats(WIDTH)?,
            normalized: floats(WIDTH)?,
            rotated_hidden: floats(WIDTH)?,
            branch: floats(WIDTH)?,
            query_gate: work_a.clone(),
            query,
            key: work_b.clone(),
            value: work_c.clone(),
            gate,
            convolved: work_b.clone(),
            alpha: floats(48)?,
            raw_beta: floats(48)?,
            decay: floats(48)?,
            beta: floats(48)?,
            recurrent_output: work_a.clone(),
            attention_output: work_b.clone(),
            rotated_attention: work_a.clone(),
            ffn_gate: work_a.clone(),
            ffn_up: work_b,
            ffn_product: work_c,
            rotated_ffn: work_a,
            logits: MetalBuffer::empty(context.device(), VOCAB * size_of::<f32>())?,
            attention: AttentionWorkspace::new(context, kv_allocated as u32)?,
        })
    }
}

const RECURRENCE_FACTOR_WIDTH: usize = 10_240;
const RECURRENCE_SCALAR_WIDTH: usize = 48;

/// Start-of-round state and compact recurrence inputs used to reconstruct an
/// accepted prefix without retaining a full state checkpoint for every row.
struct RollbackLayer {
    state: MetalBuffer,
    history: MetalBuffer,
    inputs: MetalBuffer,
    decay: MetalBuffer,
    beta: MetalBuffer,
}

/// Target-side verification buffers for a block of up to `depth + 1` rows.
/// Independent of the MTP head: the target owns the checkpoints it rolls
/// back to and the per-row logits it samples, whatever proposed the drafts.
struct Verifier {
    /// One snapshot plus `depth` rows of recurrence factors per GDN layer.
    rollback: Vec<RollbackLayer>,
    rows: usize,
    max_rows: usize,
    /// Contiguous per-row target logits produced by the block matmul.
    verify_logits: MetalBuffer,
}

impl Verifier {
    fn new(context: &MetalContext, initial_depth: usize, max_depth: usize) -> crate::Result<Self> {
        let empty = |bytes| MetalBuffer::empty(context.device(), bytes);
        let rows = initial_depth + 1;
        let rollback = (0..LAYERS - LAYERS / FULL_INTERVAL)
            .map(|_| {
                Ok(RollbackLayer {
                    state: empty(GDN_STATE_BYTES)?,
                    history: empty(CONV_STATE_BYTES)?,
                    inputs: empty(initial_depth * RECURRENCE_FACTOR_WIDTH * size_of::<f32>())?,
                    decay: empty(initial_depth * RECURRENCE_SCALAR_WIDTH * size_of::<f32>())?,
                    beta: empty(initial_depth * RECURRENCE_SCALAR_WIDTH * size_of::<f32>())?,
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(Self {
            rollback,
            rows,
            max_rows: max_depth + 1,
            verify_logits: empty(rows * VOCAB * size_of::<f32>())?,
        })
    }

    fn reserve(&mut self, context: &MetalContext, rows: usize) -> crate::Result<()> {
        if rows <= self.rows {
            return Ok(());
        }
        if rows > self.max_rows {
            return Err(crate::Error::InvalidArgument(
                "verify block exceeds configured depth".into(),
            ));
        }
        let rows = rows.next_power_of_two().min(self.max_rows);
        *self = Self::new(context, rows - 1, self.max_rows - 1)?;
        Ok(())
    }

    const fn checkpoint_bytes(depth: usize) -> usize {
        if depth == 0 {
            return 0;
        }
        (LAYERS - LAYERS / FULL_INTERVAL)
            * (GDN_STATE_BYTES
                + CONV_STATE_BYTES
                + depth
                    * (RECURRENCE_FACTOR_WIDTH + 2 * RECURRENCE_SCALAR_WIDTH)
                    * size_of::<f32>())
    }
}

/// Native speculation resources. The target verifies the seed and every draft
/// in one block; only rows it sampled itself are ever committed.
struct Speculation {
    mtp: BonsaiMtp,
    verifier: Verifier,
}

impl Speculation {
    fn new(context: &MetalContext, mtp: BonsaiMtp, verify_depth: usize) -> crate::Result<Self> {
        let initial_depth = mtp.depth.min(verify_depth);
        Ok(Self {
            verifier: Verifier::new(context, initial_depth, verify_depth)?,
            mtp,
        })
    }

    const fn checkpoint_bytes(depth: usize) -> usize {
        Verifier::checkpoint_bytes(depth)
    }

    /// Bytes beyond the head weights: its KV cache, checkpoints, and verify logits.
    const fn extra_bytes(depth: usize, capacity: usize) -> usize {
        BonsaiMtp::state_bytes(capacity)
            + Self::checkpoint_bytes(depth)
            + (depth + 1) * VOCAB * size_of::<f32>()
    }
}

/// What a forward block leaves behind for the caller besides committed state.
#[derive(Clone, Copy)]
enum BlockOutput<'a> {
    None,
    /// Every row output-normalized into `scratch.normalized`, no logits: what
    /// the MTP head needs from a committed block that produces no sample.
    Hidden,
    /// Logits of the final row in `scratch.logits`.
    LastLogits,
    /// Logits of every row in `Verifier::verify_logits`, with recurrent
    /// compact rollback factors for each non-final row. Needs at least two rows.
    Verify(&'a Verifier),
}

/// The loaded native graph plus its per-request state.
pub struct BonsaiModel {
    context: MetalContext,
    package: BonsaiPackage,
    info: BonsaiInfo,
    kernels: BonsaiKernels,
    ops: BonsaiOps,
    sampling: GpuTopK,
    input_rotation: SignedHadamard,
    attention_rotation: SignedHadamard,
    ffn_rotation: SignedHadamard,
    layers: Vec<Layer>,
    output: BonsaiMetalTensor,
    output_norm: MetalBuffer,
    scratch: Scratch,
    /// Row capacity of `scratch`: the prefill chunk or a verify block.
    block_rows: usize,
    kv_layout: KvLayout,
    /// Tokens every K/V cache (target layers and head) currently holds.
    kv_allocated: usize,
    speculation: Option<Speculation>,
    ngram_verifier: Option<Verifier>,
    epsilon: f32,
    rope_base: f32,
    position: usize,
    /// Cancellation flag the engine polls between GPU dispatches. Installed per
    /// request by [`Self::set_cancel`].
    cancel: CancelToken,
    /// Whether a cancelled request made the model stop before submitting a
    /// chunk or a round. Reported once, by [`Self::take_cancel_observed`].
    cancel_observed: bool,
    /// GPU execution time of every command buffer this model waited on since
    /// the last [`Self::take_gpu_time`], from Metal's own timestamps: what the
    /// GPU was busy for, excluding CPU encoding and inter-buffer gaps.
    gpu_time: std::cell::Cell<std::time::Duration>,
}

impl BonsaiModel {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn load(
        package: BonsaiPackage,
        capacity: usize,
        prefill_chunk: usize,
        attention_kernel: Option<AttentionKernel>,
        mtp: Option<&MtpSettings>,
        ngram: NgramSettings,
        mut kv: KvOptions,
    ) -> crate::Result<Self> {
        validate_profile(&package)?;
        if kv.initial_tokens == 0 {
            return Err(crate::Error::InvalidArgument(
                "initial KV allocation must hold at least one token".into(),
            ));
        }
        let context = MetalContext::new()?;
        let ngram_depth = if ngram.enabled { ngram.max_drafts } else { 0 };
        let verify_depth = mtp.map_or(0, |settings| settings.depth).max(ngram_depth);
        let mtp_bytes = mtp
            .map(|settings| -> crate::Result<u64> {
                Ok(std::fs::metadata(&settings.path)?.len()
                    + Speculation::checkpoint_bytes(verify_depth) as u64
                    + ((verify_depth + 1) * VOCAB * size_of::<f32>()) as u64)
            })
            .transpose()?
            .unwrap_or_else(|| {
                if verify_depth == 0 {
                    0
                } else {
                    (Verifier::checkpoint_bytes(verify_depth)
                        + (verify_depth + 1) * VOCAB * size_of::<f32>()) as u64
                }
            });
        let caps = context.caps();
        let mut info = memory_plan(
            &package,
            capacity,
            prefill_chunk,
            mtp_bytes,
            mtp.is_some(),
            kv.layout,
            &caps,
        )?;
        if capacity == 0 && info.context < 32_768 && kv.layout == KvLayout::F16 {
            kv.layout = KvLayout {
                key: KvFormat::Q8,
                value: KvFormat::Q8,
            };
            info = memory_plan(
                &package,
                capacity,
                prefill_chunk,
                mtp_bytes,
                mtp.is_some(),
                kv.layout,
                &caps,
            )?;
            info.context_reason.push_str(
                "; selected q8 K/V because F16 could not reach the 32,768-token minimum useful coding context",
            );
        }
        let capacity = info.context;
        let kv_allocated = kv.initial_tokens.min(capacity);
        let block_rows = prefill_chunk.max(verify_depth + 1);
        let shaders = ShaderLibrary::new(context.device())?;
        let matrix = |name: &str| package.metal_tensor(context.device(), name);
        let plain = |name: &str| -> crate::Result<MetalBuffer> {
            let bytes = package.bytes(name)?;
            if bytes
                .as_chunks::<4>()
                .0
                .iter()
                .any(|&value| !f32::from_le_bytes(value).is_finite())
            {
                return invalid(format!("non-finite F32 weight {name}"));
            }
            // Tiny F32 exceptions are copied unchanged, avoiding offset plumbing
            // in every normalization/recurrence kernel. Never add one to GGUF norms.
            Ok(MetalBuffer::from_slice(context.device(), bytes)?)
        };
        let empty = |bytes| MetalBuffer::empty(context.device(), bytes);
        let mut layers = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            let matrix = |suffix| matrix(&format!("blk.{index}.{suffix}"));
            let plain = |suffix| plain(&format!("blk.{index}.{suffix}"));
            let attention = if (index + 1).is_multiple_of(FULL_INTERVAL) {
                AttentionLayer::Full(FullAttentionLayer {
                    query_gate: matrix("attn_q.weight")?,
                    key: matrix("attn_k.weight")?,
                    value: matrix("attn_v.weight")?,
                    query_norm: plain("attn_q_norm.weight")?,
                    key_norm: plain("attn_k_norm.weight")?,
                    output: matrix("attn_output.weight")?,
                    key_cache: empty(kv_allocated * kv.layout.key.token_bytes())?,
                    value_cache: empty(kv_allocated * kv.layout.value.token_bytes())?,
                })
            } else {
                AttentionLayer::Recurrent(RecurrentLayer {
                    qkv: matrix("attn_qkv.weight")?,
                    gate: matrix("attn_gate.weight")?,
                    alpha: matrix("ssm_alpha.weight")?,
                    beta: matrix("ssm_beta.weight")?,
                    convolution: plain("ssm_conv1d.weight")?,
                    decay: plain("ssm_a")?,
                    dt: plain("ssm_dt.bias")?,
                    norm: plain("ssm_norm.weight")?,
                    output: matrix("ssm_out.weight")?,
                    history: empty(CONV_STATE_BYTES)?,
                    state: empty(GDN_STATE_BYTES)?,
                })
            };
            layers.push(Layer {
                attention,
                attention_norm: plain("attn_norm.weight")?,
                post_attention_norm: plain("post_attention_norm.weight")?,
                gate: matrix("ffn_gate.weight")?,
                up: matrix("ffn_up.weight")?,
                down: matrix("ffn_down.weight")?,
            });
        }
        let speculation = mtp
            .map(|settings| {
                let head = BonsaiMtp::load(
                    &context,
                    settings,
                    block_rows,
                    capacity,
                    kv_allocated,
                    VOCAB,
                )?;
                Speculation::new(&context, head, verify_depth)
            })
            .transpose()?;
        if let Some(settings) = mtp {
            let _ = settings;
            info.state_bytes += Speculation::extra_bytes(verify_depth, capacity);
        } else if verify_depth > 0 {
            info.state_bytes += Verifier::checkpoint_bytes(verify_depth)
                + (verify_depth + 1) * VOCAB * size_of::<f32>();
        }
        let ngram_verifier = (speculation.is_none() && verify_depth > 0)
            .then(|| Verifier::new(&context, verify_depth.min(3), verify_depth))
            .transpose()?;
        Ok(Self {
            info,
            kernels: BonsaiKernels::new(&context, &shaders)?,
            ops: match attention_kernel {
                Some(kernel) => BonsaiOps::new_with_attention_kernel(&context, &shaders, kernel)?,
                None => BonsaiOps::new(&context, &shaders)?,
            },
            sampling: GpuTopK::new(&context, &shaders, VOCAB)?,
            input_rotation: package.hadamard().upload(&context, WIDTH as u32)?,
            attention_rotation: package.hadamard().upload(&context, 6144)?,
            ffn_rotation: package.hadamard().upload(&context, FFN as u32)?,
            layers,
            output: matrix("output.weight")?,
            output_norm: plain("output_norm.weight")?,
            scratch: Scratch::new(&context, kv_allocated, block_rows)?,
            block_rows,
            kv_layout: kv.layout,
            kv_allocated,
            speculation,
            ngram_verifier,
            epsilon: package.metadata_f32("qwen35.attention.layer_norm_rms_epsilon")?,
            rope_base: package.metadata_f32("qwen35.rope.freq_base")?,
            position: 0,
            cancel: CancelToken::new(),
            cancel_observed: false,
            gpu_time: std::cell::Cell::new(std::time::Duration::ZERO),
            context,
            package,
        })
    }

    /// Commit `batch`, wait for it, and add its GPU execution time to the
    /// model's accumulator.
    fn finish(&self, batch: CommandBatch) -> crate::Result<()> {
        let gpu = batch.commit_async().wait_with_gpu_time()?;
        self.gpu_time.set(self.gpu_time.get().saturating_add(gpu));
        Ok(())
    }

    /// GPU-busy time accumulated since the previous call, then reset to zero.
    pub(crate) const fn take_gpu_time(&self) -> std::time::Duration {
        self.gpu_time.replace(std::time::Duration::ZERO)
    }

    pub(crate) fn reset(&mut self) {
        for layer in &self.layers {
            if let AttentionLayer::Recurrent(recurrent) = &layer.attention {
                recurrent.state.clear();
                recurrent.history.clear();
            }
        }
        if let Some(speculation) = &self.speculation {
            speculation.mtp.reset();
        }
        // Attention reads only the newly written prefix; unused KV tails need
        // not be touched, even when a shorter request follows a longer one.
        self.position = 0;
    }

    pub(crate) const fn position(&self) -> usize {
        self.position
    }

    /// Install the token polled between GPU dispatches, dropping any stop the
    /// previous request recorded.
    pub(crate) fn set_cancel(&mut self, cancel: CancelToken) {
        self.cancel = cancel;
        self.cancel_observed = false;
    }

    /// Whether a cancelled request stopped this model short of the work it
    /// asked for, consuming the record so it is reported exactly once.
    pub(crate) fn take_cancel_observed(&mut self) -> bool {
        std::mem::take(&mut self.cancel_observed)
    }

    /// Whether the next chunk or round must not be submitted, recording the stop
    /// for [`Self::take_cancel_observed`].
    ///
    /// This is the only place the token is read before submitting a dispatch.
    /// Polling here means a command buffer that is already committed always runs
    /// to completion: Metal has no cancellation API, so the realizable
    /// semantics are "submit no further work", never "abort work in flight".
    fn stop_for_cancel(&mut self) -> bool {
        if !self.cancel.is_cancelled() {
            return false;
        }
        self.cancel_observed = true;
        true
    }

    /// Copy all non-positional state at the current token boundary.
    pub(crate) fn prompt_checkpoint(&self) -> crate::Result<PromptCheckpoint> {
        let recurrent = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attention {
                AttentionLayer::Recurrent(layer) => Some(layer),
                AttentionLayer::Full(_) => None,
            })
            .map(|_| {
                Ok((
                    MetalBuffer::empty(self.context.device(), GDN_STATE_BYTES)?,
                    MetalBuffer::empty(self.context.device(), CONV_STATE_BYTES)?,
                ))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let mtp_prev_hidden = self
            .speculation
            .as_ref()
            .map(|_| MetalBuffer::empty(self.context.device(), WIDTH * size_of::<f32>()))
            .transpose()?;
        let checkpoint = PromptCheckpoint {
            position: self.position,
            recurrent,
            mtp_prev_hidden,
        };
        self.copy_prompt_state(&checkpoint, false)?;
        checkpoint.make_volatile();
        Ok(checkpoint)
    }

    /// Restore a checkpoint and logically truncate every positional K/V cache.
    pub(crate) fn restore_prompt_checkpoint(
        &mut self,
        checkpoint: &PromptCheckpoint,
    ) -> crate::Result<bool> {
        if !checkpoint.make_nonvolatile() {
            return Ok(false);
        }
        self.copy_prompt_state(checkpoint, true)?;
        checkpoint.make_volatile();
        self.position = checkpoint.position;
        Ok(true)
    }

    pub(crate) fn host_prompt_snapshot(
        &self,
        snapshot: &PromptSnapshot,
    ) -> crate::Result<HostPromptSnapshot> {
        HostPromptSnapshot::from_snapshot(self, snapshot)
    }

    pub(crate) fn restore_host_prompt_snapshot(
        &mut self,
        snapshot: &HostPromptSnapshot,
    ) -> crate::Result<bool> {
        if !snapshot.make_nonvolatile() {
            return Ok(false);
        }
        let state = snapshot.snapshot()?;
        let result = self.restore_prompt_snapshot(&state);
        snapshot.make_volatile();
        result.map(|()| true)
    }

    /// Copy an exact boundary to CPU memory. Cache tails are deliberately omitted.
    pub(crate) fn prompt_snapshot(&self) -> crate::Result<PromptSnapshot> {
        let mut recurrent = Vec::new();
        let mut target_kv = Vec::new();
        for layer in &self.layers {
            match &layer.attention {
                AttentionLayer::Recurrent(layer) => {
                    recurrent.push(layer.state.as_slice::<u8>());
                    recurrent.push(layer.history.as_slice::<u8>());
                }
                AttentionLayer::Full(layer) => {
                    target_kv.push(
                        &layer.key_cache.as_slice::<u8>()
                            [..self.position * self.kv_layout.key.token_bytes()],
                    );
                    target_kv.push(
                        &layer.value_cache.as_slice::<u8>()
                            [..self.position * self.kv_layout.value.token_bytes()],
                    );
                }
            }
        }
        let (mtp_prev_hidden, mtp_kv) = match &self.speculation {
            None => (PageBytes::zeroed(0)?, PageBytes::zeroed(0)?),
            Some(speculation) => {
                let mtp = &speculation.mtp;
                let key_bytes =
                    self.position * local_metal::bonsai_ops::KvFormat::F16.token_bytes();
                let (keys, values) = mtp.kv_caches();
                (
                    PageBytes::concat([mtp.prev_hidden().as_slice::<u8>()])?,
                    PageBytes::concat([
                        &keys.as_slice::<u8>()[..key_bytes],
                        &values.as_slice::<u8>()[..key_bytes],
                    ])?,
                )
            }
        };
        Ok(PromptSnapshot {
            position: self.position,
            layout: self.kv_layout.name(),
            recurrent: PageBytes::concat(recurrent)?,
            mtp_prev_hidden,
            target_kv: PageBytes::concat(target_kv)?,
            mtp_kv,
        })
    }

    /// Restore a CPU/disk snapshot, including all position-indexed K/V rows.
    pub(crate) fn restore_prompt_snapshot(
        &mut self,
        snapshot: &PromptSnapshot,
    ) -> crate::Result<()> {
        if snapshot.layout != self.kv_layout.name() || snapshot.position > self.info.context {
            return Err(crate::Error::InvalidArgument(
                "prompt snapshot K/V layout or position does not match this model".into(),
            ));
        }
        self.reserve_kv(snapshot.position, None)?;
        let mut recurrent = &*snapshot.recurrent;
        let mut target_kv = &*snapshot.target_kv;
        for layer in &mut self.layers {
            match &mut layer.attention {
                AttentionLayer::Recurrent(layer) => {
                    let (state, rest) =
                        recurrent.split_at_checked(GDN_STATE_BYTES).ok_or_else(|| {
                            crate::Error::InvalidFormat("truncated prompt snapshot state".into())
                        })?;
                    let (history, rest) =
                        rest.split_at_checked(CONV_STATE_BYTES).ok_or_else(|| {
                            crate::Error::InvalidFormat("truncated prompt snapshot history".into())
                        })?;
                    layer.state.copy_from_bytes(state, 0);
                    layer.history.copy_from_bytes(history, 0);
                    recurrent = rest;
                }
                AttentionLayer::Full(layer) => {
                    let key_bytes = snapshot.position * self.kv_layout.key.token_bytes();
                    let value_bytes = snapshot.position * self.kv_layout.value.token_bytes();
                    let (key, rest) = target_kv.split_at_checked(key_bytes).ok_or_else(|| {
                        crate::Error::InvalidFormat("truncated prompt snapshot keys".into())
                    })?;
                    let (value, rest) = rest.split_at_checked(value_bytes).ok_or_else(|| {
                        crate::Error::InvalidFormat("truncated prompt snapshot values".into())
                    })?;
                    layer.key_cache.copy_from_bytes(key, 0);
                    layer.value_cache.copy_from_bytes(value, 0);
                    target_kv = rest;
                }
            }
        }
        if !recurrent.is_empty() || !target_kv.is_empty() {
            return Err(crate::Error::InvalidFormat(
                "prompt snapshot has trailing state bytes".into(),
            ));
        }
        match (&mut self.speculation, snapshot.mtp_prev_hidden.is_empty()) {
            (Some(speculation), false) => {
                if snapshot.mtp_prev_hidden.len() != WIDTH * size_of::<f32>() {
                    return Err(crate::Error::InvalidFormat(
                        "invalid prompt snapshot MTP hidden size".into(),
                    ));
                }
                let key_bytes =
                    snapshot.position * local_metal::bonsai_ops::KvFormat::F16.token_bytes();
                if snapshot.mtp_kv.len() != key_bytes * 2 {
                    return Err(crate::Error::InvalidFormat(
                        "invalid prompt snapshot MTP K/V size".into(),
                    ));
                }
                speculation
                    .mtp
                    .prev_hidden()
                    .copy_from_bytes(&snapshot.mtp_prev_hidden, 0);
                speculation
                    .mtp
                    .restore_kv(snapshot.position, &snapshot.mtp_kv)?;
            }
            (None, true) => {}
            _ => {
                return Err(crate::Error::InvalidArgument(
                    "prompt snapshot MTP configuration does not match this model".into(),
                ));
            }
        }
        self.position = snapshot.position;
        Ok(())
    }

    const fn mtp_shared(&self) -> MtpShared<'_> {
        MtpShared {
            kernels: &self.kernels,
            ops: &self.ops,
            input_rotation: &self.input_rotation,
            output: &self.output,
            epsilon: self.epsilon,
            rope_base: self.rope_base,
            capacity: self.kv_allocated,
        }
    }

    pub(crate) const fn kv_layout(&self) -> KvLayout {
        self.kv_layout
    }

    /// Tokens the K/V caches currently hold; grows on demand up to the context.
    pub(crate) const fn kv_allocated(&self) -> usize {
        self.kv_allocated
    }

    /// Bytes currently allocated to the target's K/V caches.
    pub(crate) const fn kv_allocated_bytes(&self) -> usize {
        LAYERS / FULL_INTERVAL * self.kv_allocated * self.kv_layout.token_bytes()
    }

    /// Target K/V bytes once the caches have grown to the full context.
    pub(crate) const fn kv_max_bytes(&self) -> usize {
        LAYERS / FULL_INTERVAL * self.info.context * self.kv_layout.token_bytes()
    }

    pub(crate) const fn package(&self) -> &BonsaiPackage {
        &self.package
    }

    pub(crate) const fn info(&self) -> &BonsaiInfo {
        &self.info
    }

    pub(crate) const fn attention_kernel_name(&self) -> &'static str {
        self.ops.attention_kernel().name()
    }

    pub(crate) const fn prefill_kernel_name(&self) -> &'static str {
        self.kernels.prefill_kernel().name()
    }

    /// The loaded head's depth, weight bytes and checkpoint budget, when
    /// speculating.
    pub(crate) fn speculation(&self) -> Option<SpeculationInfo> {
        self.speculation
            .as_ref()
            .map(|speculation| SpeculationInfo {
                depth: speculation.mtp.depth,
                max_draft_rows: speculation.verifier.max_rows - 1,
                head_bytes: speculation.mtp.weight_bytes,
                head_cache: speculation.mtp.head_cache.name(),
                checkpoint_bytes: Speculation::checkpoint_bytes(speculation.verifier.max_rows - 1),
            })
    }

    /// Append prompt tokens in prefill-chunk blocks, leaving the final token's
    /// logits in scratch and the head (if any) caught up.
    ///
    /// A cancelled request stops before encoding the next chunk. The chunks
    /// already committed are not in-flight work this call can reclaim — Metal
    /// runs a committed command buffer to completion — so they stay committed
    /// and `position` keeps their exact total, which is a boundary the next
    /// request can still resume from. `Ok` rather than an error: cancellation
    /// is a successful outcome, reported by [`Self::take_cancel_observed`].
    pub(crate) fn prefill(&mut self, prompt: &[u32]) -> crate::Result<()> {
        let end = self.position + prompt.len();
        for block in prompt.chunks(self.info.prefill_chunk_size) {
            if self.stop_for_cancel() {
                return Ok(());
            }
            let output = if self.position + block.len() == end {
                BlockOutput::LastLogits
            } else if self.speculation.is_some() {
                // The head ingests every committed row's output-normalized
                // hidden, so intermediate blocks must produce them too.
                BlockOutput::Hidden
            } else {
                BlockOutput::None
            };
            // Head ingestion rides in the same command batch as the target
            // block: one GPU round trip per chunk instead of two.
            let mut speculation = self.speculation.take();
            let result = self.encode_block(block, output, speculation.as_mut());
            if let Some(speculation) = speculation {
                self.speculation = Some(speculation);
            }
            result?;
        }
        Ok(())
    }

    /// Plain one-token decode step producing logits.
    pub(crate) fn decode(&mut self, token: u32) -> crate::Result<()> {
        self.forward(token, true)
    }

    /// Sample from the logits of the newest committed token on the GPU.
    pub(crate) fn sample(&mut self, sampler: &mut Sampler) -> crate::Result<SamplingResult> {
        sampler.sample_buffer(&mut self.scratch.logits, 0, &self.context, &self.sampling)
    }

    /// One speculative round: draft up to `depth` tokens with the head, verify
    /// the seed and drafts in a single target block, commit the accepted
    /// prefix, and return the target's own samples for every committed row.
    pub(crate) fn speculative_step(
        &mut self,
        seed: u32,
        sampler: &mut Sampler,
        remaining: usize,
    ) -> crate::Result<SpeculativeBatch> {
        if self.stop_for_cancel() {
            return Ok(cancelled_speculative_batch());
        }
        let Some(mut speculation) = self.speculation.take() else {
            return Err(crate::Error::InvalidArgument(
                "MTP speculation is not enabled".into(),
            ));
        };
        let result = self.speculative_round(&mut speculation, seed, sampler, remaining);
        self.speculation = Some(speculation);
        result
    }

    /// Verify prompt-lookup drafts with the target and feed every committed
    /// row to the optional MTP head so its cache remains ready for fallback.
    ///
    /// A cancelled request submits no verify block: the model keeps the
    /// position it had before this call, and the empty batch says so.
    pub(crate) fn ngram_step(
        &mut self,
        seed: u32,
        drafts: &[u32],
        sampler: &mut Sampler,
    ) -> crate::Result<SpeculativeBatch> {
        if drafts.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "n-gram speculation needs at least one draft".into(),
            ));
        }
        if self.stop_for_cancel() {
            return Ok(cancelled_speculative_batch());
        }
        let mut speculation = self.speculation.take();
        let mut standalone = self.ngram_verifier.take();
        let inputs = std::iter::once(seed)
            .chain(drafts.iter().copied())
            .collect::<Vec<_>>();
        let start = self.position;
        self.reserve_kv(start + inputs.len(), speculation.as_mut())?;
        if let Some(value) = speculation.as_mut() {
            value.verifier.reserve(&self.context, inputs.len())?;
        } else if let Some(value) = standalone.as_mut() {
            value.reserve(&self.context, inputs.len())?;
        }
        let (verified, sampling) = {
            let verifier = speculation
                .as_mut()
                .map(|value| &mut value.verifier)
                .or(standalone.as_mut())
                .ok_or_else(|| {
                    crate::Error::InvalidArgument("n-gram verifier is unavailable".into())
                })?;
            self.forward_block(&inputs, BlockOutput::Verify(verifier))?;
            let sampling_started = Instant::now();
            let result = verify_greedy_drafts(sampler, drafts, |row, sampler| {
                sampler.sample_buffer(
                    &mut verifier.verify_logits,
                    row * VOCAB * size_of::<f32>(),
                    &self.context,
                    &self.sampling,
                )
            })?;
            if result.accepted + 1 < inputs.len() {
                self.restore_checkpoint(verifier, result.accepted + 1)?;
            }
            (result, sampling_started.elapsed())
        };
        let committed = verified.accepted + 1;
        if committed < inputs.len() {
            self.position = start + committed;
        }
        if let Some(head) = speculation.as_mut() {
            self.encode_committed(head, &inputs[..committed])?;
        }
        self.speculation = speculation;
        self.ngram_verifier = standalone;
        Ok(SpeculativeBatch {
            samples: verified.samples,
            stats: MtpStats::default(),
            ngram: NgramStats {
                rounds: 1,
                proposed_tokens: drafts.len(),
                accepted_tokens: verified.accepted,
                ..NgramStats::default()
            },
            sampling,
        })
    }
}

/// The batch a cancelled round hands back: no samples and no counters, so the
/// caller observes a round that stopped before submitting any work. The empty
/// `samples` is what keeps the generation loop from sampling stale logits.
fn cancelled_speculative_batch() -> SpeculativeBatch {
    SpeculativeBatch {
        samples: Vec::new(),
        stats: MtpStats::default(),
        ngram: NgramStats::default(),
        sampling: std::time::Duration::ZERO,
    }
}

/// Select the checkpoints a verify block of `tokens` rows needs (one per
/// non-final row) and seed each from the layer's current state and history,
/// so the prefix runs in [`BonsaiModel::recurrent`] start where the full run
/// starts. Without checkpoints this is a plain decode/prefill block.
fn seed_rollback(
    batch: &mut CommandBatch,
    layer: &RecurrentLayer,
    rollback: Option<&RollbackLayer>,
) -> crate::Result<()> {
    let Some(rollback) = rollback else {
        return Ok(());
    };
    batch.blit_buffer_copies([
        BufferCopyRequest {
            source: &layer.state,
            source_offset: 0,
            destination: &rollback.state,
            destination_offset: 0,
            size: GDN_STATE_BYTES,
        },
        BufferCopyRequest {
            source: &layer.history,
            source_offset: 0,
            destination: &rollback.history,
            destination_offset: 0,
            size: CONV_STATE_BYTES,
        },
    ])?;
    Ok(())
}

/// Decode PTQ1 embedding rows for `tokens` into consecutive F32 rows of `out`.
fn decode_embeddings(
    package: &BonsaiPackage,
    tokens: &[u32],
    out: &mut [f32],
) -> crate::Result<()> {
    let row_bytes = WIDTH / PTQ1_BLOCK_ELEMENTS * PTQ1_BLOCK_BYTES;
    let embeddings = package.bytes("token_embd.weight")?;
    if out.len() < tokens.len() * WIDTH {
        return invalid("embedding scratch is shorter than the token block");
    }
    let (rows, _) = out.as_chunks_mut::<WIDTH>();
    for (&token, row) in tokens.iter().zip(rows) {
        let offset = token as usize * row_bytes;
        let bytes = embeddings.get(offset..offset + row_bytes).ok_or_else(|| {
            crate::Error::InvalidArgument("Bonsai token beyond vocabulary".into())
        })?;
        decode_ptq1_row(bytes, row)?;
    }
    Ok(())
}

/// `extra_bytes` covers optional resident additions such as an MTP head.
fn memory_plan(
    package: &BonsaiPackage,
    context: usize,
    prefill_chunk: usize,
    extra_bytes: u64,
    mtp_enabled: bool,
    kv: KvLayout,
    caps: &DeviceCaps,
) -> crate::Result<BonsaiInfo> {
    let context_explicit = context != 0;
    if context_explicit && !(1..=TRAINING_CONTEXT).contains(&context) {
        return Err(crate::Error::InvalidArgument(
            "Bonsai context must be within 1..=262144".into(),
        ));
    }
    if !(1..=MAX_PREFILL_TOKENS as usize).contains(&prefill_chunk) {
        return Err(crate::Error::InvalidArgument(
            "Bonsai prefill chunk must be within 1..=128".into(),
        ));
    }
    let tensor_bytes = package
        .tensors()
        .iter()
        .map(crate::bonsai::BonsaiTensor::bytes)
        .sum::<u64>();
    let f32_bytes = package
        .tensors()
        .iter()
        .filter(|tensor| tensor.tensor_type() == BonsaiTensorType::F32)
        .map(crate::bonsai::BonsaiTensor::bytes)
        .sum::<u64>();
    let largest_tensor = package
        .tensors()
        .iter()
        .map(crate::bonsai::BonsaiTensor::bytes)
        .max()
        .unwrap_or(0);
    let cache_bytes = context * kv.key.token_bytes().max(kv.value.token_bytes());
    if caps.max_buffer_length < largest_tensor.max(cache_bytes as u64) {
        return Err(crate::Error::Context(
            "Bonsai tensor/KV cache exceeds device buffer limit".into(),
        ));
    }
    // Admission assumes the caches fully grown: a request may reach the
    // context, and refusing it mid-generation would be worse than refusing
    // the configuration up front.
    let fixed_state_bytes =
        (LAYERS - LAYERS / FULL_INTERVAL) * (GDN_STATE_BYTES + CONV_STATE_BYTES);
    // Conservative first implementation: count the entire mapped checkpoint,
    // a possible copied final tensor, F32 exceptions, and 512 MiB for tokenizer,
    // workspace, pipelines and command buffers, plus 1 MiB per prefill row
    // (more than all row scratch combined).
    let fixed_working_set = tensor_bytes
        + package.data_start()
        + largest_tensor
        + f32_bytes
        + extra_bytes
        + (512 + prefill_chunk as u64) * 1024 * 1024;
    let per_token = (LAYERS / FULL_INTERVAL * kv.token_bytes()
        + usize::from(mtp_enabled) * 2 * local_metal::bonsai_ops::KvFormat::F16.token_bytes())
        as u64;
    let admitted_context = if context_explicit {
        context
    } else {
        fit_context(
            fixed_working_set + fixed_state_bytes as u64,
            per_token,
            caps.recommended_working_set,
        )
        .ok_or_else(|| {
            crate::Error::Context(format!(
                "Bonsai fixed allocations need an estimated {} bytes; {} recommends {}",
                fixed_working_set + fixed_state_bytes as u64,
                caps.name,
                caps.recommended_working_set
            ))
        })?
    };
    let state_bytes =
        fixed_state_bytes + LAYERS / FULL_INTERVAL * admitted_context * kv.token_bytes();
    let estimated_working_set =
        fixed_working_set + fixed_state_bytes as u64 + admitted_context as u64 * per_token;
    if estimated_working_set > caps.recommended_working_set {
        return Err(crate::Error::Context(format!(
            "Bonsai with {context}-token {} KV needs an estimated {estimated_working_set} bytes; \
             {} recommends {}. Context/precision were not reduced; choose a smaller context, a \
             smaller KV format (q8), or another model",
            kv.name(),
            caps.name,
            caps.recommended_working_set
        )));
    }
    Ok(BonsaiInfo {
        precision: String::new(),
        policy: serde_json::Value::Null,
        device: caps.name.clone(),
        context: admitted_context,
        context_explicit,
        context_reason: if context_explicit {
            format!("explicit {admitted_context}-token context; strict full-growth admission")
        } else {
            format!(
                "automatically selected largest context fitting 90% of the {}-byte recommended working set",
                caps.recommended_working_set
            )
        },
        prefill_chunk_size: prefill_chunk,
        tensor_bytes,
        state_bytes,
        estimated_working_set,
        working_set_limit: caps.recommended_working_set,
    })
}

/// Largest whole-token context fitting a 10% safety reserve, capped at training context.
const fn fit_context(fixed: u64, per_token: u64, recommended: u64) -> Option<usize> {
    let budget = recommended / 10 * 9;
    if per_token == 0 || fixed >= budget {
        return None;
    }
    let tokens = ((budget - fixed) / per_token) as usize;
    if tokens == 0 {
        None
    } else {
        Some(if tokens < TRAINING_CONTEXT {
            tokens
        } else {
            TRAINING_CONTEXT
        })
    }
}

fn invalid<T>(message: impl Into<String>) -> crate::Result<T> {
    Err(crate::Error::InvalidFormat(message.into()))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
