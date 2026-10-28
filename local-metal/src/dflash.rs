//! Kernels of the `DFlash2` block drafter and the `KFlash` tree builder
//! (`shaders/dflash.metal`).
//!
//! The drafter's dense projections use the per-row int8 kernels
//! ([`crate::bonsai::BonsaiKernels::int8_matmul`]) and its output head the
//! target's `PTQ1_0` kernels; this module encodes the rest of the pass:
//! norms, the grouped dynamic convolutions, rotary embeddings, the context
//! K/V ring, block attention, the reduced output head's candidate merge, the
//! candidate lattice and the tree. Like every kernel set here, encoding only
//! records work: the caller completes the batch before reading results.

use core::ffi::c_void;
use std::ptr::NonNull;

use bytemuck::{Pod, Zeroable};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::Error;
use crate::batch::CommandBatch;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::sampling::MAX_TOP_K;
use crate::shaders::ShaderLibrary;

/// Residual width of the drafter (and of the target).
pub const DFLASH_WIDTH: usize = 5120;
pub const DFLASH_HEAD_DIM: usize = 128;
pub const DFLASH_Q_HEADS: usize = 32;
pub const DFLASH_KV_HEADS: usize = 8;
pub const DFLASH_Q_WIDTH: usize = DFLASH_Q_HEADS * DFLASH_HEAD_DIM;
pub const DFLASH_KV_WIDTH: usize = DFLASH_KV_HEADS * DFLASH_HEAD_DIM;
/// Rows of one drafting block: the anchor and seven mask rows.
pub const DFLASH_BLOCK: usize = 8;
/// Draft positions after the anchor.
pub const DFLASH_SLOTS: usize = DFLASH_BLOCK - 1;
/// Candidates per draft position.
pub const DFLASH_TOPK: usize = 16;
/// Most added tokens [`DflashKernels::head_merge`] merges (`DF_MAX_SPECIALS`).
pub const DFLASH_MAX_SPECIALS: usize = 48;
/// Rank of the candidate selector's codebooks.
pub const DFLASH_RANK: usize = 256;
/// Width of one grouped-convolution kernel projection: two halves (before and
/// after the wrapped block) of two taps of 320 groups.
pub const DFLASH_CONV_PROJECTION: usize = 4 * DFLASH_WIDTH / 16;
/// The drafter's sliding attention window.
pub const DFLASH_WINDOW: usize = 2048;
/// Most rows [`DflashKernels::tree`] builds: the anchor and seven nodes.
pub const DFLASH_MAX_TREE_ROWS: usize = 8;
/// Keys one attention threadgroup scores (`DF_CHUNK`).
const ATTENTION_CHUNK: usize = 64;
/// Positions per entry of the fine rotary table (`DF_ROPE_FINE`).
const ROPE_FINE: usize = 256;
/// Largest position the rotary tables cover, exclusive.
pub const DFLASH_MAX_POSITION: usize = 1 << 18;

/// The candidate lattice of one drafting pass, as the GPU writes it.
///
/// `cand[s][k]` is draft position `s`'s `k`-th best token (best first) by the
/// drafter's logits through the rows of the shared output head it scores,
/// `logit[s][k]` that raw logit; `edges[s][j][k]` is the selector's pairwise
/// score of candidate `k` after candidate `j` of position `s - 1` (after the
/// anchor for `s = 0`, identical for every `j`), and `cond_logq[s][j][k]` the
/// conditional log probability `log_softmax_k(logit[s][k] + edges[s][j][k])`
/// the tree uses.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct DflashLattice {
    pub cand: [[u32; DFLASH_TOPK]; DFLASH_SLOTS],
    pub logit: [[f32; DFLASH_TOPK]; DFLASH_SLOTS],
    pub edges: [[[f32; DFLASH_TOPK]; DFLASH_TOPK]; DFLASH_SLOTS],
    pub cond_logq: [[[f32; DFLASH_TOPK]; DFLASH_TOPK]; DFLASH_SLOTS],
}

/// A best-first tree as the GPU writes it. Row 0 is the anchor; rows
/// `1..rows` are nodes in the order they were taken, so a parent always
/// precedes its children and the first `n` rows are the best `n`-row tree.
///
/// Per row: its `token`, `parent` row (-1 for the anchor and unused rows),
/// `depth` (positions after the anchor; its sequence position is the anchor's
/// plus this), `slot` (its candidate index `k` in lattice position
/// `depth - 1`), `ancestors` (bit `i` set for every row on its path, itself
/// included: the verify's attention mask), `log_q` (its conditional drafter
/// log probability) and `path_log_p` (its path's calibrated log acceptance
/// probability). `expected[n - 1]` is `E(n) = 1 + Σ_{r=1}^{n-1}
/// exp(path_log_p[r])`, the tokens an `n`-row verify is expected to emit.
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct DflashTree {
    pub rows: u32,
    pub pad: [u32; 3],
    pub token: [u32; DFLASH_MAX_TREE_ROWS],
    pub parent: [i32; DFLASH_MAX_TREE_ROWS],
    pub depth: [u32; DFLASH_MAX_TREE_ROWS],
    pub slot: [u32; DFLASH_MAX_TREE_ROWS],
    pub ancestors: [u32; DFLASH_MAX_TREE_ROWS],
    pub log_q: [f32; DFLASH_MAX_TREE_ROWS],
    pub path_log_p: [f32; DFLASH_MAX_TREE_ROWS],
    pub expected: [f32; DFLASH_MAX_TREE_ROWS],
}

/// Per-depth calibration `p = σ(alpha[d] + beta[d] · logit q)` for depths
/// `d = 1..=7` (index 0 unused). `alpha = 0, beta = 1` is `p = q`.
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct DflashCalibration {
    pub alpha: [f32; DFLASH_MAX_TREE_ROWS],
    pub beta: [f32; DFLASH_MAX_TREE_ROWS],
}

impl Default for DflashCalibration {
    fn default() -> Self {
        Self {
            alpha: [0.0; DFLASH_MAX_TREE_ROWS],
            beta: [1.0; DFLASH_MAX_TREE_ROWS],
        }
    }
}

/// Rotary tables: `(cos, sin)` of `position · θ^(-2i/128)` for `i < 64`.
///
/// They are computed in F64 on the host: entry `a` of the coarse table is
/// position `256 a`, entry `b` of the fine one position `b`, and the kernels
/// combine them by the angle sum, so every position up to
/// [`DFLASH_MAX_POSITION`] keeps F32 accuracy.
pub struct RopeTables {
    coarse: MetalBuffer,
    fine: MetalBuffer,
}

impl RopeTables {
    pub fn new(context: &MetalContext, theta: f64) -> crate::Result<Self> {
        let pairs = DFLASH_HEAD_DIM / 2;
        let table = |entries: usize, stride: usize| {
            let mut values = Vec::with_capacity(entries * pairs * 2);
            for entry in 0..entries {
                for i in 0..pairs {
                    let inv = theta.powf(-((2 * i) as f64) / DFLASH_HEAD_DIM as f64);
                    let angle = (entry * stride) as f64 * inv;
                    values.push(angle.cos() as f32);
                    values.push(angle.sin() as f32);
                }
            }
            MetalBuffer::from_slice(context.device(), &values)
        };
        Ok(Self {
            coarse: table(DFLASH_MAX_POSITION / ROPE_FINE, ROPE_FINE)?,
            fine: table(ROPE_FINE, 1)?,
        })
    }
}

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub struct DflashKernels {
    rms: Pipeline,
    add: Pipeline,
    silu_mul: Pipeline,
    taps_f16: Pipeline,
    taps_f32: Pipeline,
    conv: Pipeline,
    qk_norm_rope: Pipeline,
    ctx_store: Pipeline,
    attention_partial: Pipeline,
    attention_reduce: Pipeline,
    head_merge: Pipeline,
    lattice: Pipeline,
    tree: Pipeline,
}

/// How many key chunks block attention splits `nctx` context keys into.
#[must_use]
pub const fn attention_chunks(nctx: usize) -> usize {
    (nctx + DFLASH_BLOCK).div_ceil(ATTENTION_CHUNK)
}

/// Bytes of the attention partial outputs and statistics for `nctx` keys.
#[must_use]
pub const fn attention_partial_bytes(nctx: usize) -> (usize, usize) {
    let queries = attention_chunks(nctx) * DFLASH_KV_HEADS * 4 * DFLASH_BLOCK;
    (queries * DFLASH_HEAD_DIM * 4, queries * 8)
}

fn arg(message: &str) -> Error {
    Error::InvalidArgument(format!("dflash: {message}"))
}

fn need(buffer: &MetalBuffer, offset: usize, bytes: usize) -> crate::Result<()> {
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > buffer.length())
    {
        return Err(arg("buffer is too short"));
    }
    Ok(())
}

fn same(a: &MetalBuffer, b: &MetalBuffer) -> bool {
    std::ptr::eq(a.raw(), b.raw())
}

impl DflashKernels {
    pub fn new(context: &MetalContext, shaders: &ShaderLibrary) -> crate::Result<Self> {
        let pipeline = |name: &str| -> crate::Result<Pipeline> {
            let function = shaders.get_function(name)?;
            let pipeline = context
                .device()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::PipelineCreation(error.to_string()))?;
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < 256
            {
                return Err(arg("pipeline cannot run 256-thread threadgroups"));
            }
            Ok(pipeline)
        };
        Ok(Self {
            rms: pipeline("dflash_rms_rows")?,
            add: pipeline("dflash_add")?,
            silu_mul: pipeline("dflash_silu_mul")?,
            taps_f16: pipeline("dflash_taps_f16")?,
            taps_f32: pipeline("dflash_taps_f32")?,
            conv: pipeline("dflash_conv")?,
            qk_norm_rope: pipeline("dflash_qk_norm_rope")?,
            ctx_store: pipeline("dflash_ctx_store")?,
            attention_partial: pipeline("dflash_attention_partial")?,
            attention_reduce: pipeline("dflash_attention_reduce")?,
            head_merge: pipeline("dflash_head_merge")?,
            lattice: pipeline("dflash_lattice")?,
            tree: pipeline("dflash_tree")?,
        })
    }

    /// `RMSNorm` `rows` rows of `width` from `input` (at byte `offset`) by a
    /// plain weight into `output`.
    #[expect(unsafe_code, clippy::too_many_arguments)]
    pub fn rms_rows(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        offset: usize,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> crate::Result<()> {
        need(input, offset, rows * width * 4)?;
        need(weight, 0, width * 4)?;
        need(output, 0, rows * width * 4)?;
        if rows == 0 || same(input, output) || same(weight, output) {
            return Err(arg("RMSNorm rows are empty or aliased"));
        }
        let width = width as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.rms);
        unsafe {
            bind(encoder, input, offset, 0);
            bind(encoder, weight, 0, 1);
            bind(encoder, output, 0, 2);
            set(encoder, &width, 3);
            set(encoder, &epsilon, 4);
        }
        dispatch(encoder, (rows, 1), 256);
        batch.record_dispatch();
        Ok(())
    }

    /// `x += y` over `count` F32 values.
    pub fn add(
        &self,
        batch: &mut CommandBatch,
        x: &MetalBuffer,
        y: &MetalBuffer,
        count: usize,
    ) -> crate::Result<()> {
        need(x, 0, count * 4)?;
        need(y, 0, count * 4)?;
        if same(x, y) {
            return Err(arg("residual add aliases"));
        }
        Self::elementwise(batch, &self.add, &[x, y], count);
        Ok(())
    }

    /// `output = silu(gate) · up` over `count` F32 values.
    pub fn silu_mul(
        &self,
        batch: &mut CommandBatch,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        output: &MetalBuffer,
        count: usize,
    ) -> crate::Result<()> {
        for buffer in [gate, up, output] {
            need(buffer, 0, count * 4)?;
        }
        Self::elementwise(batch, &self.silu_mul, &[gate, up, output], count);
        Ok(())
    }

    /// `count` tap values (F16 when `half`, else F32) at byte `offset` of
    /// `input` to F32 `output`.
    #[expect(unsafe_code)]
    pub fn taps(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        offset: usize,
        half: bool,
        output: &MetalBuffer,
        count: usize,
    ) -> crate::Result<()> {
        need(input, offset, count * if half { 2 } else { 4 })?;
        need(output, 0, count * 4)?;
        if same(input, output) || !offset.is_multiple_of(if half { 2 } else { 4 }) {
            return Err(arg("taps alias the staging rows or are misaligned"));
        }
        let count32 = count as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(if half { &self.taps_f16 } else { &self.taps_f32 });
        unsafe {
            bind(encoder, input, offset, 0);
            bind(encoder, output, 0, 1);
            set(encoder, &count32, 2);
        }
        dispatch(encoder, (count.div_ceil(256), 1), 256);
        batch.record_dispatch();
        Ok(())
    }

    /// One half of a grouped dynamic convolution over `rows` block rows:
    /// `dyn` holds the rows' 1280-wide kernel projections and `half` (0
    /// before the wrapped block, 1 after it) selects their taps and the base
    /// kernel's (`base` is `[2, 2, 5120]` F32).
    #[expect(unsafe_code, clippy::too_many_arguments)]
    pub fn conv(
        &self,
        batch: &mut CommandBatch,
        x: &MetalBuffer,
        dyn_kernels: &MetalBuffer,
        base: &MetalBuffer,
        half: usize,
        output: &MetalBuffer,
        rows: usize,
    ) -> crate::Result<()> {
        need(x, 0, rows * DFLASH_WIDTH * 4)?;
        need(dyn_kernels, 0, rows * DFLASH_CONV_PROJECTION * 4)?;
        need(base, 0, 4 * DFLASH_WIDTH * 4)?;
        need(output, 0, rows * DFLASH_WIDTH * 4)?;
        if half > 1 || same(x, output) || same(dyn_kernels, output) {
            return Err(arg("convolution half is invalid or the output aliases"));
        }
        let count = rows * DFLASH_WIDTH;
        let rows32 = rows as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.conv);
        unsafe {
            bind(encoder, x, 0, 0);
            bind(
                encoder,
                dyn_kernels,
                half * DFLASH_CONV_PROJECTION / 2 * 4,
                1,
            );
            bind(encoder, base, half * 2 * DFLASH_WIDTH * 4, 2);
            bind(encoder, output, 0, 3);
            set(encoder, &rows32, 4);
        }
        dispatch(encoder, (count.div_ceil(256), 1), 256);
        batch.record_dispatch();
        Ok(())
    }

    /// Per-head `RMSNorm` and rotary embedding of the block's queries `[rows,
    /// 4096]` and keys `[rows, 1024]` in place, row `t` at `position + t`.
    #[expect(unsafe_code, clippy::too_many_arguments)]
    pub fn qk_norm_rope(
        &self,
        batch: &mut CommandBatch,
        q: &MetalBuffer,
        k: &MetalBuffer,
        q_norm: &MetalBuffer,
        k_norm: &MetalBuffer,
        rope: &RopeTables,
        position: usize,
        rows: usize,
        epsilon: f32,
    ) -> crate::Result<()> {
        need(q, 0, rows * DFLASH_Q_WIDTH * 4)?;
        need(k, 0, rows * DFLASH_KV_WIDTH * 4)?;
        need(q_norm, 0, DFLASH_HEAD_DIM * 4)?;
        need(k_norm, 0, DFLASH_HEAD_DIM * 4)?;
        if position + rows > DFLASH_MAX_POSITION || same(q, k) {
            return Err(arg(
                "block position is beyond the rotary tables or q aliases k",
            ));
        }
        let position = position as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.qk_norm_rope);
        unsafe {
            bind(encoder, q, 0, 0);
            bind(encoder, k, 0, 1);
            bind(encoder, q_norm, 0, 2);
            bind(encoder, k_norm, 0, 3);
            bind(encoder, &rope.coarse, 0, 4);
            bind(encoder, &rope.fine, 0, 5);
            set(encoder, &position, 6);
            set(encoder, &epsilon, 7);
        }
        dispatch(
            encoder,
            (DFLASH_Q_HEADS + DFLASH_KV_HEADS, rows),
            DFLASH_HEAD_DIM,
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Normalize and rotate `rows` context keys (`k`, `[rows, 1024]`) and
    /// store them and their values into one layer's F16 ring of `capacity`
    /// positions, row `t` (position `position + t`) at slot
    /// `(position + t) % capacity`.
    #[expect(unsafe_code, clippy::too_many_arguments)]
    pub fn ctx_store(
        &self,
        batch: &mut CommandBatch,
        k: &MetalBuffer,
        v: &MetalBuffer,
        k_norm: &MetalBuffer,
        rope: &RopeTables,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        position: usize,
        rows: usize,
        capacity: usize,
        epsilon: f32,
    ) -> crate::Result<()> {
        need(k, 0, rows * DFLASH_KV_WIDTH * 4)?;
        need(v, 0, rows * DFLASH_KV_WIDTH * 4)?;
        need(k_norm, 0, DFLASH_HEAD_DIM * 4)?;
        need(k_cache, 0, capacity * DFLASH_KV_WIDTH * 2)?;
        need(v_cache, 0, capacity * DFLASH_KV_WIDTH * 2)?;
        if rows == 0
            || rows > capacity
            || position + rows > DFLASH_MAX_POSITION
            || same(k_cache, v_cache)
        {
            return Err(arg("context rows exceed the ring or the rotary tables"));
        }
        let (position, capacity) = (position as u32, capacity as u32);
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.ctx_store);
        unsafe {
            bind(encoder, k, 0, 0);
            bind(encoder, v, 0, 1);
            bind(encoder, k_norm, 0, 2);
            bind(encoder, &rope.coarse, 0, 3);
            bind(encoder, &rope.fine, 0, 4);
            bind(encoder, k_cache, 0, 5);
            bind(encoder, v_cache, 0, 6);
            set(encoder, &position, 7);
            set(encoder, &capacity, 8);
            set(encoder, &epsilon, 9);
        }
        dispatch(encoder, (DFLASH_KV_HEADS, rows), DFLASH_HEAD_DIM);
        batch.record_dispatch();
        Ok(())
    }

    /// Block attention of the 8 block rows: queries `q` `[8, 4096]` against
    /// the `nctx` context positions before `position` in the ring and the
    /// block's own keys and values (`[8, 1024]`), output `[8, 4096]`.
    /// `partial_out` and `partial_stats` hold [`attention_partial_bytes`].
    #[expect(unsafe_code, clippy::too_many_arguments)]
    pub fn attention(
        &self,
        batch: &mut CommandBatch,
        q: &MetalBuffer,
        block_k: &MetalBuffer,
        block_v: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        partial: (&MetalBuffer, &MetalBuffer),
        output: &MetalBuffer,
        position: usize,
        nctx: usize,
        capacity: usize,
    ) -> crate::Result<()> {
        let (partial_out, partial_stats) = partial;
        let (out_bytes, stats_bytes) = attention_partial_bytes(nctx);
        need(q, 0, DFLASH_BLOCK * DFLASH_Q_WIDTH * 4)?;
        need(block_k, 0, DFLASH_BLOCK * DFLASH_KV_WIDTH * 4)?;
        need(block_v, 0, DFLASH_BLOCK * DFLASH_KV_WIDTH * 4)?;
        need(k_cache, 0, capacity * DFLASH_KV_WIDTH * 2)?;
        need(v_cache, 0, capacity * DFLASH_KV_WIDTH * 2)?;
        need(partial_out, 0, out_bytes)?;
        need(partial_stats, 0, stats_bytes)?;
        need(output, 0, DFLASH_BLOCK * DFLASH_Q_WIDTH * 4)?;
        if nctx > position || nctx > capacity || nctx >= DFLASH_WINDOW || same(q, output) {
            return Err(arg(
                "attention context exceeds the position, ring or window",
            ));
        }
        let chunks = attention_chunks(nctx);
        let chunks32 = chunks as u32;
        let (position, nctx, capacity) = (position as u32, nctx as u32, capacity as u32);
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.attention_partial);
        unsafe {
            bind(encoder, q, 0, 0);
            bind(encoder, block_k, 0, 1);
            bind(encoder, block_v, 0, 2);
            bind(encoder, k_cache, 0, 3);
            bind(encoder, v_cache, 0, 4);
            bind(encoder, partial_out, 0, 5);
            bind(encoder, partial_stats, 0, 6);
            set(encoder, &position, 7);
            set(encoder, &nctx, 8);
            set(encoder, &capacity, 9);
        }
        dispatch(encoder, (chunks, DFLASH_KV_HEADS), 256);
        batch.record_dispatch();
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.attention_reduce);
        unsafe {
            bind(encoder, partial_out, 0, 0);
            bind(encoder, partial_stats, 0, 1);
            bind(encoder, output, 0, 2);
            set(encoder, &chunks32, 3);
        }
        dispatch(encoder, (DFLASH_Q_HEADS, DFLASH_BLOCK), DFLASH_HEAD_DIM);
        batch.record_dispatch();
        Ok(())
    }

    /// The reduced output head's top 16 per draft position: the top-16
    /// selections `top` of the first token ids (rows of [`MAX_TOP_K`]
    /// `TopKCandidate`s) merged with the logits `special` `[7, specials]` of
    /// the added tokens `special_first..special_first + specials`, into rows
    /// of the same stride in `merged`, ranked as the top-k kernels rank.
    #[expect(unsafe_code)]
    pub fn head_merge(
        &self,
        batch: &mut CommandBatch,
        top: &MetalBuffer,
        special: &MetalBuffer,
        special_first: u32,
        specials: usize,
        merged: &MetalBuffer,
    ) -> crate::Result<()> {
        need(top, 0, DFLASH_SLOTS * MAX_TOP_K * 8)?;
        need(special, 0, DFLASH_SLOTS * specials * 4)?;
        need(merged, 0, DFLASH_SLOTS * MAX_TOP_K * 8)?;
        if !(1..=DFLASH_MAX_SPECIALS).contains(&specials) || same(top, merged) {
            return Err(arg(
                "added tokens must be 1..=48 and the merge must not alias",
            ));
        }
        let (stride, specials) = (MAX_TOP_K as u32, specials as u32);
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.head_merge);
        unsafe {
            bind(encoder, top, 0, 0);
            bind(encoder, special, 0, 1);
            bind(encoder, merged, 0, 2);
            set(encoder, &stride, 3);
            set(encoder, &special_first, 4);
            set(encoder, &specials, 5);
        }
        dispatch(
            encoder,
            (DFLASH_SLOTS, 1),
            DFLASH_TOPK + DFLASH_MAX_SPECIALS,
        );
        batch.record_dispatch();
        Ok(())
    }

    /// The candidate lattice from the top-16 selections `top` (rows of
    /// [`MAX_TOP_K`] `TopKCandidate`s, one per draft position), the selector
    /// projections `selector` `[7, 256]`, the BF16 predecessor and successor
    /// codebooks `[vocab, 256]`, and the anchor at `tokens[0]`.
    #[expect(unsafe_code, clippy::too_many_arguments)]
    pub fn lattice(
        &self,
        batch: &mut CommandBatch,
        top: &MetalBuffer,
        selector: &MetalBuffer,
        predecessor: &MetalBuffer,
        successor: &MetalBuffer,
        tokens: &MetalBuffer,
        lattice: &MetalBuffer,
    ) -> crate::Result<()> {
        need(top, 0, DFLASH_SLOTS * MAX_TOP_K * 8)?;
        need(selector, 0, DFLASH_SLOTS * DFLASH_RANK * 4)?;
        need(tokens, 0, 4)?;
        need(lattice, 0, size_of::<DflashLattice>())?;
        if predecessor.length() < DFLASH_RANK * 2 || successor.length() < DFLASH_RANK * 2 {
            return Err(arg("selector codebooks are empty"));
        }
        let stride = MAX_TOP_K as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.lattice);
        unsafe {
            bind(encoder, top, 0, 0);
            bind(encoder, selector, 0, 1);
            bind(encoder, predecessor, 0, 2);
            bind(encoder, successor, 0, 3);
            bind(encoder, tokens, 0, 4);
            bind(encoder, lattice, 0, 5);
            set(encoder, &stride, 6);
        }
        dispatch(encoder, (DFLASH_SLOTS, 1), DFLASH_TOPK * DFLASH_TOPK);
        batch.record_dispatch();
        Ok(())
    }

    /// The best-first tree of up to `max_rows` rows (anchor `tokens[0]`
    /// included) over `lattice`, scored with `calibration`
    /// ([`DflashCalibration`]), into `tree` ([`DflashTree`]).
    #[expect(unsafe_code)]
    pub fn tree(
        &self,
        batch: &mut CommandBatch,
        lattice: &MetalBuffer,
        tokens: &MetalBuffer,
        calibration: &MetalBuffer,
        tree: &MetalBuffer,
        max_rows: usize,
    ) -> crate::Result<()> {
        need(lattice, 0, size_of::<DflashLattice>())?;
        need(tokens, 0, 4)?;
        need(calibration, 0, size_of::<DflashCalibration>())?;
        need(tree, 0, size_of::<DflashTree>())?;
        if !(1..=DFLASH_MAX_TREE_ROWS).contains(&max_rows) || same(tree, lattice) {
            return Err(arg("tree rows must be 1..=8"));
        }
        let rows = max_rows as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.tree);
        unsafe {
            bind(encoder, lattice, 0, 0);
            bind(encoder, tokens, 0, 1);
            bind(encoder, calibration, 0, 2);
            bind(encoder, tree, 0, 3);
            set(encoder, &rows, 4);
        }
        dispatch(encoder, (1, 1), 32);
        batch.record_dispatch();
        Ok(())
    }

    #[expect(unsafe_code)]
    fn elementwise(
        batch: &mut CommandBatch,
        pipeline: &Pipeline,
        buffers: &[&MetalBuffer],
        count: usize,
    ) {
        let count32 = count as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            for (index, buffer) in buffers.iter().enumerate() {
                bind(encoder, buffer, 0, index);
            }
            set(encoder, &count32, buffers.len());
        }
        dispatch(encoder, (count.div_ceil(256), 1), 256);
        batch.record_dispatch();
    }
}

fn dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    groups: (usize, usize),
    threads: usize,
) {
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups.0,
            height: groups.1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

#[expect(unsafe_code)]
unsafe fn bind(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffer: &MetalBuffer,
    offset: usize,
    index: usize,
) {
    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.raw()), offset, index) };
}

#[expect(unsafe_code)]
unsafe fn set<T: Pod>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: &T,
    index: usize,
) {
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(std::ptr::from_ref(value).cast_mut().cast::<c_void>()),
            size_of::<T>(),
            index,
        );
    }
}
