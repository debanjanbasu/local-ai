//! The drafter's GPU state and its two passes: appending context and drafting.

use std::path::{Path, PathBuf};
use std::time::Duration;

use local_metal::batch::{BufferCopyRequest, CommandBatch};
use local_metal::bonsai::{BonsaiKernels, HadamardDirection, Ptq1Matrix, SignedHadamard};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::dflash::{
    DFLASH_BLOCK, DFLASH_MAX_TREE_ROWS, DFLASH_SLOTS, DFLASH_TOPK, DFLASH_WINDOW,
    DflashCalibration, DflashKernels, DflashLattice, DflashTree, RopeTables,
    attention_partial_bytes,
};
use local_metal::draft::DraftRows;
use local_metal::sampling::{MAX_TOP_K, TopKCandidate, TopKRows};
use local_metal::shaders::ShaderLibrary;

use super::weights::Weights;
use super::{
    CONV_PROJECTION, EPSILON, FFN, HEAD_ROWS, KV_WIDTH, MASK_TOKEN, Q_WIDTH, RANK,
    REDUCED_HEAD_ANCHORS, ROPE_THETA, SPECIAL_FIRST, SPECIALS, TAPS, VOCAB, WIDTH,
    reduced_head_scores,
};

/// Positions [`DflashDrafter::truncate_context`] can roll back. The ring
/// holds this many positions beyond the 2047 the window reads.
pub const MAX_ROLLBACK: usize = 64;
/// Context positions the drafter's K/V ring holds.
pub const RING_CAPACITY: usize = DFLASH_WINDOW + MAX_ROLLBACK;
/// Context rows one encoded chunk of [`DflashDrafter::append_context`] covers.
const CONTEXT_CHUNK: usize = 64;

/// Target-owned resources the drafter shares, borrowed per call.
///
/// They are the kernels, the `PTQ1_0` embedding table and output matrix, and the 5120-wide rotation
/// both live in (`token_embd.weight`, `output.weight`, and the checkpoint's
/// signs for width 5120).
pub struct DflashShared<'a> {
    pub kernels: &'a BonsaiKernels,
    pub embedding: Ptq1Matrix<'a>,
    pub output: Ptq1Matrix<'a>,
    pub rotation: &'a SignedHadamard,
}

/// The element type of the target's hidden taps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapFormat {
    F16,
    F32,
}

impl TapFormat {
    const fn bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

/// The anchor token of a drafting pass: known on the host, or a `u32` the
/// target wrote at `offset` of a GPU buffer earlier in the same batch.
pub enum DraftAnchor<'a> {
    Token(u32),
    Device {
        buffer: &'a MetalBuffer,
        offset: usize,
    },
}

struct Scratch {
    x: MetalBuffer,
    h: MetalBuffer,
    hc: MetalBuffer,
    branch: MetalBuffer,
    kernels: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    attention: MetalBuffer,
    gate: MetalBuffer,
    up: MetalBuffer,
    product: MetalBuffer,
    hidden: MetalBuffer,
    rotated: MetalBuffer,
    selector: MetalBuffer,
    logits: MetalBuffer,
    special_logits: MetalBuffer,
    partial_out: MetalBuffer,
    partial_stats: MetalBuffer,
    taps: MetalBuffer,
    context: MetalBuffer,
    context_normed: MetalBuffer,
    context_k: MetalBuffer,
    context_v: MetalBuffer,
}

impl Scratch {
    fn new(context: &MetalContext) -> crate::Result<Self> {
        let floats = |count: usize| MetalBuffer::empty(context.device(), count * 4);
        let (out_bytes, stats_bytes) = attention_partial_bytes(DFLASH_WINDOW - 1);
        Ok(Self {
            x: floats(DFLASH_BLOCK * WIDTH)?,
            h: floats(DFLASH_BLOCK * WIDTH)?,
            hc: floats(DFLASH_BLOCK * WIDTH)?,
            branch: floats(DFLASH_BLOCK * WIDTH)?,
            kernels: floats(DFLASH_BLOCK * CONV_PROJECTION)?,
            q: floats(DFLASH_BLOCK * Q_WIDTH)?,
            k: floats(DFLASH_BLOCK * KV_WIDTH)?,
            v: floats(DFLASH_BLOCK * KV_WIDTH)?,
            attention: floats(DFLASH_BLOCK * Q_WIDTH)?,
            gate: floats(DFLASH_BLOCK * FFN)?,
            up: floats(DFLASH_BLOCK * FFN)?,
            product: floats(DFLASH_BLOCK * FFN)?,
            hidden: floats(DFLASH_SLOTS * WIDTH)?,
            rotated: floats(DFLASH_SLOTS * WIDTH)?,
            selector: floats(DFLASH_SLOTS * RANK)?,
            logits: floats(DFLASH_SLOTS * VOCAB)?,
            special_logits: floats(DFLASH_SLOTS * SPECIALS)?,
            partial_out: MetalBuffer::empty(context.device(), out_bytes)?,
            partial_stats: MetalBuffer::empty(context.device(), stats_bytes)?,
            taps: floats(CONTEXT_CHUNK * TAPS * WIDTH)?,
            context: floats(CONTEXT_CHUNK * WIDTH)?,
            context_normed: floats(CONTEXT_CHUNK * WIDTH)?,
            context_k: floats(CONTEXT_CHUNK * KV_WIDTH)?,
            context_v: floats(CONTEXT_CHUNK * KV_WIDTH)?,
        })
    }

    fn bytes(&self) -> usize {
        [
            &self.x,
            &self.h,
            &self.hc,
            &self.branch,
            &self.kernels,
            &self.q,
            &self.k,
            &self.v,
            &self.attention,
            &self.gate,
            &self.up,
            &self.product,
            &self.hidden,
            &self.rotated,
            &self.selector,
            &self.logits,
            &self.special_logits,
            &self.partial_out,
            &self.partial_stats,
            &self.taps,
            &self.context,
            &self.context_normed,
            &self.context_k,
            &self.context_v,
        ]
        .iter()
        .map(|buffer| buffer.length())
        .sum()
    }
}

/// The loaded drafter and one sequence's drafting state: its context K/V
/// ring and the outputs of the last pass.
///
/// Encoding only records work. The caller completes a batch before reading
/// [`Self::read_lattice`] or [`Self::read_tree`], and must not encode a pass
/// while a previous pass of this drafter is still executing: the passes
/// share scratch, and [`Self::draft`] with [`DraftAnchor::Token`] and
/// [`Self::set_calibration`] write host-visible buffers at encode time.
pub struct DflashDrafter {
    weights: Weights,
    path: PathBuf,
    kernels: DflashKernels,
    embed: DraftRows,
    /// Top 16 of the reduced head's first rows, and that merged with the
    /// added tokens: the candidates the lattice reads.
    top: TopKRows,
    head_top: MetalBuffer,
    /// Top 16 of the full head.
    full_top: TopKRows,
    /// Consecutive host anchors the reduced head scores, up to
    /// [`REDUCED_HEAD_ANCHORS`]; below that, passes score the full head.
    scored_anchors: usize,
    rope: RopeTables,
    key_ring: Vec<MetalBuffer>,
    value_ring: Vec<MetalBuffer>,
    scratch: Scratch,
    /// `[anchor, mask]` token ids the embedding gather, the selector and the
    /// tree read.
    tokens: MetalBuffer,
    lattice: MetalBuffer,
    tree: MetalBuffer,
    calibration: MetalBuffer,
    context_len: usize,
    /// The first position the context holds (see [`Self::restart_at`]).
    floor: usize,
    draft_gpu_time: Option<Duration>,
}

impl DflashDrafter {
    /// Map, validate and prepare the drafter artifact at `path`. A corrupt,
    /// truncated or foreign file is an error naming the file and the reason.
    pub fn load(
        context: &MetalContext,
        shaders: &ShaderLibrary,
        path: &Path,
    ) -> crate::Result<Self> {
        let weights = Weights::load(context.device(), path)?;
        let ring = || {
            (0..super::LAYERS)
                .map(|_| MetalBuffer::empty(context.device(), RING_CAPACITY * KV_WIDTH * 2))
                .collect::<local_metal::Result<Vec<_>>>()
        };
        let tokens = MetalBuffer::from_slice(context.device(), &[0_u32, MASK_TOKEN])?;
        let calibration =
            MetalBuffer::from_slice(context.device(), &[DflashCalibration::default()])?;
        Ok(Self {
            weights,
            path: path.to_owned(),
            kernels: DflashKernels::new(context, shaders)?,
            embed: DraftRows::new(context, shaders, VOCAB, DFLASH_BLOCK)?,
            top: TopKRows::new(context, shaders, HEAD_ROWS, DFLASH_SLOTS)?,
            head_top: MetalBuffer::empty(
                context.device(),
                DFLASH_SLOTS * MAX_TOP_K * size_of::<TopKCandidate>(),
            )?,
            full_top: TopKRows::new(context, shaders, VOCAB, DFLASH_SLOTS)?,
            scored_anchors: 0,
            rope: RopeTables::new(context, ROPE_THETA)?,
            key_ring: ring()?,
            value_ring: ring()?,
            scratch: Scratch::new(context)?,
            tokens,
            lattice: MetalBuffer::empty(context.device(), size_of::<DflashLattice>())?,
            tree: MetalBuffer::empty(context.device(), size_of::<DflashTree>())?,
            calibration,
            context_len: 0,
            floor: 0,
            draft_gpu_time: None,
        })
    }

    /// The artifact this drafter was loaded from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The payload digest the load re-derived: the drafter's identity.
    #[must_use]
    pub fn payload_sha256(&self) -> &str {
        &self.weights.payload_sha256
    }

    /// Mapped weight bytes.
    #[must_use]
    pub const fn weight_bytes(&self) -> u64 {
        self.weights.bytes
    }

    /// Bytes this drafter allocates beside its weights: the context ring and
    /// the scratch of both passes.
    #[must_use]
    pub fn state_bytes(&self) -> usize {
        self.key_ring
            .iter()
            .chain(&self.value_ring)
            .map(MetalBuffer::length)
            .sum::<usize>()
            + self.scratch.bytes()
    }

    /// Positions of context the drafter holds: the next draft's anchor
    /// position.
    #[must_use]
    pub const fn context_len(&self) -> usize {
        self.context_len
    }

    /// The first position the context holds: 0, unless [`Self::restart_at`]
    /// started it later.
    #[must_use]
    pub const fn context_floor(&self) -> usize {
        self.floor
    }

    /// Forget the context (a new sequence).
    pub const fn reset(&mut self) {
        self.context_len = 0;
        self.floor = 0;
        self.scored_anchors = 0;
    }

    /// Forget the context and continue it from `position`: the next append
    /// holds positions from `position` on, and drafts attend only the
    /// positions appended since (at most the window), as a sequence that
    /// started there would. For a target whose hidden taps exist only from
    /// `position` on (a prompt longer than the window, a restored prefix, or
    /// positions decoded without taps).
    pub const fn restart_at(&mut self, position: usize) {
        self.context_len = position;
        self.floor = position;
    }

    /// Roll the context back to `len` positions, however far: the floor
    /// rises past the positions the ring no longer holds (it holds the last
    /// [`RING_CAPACITY`]), so drafts attend at most the window of what
    /// remains. For a prefix of this context restored into the target from
    /// a prompt cache. Fails when nothing below `len` would remain.
    pub fn rewind_to(&mut self, len: usize) -> crate::Result<()> {
        let floor = self
            .floor
            .max(self.context_len.saturating_sub(RING_CAPACITY));
        if len > self.context_len || len <= floor {
            return Err(crate::Error::InvalidArgument(format!(
                "DFlash2 context of positions {floor}..{} cannot rewind to {len}",
                self.context_len
            )));
        }
        self.floor = floor;
        self.context_len = len;
        Ok(())
    }

    /// Roll the context back to `len` positions, at most [`MAX_ROLLBACK`]
    /// behind: the ring still holds every position the window needs.
    pub fn truncate_context(&mut self, len: usize) -> crate::Result<()> {
        if len > self.context_len || self.context_len - len > MAX_ROLLBACK || len < self.floor {
            return Err(crate::Error::InvalidArgument(format!(
                "DFlash2 context of {} positions cannot roll back to {len}",
                self.context_len
            )));
        }
        self.context_len = len;
        Ok(())
    }

    /// Append `rows` committed positions (`context_len .. context_len +
    /// rows`) from the target's hidden taps: `[rows, 5, 5120]` values of
    /// `format` starting at byte `offset` of `taps`, layers 5, 19, 33, 47 and
    /// 61 in that order. Only the newest [`RING_CAPACITY`] rows of a longer
    /// block are computed; older ones could never be read.
    pub fn append_context(
        &mut self,
        batch: &mut CommandBatch,
        shared: &DflashShared<'_>,
        taps: &MetalBuffer,
        offset: usize,
        rows: usize,
        format: TapFormat,
    ) -> crate::Result<()> {
        let row_bytes = TAPS * WIDTH * format.bytes();
        if offset
            .checked_add(rows * row_bytes)
            .is_none_or(|end| end > taps.length())
        {
            return Err(crate::Error::InvalidArgument(
                "DFlash2 taps exceed their buffer".into(),
            ));
        }
        let skip = rows.saturating_sub(RING_CAPACITY);
        let mut done = skip;
        while done < rows {
            let count = (rows - done).min(CONTEXT_CHUNK);
            let position = self.context_len + done;
            self.append_chunk(
                batch,
                shared,
                taps,
                offset + done * row_bytes,
                count,
                format,
                position,
            )?;
            done += count;
        }
        self.context_len += rows;
        Ok(())
    }

    #[expect(clippy::too_many_arguments)]
    fn append_chunk(
        &self,
        batch: &mut CommandBatch,
        shared: &DflashShared<'_>,
        taps: &MetalBuffer,
        offset: usize,
        rows: usize,
        format: TapFormat,
        position: usize,
    ) -> crate::Result<()> {
        let s = &self.scratch;
        let w = &self.weights;
        let k = &self.kernels;
        let tokens = rows as u32;
        k.taps(
            batch,
            taps,
            offset,
            format == TapFormat::F16,
            &s.taps,
            rows * TAPS * WIDTH,
        )?;
        shared
            .kernels
            .int8_matmul(batch, w.fc.matrix()?, &s.taps, &s.context, tokens)?;
        k.rms_rows(
            batch,
            &s.context,
            0,
            &w.hidden_norm,
            &s.context_normed,
            rows,
            WIDTH,
            EPSILON,
        )?;
        for (layer, (keys, values)) in w
            .layers
            .iter()
            .zip(self.key_ring.iter().zip(&self.value_ring))
        {
            batch.independent(|batch| -> crate::Result<()> {
                shared.kernels.int8_matmul(
                    batch,
                    layer.k.matrix()?,
                    &s.context_normed,
                    &s.context_k,
                    tokens,
                )?;
                shared.kernels.int8_matmul(
                    batch,
                    layer.v.matrix()?,
                    &s.context_normed,
                    &s.context_v,
                    tokens,
                )?;
                Ok(())
            })?;
            k.ctx_store(
                batch,
                &s.context_k,
                &s.context_v,
                &layer.k_norm,
                &self.rope,
                keys,
                values,
                position,
                rows,
                RING_CAPACITY,
                EPSILON,
            )?;
        }
        Ok(())
    }

    /// Encode the drafting pass for `anchor` at sequence position `position`
    /// (which must equal [`Self::context_len`]), leaving the lattice in
    /// [`Self::lattice_buffer`]. Follow it with [`Self::build_tree`].
    ///
    /// The pass scores the reduced output head once the last
    /// [`REDUCED_HEAD_ANCHORS`] host anchors since [`Self::reset`] were all
    /// tokens it scores, and the full head otherwise; device anchors leave
    /// that count as it is.
    #[expect(clippy::too_many_lines)]
    pub fn draft(
        &mut self,
        batch: &mut CommandBatch,
        shared: &DflashShared<'_>,
        anchor: &DraftAnchor<'_>,
        position: usize,
    ) -> crate::Result<()> {
        if position != self.context_len {
            return Err(crate::Error::InvalidArgument(format!(
                "DFlash2 draft at position {position} needs context through {}, the drafter \
                 holds {}",
                position.saturating_sub(1),
                self.context_len
            )));
        }
        match anchor {
            DraftAnchor::Token(token) => {
                self.tokens.copy_from_bytes(&token.to_le_bytes(), 0);
                self.scored_anchors = if reduced_head_scores(*token) {
                    (self.scored_anchors + 1).min(REDUCED_HEAD_ANCHORS)
                } else {
                    0
                };
            }
            DraftAnchor::Device { buffer, offset } => {
                batch.blit_buffer_copies([BufferCopyRequest {
                    source: buffer,
                    source_offset: *offset,
                    destination: &self.tokens,
                    destination_offset: 0,
                    size: 4,
                }])?;
            }
        }
        let s = &self.scratch;
        let w = &self.weights;
        let k = &self.kernels;
        let kernels = shared.kernels;
        let rows = DFLASH_BLOCK as u32;
        let nctx = (position - self.floor).min(DFLASH_WINDOW - 1);
        self.embed.embed_inverse(
            batch,
            shared.embedding,
            shared.rotation,
            &self.tokens,
            &[0, 1, 1, 1, 1, 1, 1, 1],
            &s.x,
        )?;
        for ((layer, keys), values) in w.layers.iter().zip(&self.key_ring).zip(&self.value_ring) {
            // Attention, wrapped in its grouped convolution.
            k.rms_rows(
                batch,
                &s.x,
                0,
                &layer.input_norm,
                &s.h,
                DFLASH_BLOCK,
                WIDTH,
                EPSILON,
            )?;
            kernels.int8_matmul(
                batch,
                layer.attention_kernels.matrix()?,
                &s.h,
                &s.kernels,
                rows,
            )?;
            k.conv(
                batch,
                &s.h,
                &s.kernels,
                &layer.attention_base,
                0,
                &s.hc,
                DFLASH_BLOCK,
            )?;
            batch.independent(|batch| -> crate::Result<()> {
                kernels.int8_matmul(batch, layer.q.matrix()?, &s.hc, &s.q, rows)?;
                kernels.int8_matmul(batch, layer.k.matrix()?, &s.hc, &s.k, rows)?;
                kernels.int8_matmul(batch, layer.v.matrix()?, &s.hc, &s.v, rows)?;
                Ok(())
            })?;
            k.qk_norm_rope(
                batch,
                &s.q,
                &s.k,
                &layer.q_norm,
                &layer.k_norm,
                &self.rope,
                position,
                DFLASH_BLOCK,
                EPSILON,
            )?;
            k.attention(
                batch,
                &s.q,
                &s.k,
                &s.v,
                keys,
                values,
                (&s.partial_out, &s.partial_stats),
                &s.attention,
                position,
                nctx,
                RING_CAPACITY,
            )?;
            kernels.int8_matmul(batch, layer.o.matrix()?, &s.attention, &s.branch, rows)?;
            k.conv(
                batch,
                &s.branch,
                &s.kernels,
                &layer.attention_base,
                1,
                &s.hc,
                DFLASH_BLOCK,
            )?;
            k.add(batch, &s.x, &s.hc, DFLASH_BLOCK * WIDTH)?;
            // The MLP, wrapped in its own convolution.
            k.rms_rows(
                batch,
                &s.x,
                0,
                &layer.post_attention_norm,
                &s.h,
                DFLASH_BLOCK,
                WIDTH,
                EPSILON,
            )?;
            kernels.int8_matmul(batch, layer.mlp_kernels.matrix()?, &s.h, &s.kernels, rows)?;
            k.conv(
                batch,
                &s.h,
                &s.kernels,
                &layer.mlp_base,
                0,
                &s.hc,
                DFLASH_BLOCK,
            )?;
            batch.independent(|batch| -> crate::Result<()> {
                kernels.int8_matmul(batch, layer.gate.matrix()?, &s.hc, &s.gate, rows)?;
                kernels.int8_matmul(batch, layer.up.matrix()?, &s.hc, &s.up, rows)?;
                Ok(())
            })?;
            k.silu_mul(batch, &s.gate, &s.up, &s.product, DFLASH_BLOCK * FFN)?;
            kernels.int8_matmul(batch, layer.down.matrix()?, &s.product, &s.branch, rows)?;
            k.conv(
                batch,
                &s.branch,
                &s.kernels,
                &layer.mlp_base,
                1,
                &s.hc,
                DFLASH_BLOCK,
            )?;
            k.add(batch, &s.x, &s.hc, DFLASH_BLOCK * WIDTH)?;
        }
        // Rows 1..8 are the drafts: final norm, then the selector's projection
        // in the drafter's basis and the shared output head in the target's:
        // its first `HEAD_ROWS` rows and the added tokens, or every row while
        // the recent anchors include a token those lack.
        let slots = DFLASH_SLOTS as u32;
        k.rms_rows(
            batch,
            &s.x,
            WIDTH * 4,
            &w.norm,
            &s.hidden,
            DFLASH_SLOTS,
            WIDTH,
            EPSILON,
        )?;
        batch.independent(|batch| -> crate::Result<()> {
            kernels.int8_matmul(batch, w.selector.matrix()?, &s.hidden, &s.selector, slots)?;
            kernels.transform(
                batch,
                shared.rotation,
                &s.hidden,
                &s.rotated,
                slots,
                HadamardDirection::Forward,
            )?;
            Ok(())
        })?;
        let candidates = if self.scored_anchors < REDUCED_HEAD_ANCHORS {
            kernels.matmul(batch, shared.output, &s.rotated, &s.logits, slots)?;
            self.full_top
                .encode(batch, &s.logits, DFLASH_SLOTS, DFLASH_TOPK)?;
            self.full_top.output_buffer()
        } else {
            let head = shared.output.row_slice(0, HEAD_ROWS as u32)?;
            let special = shared.output.row_slice(SPECIAL_FIRST, SPECIALS as u32)?;
            batch.independent(|batch| -> crate::Result<()> {
                kernels.matmul(batch, head, &s.rotated, &s.logits, slots)?;
                kernels.matmul(batch, special, &s.rotated, &s.special_logits, slots)?;
                Ok(())
            })?;
            self.top
                .encode(batch, &s.logits, DFLASH_SLOTS, DFLASH_TOPK)?;
            k.head_merge(
                batch,
                self.top.output_buffer(),
                &s.special_logits,
                SPECIAL_FIRST,
                SPECIALS,
                &self.head_top,
            )?;
            &self.head_top
        };
        k.lattice(
            batch,
            candidates,
            &s.selector,
            &w.predecessor,
            &w.successor,
            &self.tokens,
            &self.lattice,
        )?;
        Ok(())
    }

    /// Encode the best-first tree of up to `max_rows` (1..=8) rows over the
    /// lattice the preceding [`Self::draft`] encoded, into
    /// [`Self::tree_buffer`].
    pub fn build_tree(&self, batch: &mut CommandBatch, max_rows: usize) -> crate::Result<()> {
        self.kernels.tree(
            batch,
            &self.lattice,
            &self.tokens,
            &self.calibration,
            &self.tree,
            max_rows,
        )?;
        Ok(())
    }

    /// The GPU lattice ([`DflashLattice`]) of the last drafting pass.
    #[must_use]
    pub const fn lattice_buffer(&self) -> &MetalBuffer {
        &self.lattice
    }

    /// The GPU tree ([`DflashTree`]) of the last [`Self::build_tree`].
    #[must_use]
    pub const fn tree_buffer(&self) -> &MetalBuffer {
        &self.tree
    }

    /// Whether the last pass scored the full head.
    #[cfg(test)]
    pub(super) const fn scored_full_head(&self) -> bool {
        self.scored_anchors < REDUCED_HEAD_ANCHORS
    }

    /// The forward-rotated draft rows `[7, 5120]` the last pass scored.
    #[cfg(test)]
    pub(super) const fn rotated_rows(&self) -> &MetalBuffer {
        &self.scratch.rotated
    }

    /// Copy of the last lattice; call only after its batch completed.
    #[must_use]
    #[expect(unsafe_code)]
    pub fn read_lattice(&self) -> DflashLattice {
        // SAFETY: the caller reads after the drafting batch completed and
        // encodes nothing that writes the lattice meanwhile.
        unsafe { self.lattice.as_slice::<DflashLattice>()[0] }
    }

    /// Copy of the last tree; call only after its batch completed.
    #[must_use]
    #[expect(unsafe_code)]
    pub fn read_tree(&self) -> DflashTree {
        // SAFETY: as `read_lattice`.
        unsafe { self.tree.as_slice::<DflashTree>()[0] }
    }

    /// Use `calibration` for the next trees. Call between passes.
    pub fn set_calibration(&self, calibration: &DflashCalibration) {
        self.calibration
            .copy_from_bytes(bytemuck::bytes_of(calibration), 0);
    }

    /// Draft and build an `max_rows`-row tree in a command buffer of their
    /// own, wait for it, and fold its GPU time into [`Self::draft_gpu_time`].
    pub fn draft_and_wait(
        &mut self,
        context: &MetalContext,
        shared: &DflashShared<'_>,
        anchor: &DraftAnchor<'_>,
        position: usize,
        max_rows: usize,
    ) -> crate::Result<Duration> {
        let mut batch = CommandBatch::new(context)?;
        self.draft(&mut batch, shared, anchor, position)?;
        self.build_tree(&mut batch, max_rows)?;
        let elapsed = batch.commit_async().wait_with_gpu_time()?;
        self.draft_gpu_time = Some(self.draft_gpu_time.map_or(elapsed, |previous| {
            // An exponential average over about eight passes.
            previous.mul_f64(0.875) + elapsed.mul_f64(0.125)
        }));
        Ok(elapsed)
    }

    /// Measure the drafting pass (draft and 8-row tree) at the current
    /// context, over the reduced head that English text and code draft
    /// with: `passes` passes, each in its own command buffer, after one
    /// warm-up. Returns the median GPU time and records it as
    /// [`Self::draft_gpu_time`]. Neither the context nor the head choice is
    /// changed.
    pub fn measure_draft_time(
        &mut self,
        context: &MetalContext,
        shared: &DflashShared<'_>,
        passes: usize,
    ) -> crate::Result<Duration> {
        let position = self.context_len;
        let anchor = DraftAnchor::Token(MASK_TOKEN);
        let scored_anchors = std::mem::replace(&mut self.scored_anchors, REDUCED_HEAD_ANCHORS);
        let times = (0..=passes.max(1))
            .map(|_| self.draft_and_wait(context, shared, &anchor, position, DFLASH_MAX_TREE_ROWS))
            .collect::<crate::Result<Vec<_>>>();
        self.scored_anchors = scored_anchors;
        let mut times = times?.split_off(1);
        times.sort();
        let median = times[times.len() / 2];
        self.draft_gpu_time = Some(median);
        Ok(median)
    }

    /// The measured GPU time of one drafting pass, once
    /// [`Self::measure_draft_time`] or [`Self::draft_and_wait`] has run.
    #[must_use]
    pub const fn draft_gpu_time(&self) -> Option<Duration> {
        self.draft_gpu_time
    }
}
