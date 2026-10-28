//! `KFlash` drafting for Bonsai 2: a `DFlash2` block drafter, its GPU-resident
//! candidate lattice, and a best-first tree the target verifies in one pass.
//!
//! The drafter is z-lab's `DFlash2DraftModel`; the pinned weights are
//! ProCreations' `Ternary-Bonsai-2-27B-DFlash2` ([`PINNED_DFLASH_SHA256`]),
//! z-lab's `Qwen3.8-27B-DFlash2` trained further against the frozen Bonsai
//! target. It has five bidirectional Qwen3 layers with
//! grouped dynamic convolutions, conditioned on the target's residual stream
//! at layers 5, 19, 33, 47 and 61, drafting an 8-row block (the anchor and
//! seven mask tokens) in one pass. Its rank-256 candidate selector scores
//! adjacent candidates, which turns the top-16 candidates of each of the
//! seven draft positions into a first-order lattice. A best-first tree of up
//! to 8 rows over that lattice, scored with per-position calibrated path
//! probabilities, is what the target verifies (Kapitsa's `KFlash`,
//! `docs/BONSAI.md`). It shares the target's embedding and output matrix, so
//! no second copy of either ships.
//!
//! # Contract for the integrator
//!
//! 1. **Load.** [`DflashMode::resolve`] finds the installed artifact: the
//!    4-bit drafter ([`DEFAULT_BONSAI_DFLASH_INT4_ARTIFACT`]) when it is
//!    there, the int8 one ([`DEFAULT_BONSAI_DFLASH_ARTIFACT`]) otherwise,
//!    exact files beside the target's model directory, written by `local-ai
//!    bonsai --export dflash-head-int4=DIR` and `--export dflash-head=DIR`.
//!    Both absent, it is an error: the drafter is required. Present,
//!    [`DflashDrafter::load`] re-derives the payload digest and fails hard on
//!    any corruption: there is no silent fallback for a broken install. Only
//!    an explicit [`DflashMode::Off`] (a plain-decoding reference) resolves
//!    to [`DflashResolution::Fallback`].
//! 2. **Context.** [`DflashDrafter::append_context`] takes, for newly
//!    committed positions, the target's residual-stream hidden states at the
//!    five tap layers, `[n, 5, 5120]` F16 or F32 rows in a Metal buffer the
//!    target writes (standard Qwen basis: the taps need no unrotation). The
//!    drafter keeps its own F16 context K/V in a ring covering the
//!    2048-token sliding window; append only accepted positions, or roll
//!    back up to [`MAX_ROLLBACK`] positions with
//!    [`DflashDrafter::truncate_context`]. When taps exist only from some
//!    position on (a prompt longer than the window, a restored prefix,
//!    positions decoded in a batched pass), [`DflashDrafter::restart_at`]
//!    starts the context there and drafts attend only what was appended
//!    since.
//! 3. **Draft.** [`DflashDrafter::draft`] encodes the drafting pass for an
//!    anchor token at sequence position `P` (the context must hold exactly
//!    positions `0..P`) onto the caller's `CommandBatch`, leaving the
//!    [`DflashLattice`] on the GPU ([`DflashDrafter::lattice_buffer`]): 7
//!    positions × 16 candidate ids with their logits, the selector's pairwise
//!    scores between adjacent positions, and the conditional log
//!    probabilities the tree uses.
//! 4. **Tree.** [`DflashDrafter::build_tree`] encodes the best-first tree
//!    kernel behind the draft in the same batch, with no CPU round trip: up
//!    to 8 rows (anchor included) with token ids, parent rows, depths,
//!    ancestor masks, and the expected-token prefix sums `E(n)` for
//!    `n = 1..=8` ([`DflashTree`], [`DflashDrafter::tree_buffer`]). The first
//!    `n` rows are the best `n`-row tree for every `n`, parents precede
//!    children, and row `r` sits at sequence position `P + depth[r]`.
//! 5. **Calibration.** [`Calibrator`] holds the per-position logistic
//!    calibration `p = σ(α_d + β_d · logit q)`, starting from `p = q`. Feed it
//!    each cycle's verify outcome ([`Calibrator::record_cycle`]); it refits
//!    every 16 cycles over the last 2,048 records per position and persists
//!    the fit per model, drafter and chip under `~/Library/Caches/local-ai/`.
//!    Upload it with [`DflashDrafter::set_calibration`] between cycles.
//! 6. **Draft cost.** [`DflashDrafter::draft_gpu_time`] reports the measured
//!    GPU time of a drafting pass ([`DflashDrafter::measure_draft_time`]
//!    runs passes in their own command buffers to measure it).
//!
//! Output stays lossless whatever the drafter computes: the target verifies
//! every token.

mod artifact;
mod calibration;
mod drafter;
mod policy;
#[cfg(test)]
#[expect(unsafe_code)]
mod tests;
mod tree;
mod weights;

pub use self::artifact::{
    DFLASH_HEAD_ARTIFACT, DFLASH_INT4_ARTIFACT, DflashHeadArtifact, DflashPrecision,
    export_dflash_head,
};
pub use self::calibration::{
    CALIBRATION_RECORDS, Calibrator, REFIT_CYCLES, calibration_cache_path, fit_depth,
};
pub use self::drafter::{
    DflashDrafter, DflashShared, DraftAnchor, MAX_ROLLBACK, RING_CAPACITY, TapFormat,
};
pub use self::policy::{
    DEFAULT_BONSAI_DFLASH_ARTIFACT, DEFAULT_BONSAI_DFLASH_INT4_ARTIFACT, DflashMode,
    DflashResolution, PINNED_DFLASH_INT4_SHA256, PINNED_DFLASH_SHA256, installed,
};
pub use self::tree::{TreeNode, build_tree_reference, tested_nodes};
pub use local_metal::dflash::{
    DFLASH_BLOCK, DFLASH_MAX_TREE_ROWS, DFLASH_SLOTS, DFLASH_TOPK, DflashCalibration,
    DflashLattice, DflashTree,
};

/// Target layers whose residual stream the drafter reads, in tap order.
pub const TARGET_LAYER_IDS: [usize; 5] = [5, 19, 33, 47, 61];
const TAPS: usize = TARGET_LAYER_IDS.len();
const WIDTH: usize = 5120;
const FFN: usize = 17408;
const LAYERS: usize = 5;
const Q_WIDTH: usize = 4096;
const KV_WIDTH: usize = 1024;
const HEAD_DIM: usize = 128;
const RANK: usize = 256;
const CONV_PROJECTION: usize = 1280;
const VOCAB: usize = 248_320;
/// Output-head rows the drafter scores: the lowest token ids, which for this
/// byte-level BPE vocabulary are its earliest merges, the frequent tokens
/// (Kapitsa's id-order cut). They hold 99.9% of the English text, code and
/// tool calls Bonsai generates and draft like the full head in 2.1 ms less
/// per pass (`docs/BONSAI.md#reduced-draft-head`); the target verifies over
/// the full vocabulary, so output does not change.
const HEAD_ROWS: usize = 98_304;
/// The added tokens the drafter scores beside [`HEAD_ROWS`]:
/// `<|endoftext|>` (248044) through `<|audio_pad|>` (248076), which include
/// the end of turn, the think tags and the tool-call tags. The 243 rows
/// after them are unused padding.
const SPECIAL_FIRST: u32 = 248_044;
const SPECIALS: usize = 33;
/// Consecutive drafting anchors the reduced head must score before the
/// drafter scores it instead of the full head. In a tree cycle a token the
/// reduced head lacks can only enter the text as the target's correction,
/// which is the next tree's anchor, so text made of such tokens (Chinese,
/// Japanese, accented Latin script) keeps the full head, through head
/// cycles too, while English text and code switch at their eighth tree and
/// draft eight trees on the full head after a rare token (one in about a
/// thousand).
const REDUCED_HEAD_ANCHORS: usize = 8;

/// Whether the reduced head scores `token`.
const fn reduced_head_scores(token: u32) -> bool {
    (token as usize) < HEAD_ROWS
        || (token >= SPECIAL_FIRST && token < SPECIAL_FIRST + SPECIALS as u32)
}
/// The drafter's `mask_token_id`: every draft row after the anchor embeds it.
pub const MASK_TOKEN: u32 = 248_070;
const EPSILON: f32 = 1e-6;
const ROPE_THETA: f64 = 10_000_000.0;

fn invalid<T>(message: impl Into<String>) -> crate::Result<T> {
    Err(crate::Error::InvalidFormat(message.into()))
}
