//! Native speculative decoding for Bonsai 2 with a trained MTP head.
//!
//! The head is one dense Qwen3.5 full-attention decoder layer, distilled from
//! the community `ProCreations/Ternary-Bonsai-2-27B-MTP` head into the
//! target's Hadamard-rotated basis. Each matrix is exactly ternary (one F16
//! scale per 128 columns), shipped as the target's own `PTQ1_0` blocks and
//! multiplied by the target's kernels, or — for the few the trainer keeps at
//! higher precision — int8 with one F32 scale per row, multiplied by the int8
//! kernels from the same rotated activations. All of it ships in
//! `mtp-head-ptq1-v1.bin`.
//! Row `j` consumes the target's inverse-rotated embedding of token `j` and the
//! target's *output-normalized* final hidden of token `j - 1` (zero for row 0),
//! and predicts token `j + 1` through the shared PTQ1 output matrix. Deeper
//! drafts feed the head's own normalized output back as the hidden.
//!
//! Drafts are proposals only: every emitted token is sampled by the target,
//! which verifies the seed and drafts in one block with per-row logits. The
//! head keeps its own F16 KV cache indexed by absolute token position.
//!
//! The artifact is written by `local-ai bonsai --export mtp-head=DIR` from
//! `DIR/model_mtp_ternary.safetensors`, and records its section names, section
//! sizes and payload digest, which every load validates.

mod artifact;
mod format;
mod head;
mod policy;
mod ternary;
#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
mod weights;

#[cfg(test)]
use std::path::PathBuf;

#[cfg(test)]
pub use self::artifact::requantize_head;
pub use self::{
    artifact::{MtpHeadArtifact, export_head},
    format::MTP_HEAD_ARTIFACT,
    head::{BonsaiMtp, HeadSequence, Shared},
    policy::{
        DEFAULT_BONSAI_MTP_ARTIFACT, DEFAULT_MTP_DEPTH, DEFAULT_MTP_ENABLED,
        DRAFT_CHAIN_MIN_MARGIN, MAX_MTP_DEPTH, MtpMode, MtpResolution, MtpSettings,
    },
};

/// One section of a head file: the name the loader binds it to and its exact
/// byte length, in artifact order. A head file records both, so it names its own
/// contents rather than relying on a bare `(offset, length)` table.
type Section = (String, usize);

// The cases in `tests` reach this module through a glob and cover internals no
// other reader names, so those bindings exist only in test builds.
#[cfg(test)]
use self::{
    format::{
        HEADER_BYTES, PAGE, PTQ1_VERSION, align_page, read_head, read_head_with, write_artifact,
    },
    policy::{MTP_DEFAULT_OFF_REASON, MTP_OFF_REASON},
    ternary::{
        MTP_TERNARY_SOURCE, MatrixFormat, PTQ1_BLOCK_BYTES, PTQ1_BLOCK_ELEMENTS,
        SAFETENSORS_MAX_HEADER, formats_from_names, head_sections, head_spec, pack_ptq1_block,
        pack_ptq1_matrix, ternary_spec,
    },
    weights::fold_norm,
};

const WIDTH: usize = 5120;
const FFN: usize = 17408;
const QUERY_GATE: usize = 12288;
const KV: usize = 1024;
const ATTENTION: usize = 6144;
const HEAD_DIM: usize = 256;
const KV_TOKEN_BYTES: usize = 4 * 256 * 2;

fn invalid<T>(message: impl Into<String>) -> crate::Result<T> {
    Err(crate::Error::InvalidFormat(message.into()))
}
