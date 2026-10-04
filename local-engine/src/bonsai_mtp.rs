//! Native speculative decoding for Bonsai 2 with the community MTP head.
//!
//! The head (`ProCreations/Ternary-Bonsai-2-27B-MTP`, `model_mtp.safetensors`)
//! is one dense Qwen3.5 full-attention decoder layer. Its BF16 matrices are
//! quantized at load time to symmetric per-row int8 weights with F32 scales.
//! Row `j` consumes the target's inverse-rotated embedding of token
//! `j` and the target's *output-normalized* final hidden of token `j - 1` (zero
//! for row 0), and predicts token `j + 1` through the shared PTQ1 output matrix.
//! Draft depth 2 feeds the head's own normalized output back as the hidden.
//!
//! Drafts are proposals only: every emitted token is sampled by the target,
//! which verifies the seed and drafts in one block with per-row logits. The
//! head keeps its own F16 KV cache indexed by absolute token position.
//!
//! The quantized form is also a distributable artifact: the same bytes the
//! loader builds for itself are written by `local-ai bonsai --export mtp-head=DIR`,
//! and a head file records its section names, section sizes and payload digest,
//! so an install can ship the artifact and validate it without the 849 MB source.

mod artifact;
mod cache;
mod head;
mod layout;
mod policy;
#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
mod weights;

#[cfg(test)]
use std::path::PathBuf;

pub use self::{
    artifact::{MtpHeadArtifact, export_head, export_head_zstd},
    cache::MTP_HEAD_ARTIFACT,
    head::{BonsaiMtp, Shared},
    policy::{
        DEFAULT_BONSAI_MTP_ARTIFACT, DEFAULT_BONSAI_MTP_HEAD, DEFAULT_MTP_DEPTH,
        DEFAULT_MTP_ENABLED, DRAFT_CHAIN_MIN_MARGIN, MAX_MTP_DEPTH, MtpMode, MtpResolution,
        MtpSettings,
    },
};

/// One section of an int8 head: the name the loader binds it to and its exact
/// byte length, in artifact order. A head file records both, so it names its own
/// contents rather than relying on a bare `(offset, length)` table.
type Section = (String, usize);

// The cases in `tests` reach this module through a glob and cover internals no
// other reader names, so those bindings exist only in test builds.
#[cfg(test)]
use self::{
    cache::{
        CACHE_HEADER_BYTES, CACHE_PAGE, CACHE_SECTIONS, CACHE_VERSION, Codec, HeadKind,
        align_cache, find_cache, head_kind, read_cache, read_head, write_artifact, write_cache,
        write_zstd_artifact,
    },
    layout::{SAFETENSORS_MAX_HEADER, SafetensorsLayout, TENSORS, open_layout},
    policy::{MTP_DEFAULT_OFF_REASON, MTP_OFF_REASON},
    weights::{fold_norm, quantize_rows, spec, split_fc},
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
