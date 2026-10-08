use std::ptr::NonNull;
use std::sync::Arc;

use local_metal::buffer::MetalBuffer;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::MtpSettings;
use super::format::{align_page, read_head_with};
use super::ternary::{INT8_SUFFIX, formats_from_names, head_spec};
use super::{ATTENTION, FFN, HEAD_DIM, KV, QUERY_GATE, WIDTH, invalid};

/// Zero-centered Qwen3.5 `RMSNorm` weights become `1 + w` for the runtime kernel,
/// matching the GGUF converter's folding of the target's norms.
pub fn fold_norm(bf16: &[u8]) -> crate::Result<Vec<f32>> {
    let mut folded = Vec::with_capacity(bf16.len() / 2);
    for pair in bf16.as_chunks::<2>().0 {
        let value = half::bf16::from_bits(u16::from_le_bytes(*pair)).to_f32();
        if !value.is_finite() {
            return invalid("non-finite MTP norm weight");
        }
        folded.push(value + 1.0);
    }
    Ok(folded)
}

/// One head matrix as the GPU multiplies it, in the target's rotated basis:
/// the input is rotated with the target's forward transform for `columns`,
/// then multiplied by the target's own `PTQ1_0` kernels or the int8 ones.
pub(super) enum MatrixWeight {
    Ptq1 {
        packed: MetalBuffer,
        rows: u32,
        columns: u32,
    },
    /// Row-major int8 values and one F32 scale per row.
    Int8 {
        weights: MetalBuffer,
        scales: MetalBuffer,
        rows: u32,
        columns: u32,
    },
}

/// One head matrix: the name its artifact section and source tensors carry, and
/// its pinned shape.
pub(super) struct MatrixSpec {
    pub(super) section: &'static str,
    pub(super) rows: usize,
    pub(super) columns: usize,
}

/// Artifact order is `Weights` field order: section `i` is matrix `i`.
pub(super) const MATRIX_SPECS: [MatrixSpec; 9] = [
    MatrixSpec {
        section: "mtp.fc.weight.embedding",
        rows: WIDTH,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.fc.weight.hidden",
        rows: WIDTH,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.layers.0.self_attn.q_proj.weight",
        rows: QUERY_GATE,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.layers.0.self_attn.k_proj.weight",
        rows: KV,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.layers.0.self_attn.v_proj.weight",
        rows: KV,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.layers.0.self_attn.o_proj.weight",
        rows: WIDTH,
        columns: ATTENTION,
    },
    MatrixSpec {
        section: "mtp.layers.0.mlp.gate_proj.weight",
        rows: FFN,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.layers.0.mlp.up_proj.weight",
        rows: FFN,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.layers.0.mlp.down_proj.weight",
        rows: WIDTH,
        columns: FFN,
    },
];

/// One folded norm. Its element count is stated rather than inferred from the
/// name, because `q_norm` and `k_norm` are `HEAD_DIM` while every other norm is
/// `WIDTH` — and the section name has to be the tensor name, not a decoration.
pub(super) struct NormSpec {
    pub(super) section: &'static str,
    pub(super) elements: usize,
}

pub(super) const NORM_SPECS: [NormSpec; 7] = [
    NormSpec {
        section: "mtp.pre_fc_norm_embedding.weight",
        elements: WIDTH,
    },
    NormSpec {
        section: "mtp.pre_fc_norm_hidden.weight",
        elements: WIDTH,
    },
    NormSpec {
        section: "mtp.layers.0.input_layernorm.weight",
        elements: WIDTH,
    },
    NormSpec {
        section: "mtp.layers.0.post_attention_layernorm.weight",
        elements: WIDTH,
    },
    NormSpec {
        section: "mtp.layers.0.self_attn.q_norm.weight",
        elements: HEAD_DIM,
    },
    NormSpec {
        section: "mtp.layers.0.self_attn.k_norm.weight",
        elements: HEAD_DIM,
    },
    NormSpec {
        section: "mtp.norm.weight",
        elements: WIDTH,
    },
];

pub(super) struct Weights {
    pub(super) fc_embedding: MatrixWeight,
    pub(super) fc_hidden: MatrixWeight,
    pub(super) embedding_norm: MetalBuffer,
    pub(super) hidden_norm: MetalBuffer,
    pub(super) input_norm: MetalBuffer,
    pub(super) post_attention_norm: MetalBuffer,
    pub(super) query_gate: MatrixWeight,
    pub(super) key: MatrixWeight,
    pub(super) value: MatrixWeight,
    pub(super) output: MatrixWeight,
    pub(super) query_norm: MetalBuffer,
    pub(super) key_norm: MetalBuffer,
    pub(super) gate: MatrixWeight,
    pub(super) up: MatrixWeight,
    pub(super) down: MatrixWeight,
    pub(super) final_norm: MetalBuffer,
}

impl Weights {
    /// Map and validate the head artifact at `settings.path`.
    ///
    /// An artifact that fails validation is a hard error, not a fallback:
    /// decoding without speculation would look like a healthy install rather
    /// than a broken one.
    pub(super) fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        settings: &MtpSettings,
    ) -> crate::Result<(Self, u64)> {
        let path = &settings.path;
        let bind = |names: &[&str]| formats_from_names(names).map(|formats| head_spec(&formats));
        let head = read_head_with(path, &bind).map_err(|reason| {
            crate::Error::InvalidFormat(format!(
                "MTP head artifact {} is unusable: {reason}; re-export it with \
                 `local-ai bonsai --export mtp-head=<dir>` or remove it",
                path.display()
            ))
        })?;
        let buffer = |index: usize| -> crate::Result<MetalBuffer> {
            let (offset, length) = head.sections[index];
            let pointer = NonNull::new(head.map[offset..].as_ptr().cast_mut())
                .ok_or_else(|| crate::Error::InvalidFormat("empty MTP head mapping".into()))?;
            let owner: Arc<dyn Send + Sync> = head.map.clone();
            // SAFETY: sections begin on VM pages, their padded ranges are
            // validated within the immutable mapping, and `owner` retains it.
            #[allow(unsafe_code)]
            unsafe {
                Ok(MetalBuffer::from_bytes_no_copy(
                    device,
                    pointer,
                    align_page(length).ok_or_else(|| {
                        crate::Error::InvalidFormat("MTP head size overflow".into())
                    })?,
                    owner,
                )?)
            }
        };
        // Each matrix starts where the previous one's sections end: one
        // section for `PTQ1_0`, two (weights, row scales) for int8.
        let mut matrices = Vec::with_capacity(MATRIX_SPECS.len());
        let mut at = 0;
        for matrix in &MATRIX_SPECS {
            let (rows, columns) = (matrix.rows as u32, matrix.columns as u32);
            let int8 = head.spec[at].0.ends_with(INT8_SUFFIX);
            matrices.push(Some(if int8 {
                at += 2;
                MatrixWeight::Int8 {
                    weights: buffer(at - 2)?,
                    scales: buffer(at - 1)?,
                    rows,
                    columns,
                }
            } else {
                at += 1;
                MatrixWeight::Ptq1 {
                    packed: buffer(at - 1)?,
                    rows,
                    columns,
                }
            }));
        }
        let norm = |index: usize| buffer(at + index);
        let mut matrix = |index: usize| {
            matrices[index]
                .take()
                .ok_or_else(|| crate::Error::InvalidFormat("MTP head matrix bound twice".into()))
        };
        let weights = Self {
            fc_embedding: matrix(0)?,
            fc_hidden: matrix(1)?,
            embedding_norm: norm(0)?,
            hidden_norm: norm(1)?,
            input_norm: norm(2)?,
            post_attention_norm: norm(3)?,
            query_gate: matrix(2)?,
            key: matrix(3)?,
            value: matrix(4)?,
            output: matrix(5)?,
            query_norm: norm(4)?,
            key_norm: norm(5)?,
            gate: matrix(6)?,
            up: matrix(7)?,
            down: matrix(8)?,
            final_norm: norm(6)?,
        };
        let total = head.spec.iter().map(|(_, bytes)| *bytes as u64).sum();
        Ok((weights, total))
    }
}
