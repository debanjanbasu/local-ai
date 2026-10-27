//! Mapping and validating the drafter artifact into Metal buffers.

use std::collections::HashMap;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use local_metal::bonsai::Int8Matrix;
use local_metal::buffer::MetalBuffer;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::artifact::{DFLASH_CONTAINER, Kind, PREDECESSOR, SUCCESSOR, spec, tensors};
use super::{LAYERS, invalid};
use crate::bonsai_mtp::{align_page, read_container_with};

/// One per-row int8 matrix.
pub(super) struct Int8 {
    weights: MetalBuffer,
    scales: MetalBuffer,
    rows: u32,
    columns: u32,
}

impl Int8 {
    pub(super) fn matrix(&self) -> crate::Result<Int8Matrix<'_>> {
        Ok(Int8Matrix::new(
            &self.weights,
            &self.scales,
            self.rows,
            self.columns,
        )?)
    }
}

pub(super) struct Layer {
    pub(super) input_norm: MetalBuffer,
    pub(super) post_attention_norm: MetalBuffer,
    pub(super) q_norm: MetalBuffer,
    pub(super) k_norm: MetalBuffer,
    /// `[2, 2, 5120]` F32 base kernels of the two convolutions.
    pub(super) attention_base: MetalBuffer,
    pub(super) mlp_base: MetalBuffer,
    pub(super) attention_kernels: Int8,
    pub(super) q: Int8,
    pub(super) k: Int8,
    pub(super) v: Int8,
    pub(super) o: Int8,
    pub(super) mlp_kernels: Int8,
    pub(super) gate: Int8,
    pub(super) up: Int8,
    pub(super) down: Int8,
}

pub(super) struct Weights {
    pub(super) fc: Int8,
    pub(super) hidden_norm: MetalBuffer,
    pub(super) norm: MetalBuffer,
    pub(super) selector: Int8,
    /// `[vocab, 256]` BF16 codebooks.
    pub(super) predecessor: MetalBuffer,
    pub(super) successor: MetalBuffer,
    pub(super) layers: Vec<Layer>,
    /// Payload bytes, which stay mapped (and resident once drafting runs).
    pub(super) bytes: u64,
    pub(super) payload_sha256: String,
}

impl Weights {
    /// Map and validate the artifact at `path`: framing, every section name
    /// and size, and the payload digest, re-derived from the file. Any
    /// failure is a hard error naming the file.
    #[expect(clippy::too_many_lines)]
    pub(super) fn load(device: &ProtocolObject<dyn MTLDevice>, path: &Path) -> crate::Result<Self> {
        let expected = spec();
        let head = read_container_with(path, &DFLASH_CONTAINER, &|_| Ok(expected.clone()))
            .map_err(|reason| {
                crate::Error::InvalidFormat(format!(
                    "DFlash2 drafter artifact {} is unusable: {reason}; re-export it with \
                     `local-ai bonsai --export dflash-head=<dir>` or remove it",
                    path.display()
                ))
            })?;
        let buffer = |index: usize| -> crate::Result<MetalBuffer> {
            let (offset, length) = head.sections[index];
            let pointer = NonNull::new(head.map[offset..].as_ptr().cast_mut())
                .ok_or_else(|| crate::Error::InvalidFormat("empty DFlash2 mapping".into()))?;
            let owner: Arc<dyn Send + Sync> = head.map.clone();
            let padded = align_page(length)
                .ok_or_else(|| crate::Error::InvalidFormat("DFlash2 size overflow".into()))?;
            // SAFETY: sections begin on VM pages, their padded ranges are
            // validated within the immutable mapping, and `owner` retains it.
            #[expect(unsafe_code)]
            unsafe {
                Ok(MetalBuffer::from_bytes_no_copy(
                    device, pointer, padded, owner,
                )?)
            }
        };
        let mut plain = HashMap::new();
        let mut matrices = HashMap::new();
        let mut at = 0;
        for tensor in tensors() {
            match tensor.kind {
                Kind::Int8 { rows, columns } => {
                    matrices.insert(
                        tensor.name,
                        Int8 {
                            weights: buffer(at)?,
                            scales: buffer(at + 1)?,
                            rows: rows as u32,
                            columns: columns as u32,
                        },
                    );
                    at += 2;
                }
                Kind::F32 { .. } | Kind::Bf16 { .. } => {
                    plain.insert(tensor.name, buffer(at)?);
                    at += 1;
                }
            }
        }
        let mut take_plain = |name: &str| {
            plain
                .remove(name)
                .map_or_else(|| invalid(format!("DFlash2 tensor {name} is unbound")), Ok)
        };
        let fc_norms = (
            take_plain("hidden_norm.weight")?,
            take_plain("norm.weight")?,
        );
        let codebooks = (take_plain(PREDECESSOR)?, take_plain(SUCCESSOR)?);
        let mut layer_plain = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let mut get = |suffix: &str| take_plain(&format!("layers.{layer}.{suffix}"));
            layer_plain.push([
                get("input_layernorm.weight")?,
                get("post_attention_layernorm.weight")?,
                get("self_attn.q_norm.weight")?,
                get("self_attn.k_norm.weight")?,
                get("attention_conv.base_kernel")?,
                get("mlp_conv.base_kernel")?,
            ]);
        }
        let mut take = |name: &str| {
            matrices
                .remove(name)
                .map_or_else(|| invalid(format!("DFlash2 matrix {name} is unbound")), Ok)
        };
        let mut layers = Vec::with_capacity(LAYERS);
        for (layer, [input, post, q_norm, k_norm, attention_base, mlp_base]) in
            layer_plain.into_iter().enumerate()
        {
            let mut get = |suffix: &str| take(&format!("layers.{layer}.{suffix}"));
            layers.push(Layer {
                input_norm: input,
                post_attention_norm: post,
                q_norm,
                k_norm,
                attention_base,
                mlp_base,
                attention_kernels: get("attention_conv.kernel_projection.weight")?,
                q: get("self_attn.q_proj.weight")?,
                k: get("self_attn.k_proj.weight")?,
                v: get("self_attn.v_proj.weight")?,
                o: get("self_attn.o_proj.weight")?,
                mlp_kernels: get("mlp_conv.kernel_projection.weight")?,
                gate: get("mlp.gate_proj.weight")?,
                up: get("mlp.up_proj.weight")?,
                down: get("mlp.down_proj.weight")?,
            });
        }
        Ok(Self {
            fc: take("fc.weight")?,
            hidden_norm: fc_norms.0,
            norm: fc_norms.1,
            selector: take("candidate_selector.hidden_projection.weight")?,
            predecessor: codebooks.0,
            successor: codebooks.1,
            layers,
            bytes: expected.iter().map(|(_, bytes)| *bytes as u64).sum(),
            payload_sha256: head.payload_sha256,
        })
    }
}
