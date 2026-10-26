//! The BF16 vision artifact written by `tools/convert_embeddinggemma2_vision.py`.
//!
//! Same layout as the text artifact (8-byte magic, `u64` header length, JSON
//! header, payload) with its own magic and format. Opening checks the magic,
//! format, version, architecture constants, every tensor's bounds, dtype and
//! shape, and the payload's SHA-256, so a truncated, foreign or edited file is
//! refused rather than misread.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde::Deserialize;
use sha2::{Digest as _, Sha256};

use super::{HEAD_DIM, HEADS, LAYERS, MLP, POSITIONS, WIDTH};
use crate::Error;

const MAGIC: &[u8; 8] = b"EG2VIS\x00\x01";
const FORMAT: &str = "embeddinggemma2-vision";
const VERSION: u32 = 1;
const MAX_HEADER: u64 = 1 << 20;

#[derive(Deserialize)]
struct Header {
    format: String,
    version: u32,
    config: Config,
    tensors: BTreeMap<String, RawTensor>,
    payload_length: u64,
    payload_sha256: String,
}

#[derive(Deserialize)]
struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_attention_heads: usize,
    head_dim: usize,
    patch_size: usize,
    pooling_kernel_size: usize,
    position_embedding_size: usize,
    rope_theta: f64,
    text_hidden_size: usize,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Dtype {
    Bf16,
    F32,
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

#[derive(Clone, Copy)]
struct TensorInfo {
    dtype: Dtype,
    shape: [usize; 2],
    offset: usize,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidFormat(format!("vision artifact: {}", message.into()))
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(64), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

/// A verified, memory-mapped vision artifact.
pub(super) struct VisionArtifact {
    map: Mmap,
    payload_start: usize,
    tensors: BTreeMap<String, TensorInfo>,
    pub(super) payload_sha256: String,
}

impl VisionArtifact {
    pub(super) fn open(path: &Path) -> crate::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the file is opened read-only and treated as immutable; the
        // installation is never rewritten in place (new installs are new files).
        #[expect(unsafe_code)]
        let map = unsafe { Mmap::map(&file)? };
        if map.len() < 16 || &map[..8] != MAGIC {
            return Err(invalid("bad magic"));
        }
        let header_length =
            u64::from_le_bytes(map[8..16].try_into().map_err(|_| invalid("short"))?);
        if header_length > MAX_HEADER || 16 + header_length > map.len() as u64 {
            return Err(invalid("bad header length"));
        }
        let payload_start = 16 + header_length as usize;
        let header: Header = serde_json::from_slice(&map[16..payload_start])?;
        if header.format != FORMAT || header.version != VERSION {
            return Err(invalid(format!(
                "format {} v{} is not {FORMAT} v{VERSION}",
                header.format, header.version
            )));
        }
        let c = &header.config;
        if c.hidden_size != WIDTH
            || c.intermediate_size != MLP
            || c.num_layers != LAYERS
            || c.num_attention_heads != HEADS
            || c.head_dim != HEAD_DIM
            || c.patch_size != super::PATCH_SIZE
            || c.pooling_kernel_size != super::POOLING
            || c.position_embedding_size != POSITIONS
            || (c.rope_theta - f64::from(super::ROPE_THETA)).abs() > 0.0
            || c.text_hidden_size != super::TEXT_HIDDEN
        {
            return Err(invalid("architecture differs from EmbeddingGemma 2"));
        }
        let payload = &map[payload_start..];
        if payload.len() as u64 != header.payload_length {
            return Err(invalid("payload length differs from the header"));
        }
        let digest = hex(Sha256::digest(payload).as_slice());
        if digest != header.payload_sha256 {
            return Err(invalid(format!(
                "payload hashes to {digest}, header records {}",
                header.payload_sha256
            )));
        }
        let mut tensors = BTreeMap::new();
        for (name, tensor) in header.tensors {
            let shape = match tensor.shape.as_slice() {
                [n] => [1, *n],
                [rows, cols] => [*rows, *cols],
                _ => return Err(invalid(format!("tensor {name} has rank > 2"))),
            };
            let element = match tensor.dtype {
                Dtype::Bf16 => 2,
                Dtype::F32 => 4,
            };
            let expected = shape[0]
                .checked_mul(shape[1])
                .and_then(|n| n.checked_mul(element));
            if expected != Some(tensor.length)
                || tensor
                    .offset
                    .checked_add(tensor.length)
                    .is_none_or(|end| end > payload.len())
                || !tensor.offset.is_multiple_of(16)
            {
                return Err(invalid(format!(
                    "tensor {name} is out of bounds or misshapen"
                )));
            }
            tensors.insert(
                name,
                TensorInfo {
                    dtype: tensor.dtype,
                    shape,
                    offset: tensor.offset,
                },
            );
        }
        Ok(Self {
            map,
            payload_start,
            tensors,
            payload_sha256: header.payload_sha256,
        })
    }

    pub(super) fn payload(&self) -> &[u8] {
        &self.map[self.payload_start..]
    }

    fn info(&self, name: &str) -> crate::Result<TensorInfo> {
        self.tensors
            .get(name)
            .copied()
            .ok_or_else(|| Error::MissingTensor(format!("vision {name}")))
    }

    /// Payload offset of a BF16 `[rows, cols]` matrix.
    pub(super) fn bf16(&self, name: &str, rows: usize, cols: usize) -> crate::Result<usize> {
        let info = self.info(name)?;
        if info.dtype != Dtype::Bf16 || info.shape != [rows, cols] {
            return Err(invalid(format!(
                "tensor {name} has an unexpected dtype or shape"
            )));
        }
        Ok(info.offset)
    }

    /// Payload offset of an F32 vector of `count` values.
    pub(super) fn f32(&self, name: &str, count: usize) -> crate::Result<usize> {
        let info = self.info(name)?;
        if info.dtype != Dtype::F32 || info.shape[0] * info.shape[1] != count {
            return Err(invalid(format!(
                "tensor {name} has an unexpected dtype or size"
            )));
        }
        Ok(info.offset)
    }
}
