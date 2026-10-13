//! The BF16 text artifact written by `tools/convert_embeddinggemma2.py`.
//!
//! Layout: 8-byte magic, `u64` header length, a JSON header, then the payload
//! (tensor offsets are relative to it). Opening checks the magic, the format
//! and version, the architecture constants, every tensor's bounds, dtype and
//! shape, and the payload's SHA-256 against the header, so a truncated,
//! foreign or edited file is refused rather than misread.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde::Deserialize;
use sha2::{Digest as _, Sha256};

use super::{EMBED_DIM, HIDDEN, INTERMEDIATE, LAYERS, VOCAB};
use crate::Error;

const MAGIC: &[u8; 8] = b"EG2TXT\x00\x01";
const FORMAT: &str = "embeddinggemma2-text";
const VERSION: u32 = 1;
const MAX_HEADER: u64 = 1 << 20;

#[derive(Deserialize)]
struct Header {
    format: String,
    version: u32,
    config: serde_json::Value,
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
    vocab_size: usize,
    embedding_dim: usize,
    sliding_window: usize,
    sliding_attention: LayerKind,
    full_attention: LayerKind,
    layer_types: Vec<String>,
}

#[derive(Deserialize, PartialEq)]
struct LayerKind {
    head_dim: usize,
    num_key_value_heads: usize,
    rope_theta: f64,
}

#[derive(Clone, Copy)]
pub(super) struct TensorInfo {
    dtype: Dtype,
    shape: [usize; 2],
    offset: usize,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Dtype {
    Bf16,
    F32,
}

// `shape` is one or two dimensions in the file; normalize to [rows, cols].
impl<'de> Deserialize<'de> for ShapeShim {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let dims = Vec::<usize>::deserialize(deserializer)?;
        match dims.as_slice() {
            [n] => Ok(Self([1, *n])),
            [rows, cols] => Ok(Self([*rows, *cols])),
            _ => Err(serde::de::Error::custom("tensor rank must be 1 or 2")),
        }
    }
}
struct ShapeShim([usize; 2]);

#[derive(Deserialize)]
struct RawTensor {
    dtype: Dtype,
    shape: ShapeShim,
    offset: usize,
    length: usize,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidFormat(format!("embedding artifact: {}", message.into()))
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(64), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

/// A verified, memory-mapped artifact.
pub(super) struct Artifact {
    map: Mmap,
    payload_start: usize,
    tensors: BTreeMap<String, TensorInfo>,
    pub(super) payload_sha256: String,
}

impl Artifact {
    pub(super) fn open(path: &Path) -> crate::Result<Self> {
        let (artifact, config) = Self::open_format(path, *MAGIC, FORMAT, VERSION)?;
        check_config(&serde_json::from_value(config)?)?;
        Ok(artifact)
    }

    /// Open and verify an artifact of any format in this container layout,
    /// returning its (unchecked) `config` header object.
    pub(super) fn open_format(
        path: &Path,
        magic: [u8; 8],
        format: &str,
        version: u32,
    ) -> crate::Result<(Self, serde_json::Value)> {
        let file = File::open(path)?;
        // SAFETY: the file is opened read-only and treated as immutable; the
        // installation is never rewritten in place (new installs are new files).
        #[allow(unsafe_code)]
        let map = unsafe { Mmap::map(&file)? };
        if map.len() < 16 || map[..8] != magic {
            return Err(invalid("bad magic"));
        }
        let header_length =
            u64::from_le_bytes(map[8..16].try_into().map_err(|_| invalid("short"))?);
        if header_length > MAX_HEADER || 16 + header_length > map.len() as u64 {
            return Err(invalid("bad header length"));
        }
        let payload_start = 16 + header_length as usize;
        let header: Header = serde_json::from_slice(&map[16..payload_start])?;
        if header.format != format || header.version != version {
            return Err(invalid(format!(
                "format {} v{} is not {format} v{version}",
                header.format, header.version
            )));
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
            let element = match tensor.dtype {
                Dtype::Bf16 => 2,
                Dtype::F32 => 4,
            };
            let expected = tensor.shape.0[0]
                .checked_mul(tensor.shape.0[1])
                .and_then(|n| n.checked_mul(element));
            if expected != Some(tensor.length)
                || tensor
                    .offset
                    .checked_add(tensor.length)
                    .is_none_or(|end| end > payload.len())
                || !tensor.offset.is_multiple_of(element)
            {
                return Err(invalid(format!(
                    "tensor {name} is out of bounds or misshapen"
                )));
            }
            tensors.insert(
                name,
                TensorInfo {
                    dtype: tensor.dtype,
                    shape: tensor.shape.0,
                    offset: tensor.offset,
                },
            );
        }
        Ok((
            Self {
                map,
                payload_start,
                tensors,
                payload_sha256: header.payload_sha256,
            },
            header.config,
        ))
    }

    pub(super) fn payload(&self) -> &[u8] {
        &self.map[self.payload_start..]
    }

    fn info(&self, name: &str, dtype: Dtype, shape: [usize; 2]) -> crate::Result<TensorInfo> {
        let info = *self
            .tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
        if info.dtype != dtype || info.shape != shape {
            return Err(invalid(format!(
                "tensor {name} has an unexpected dtype or shape"
            )));
        }
        Ok(info)
    }

    /// Payload offset of a BF16 matrix.
    pub(super) fn bf16(&self, name: &str, rows: usize, cols: usize) -> crate::Result<usize> {
        Ok(self.info(name, Dtype::Bf16, [rows, cols])?.offset)
    }

    /// Payload offset of an F32 vector or matrix.
    pub(super) fn f32_offset(&self, name: &str, count: usize) -> crate::Result<usize> {
        let info = *self
            .tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
        if info.dtype != Dtype::F32 || info.shape[0] * info.shape[1] != count {
            return Err(invalid(format!(
                "tensor {name} has an unexpected dtype or size"
            )));
        }
        Ok(info.offset)
    }

    /// An F32 tensor copied out of the map.
    pub(super) fn f32_values(&self, name: &str, count: usize) -> crate::Result<Vec<f32>> {
        let offset = self.f32_offset(name, count)?;
        Ok(self.payload()[offset..offset + 4 * count]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    /// First payload byte past the token table (read on the CPU; the
    /// tensors after it go to the GPU).
    pub(super) fn table_end(&self) -> crate::Result<usize> {
        Ok(self.bf16("embed_tokens", VOCAB, HIDDEN)? + VOCAB * HIDDEN * 2)
    }
}

fn check_config(config: &Config) -> crate::Result<()> {
    let sliding = LayerKind {
        head_dim: 256,
        num_key_value_heads: 2,
        rope_theta: 10_000.0,
    };
    let full = LayerKind {
        head_dim: 512,
        num_key_value_heads: 1,
        rope_theta: 1_000_000.0,
    };
    let types_ok = config.layer_types.len() == LAYERS
        && config.layer_types.iter().enumerate().all(|(i, kind)| {
            kind == if super::is_global(i) {
                "full_attention"
            } else {
                "sliding_attention"
            }
        });
    if config.hidden_size != HIDDEN
        || config.intermediate_size != INTERMEDIATE
        || config.num_layers != LAYERS
        || config.num_attention_heads != super::Q_HEADS
        || config.vocab_size != VOCAB
        || config.embedding_dim != EMBED_DIM
        || config.sliding_window != super::SLIDING_WINDOW as usize
        || config.sliding_attention != sliding
        || config.full_attention != full
        || !types_ok
    {
        return Err(invalid("architecture differs from EmbeddingGemma 2"));
    }
    Ok(())
}
