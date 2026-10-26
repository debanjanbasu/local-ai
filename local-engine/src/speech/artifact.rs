//! The Kokoro-82M artifact written by `tools/convert_kokoro.py`.
//!
//! Layout: 8-byte magic, `u64` header length, a JSON header, then the payload
//! (offsets are relative to it). Opening checks the magic, format, version,
//! the architecture constants, every tensor's, voice's and blob's bounds,
//! and the payload's SHA-256 against the header, so a truncated, foreign or
//! edited file is refused rather than misread.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde::Deserialize;
use sha2::{Digest as _, Sha256};

use crate::Error;

const MAGIC: &[u8; 8] = b"KOKORO\x00\x01";
const FORMAT: &str = "kokoro-82m";
const VERSION: u32 = 1;
const MAX_HEADER: u64 = 1 << 22;
/// Rows of a voice pack (one style vector per phoneme count) and its width.
pub(super) const VOICE_ROWS: usize = 510;
pub(super) const STYLE_WIDTH: usize = 256;

#[derive(Deserialize)]
struct Header {
    format: String,
    version: u32,
    config: Config,
    tensors: BTreeMap<String, RawTensor>,
    voices: BTreeMap<String, usize>,
    blobs: BTreeMap<String, RawBlob>,
    payload_length: u64,
    payload_sha256: String,
}

#[derive(Deserialize)]
pub(super) struct Config {
    pub(super) vocab: BTreeMap<String, u32>,
    pub(super) n_token: usize,
    pub(super) hidden_dim: usize,
    pub(super) style_dim: usize,
    pub(super) n_layer: usize,
    pub(super) max_dur: usize,
    pub(super) text_encoder_kernel_size: usize,
    pub(super) plbert: PlBert,
    pub(super) istftnet: Istftnet,
}

#[derive(Deserialize)]
pub(super) struct PlBert {
    pub(super) hidden_size: usize,
    pub(super) num_attention_heads: usize,
    pub(super) intermediate_size: usize,
    pub(super) max_position_embeddings: usize,
    pub(super) num_hidden_layers: usize,
}

#[derive(Deserialize)]
pub(super) struct Istftnet {
    pub(super) upsample_kernel_sizes: Vec<usize>,
    pub(super) upsample_rates: Vec<usize>,
    pub(super) gen_istft_hop_size: usize,
    pub(super) gen_istft_n_fft: usize,
    pub(super) resblock_dilation_sizes: Vec<Vec<usize>>,
    pub(super) resblock_kernel_sizes: Vec<usize>,
    pub(super) upsample_initial_channel: usize,
}

#[derive(Deserialize)]
struct RawTensor {
    shape: Vec<usize>,
    offset: usize,
}

#[derive(Deserialize)]
struct RawBlob {
    offset: usize,
    length: usize,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidFormat(format!("speech artifact: {}", message.into()))
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
    pub(super) config: Config,
    tensors: BTreeMap<String, (Vec<usize>, usize)>,
    voices: BTreeMap<String, usize>,
    blobs: BTreeMap<String, (usize, usize)>,
    pub(super) payload_sha256: String,
}

impl Artifact {
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
        check_config(&header.config)?;
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
        let within = |offset: usize, bytes: Option<usize>| {
            bytes
                .and_then(|bytes| offset.checked_add(bytes))
                .is_some_and(|end| end <= payload.len())
        };
        let mut tensors = BTreeMap::new();
        for (name, tensor) in header.tensors {
            let count = tensor
                .shape
                .iter()
                .try_fold(1_usize, |n, d| n.checked_mul(*d));
            if !within(tensor.offset, count.and_then(|n| n.checked_mul(4)))
                || !tensor.offset.is_multiple_of(4)
            {
                return Err(invalid(format!("tensor {name} is out of bounds")));
            }
            tensors.insert(name, (tensor.shape, tensor.offset));
        }
        for (name, offset) in &header.voices {
            if !within(*offset, Some(VOICE_ROWS * STYLE_WIDTH * 4)) || !offset.is_multiple_of(4) {
                return Err(invalid(format!("voice {name} is out of bounds")));
            }
        }
        let mut blobs = BTreeMap::new();
        for (name, blob) in header.blobs {
            if !within(blob.offset, Some(blob.length)) {
                return Err(invalid(format!("blob {name} is out of bounds")));
            }
            blobs.insert(name, (blob.offset, blob.length));
        }
        Ok(Self {
            map,
            payload_start,
            config: header.config,
            tensors,
            voices: header.voices,
            blobs,
            payload_sha256: header.payload_sha256,
        })
    }

    pub(super) fn payload(&self) -> &[u8] {
        &self.map[self.payload_start..]
    }

    /// Payload offset of a tensor with exactly `shape`.
    pub(super) fn offset(&self, name: &str, shape: &[usize]) -> crate::Result<usize> {
        let (actual, offset) = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
        if actual.as_slice() != shape {
            return Err(invalid(format!(
                "tensor {name} has shape {actual:?}, expected {shape:?}"
            )));
        }
        Ok(*offset)
    }

    /// A tensor's values copied out of the map.
    pub(super) fn values(&self, name: &str, shape: &[usize]) -> crate::Result<Vec<f32>> {
        let offset = self.offset(name, shape)?;
        let count: usize = shape.iter().product();
        Ok(f32s(&self.payload()[offset..offset + 4 * count]))
    }

    /// Tensor names, sorted.
    pub(super) fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// First dimension of a tensor.
    pub(super) fn rows(&self, name: &str) -> crate::Result<usize> {
        self.tensors
            .get(name)
            .and_then(|(shape, _)| shape.first().copied())
            .ok_or_else(|| Error::MissingTensor(name.to_owned()))
    }

    /// Voice names, sorted.
    pub(super) fn voice_names(&self) -> impl Iterator<Item = &str> {
        self.voices.keys().map(String::as_str)
    }

    /// One voice pack, `[510][256]`.
    pub(super) fn voice(&self, name: &str) -> Option<Vec<f32>> {
        let offset = *self.voices.get(name)?;
        Some(f32s(
            &self.payload()[offset..offset + VOICE_ROWS * STYLE_WIDTH * 4],
        ))
    }

    /// A blob's bytes.
    pub(super) fn blob(&self, name: &str) -> crate::Result<&[u8]> {
        let (offset, length) = self
            .blobs
            .get(name)
            .ok_or_else(|| invalid(format!("missing blob {name}")))?;
        Ok(&self.payload()[*offset..*offset + *length])
    }
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn check_config(config: &Config) -> crate::Result<()> {
    let bert = &config.plbert;
    let net = &config.istftnet;
    let ok = config.n_token == 178
        && config.hidden_dim == 512
        && config.style_dim == 128
        && config.n_layer == 3
        && config.max_dur == 50
        && config.text_encoder_kernel_size == 5
        && bert.hidden_size == 768
        && bert.num_attention_heads == 12
        && bert.intermediate_size == 2048
        && bert.max_position_embeddings == 512
        && bert.num_hidden_layers == 12
        && net.upsample_kernel_sizes == [20, 12]
        && net.upsample_rates == [10, 6]
        && net.gen_istft_hop_size == 5
        && net.gen_istft_n_fft == 20
        && net.resblock_kernel_sizes == [3, 7, 11]
        && net.resblock_dilation_sizes.iter().all(|d| d == &[1, 3, 5])
        && net.resblock_dilation_sizes.len() == 3
        && net.upsample_initial_channel == 512
        && config
            .vocab
            .values()
            .all(|id| (*id as usize) < config.n_token);
    if ok {
        Ok(())
    } else {
        Err(invalid("architecture differs from Kokoro-82M v1.0"))
    }
}
