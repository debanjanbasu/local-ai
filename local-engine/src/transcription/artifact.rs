//! The F16 speech artifact written by `tools/convert_whisper.py`.
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

use super::tokens::SpecialTokens;
use super::{CTX_AUDIO, CTX_TEXT, FFN, HEADS, LAYERS, N_MELS, STATE, VOCAB};
use crate::Error;

const MAGIC: &[u8; 8] = b"WHSPRV3\x01";
const FORMAT: &str = "whisper-large-v3";
const VERSION: u32 = 1;
const MAX_HEADER: u64 = 1 << 20;

#[derive(Deserialize)]
struct Header {
    format: String,
    version: u32,
    config: Config,
    tokens: SpecialTokens,
    tensors: BTreeMap<String, RawTensor>,
    payload_length: u64,
    payload_sha256: String,
}

#[derive(Deserialize, PartialEq, Eq, Debug)]
#[allow(clippy::struct_field_names)]
struct Config {
    n_mels: usize,
    n_audio_ctx: usize,
    n_audio_state: usize,
    n_audio_head: usize,
    n_audio_layer: usize,
    n_vocab: usize,
    n_text_ctx: usize,
    n_text_state: usize,
    n_text_head: usize,
    n_text_layer: usize,
    n_ffn: usize,
    n_fft_bins: usize,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub(super) enum Dtype {
    F16,
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
    tensors: BTreeMap<String, TensorInfo>,
    pub(super) tokens: SpecialTokens,
    pub(super) payload_sha256: String,
}

impl Artifact {
    pub(super) fn open(path: &Path) -> crate::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the file is opened read-only and treated as immutable; the
        // installation is never rewritten in place (new installs are new files).
        #[allow(unsafe_code)]
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
        let expected = Config {
            n_mels: N_MELS,
            n_audio_ctx: CTX_AUDIO,
            n_audio_state: STATE,
            n_audio_head: HEADS,
            n_audio_layer: LAYERS,
            n_vocab: VOCAB,
            n_text_ctx: CTX_TEXT,
            n_text_state: STATE,
            n_text_head: HEADS,
            n_text_layer: LAYERS,
            n_ffn: FFN,
            n_fft_bins: super::mel::FFT_BINS,
        };
        if header.config != expected {
            return Err(invalid("architecture differs from Whisper large-v3"));
        }
        header.tokens.validate()?;
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
                Dtype::F16 => 2,
                Dtype::F32 => 4,
            };
            let shape = match tensor.shape.as_slice() {
                [n] => [1, *n],
                [rows, cols] => [*rows, *cols],
                _ => return Err(invalid(format!("tensor {name} has rank > 2"))),
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
                    "tensor {name} is out of bounds, misaligned or misshapen"
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
            tokens: header.tokens,
            payload_sha256: header.payload_sha256,
        })
    }

    pub(super) fn payload(&self) -> &[u8] {
        &self.map[self.payload_start..]
    }

    /// Payload offset of a tensor with this dtype and `[rows, cols]` shape
    /// (vectors are `[1, n]`).
    pub(super) fn offset(
        &self,
        name: &str,
        dtype: Dtype,
        rows: usize,
        cols: usize,
    ) -> crate::Result<usize> {
        let info = *self
            .tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
        if info.dtype != dtype || info.shape != [rows, cols] {
            return Err(invalid(format!(
                "tensor {name} is {:?} {:?}, expected {dtype:?} [{rows}, {cols}]",
                info.dtype, info.shape
            )));
        }
        Ok(info.offset)
    }

    /// An F32 tensor copied out of the map.
    pub(super) fn f32_values(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> crate::Result<Vec<f32>> {
        let offset = self.offset(name, Dtype::F32, rows, cols)?;
        Ok(self.payload()[offset..offset + 4 * rows * cols]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
}
