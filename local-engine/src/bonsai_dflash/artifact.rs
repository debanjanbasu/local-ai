//! The drafter artifact: what it holds, and exporting it from a `DFlash2`
//! safetensors checkpoint.
//!
//! `local-ai bonsai --export dflash-head=DIR` reads `DIR/config.json` and
//! `DIR/model.safetensors` (any `DFlash2DraftModel` with the pinned shapes,
//! in BF16, F16 or F32: the published drafter or a fine-tuned one) and writes
//! `DIR/dflash2-drafter-int8-v1.bin`: no GPU and no target checkpoint needed.
//!
//! Every dense matrix becomes per-row int8 (`row_scale = absmax / 127`,
//! round to nearest), multiplied by the int8 kernels; norms and the
//! convolutions' base kernels are F32; the selector's two codebooks, which
//! are only gathered (16 rows per position), stay BF16. The file is the MTP
//! head's container (named page-aligned sections and a payload digest every
//! load re-derives) under its own magic.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::MmapOptions;
use serde::Deserialize;

use super::{
    CONV_PROJECTION, FFN, HEAD_DIM, KV_WIDTH, LAYERS, MASK_TOKEN, Q_WIDTH, RANK, TAPS,
    TARGET_LAYER_IDS, VOCAB, WIDTH, invalid,
};
use crate::bonsai_mtp::{Container, PAGE, write_container};

/// The distributable drafter, installed in `models/bonsai2-27b-dflash2/`.
/// The format and version are in the name, so a future format is a
/// different file rather than a misparse of this one.
pub const DFLASH_HEAD_ARTIFACT: &str = "dflash2-drafter-int8-v1.bin";

/// The source the exporter reads from its directory.
const SOURCE_WEIGHTS: &str = "model.safetensors";
const SOURCE_CONFIG: &str = "config.json";

pub(super) const DFLASH_CONTAINER: Container = Container {
    magic: b"DFLSH2\0\x01",
    version: 1,
    // 128 named sections outgrow a 4 KiB page; the first section starts at
    // `PAGE` anyway.
    header_bytes: PAGE,
    name: "DFlash2 drafter",
    a_name: "a DFlash2 drafter",
    temp_prefix: ".dflash2-drafter",
};

const SAFETENSORS_MAX_HEADER: u64 = 1 << 24;

/// How one source tensor is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// `[rows, columns]` per-row int8: sections `<name>.int8` (row-major
    /// int8 in [-127, 127]) and `<name>.row_scales` (F32 per row).
    Int8 { rows: usize, columns: usize },
    /// `elements` F32 values: section `<name>.f32`.
    F32 { elements: usize },
    /// `[rows, columns]` BF16 bits: section `<name>.bf16`.
    Bf16 { rows: usize, columns: usize },
}

/// One source tensor: its checkpoint name and how it is stored.
pub(super) struct Tensor {
    pub(super) name: String,
    pub(super) kind: Kind,
    /// The shape it has in the checkpoint.
    pub(super) shape: Vec<usize>,
}

pub(super) const PREDECESSOR: &str = "candidate_selector.predecessor_codebook";
pub(super) const SUCCESSOR: &str = "candidate_selector.successor_codebook";

/// Every tensor of a `DFlash2` drafter, in artifact order.
pub(super) fn tensors() -> Vec<Tensor> {
    let int8 = |name: String, rows, columns| Tensor {
        name,
        kind: Kind::Int8 { rows, columns },
        shape: vec![rows, columns],
    };
    let f32 = |name: String, shape: Vec<usize>| Tensor {
        name,
        kind: Kind::F32 {
            elements: shape.iter().product(),
        },
        shape,
    };
    let mut all = vec![
        int8("fc.weight".into(), WIDTH, TAPS * WIDTH),
        f32("hidden_norm.weight".into(), vec![WIDTH]),
        f32("norm.weight".into(), vec![WIDTH]),
        int8(
            "candidate_selector.hidden_projection.weight".into(),
            RANK,
            WIDTH,
        ),
        Tensor {
            name: PREDECESSOR.into(),
            kind: Kind::Bf16 {
                rows: VOCAB,
                columns: RANK,
            },
            shape: vec![VOCAB, RANK],
        },
        Tensor {
            name: SUCCESSOR.into(),
            kind: Kind::Bf16 {
                rows: VOCAB,
                columns: RANK,
            },
            shape: vec![VOCAB, RANK],
        },
    ];
    for layer in 0..LAYERS {
        let name = |suffix: &str| format!("layers.{layer}.{suffix}");
        all.extend([
            f32(name("input_layernorm.weight"), vec![WIDTH]),
            f32(name("post_attention_layernorm.weight"), vec![WIDTH]),
            f32(name("self_attn.q_norm.weight"), vec![HEAD_DIM]),
            f32(name("self_attn.k_norm.weight"), vec![HEAD_DIM]),
            f32(name("attention_conv.base_kernel"), vec![2, 2, WIDTH]),
            f32(name("mlp_conv.base_kernel"), vec![2, 2, WIDTH]),
            int8(
                name("attention_conv.kernel_projection.weight"),
                CONV_PROJECTION,
                WIDTH,
            ),
            int8(name("self_attn.q_proj.weight"), Q_WIDTH, WIDTH),
            int8(name("self_attn.k_proj.weight"), KV_WIDTH, WIDTH),
            int8(name("self_attn.v_proj.weight"), KV_WIDTH, WIDTH),
            int8(name("self_attn.o_proj.weight"), WIDTH, Q_WIDTH),
            int8(
                name("mlp_conv.kernel_projection.weight"),
                CONV_PROJECTION,
                WIDTH,
            ),
            int8(name("mlp.gate_proj.weight"), FFN, WIDTH),
            int8(name("mlp.up_proj.weight"), FFN, WIDTH),
            int8(name("mlp.down_proj.weight"), WIDTH, FFN),
        ]);
    }
    all
}

/// The `(section name, bytes)` list every artifact must hold, in order.
pub(super) fn spec() -> Vec<(String, usize)> {
    let mut sections = Vec::new();
    for tensor in tensors() {
        match tensor.kind {
            Kind::Int8 { rows, columns } => {
                sections.push((format!("{}.int8", tensor.name), rows * columns));
                sections.push((format!("{}.row_scales", tensor.name), rows * 4));
            }
            Kind::F32 { elements } => sections.push((format!("{}.f32", tensor.name), elements * 4)),
            Kind::Bf16 { rows, columns } => {
                sections.push((format!("{}.bf16", tensor.name), rows * columns * 2));
            }
        }
    }
    sections
}

/// What one exported drafter contains.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct DflashHeadArtifact {
    /// The artifact that was written.
    pub path: PathBuf,
    /// The safetensors it was converted from.
    pub source: PathBuf,
    /// Sections in artifact order; the header names every one of them.
    pub sections: usize,
    /// Section bytes with the padding skipped: what a load hashes, and the
    /// artifact's content identity.
    pub payload_bytes: u64,
    /// Bytes on disk, header page and padding included.
    pub file_bytes: u64,
    pub payload_sha256: String,
}

/// The `config.json` fields this build's drafter pins. A checkpoint whose
/// config differs anywhere here is a different architecture.
#[derive(Deserialize)]
struct Config {
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    sliding_window: usize,
    rope_parameters: RopeConfig,
    #[serde(rename = "dflash_config")]
    dflash: DflashConfig,
}

#[derive(Deserialize)]
struct RopeConfig {
    rope_theta: f64,
    rope_type: String,
}

#[derive(Deserialize)]
struct DflashConfig {
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    mask_token_id: u32,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
}

fn check_config(path: &Path) -> crate::Result<()> {
    let config: Config = serde_json::from_slice(&std::fs::read(path)?).map_err(|error| {
        crate::Error::InvalidFormat(format!(
            "{} is not a DFlash2 config: {error}",
            path.display()
        ))
    })?;
    let d = &config.dflash;
    let checks = [
        (
            config.architectures == ["DFlash2DraftModel"],
            "architectures must be [DFlash2DraftModel]",
        ),
        (config.hidden_size == WIDTH, "hidden_size must be 5120"),
        (
            config.intermediate_size == FFN,
            "intermediate_size must be 17408",
        ),
        (
            config.num_hidden_layers == LAYERS,
            "num_hidden_layers must be 5",
        ),
        (
            config.num_attention_heads == 32,
            "num_attention_heads must be 32",
        ),
        (
            config.num_key_value_heads == 8,
            "num_key_value_heads must be 8",
        ),
        (config.head_dim == HEAD_DIM, "head_dim must be 128"),
        (config.vocab_size == VOCAB, "vocab_size must be 248320"),
        (
            (config.rms_norm_eps - 1e-6).abs() < 1e-12,
            "rms_norm_eps must be 1e-6",
        ),
        (config.sliding_window == 2048, "sliding_window must be 2048"),
        (
            (config.rope_parameters.rope_theta - 1e7).abs() < 1e-3
                && config.rope_parameters.rope_type == "default",
            "rope must be default with theta 1e7",
        ),
        (d.block_size == 8, "dflash block_size must be 8"),
        (d.conv_group_size == 16, "dflash conv_group_size must be 16"),
        (d.conv_kernel_size == 2, "dflash conv_kernel_size must be 2"),
        (
            d.mask_token_id == MASK_TOKEN,
            "dflash mask_token_id must be 248070",
        ),
        (d.selector_rank == RANK, "dflash selector_rank must be 256"),
        (d.selector_top_k == 16, "dflash selector_top_k must be 16"),
        (
            d.target_layer_ids == TARGET_LAYER_IDS,
            "dflash target_layer_ids must be [5, 19, 33, 47, 61]",
        ),
    ];
    for (ok, requirement) in checks {
        if !ok {
            return invalid(format!("{}: {requirement}", path.display()));
        }
    }
    Ok(())
}

/// Source tensor values as F32, whatever the checkpoint's float type.
fn to_f32(dtype: &str, bytes: &[u8], output: &mut [f32]) {
    match dtype {
        "BF16" => {
            for (pair, value) in bytes.as_chunks::<2>().0.iter().zip(output) {
                *value = half::bf16::from_le_bytes(*pair).to_f32();
            }
        }
        "F16" => {
            for (pair, value) in bytes.as_chunks::<2>().0.iter().zip(output) {
                *value = half::f16::from_le_bytes(*pair).to_f32();
            }
        }
        _ => {
            for (quad, value) in bytes.as_chunks::<4>().0.iter().zip(output) {
                *value = f32::from_le_bytes(*quad);
            }
        }
    }
}

/// Per-row int8 of a `[rows, columns]` source matrix: `row_scale = absmax /
/// 127` (1 for an all-zero row), `q = round(x / row_scale)`. Rows run on
/// every core.
fn quantize_int8(
    name: &str,
    dtype: &str,
    bytes: &[u8],
    rows: usize,
    columns: usize,
) -> crate::Result<(Vec<u8>, Vec<u8>)> {
    let element = bytes.len() / (rows * columns);
    let mut weights = vec![0_u8; rows * columns];
    let mut scales = vec![0_u8; rows * 4];
    let threads = std::thread::available_parallelism().map_or(8, std::num::NonZero::get);
    let per = rows.div_ceil(threads);
    let finite = std::thread::scope(|scope| {
        let workers: Vec<_> = weights
            .chunks_mut(per * columns)
            .zip(scales.chunks_mut(per * 4))
            .zip(bytes.chunks(per * columns * element))
            .map(|((weights, scales), bytes)| {
                scope.spawn(move || {
                    let mut row = vec![0.0_f32; columns];
                    for ((weights, scale), bytes) in weights
                        .chunks_mut(columns)
                        .zip(scales.chunks_mut(4))
                        .zip(bytes.chunks(columns * element))
                    {
                        to_f32(dtype, bytes, &mut row);
                        if row.iter().any(|value| !value.is_finite()) {
                            return false;
                        }
                        let peak = row.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
                        let step = if peak > 0.0 { peak / 127.0 } else { 1.0 };
                        for (out, value) in weights.iter_mut().zip(&row) {
                            *out = (value / step).round().clamp(-127.0, 127.0) as i8 as u8;
                        }
                        scale.copy_from_slice(&step.to_le_bytes());
                    }
                    true
                })
            })
            .collect();
        workers
            .into_iter()
            .all(|worker| worker.join().unwrap_or(false))
    });
    if !finite {
        return invalid(format!("DFlash2 tensor {name} holds non-finite values"));
    }
    Ok((weights, scales))
}

/// Convert the `DFlash2` drafter at `<directory>/model.safetensors` (checked
/// against `<directory>/config.json`) into `<directory>/dflash2-drafter-int8-v1.bin`.
///
/// The checkpoint must hold exactly the drafter's 81 tensors (the selector
/// codebooks with or without a `.weight` suffix) in BF16, F16 or F32. Install
/// the result in `models/bonsai2-27b-dflash2/`.
#[expect(clippy::too_many_lines)]
pub fn export_dflash_head(directory: &Path) -> crate::Result<DflashHeadArtifact> {
    #[derive(Deserialize)]
    struct Entry {
        dtype: String,
        shape: Vec<usize>,
        data_offsets: [usize; 2],
    }
    check_config(&directory.join(SOURCE_CONFIG))?;
    let source = directory.join(SOURCE_WEIGHTS);
    let file = File::open(&source)?;
    // SAFETY: read-only mapping of an input file this process never writes.
    #[expect(unsafe_code)]
    let map = unsafe { MmapOptions::new().map(&file)? };
    let header_len = map
        .first_chunk::<8>()
        .map(|bytes| u64::from_le_bytes(*bytes))
        .ok_or_else(|| crate::Error::InvalidFormat("DFlash2 source is too short".into()))?;
    if header_len == 0 || header_len > SAFETENSORS_MAX_HEADER || header_len + 8 > map.len() as u64 {
        return invalid("DFlash2 source safetensors header length is invalid");
    }
    let data_start = 8 + header_len as usize;
    let data = &map[data_start..];
    let mut entries: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&map[8..data_start])?;
    entries.remove("__metadata__");
    let expected = tensors();
    if entries.len() != expected.len() {
        return invalid(format!(
            "DFlash2 source must contain exactly {} tensors, found {}",
            expected.len(),
            entries.len()
        ));
    }
    let mut sections = Vec::with_capacity(spec().len());
    for tensor in &expected {
        let value = entries
            .remove(&tensor.name)
            .or_else(|| entries.remove(&format!("{}.weight", tensor.name)))
            .ok_or_else(|| crate::Error::MissingTensor(tensor.name.clone()))?;
        let entry: Entry = serde_json::from_value(value)?;
        let element = match entry.dtype.as_str() {
            "BF16" | "F16" => 2,
            "F32" => 4,
            other => {
                return invalid(format!(
                    "DFlash2 tensor {} is {other}, expected BF16, F16 or F32",
                    tensor.name
                ));
            }
        };
        if entry.shape != tensor.shape {
            return invalid(format!(
                "DFlash2 tensor {} must be {:?}, found {:?}",
                tensor.name, tensor.shape, entry.shape
            ));
        }
        let [start, end] = entry.data_offsets;
        if end < start
            || end - start != tensor.shape.iter().product::<usize>() * element
            || end > data.len()
        {
            return invalid(format!(
                "DFlash2 tensor {} has an invalid byte range",
                tensor.name
            ));
        }
        let bytes = &data[start..end];
        match tensor.kind {
            Kind::Int8 { rows, columns } => {
                let (weights, scales) =
                    quantize_int8(&tensor.name, &entry.dtype, bytes, rows, columns)?;
                sections.push(weights);
                sections.push(scales);
            }
            Kind::F32 { elements } => {
                let mut values = vec![0.0_f32; elements];
                to_f32(&entry.dtype, bytes, &mut values);
                if values.iter().any(|value| !value.is_finite()) {
                    return invalid(format!("DFlash2 tensor {} is not finite", tensor.name));
                }
                sections.push(values.iter().flat_map(|v| v.to_le_bytes()).collect());
            }
            Kind::Bf16 { .. } => {
                let section = if entry.dtype == "BF16" {
                    bytes.to_vec()
                } else {
                    let mut values = vec![0.0_f32; bytes.len() / element];
                    to_f32(&entry.dtype, bytes, &mut values);
                    values
                        .iter()
                        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                        .collect()
                };
                if section
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .any(|pair| !half::bf16::from_le_bytes(*pair).is_finite())
                {
                    return invalid(format!("DFlash2 tensor {} is not finite", tensor.name));
                }
                sections.push(section);
            }
        }
    }
    let count = sections.len();
    let written = write_container(
        &directory.join(DFLASH_HEAD_ARTIFACT),
        &DFLASH_CONTAINER,
        &spec(),
        &sections,
    )?;
    let widen = |bytes: usize| {
        u64::try_from(bytes)
            .map_err(|_| crate::Error::InvalidFormat("DFlash2 artifact exceeds 64 bits".into()))
    };
    Ok(DflashHeadArtifact {
        path: written.path,
        source,
        sections: count,
        payload_bytes: widen(written.payload_bytes)?,
        file_bytes: widen(written.file_bytes)?,
        payload_sha256: written.payload_sha256,
    })
}
