use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use local_metal::buffer::MetalBuffer;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::MtpSettings;
use super::Section;
use super::cache::{
    CACHE_SECTIONS, CacheMap, HeadCacheStatus, HeadKind, align_cache, find_cache, head_kind,
    read_cache, read_head, write_cache,
};
use super::layout::open_layout;
use super::{ATTENTION, FFN, HEAD_DIM, KV, QUERY_GATE, WIDTH, invalid};

/// The fused projection both fc sections are split from. The only tensor in the
/// head that two sections read, so those sections name the stream they multiply
/// rather than the tensor.
const FC_TENSOR: &str = "mtp.fc.weight";

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

/// Split the fused `[WIDTH, 2 * WIDTH]` projection into two contiguous halves so
/// each half multiplies its own normalized stream without concatenation.
pub fn split_fc(fused: &[u8]) -> crate::Result<(Vec<u8>, Vec<u8>)> {
    let row_bytes = 2 * WIDTH * 2;
    if fused.len() != WIDTH * row_bytes {
        return invalid("MTP fc projection has an unexpected size");
    }
    let mut embedding = Vec::with_capacity(WIDTH * WIDTH * 2);
    let mut hidden = Vec::with_capacity(WIDTH * WIDTH * 2);
    for row in fused.chunks_exact(row_bytes) {
        embedding.extend_from_slice(&row[..WIDTH * 2]);
        hidden.extend_from_slice(&row[WIDTH * 2..]);
    }
    Ok((embedding, hidden))
}

pub(super) struct MatrixWeight {
    pub(super) weights: MetalBuffer,
    pub(super) scales: MetalBuffer,
}

pub(super) fn quantize_rows(
    bf16: &[u8],
    rows: usize,
    columns: usize,
) -> crate::Result<(Vec<i8>, Vec<f32>)> {
    if bf16.len() != rows * columns * 2 {
        return invalid("MTP matrix has an unexpected size");
    }
    let mut quantized = Vec::with_capacity(rows * columns);
    let mut scales = Vec::with_capacity(rows);
    for row in bf16.chunks_exact(columns * 2) {
        let values = row
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| half::bf16::from_bits(u16::from_le_bytes(*pair)).to_f32());
        let absmax = values
            .clone()
            .try_fold(0.0_f32, |max, value| {
                value.is_finite().then_some(max.max(value.abs()))
            })
            .ok_or_else(|| crate::Error::InvalidFormat("non-finite MTP matrix weight".into()))?;
        let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
        scales.push(scale);
        quantized.extend(values.map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8));
    }
    Ok((quantized, scales))
}

/// One quantized matrix: the name its artifact sections carry, the safetensors
/// tensor they are quantized from (`None` when the section name already is that
/// tensor), and its pinned shape.
struct MatrixSpec {
    section: &'static str,
    tensor: Option<&'static str>,
    rows: usize,
    columns: usize,
}

/// Artifact order is `Weights` field order, so section `2 * i` is matrix `i`'s
/// int8 weights and `2 * i + 1` its f32 scales.
const MATRIX_SPECS: [MatrixSpec; 9] = [
    MatrixSpec {
        section: "mtp.fc.weight.embedding",
        tensor: Some(FC_TENSOR),
        rows: WIDTH,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.fc.weight.hidden",
        tensor: Some(FC_TENSOR),
        rows: WIDTH,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.layers.0.self_attn.q_proj.weight",
        tensor: None,
        rows: QUERY_GATE,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.layers.0.self_attn.k_proj.weight",
        tensor: None,
        rows: KV,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.layers.0.self_attn.v_proj.weight",
        tensor: None,
        rows: KV,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.layers.0.self_attn.o_proj.weight",
        tensor: None,
        rows: WIDTH,
        columns: ATTENTION,
    },
    MatrixSpec {
        section: "mtp.layers.0.mlp.gate_proj.weight",
        tensor: None,
        rows: FFN,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.layers.0.mlp.up_proj.weight",
        tensor: None,
        rows: FFN,
        columns: WIDTH,
    },
    MatrixSpec {
        section: "mtp.layers.0.mlp.down_proj.weight",
        tensor: None,
        rows: WIDTH,
        columns: FFN,
    },
];

/// One folded norm. Its element count is stated rather than inferred from the
/// name, because `q_norm` and `k_norm` are `HEAD_DIM` while every other norm is
/// `WIDTH` — and the section name has to be the tensor name, not a decoration.
struct NormSpec {
    section: &'static str,
    elements: usize,
}

const NORM_SPECS: [NormSpec; 7] = [
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

/// Section names and exact byte lengths, in artifact order.
///
/// Derived from the pinned specs rather than listed separately, so a head file's
/// name table can never drift from the transform that fills it, and so this
/// depends only on the pinned constants — never on a file.
pub(super) fn spec() -> Vec<Section> {
    let mut spec = Vec::with_capacity(CACHE_SECTIONS);
    for matrix in &MATRIX_SPECS {
        spec.push((
            format!("{}.i8", matrix.section),
            matrix.rows * matrix.columns,
        ));
        spec.push((
            format!("{}.scales", matrix.section),
            matrix.rows * size_of::<f32>(),
        ));
    }
    for norm in &NORM_SPECS {
        spec.push((norm.section.to_owned(), norm.elements * size_of::<f32>()));
    }
    spec
}

/// Quantize the BF16 head at `source` into artifact-order sections.
///
/// `Weights::load` and `local-ai bonsai --export-mtp-head` both call this, which
/// is what makes a shipped artifact byte-identical to what the loader would have
/// built for itself: same split, same per-row quantization, same section order.
pub(super) fn transform_sections(source: &Path) -> crate::Result<Vec<Vec<u8>>> {
    let (map, layout) = open_layout(source)?;
    let bytes = |name: &str| -> crate::Result<&[u8]> {
        let range = layout.range(name)?;
        Ok(&map[layout.data_start + range.start..layout.data_start + range.end])
    };
    let (fc_embedding, fc_hidden) = split_fc(bytes(FC_TENSOR)?)?;
    let mut sections = Vec::with_capacity(MATRIX_SPECS.len() * 2 + NORM_SPECS.len());
    for (index, matrix) in MATRIX_SPECS.iter().enumerate() {
        let data = match index {
            0 => fc_embedding.as_slice(),
            1 => fc_hidden.as_slice(),
            _ => bytes(matrix.tensor.unwrap_or(matrix.section))?,
        };
        let (weights, scales) = quantize_rows(data, matrix.rows, matrix.columns)?;
        sections.push(weights.into_iter().map(|value| value as u8).collect());
        sections.push(scales.into_iter().flat_map(f32::to_le_bytes).collect());
    }
    for norm in &NORM_SPECS {
        sections.push(
            fold_norm(bytes(norm.section)?)?
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        );
    }
    Ok(sections)
}

/// Read `path` when it holds the int8 artifact.
///
/// Either encoding will do: [`HeadKind`] says which one a file carries from its
/// own magic, and [`read_head`] inflates a compressed one into the same bytes a
/// stored one maps. Which of the two costs RAM is the exporter's decision to make
/// explicit — see [`write_zstd_artifact`](super::cache::write_zstd_artifact) — not
/// this loader's, since a head it cannot read has no second source to fall back
/// on.
///
/// An artifact that fails validation is a hard error, not a fallback: the BF16
/// source is not installed on this path, so there is nothing to rebuild from, and
/// decoding without speculation looks like a healthy install rather than a broken
/// one.
fn read_artifact(path: &Path, spec: &[Section]) -> crate::Result<Option<CacheMap>> {
    match head_kind(path)? {
        HeadKind::Safetensors => return Ok(None),
        HeadKind::Unrecognizable(bytes) => {
            return Err(crate::Error::InvalidFormat(format!(
                "MTP head {} is {bytes} bytes: too short to be an int8 artifact or a \
                 safetensors source",
                path.display()
            )));
        }
        HeadKind::Artifact | HeadKind::ZstdArtifact => {}
    }
    read_head(path, spec).map(Some).map_err(|reason| {
        crate::Error::InvalidFormat(format!(
            "MTP int8 head artifact {} is unusable: {reason}; re-export it with \
             `local-ai bonsai --export-mtp-head <dir>` or remove it to quantize from a \
             BF16 source",
            path.display()
        ))
    })
}

/// The int8 head a load ended up with: the validated file when there is one, the
/// freshly built sections when there is not, and how the cache went.
type Loaded = (Option<CacheMap>, Option<Vec<Vec<u8>>>, HeadCacheStatus);

/// The int8 sections for a BF16 `source`: the machine cache when it already holds
/// them, otherwise by quantizing the source and rewriting the cache.
///
/// This is where the 849 MB mapping happens, and it runs only after no artifact
/// answered, so a hit never maps the source at all.
fn load_sections(
    source: &Path,
    cache_dir: Option<&Path>,
    spec: &[Section],
) -> crate::Result<Loaded> {
    if let Some(cache) = cache_dir
        .and_then(find_cache)
        .and_then(|candidate| read_cache(&candidate, spec).ok())
    {
        return Ok((Some(cache), None, HeadCacheStatus::Hit));
    }
    let sections = transform_sections(source)?;
    // Writing then re-reading is what keeps a miss on the same zero-copy upload
    // path as a hit: the anonymous transform is dropped in favour of the file.
    let written = cache_dir.and_then(|directory| {
        let head = write_cache(directory, spec, &sections).ok()?;
        read_cache(&head.path, spec).ok()
    });
    Ok(match written {
        Some(cache) => (Some(cache), None, HeadCacheStatus::Miss),
        None if cache_dir.is_some() => (None, Some(sections), HeadCacheStatus::Miss),
        None => (None, Some(sections), HeadCacheStatus::Disabled),
    })
}

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
    #[allow(clippy::too_many_lines)]
    pub(super) fn load(
        device: &ProtocolObject<dyn MTLDevice>,
        settings: &MtpSettings,
    ) -> crate::Result<(Self, u64, HeadCacheStatus)> {
        let spec = spec();
        debug_assert_eq!(spec.len(), CACHE_SECTIONS);
        let (cache, sections, status) = match read_artifact(&settings.path, &spec)? {
            Some(cache) => (Some(cache), None, HeadCacheStatus::Hit),
            None => load_sections(&settings.path, settings.head_cache_dir.as_deref(), &spec)?,
        };
        let buffer = |index: usize| -> crate::Result<MetalBuffer> {
            if let Some(cache) = &cache {
                let (offset, length) = cache.sections[index];
                let pointer = NonNull::new(cache.map[offset..].as_ptr().cast_mut())
                    .ok_or_else(|| crate::Error::InvalidFormat("empty MTP cache mapping".into()))?;
                let owner: Arc<dyn Send + Sync> = cache.map.clone();
                // SAFETY: sections begin on VM pages, their padded ranges are
                // validated within the immutable mapping, and `owner` retains it.
                #[allow(unsafe_code)]
                return unsafe {
                    Ok(MetalBuffer::from_bytes_no_copy(
                        device,
                        pointer,
                        align_cache(length).ok_or_else(|| {
                            crate::Error::InvalidFormat("MTP cache size overflow".into())
                        })?,
                        owner,
                    )?)
                };
            }
            let data = sections.as_ref().ok_or_else(|| {
                crate::Error::InvalidFormat("MTP cache has no backing data".into())
            })?;
            Ok(MetalBuffer::from_slice(device, &data[index])?)
        };
        let matrix = |index: usize| -> crate::Result<MatrixWeight> {
            Ok(MatrixWeight {
                weights: buffer(index * 2)?,
                scales: buffer(index * 2 + 1)?,
            })
        };
        let norm = |index: usize| buffer(MATRIX_SPECS.len() * 2 + index);
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
        let matrix_elements = 2 * WIDTH * WIDTH
            + QUERY_GATE * WIDTH
            + 2 * KV * WIDTH
            + WIDTH * ATTENTION
            + 2 * FFN * WIDTH
            + WIDTH * FFN;
        let matrix_rows = 2 * WIDTH + QUERY_GATE + 2 * KV + WIDTH + 2 * FFN + WIDTH;
        let norm_bytes = (5 * WIDTH + 2 * HEAD_DIM) * size_of::<f32>();
        let total = (matrix_elements + matrix_rows * size_of::<f32>() + norm_bytes) as u64;
        Ok((weights, total, status))
    }
}
