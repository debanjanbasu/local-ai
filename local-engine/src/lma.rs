//! `.lma` model archive: a self-describing, zstd-compressed container for the
//! Gemma4 QAT tensors.
//!
//! Each GGUF tensor is stored verbatim (raw quantized bytes) in its own frame,
//! addressed by a small JSON index in the archive's metadata frame. The index
//! also embeds the model `config.json` so the archive is fully self-contained.
//!
//! [`compress_gguf_to_lma`] produces an archive from a loaded [`GGUFModel`];
//! [`load_lma`] reverses it into the same `(config, weights)` the GGUF path
//! yields, so `.lma` and GGUF loading are interchangeable.

use std::collections::HashMap;
use std::path::Path;

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use serde::{Deserialize, Serialize};

use local_core::config::{Gemma4QATConfig, ModelConfig};
use local_metal::buffer::MetalBuffer;

use crate::archive::output::{ArchiveOutputPlan, OutputPolicy};
use crate::archive::{ArchiveWriter, CompressedArchive, FrameKind};
use crate::gguf::{GGUFModel, GGUFType};
use crate::layer::QuantWeight;
use crate::multimodal::{
    MultimodalSupport, tensor_names_indicate_audio_support, tensor_names_indicate_image_support,
};
use crate::pipeline::init_gemma4_qat::{
    self, PLE_TABLE_NAME, QuantRowTable, WeightMap, decode_tensor_to_f16,
};

/// Format tag stored in the index so future readers can detect the layout.
const FORMAT_TAG: &str = "lma-tensors-v1";

/// One tensor's placement in the archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRecord {
    pub name: String,
    /// GGUF tensor-type discriminant (see [`GGUFType::from_u32`]).
    pub ggml_type: u32,
    pub shape: Vec<u64>,
    /// Frame id within the archive (`FrameKind::Layer`).
    pub frame_id: u32,
    /// Additional frame ids for tensors split into multiple chunks (the
    /// payload is the concatenation of `frame_id` then these, in order).
    /// Splitting huge tensors lets the loader decompress them in parallel —
    /// a single zstd stream can only ever decode on one core.
    #[serde(default)]
    pub extra_frames: Vec<u32>,
}

/// Tensors larger than this are split into multiple frames at write time.
const CHUNK_BYTES: usize = 64 * 1024 * 1024;

/// JSON index stored in the archive's metadata frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmaIndex {
    pub format: String,
    /// Verbatim `config.json` contents, so the archive is self-describing.
    pub config_json: String,
    pub tensors: Vec<TensorRecord>,
    /// Bundled Gemma 4 multimodal companion/projector tensors (for example
    /// Unsloth's `mmproj-BF16.gguf`). These are preserved in the archive,
    /// used for capability detection, and uploaded into device-resident
    /// modality buffers at model load.
    #[serde(default)]
    pub companion_tensors: Vec<TensorRecord>,
    /// Legacy field retained for backward-compatible deserialization of older
    /// archives that bundled an MTP drafter. Always written empty now and
    /// ignored at load; speculative decoding has been removed.
    #[serde(default)]
    pub assistant_tensors: Vec<TensorRecord>,
    /// Frame id of the bundled `tokenizer.json` (0 = not bundled).
    #[serde(default)]
    pub tokenizer_frame: u32,
}

/// Borrow one tensor's raw quantized payload from a GGUF source.
fn payload_of<'g>(name: &str, source: &'g GGUFModel) -> crate::Result<&'g [u8]> {
    source
        .tensor_data::<u8>(name)
        .ok_or_else(|| crate::Error::InvalidFormat(format!("tensor {name} has no data")))
}

/// Compress a loaded GGUF model into a single-file `.lma` archive at `out_path`.
///
/// Optionally bundles the multimodal companion GGUF and `tokenizer.json` so the
/// archive is the **only** file the engine needs.
///
/// `config_json` is the verbatim model `config.json`, embedded so the archive
/// is self-describing. After writing, every frame is read back and
/// byte-compared against its source, so the originals are safe to delete once
/// this returns. Writes atomically via a `.partial` file that is published on
/// success.
///
/// # Errors
///
/// Returns an error if a tensor payload is missing, verification finds a
/// mismatch, or on I/O / compression failure.
#[allow(clippy::too_many_lines)]
pub fn compress_gguf_to_lma(
    gguf: &GGUFModel,
    config_json: &str,
    companion: Option<&GGUFModel>,
    tokenizer_json: Option<&[u8]>,
    out_path: &Path,
    zstd_level: i32,
) -> crate::Result<()> {
    let plan = ArchiveOutputPlan::new(out_path, OutputPolicy::default());
    plan.prepare()?;

    let mut writer = ArchiveWriter::new(plan.partial_path(), zstd_level)?;
    let mut next_frame = 1u32;
    let write_all = |source: &GGUFModel,
                     writer: &mut ArchiveWriter<_>,
                     next_frame: &mut u32,
                     skip_aux: bool|
     -> crate::Result<Vec<TensorRecord>> {
        let mut records = Vec::new();
        let mut batch: Vec<PendingTensor<'_>> = Vec::new();
        let mut batch_bytes = 0usize;
        for (name, info) in source.tensors.iter().filter(|(n, _)| {
            !skip_aux || (!n.starts_with("masked_embd") && !n.starts_with("rope_freqs"))
        }) {
            let payload = payload_of(name, source)?;
            batch_bytes += payload.len();
            batch.push(PendingTensor {
                name: name.clone(),
                ggml_type: info.ty as u32,
                shape: info.shape.clone(),
                payload: std::borrow::Cow::Borrowed(payload),
            });
            if batch_bytes >= BATCH_BYTES {
                write_tensor_batch(
                    writer,
                    next_frame,
                    zstd_level,
                    CHUNK_BYTES,
                    &batch,
                    &mut records,
                )?;
                batch.clear();
                batch_bytes = 0;
            }
        }
        if !batch.is_empty() {
            write_tensor_batch(
                writer,
                next_frame,
                zstd_level,
                CHUNK_BYTES,
                &batch,
                &mut records,
            )?;
        }
        Ok(records)
    };
    let tensors = write_all(gguf, &mut writer, &mut next_frame, false)?;
    let companion_tensors = match companion {
        Some(c) => write_all(c, &mut writer, &mut next_frame, false)?,
        None => Vec::new(),
    };
    let tokenizer_frame = match tokenizer_json {
        Some(tok) => {
            let id = next_frame;
            writer.write_frame(FrameKind::Layer, id, tok)?;
            id
        }
        None => 0,
    };

    let index = LmaIndex {
        format: FORMAT_TAG.to_owned(),
        config_json: config_json.to_owned(),
        tensors,
        companion_tensors,
        assistant_tensors: Vec::new(),
        tokenizer_frame,
    };
    let index_bytes = serde_json::to_vec(&index)
        .map_err(|e| crate::Error::InvalidFormat(format!("encode lma index: {e}")))?;
    writer.write_frame(FrameKind::Metadata, 0, &index_bytes)?;
    writer.finish()?;

    // Verify every payload against its source before publishing — the archive
    // is meant to replace the originals.
    let mut archive = CompressedArchive::open(plan.partial_path())?;
    for (record, source) in
        index
            .tensors
            .iter()
            .map(|r| (r, gguf))
            .chain(index.companion_tensors.iter().map(|r| {
                (r, companion.unwrap_or(gguf)) // companion records imply Some
            }))
    {
        let want = payload_of(&record.name, source)?;
        let got = assemble_payload(&mut archive, record)?;
        if got != want {
            return Err(crate::Error::InvalidFormat(format!(
                "lma verification failed for tensor {}",
                record.name
            )));
        }
    }
    if let Some(tok) = tokenizer_json {
        let got = archive.decompress_vec(FrameKind::Layer, tokenizer_frame)?;
        if got != tok {
            return Err(crate::Error::InvalidFormat(
                "lma verification failed for tokenizer".into(),
            ));
        }
    }
    drop(archive);
    plan.publish()
}

/// Reassemble a (possibly chunked) tensor payload from an archive.
fn assemble_payload<R: std::io::Read + std::io::Seek>(
    archive: &mut CompressedArchive<R>,
    record: &TensorRecord,
) -> crate::Result<Vec<u8>> {
    let mut out = archive.decompress_vec(FrameKind::Layer, record.frame_id)?;
    for &id in &record.extra_frames {
        out.extend_from_slice(&archive.decompress_vec(FrameKind::Layer, id)?);
    }
    Ok(out)
}

/// One tensor queued for a batched, parallel-compressed write.
struct PendingTensor<'a> {
    name: String,
    ggml_type: u32,
    shape: Vec<u64>,
    payload: std::borrow::Cow<'a, [u8]>,
}

/// Process tensor batches at roughly this many raw bytes (bounds the
/// transient memory of parallel compression).
const BATCH_BYTES: usize = 512 * 1024 * 1024;

/// Compress and write a batch of tensors. Every [`CHUNK_BYTES`]-sized chunk of
/// every tensor in the batch compresses **in parallel** (zstd's internal
/// job-splitting cannot engage on chunk-sized inputs, so cross-frame
/// parallelism is the only way to use the cores), then frames are written in
/// order and one [`TensorRecord`] per tensor is appended to `records`.
fn write_tensor_batch<W: std::io::Write + std::io::Seek>(
    writer: &mut ArchiveWriter<W>,
    next_frame: &mut u32,
    zstd_level: i32,
    chunk_bytes: usize,
    batch: &[PendingTensor<'_>],
    records: &mut Vec<TensorRecord>,
) -> crate::Result<()> {
    use rayon::prelude::*;
    // Flatten (tensor, chunk) units in order; empty payloads still get a frame.
    let units: Vec<(usize, &[u8])> = batch
        .iter()
        .enumerate()
        .flat_map(|(ti, t)| {
            if t.payload.is_empty() {
                vec![(ti, &[][..])]
            } else {
                t.payload
                    .chunks(chunk_bytes.max(1))
                    .map(|c| (ti, c))
                    .collect()
            }
        })
        .collect();
    let compressed: Vec<crate::Result<Vec<u8>>> = units
        .par_iter()
        .map(|(_, c)| crate::archive::compress_frame_payload(c, zstd_level))
        .collect();
    let mut ids_per_tensor: Vec<Vec<u32>> = vec![Vec::new(); batch.len()];
    for ((ti, chunk), comp) in units.iter().zip(compressed) {
        writer.write_raw_frame(FrameKind::Layer, *next_frame, &comp?, chunk.len() as u64)?;
        ids_per_tensor[*ti].push(*next_frame);
        *next_frame += 1;
    }
    for (t, mut ids) in batch.iter().zip(ids_per_tensor) {
        let frame_id = ids.remove(0);
        records.push(TensorRecord {
            name: t.name.clone(),
            ggml_type: t.ggml_type,
            shape: t.shape.clone(),
            frame_id,
            extra_frames: ids,
        });
    }
    Ok(())
}

/// Rewrite an existing `.lma` archive at a different zstd level.
///
/// Works tensor by tensor from the archive itself — no source files needed —
/// and (re-)applies [`CHUNK_BYTES`] splitting so huge tensors decompress in
/// parallel at load. Every payload of the new archive is byte-verified
/// against the original before it atomically replaces it.
///
/// # Errors
///
/// Returns an error on I/O failure or a verification mismatch (the original
/// archive is left untouched).
#[allow(clippy::too_many_lines)]
pub fn recompress_lma(path: &Path, zstd_level: i32) -> crate::Result<()> {
    let mut src = CompressedArchive::open(path)?;
    let old_index = read_index(&mut src)?;

    let plan = ArchiveOutputPlan::new(path, OutputPolicy::default());
    plan.prepare()?;
    let mut writer = ArchiveWriter::new(plan.partial_path(), zstd_level)?;
    let mut next_frame = 1u32;

    let rewrite = |records: &[TensorRecord],
                   src: &mut CompressedArchive<_>,
                   writer: &mut ArchiveWriter<_>,
                   next_frame: &mut u32|
     -> crate::Result<Vec<TensorRecord>> {
        let mut out = Vec::new();
        let mut batch: Vec<PendingTensor<'static>> = Vec::new();
        let mut batch_bytes = 0usize;
        for r in records {
            let payload = assemble_payload(src, r)?;
            batch_bytes += payload.len();
            batch.push(PendingTensor {
                name: r.name.clone(),
                ggml_type: r.ggml_type,
                shape: r.shape.clone(),
                payload: std::borrow::Cow::Owned(payload),
            });
            if batch_bytes >= BATCH_BYTES {
                write_tensor_batch(
                    writer,
                    next_frame,
                    zstd_level,
                    CHUNK_BYTES,
                    &batch,
                    &mut out,
                )?;
                batch.clear();
                batch_bytes = 0;
            }
        }
        if !batch.is_empty() {
            write_tensor_batch(
                writer,
                next_frame,
                zstd_level,
                CHUNK_BYTES,
                &batch,
                &mut out,
            )?;
        }
        Ok(out)
    };
    let tensors = rewrite(&old_index.tensors, &mut src, &mut writer, &mut next_frame)?;
    let companion_tensors = rewrite(
        &old_index.companion_tensors,
        &mut src,
        &mut writer,
        &mut next_frame,
    )?;
    let assistant_tensors = rewrite(
        &old_index.assistant_tensors,
        &mut src,
        &mut writer,
        &mut next_frame,
    )?;
    let tokenizer_frame = if old_index.tokenizer_frame == 0 {
        0
    } else {
        let tok = src.decompress_vec(FrameKind::Layer, old_index.tokenizer_frame)?;
        let id = next_frame;
        writer.write_frame(FrameKind::Layer, id, &tok)?;
        id
    };

    let index = LmaIndex {
        format: old_index.format.clone(),
        config_json: old_index.config_json.clone(),
        tensors,
        companion_tensors,
        assistant_tensors,
        tokenizer_frame,
    };
    let index_bytes = serde_json::to_vec(&index)
        .map_err(|e| crate::Error::InvalidFormat(format!("encode lma index: {e}")))?;
    writer.write_frame(FrameKind::Metadata, 0, &index_bytes)?;
    writer.finish()?;

    // Verify every payload against the original before replacing it.
    let mut new = CompressedArchive::open(plan.partial_path())?;
    for (old, fresh) in old_index
        .tensors
        .iter()
        .chain(&old_index.companion_tensors)
        .chain(&old_index.assistant_tensors)
        .zip(
            index
                .tensors
                .iter()
                .chain(&index.companion_tensors)
                .chain(&index.assistant_tensors),
        )
    {
        let want = assemble_payload(&mut src, old)?;
        let got = assemble_payload(&mut new, fresh)?;
        if got != want {
            return Err(crate::Error::InvalidFormat(format!(
                "recompression verification failed for tensor {}",
                old.name
            )));
        }
    }
    if old_index.tokenizer_frame != 0 {
        let want = src.decompress_vec(FrameKind::Layer, old_index.tokenizer_frame)?;
        let got = new.decompress_vec(FrameKind::Layer, index.tokenizer_frame)?;
        if got != want {
            return Err(crate::Error::InvalidFormat(
                "recompression verification failed for tokenizer".into(),
            ));
        }
    }
    drop(new);
    drop(src);
    plan.publish()
}

/// Read and parse the index from an open `.lma` archive.
fn read_index<R: std::io::Read + std::io::Seek>(
    archive: &mut CompressedArchive<R>,
) -> crate::Result<LmaIndex> {
    let bytes = archive.decompress_vec(FrameKind::Metadata, 0)?;
    let index: LmaIndex = serde_json::from_slice(&bytes)
        .map_err(|e| crate::Error::InvalidFormat(format!("decode lma index: {e}")))?;
    if index.format != FORMAT_TAG {
        return Err(crate::Error::InvalidFormat(format!(
            "unsupported lma format {:?}, expected {FORMAT_TAG:?}",
            index.format
        )));
    }
    Ok(index)
}

/// Read the embedded `config.json` string from a `.lma` archive.
///
/// # Errors
///
/// Returns an error on I/O or if the index cannot be decoded.
pub fn read_lma_config(path: &Path) -> crate::Result<String> {
    let mut archive = CompressedArchive::open(path)?;
    Ok(read_index(&mut archive)?.config_json)
}

/// Read every tensor from a `.lma` archive, dequantized to FP16.
///
/// Used by the loader and by round-trip tests; does not touch Metal.
///
/// # Errors
///
/// Returns an error on I/O, decode, or an unknown tensor type.
pub fn read_lma_tensors_f16(path: &Path) -> crate::Result<HashMap<String, Vec<half::f16>>> {
    let mut archive = CompressedArchive::open(path)?;
    let index = read_index(&mut archive)?;
    let mut out = HashMap::with_capacity(
        index.tensors.len() + index.companion_tensors.len() + index.assistant_tensors.len(),
    );
    for record in index
        .tensors
        .iter()
        .chain(&index.companion_tensors)
        .chain(&index.assistant_tensors)
    {
        let raw = archive.decompress_vec(FrameKind::Layer, record.frame_id)?;
        let ty = GGUFType::from_u32(record.ggml_type).ok_or_else(|| {
            crate::Error::InvalidFormat(format!(
                "tensor {} has unknown ggml type {}",
                record.name, record.ggml_type
            ))
        })?;
        let num_elements: usize = record.shape.iter().map(|&d| d as usize).product();
        out.insert(
            record.name.clone(),
            decode_tensor_to_f16(&raw, ty, num_elements)?,
        );
    }
    Ok(out)
}

/// A decoded frame payload from the parallel `.lma` decode pipeline.
enum Decoded {
    F16(Vec<half::f16>),
    /// Raw quantized payload kept verbatim for device-resident quant weights.
    Quant(Vec<u8>, GGUFType),
    PleTable(QuantRowTable),
}

/// Transient-memory budget for the load pipeline, in bytes.
///
/// Loading decompresses frames into temporary buffers before upload; without
/// a bound the peak RSS is a multiple of the steady state. The cap defaults
/// to an eighth of the memory free **right now** (clamped to 128–512 MiB) so
/// load adapts to the machine like everything else; `LOCAL_AI_LOAD_BUDGET`
/// (MiB) overrides. Smaller caps trade load speed for a lower peak.
fn load_transient_cap() -> usize {
    // Measured on M4 Pro: 512 MiB is the knee — same load time as 2 GiB but
    // ~1.5 GB lower peak RSS; below 256 MiB load time starts to climb.
    const MIN: u64 = 128 * 1024 * 1024;
    const MAX: u64 = 512 * 1024 * 1024;
    if let Some(mib) = std::env::var("LOCAL_AI_LOAD_BUDGET")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        return usize::try_from(mib * 1024 * 1024)
            .unwrap_or(usize::MAX)
            .max(64 * 1024 * 1024);
    }
    let cap = local_metal::memory::available_memory_now().map_or(MAX, |free| free / 8);
    usize::try_from(cap.clamp(MIN, MAX)).unwrap_or(usize::MAX)
}

/// Estimated transient bytes to decode one record: compressed input (bounded
/// by raw), the raw payload, and the FP16 expansion when it will be decoded.
fn record_transient_bytes(record: &TensorRecord, allow_quant: bool) -> usize {
    let elements: usize = record.shape.iter().map(|&d| d as usize).product();
    let raw =
        GGUFType::from_u32(record.ggml_type).map_or(elements * 2, |ty| ty.tensor_bytes(elements));
    let decoded = if allow_quant
        && GGUFType::from_u32(record.ggml_type)
            .is_some_and(|ty| init_gemma4_qat::keep_quantized(ty, &record.shape))
    {
        0
    } else {
        elements * 2
    };
    raw * 2 + decoded
}

/// Group records greedily so each group's estimated transient stays under
/// `cap` (oversized records form singleton groups and are wave-bounded later).
fn group_records_by_transient(
    records: &[TensorRecord],
    allow_quant: bool,
    cap: usize,
) -> Vec<Vec<&TensorRecord>> {
    let mut groups: Vec<Vec<&TensorRecord>> = Vec::new();
    let mut current: Vec<&TensorRecord> = Vec::new();
    let mut current_bytes = 0usize;
    for record in records {
        let est = record_transient_bytes(record, allow_quant);
        if !current.is_empty() && current_bytes + est > cap {
            groups.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current.push(record);
        current_bytes += est;
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// Read and decompress all frames of one record into its raw payload, with
/// piece decompression bounded to `cap` in-flight bytes (waves of parallel
/// pieces): large chunked tensors stream instead of materializing every
/// compressed and decompressed piece simultaneously.
fn assemble_record_raw<R: std::io::Read + std::io::Seek>(
    archive: &mut CompressedArchive<R>,
    record: &TensorRecord,
    cap: usize,
) -> crate::Result<Vec<u8>> {
    use rayon::prelude::*;
    let ids: Vec<u32> = std::iter::once(record.frame_id)
        .chain(record.extra_frames.iter().copied())
        .collect();
    let elements: usize = record.shape.iter().map(|&d| d as usize).product();
    let cap_hint =
        GGUFType::from_u32(record.ggml_type).map_or(elements * 2, |ty| ty.tensor_bytes(elements));
    let mut raw = Vec::with_capacity(cap_hint);
    // A wave's in-flight transient ≈ 2× the wave's raw bytes (compressed +
    // decompressed pieces) on top of the partially assembled payload.
    let mut i = 0;
    while i < ids.len() {
        let mut wave = Vec::new();
        let mut wave_raw = 0usize;
        while i < ids.len() {
            let part = archive.read_frame_compressed(FrameKind::Layer, ids[i])?;
            wave_raw += part.0;
            wave.push(part);
            i += 1;
            if wave_raw * 2 >= cap {
                break;
            }
        }
        let outs: Vec<crate::Result<Vec<u8>>> = wave
            .into_par_iter()
            .map(|(raw_len, bytes)| {
                zstd::bulk::decompress(&bytes, raw_len)
                    .map_err(|e| crate::Error::Io(format!("zstd decompress: {e}")))
            })
            .collect();
        for out in outs {
            raw.extend_from_slice(&out?);
        }
    }
    Ok(raw)
}

/// Upload a decoded tensor payload into a device-resident [`QuantWeight`].
fn upload_weight(
    device: &ProtocolObject<dyn MTLDevice>,
    name: &str,
    payload: Decoded,
    rows: u32,
    cols: u32,
) -> crate::Result<QuantWeight> {
    let mk_buf = |bytes: &[u8]| {
        MetalBuffer::from_slice(device, bytes)
            .map_err(|e| crate::Error::InvalidArgument(format!("MetalBuffer for {name}: {e}")))
    };
    if std::env::var_os("LOCAL_AI_WFMT").is_some() {
        let tag = match &payload {
            Decoded::Quant(_, ty) => format!("{ty:?}"),
            Decoded::F16(_) => "F16".to_string(),
            Decoded::PleTable(_) => "PLE".to_string(),
        };
        eprintln!("[wfmt] {name} ty={tag} rows={rows} cols={cols}");
    }
    match payload {
        Decoded::Quant(raw, ty) => QuantWeight::quantized(mk_buf(&raw)?, ty, rows, cols),
        Decoded::F16(f16_data) => Ok(QuantWeight::f16(
            mk_buf(bytemuck::cast_slice(&f16_data))?,
            rows,
            cols,
        )),
        Decoded::PleTable(_) => Err(crate::Error::InvalidFormat(format!(
            "{name}: PLE table cannot be uploaded as a weight"
        ))),
    }
}

/// Decode one record's raw payload into its in-memory form: the streamed PLE
/// table, a verbatim quantized payload, or an FP16 expansion.
fn decode_payload(
    record: &TensorRecord,
    raw: Vec<u8>,
    wants_ple: bool,
    allow_quant: bool,
) -> crate::Result<(String, Decoded)> {
    let ty = GGUFType::from_u32(record.ggml_type).ok_or_else(|| {
        crate::Error::InvalidFormat(format!(
            "tensor {} has unknown ggml type {}",
            record.name, record.ggml_type
        ))
    })?;
    // Stream the per-layer embedding table on demand (see `QuantRowTable`)
    // instead of expanding it to f16.
    if wants_ple && record.name == PLE_TABLE_NAME && record.shape.len() == 2 {
        let cols = record.shape[0] as usize;
        let rows = record.shape[1] as usize;
        let table = QuantRowTable::new(raw, ty, cols, rows)?;
        return Ok((record.name.clone(), Decoded::PleTable(table)));
    }
    if allow_quant && init_gemma4_qat::keep_quantized(ty, &record.shape) {
        return Ok((record.name.clone(), Decoded::Quant(raw, ty)));
    }
    let num_elements: usize = record.shape.iter().map(|&d| d as usize).product();
    let f16_data = decode_tensor_to_f16(&raw, ty, num_elements)?;
    Ok((record.name.clone(), Decoded::F16(f16_data)))
}

/// Load a `.lma` archive into the model config and GPU weight buffers.
///
/// # Errors
///
/// Returns an error on I/O, decode, config parse, or Metal allocation failure.
#[allow(clippy::too_many_lines)]
pub fn load_lma(path: &Path, device: &ProtocolObject<dyn MTLDevice>) -> crate::Result<LoadedLma> {
    let mut archive = CompressedArchive::open(path)?;
    let mut index = read_index(&mut archive)?;

    let ModelConfig::Gemma4QAT(config) = ModelConfig::from_json(&index.config_json)
        .map_err(|e| crate::Error::InvalidFormat(format!("lma config: {e}")))?;

    // Text-only mode (`LOCAL_AI_TEXT_ONLY=1`): drop the vision/audio towers and
    // the multimodal projector before any frame is read. These tensors are
    // resident-but-never-read during text decode (on Gemma 4 E2B they account
    // for ~0.9 GiB of FP16 weights), so skipping them cuts the memory footprint
    // and load time with zero impact on text generation. Image/audio prompts
    // are unavailable in this mode — the `multimodal` report below reflects that.
    let text_only = std::env::var("LOCAL_AI_TEXT_ONLY").ok().as_deref() == Some("1");
    let is_multimodal_tensor =
        |name: &str| name.starts_with("v.") || name.starts_with("a.") || name.starts_with("mm.");
    if text_only {
        index
            .tensors
            .retain(|record| !is_multimodal_tensor(&record.name));
        index
            .companion_tensors
            .retain(|record| !is_multimodal_tensor(&record.name));
    }
    let tensor_names = index.tensors.iter().map(|record| record.name.as_str());
    let companion_names = index
        .companion_tensors
        .iter()
        .map(|record| record.name.as_str());
    let has_image_tensors = tensor_names_indicate_image_support(tensor_names.clone())
        || tensor_names_indicate_image_support(companion_names.clone());
    let multimodal = MultimodalSupport {
        config_declares_image: config.multimodal.has_vision_config()
            || config.multimodal.image_token_id.is_some(),
        config_declares_audio: config.multimodal.has_audio_config()
            || config.multimodal.audio_token_id.is_some(),
        config_declares_video: config.multimodal.has_video_config(),
        has_image_tensors,
        has_audio_tensors: tensor_names_indicate_audio_support(tensor_names)
            || tensor_names_indicate_audio_support(companion_names),
        // Gemma 4 video reuses the image/vision companion tensors.
        has_video_tensors: has_image_tensors,
    };

    // Frames are independent zstd streams, so decompression + dequantization
    // run in parallel across cores; chunking bounds the transient memory and
    // Metal buffer creation stays on this thread. (Single-threaded decode of
    // the ~2.3 GB payload was the dominant load cost.)
    //
    // Backbone matrices in formats with device kernels (Q4_0 / TQ2_0) skip
    // dequantization entirely and upload their block payloads verbatim — they
    // run on quantized matvec/matmul kernels at inference time.
    let allow_quant = init_gemma4_qat::quant_weights_enabled();
    // "JIT" loading: groups of records are sized so the in-flight transient
    // (compressed + raw + decoded buffers) stays under a budget derived from
    // the memory free right now — peak RSS tracks the steady state instead of
    // ballooning during load. Oversized single tensors (the PLE table) stream
    // their chunked frames in bounded waves.
    let transient_cap = load_transient_cap();
    let mut decode_records = |records: &[TensorRecord],
                              ple: Option<&mut Option<QuantRowTable>>,
                              allow_quant: bool|
     -> crate::Result<HashMap<String, QuantWeight>> {
        use rayon::prelude::*;
        let mut ple = ple;
        let mut buffers = HashMap::with_capacity(records.len());
        let finish = |record: &TensorRecord,
                      item: crate::Result<(String, Decoded)>,
                      ple: &mut Option<&mut Option<QuantRowTable>>,
                      buffers: &mut HashMap<String, QuantWeight>|
         -> crate::Result<()> {
            let (name, payload) = item?;
            if let Decoded::PleTable(table) = payload {
                if let Some(ple_slot) = ple.as_deref_mut() {
                    *ple_slot = Some(table);
                }
                return Ok(());
            }
            let cols = record.shape.first().copied().unwrap_or(1) as u32;
            let rows = record.shape.get(1).copied().unwrap_or(1) as u32;
            let weight = upload_weight(device, &name, payload, rows, cols)?;
            buffers.insert(name, weight);
            Ok(())
        };
        let wants_ple = ple.is_some();
        for group in group_records_by_transient(records, allow_quant, transient_cap) {
            // Oversized single tensor: stream its frames in bounded waves and
            // decode/upload alone.
            if group.len() == 1 && record_transient_bytes(group[0], allow_quant) > transient_cap {
                let record = group[0];
                let raw = assemble_record_raw(&mut archive, record, transient_cap)?;
                let item = decode_payload(record, raw, wants_ple, allow_quant);
                finish(record, item, &mut ple, &mut buffers)?;
                continue;
            }
            // Phase 1: sequential I/O — compressed bytes for the whole group.
            let mut compressed = Vec::with_capacity(group.len());
            for record in &group {
                let mut parts =
                    vec![archive.read_frame_compressed(FrameKind::Layer, record.frame_id)?];
                for &id in &record.extra_frames {
                    parts.push(archive.read_frame_compressed(FrameKind::Layer, id)?);
                }
                compressed.push(parts);
            }
            // Phase 2: parallel decompress + decode across the group.
            let decoded: Vec<crate::Result<(String, Decoded)>> = group
                .par_iter()
                .zip(compressed)
                .map(|(record, parts)| {
                    let pieces: Vec<crate::Result<Vec<u8>>> = parts
                        .into_par_iter()
                        .map(|(raw_len, bytes)| {
                            zstd::bulk::decompress(&bytes, raw_len)
                                .map_err(|e| crate::Error::Io(format!("zstd decompress: {e}")))
                        })
                        .collect();
                    let mut raw = Vec::new();
                    for piece in pieces {
                        raw.extend_from_slice(&piece?);
                    }
                    decode_payload(record, raw, wants_ple, allow_quant)
                })
                .collect();
            // Phase 3: Metal uploads on this thread.
            for (record, item) in group.iter().zip(decoded) {
                finish(record, item, &mut ple, &mut buffers)?;
            }
        }
        Ok(buffers)
    };

    let mut ple_table = None;
    let mut buffers = decode_records(&index.tensors, Some(&mut ple_table), allow_quant)?;
    if !index.companion_tensors.is_empty() {
        let companion = decode_records(&index.companion_tensors, None, allow_quant)?;
        for (name, weight) in companion {
            buffers.entry(name).or_insert(weight);
        }
    }
    let tokenizer_json = if index.tokenizer_frame == 0 {
        None
    } else {
        Some(archive.decompress_vec(FrameKind::Layer, index.tokenizer_frame)?)
    };
    Ok(LoadedLma {
        config,
        weights: WeightMap { buffers, ple_table },
        tokenizer_json,
        multimodal,
    })
}

/// Everything a single-file `.lma` bundle provides.
pub struct LoadedLma {
    pub config: Gemma4QATConfig,
    pub weights: WeightMap,
    /// Bundled `tokenizer.json` bytes, if present.
    pub tokenizer_json: Option<Vec<u8>>,
    /// Declared and bundled Gemma 4 modality support.
    pub multimodal: MultimodalSupport,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
    use super::*;

    /// Build a minimal valid GGUF (version 3, no metadata KV) containing the
    /// given `(name, ggml_type, dims, payload)` tensors.
    fn build_gguf(tensors: &[(&str, u32, &[u64], Vec<u8>)]) -> Vec<u8> {
        const ALIGN: usize = 32;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count

        let mut offset_positions = Vec::new();
        for (name, ty, dims, _) in tensors {
            let name_bytes = format!("{name}\0");
            buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(name_bytes.as_bytes());
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&ty.to_le_bytes());
            offset_positions.push(buf.len());
            buf.extend_from_slice(&0u64.to_le_bytes()); // offset placeholder
        }

        let headers_end = buf.len();
        let data_start = (headers_end + ALIGN - 1) & !(ALIGN - 1);
        buf.resize(data_start, 0);

        let mut rel_offset = 0usize;
        for (pos, (_, _, _, payload)) in offset_positions.iter().zip(tensors) {
            buf[*pos..*pos + 8].copy_from_slice(&(rel_offset as u64).to_le_bytes());
            rel_offset += payload.len();
        }
        for (_, _, _, payload) in tensors {
            buf.extend_from_slice(payload);
        }
        buf
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect()
    }

    #[test]
    fn gguf_to_lma_round_trips_tensor_data() {
        let dir = tempfile::tempdir().expect("tempdir");
        let gguf_path = dir.path().join("model.gguf");
        let lma_path = dir.path().join("model.lma");

        // One F16 tensor and one F32 tensor with distinct, recognizable values.
        let embd = f16_bytes(&[1.0, -2.0, 3.5, 0.25]);
        let norm = [0.5f32, 1.5, 2.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        let gguf_bytes = build_gguf(&[
            ("token_embd.weight", 1, &[2, 2], embd),
            ("output_norm.weight", 0, &[3], norm),
        ]);
        std::fs::write(&gguf_path, &gguf_bytes).expect("write gguf");

        let config_json = "{\"model_type\":\"gemma4_text\",\"hidden_size\":1536}";
        let gguf = GGUFModel::open(&gguf_path).expect("open gguf");
        compress_gguf_to_lma(&gguf, config_json, None, None, &lma_path, 3).expect("compress");
        assert!(lma_path.exists(), "published .lma must exist");
        assert!(
            !dir.path().join("model.lma.partial").exists(),
            "partial cleaned up"
        );

        // The archive is self-describing: the config round-trips verbatim.
        assert_eq!(read_lma_config(&lma_path).expect("config"), config_json);

        // Decode both the GGUF directly and the .lma, and compare FP16 output.
        let from_lma = read_lma_tensors_f16(&lma_path).expect("read lma");
        assert_eq!(from_lma.len(), 2);

        for (name, info) in &gguf.tensors {
            let raw = gguf.tensor_data::<u8>(name).expect("gguf data");
            let n: usize = info.shape.iter().map(|&d| d as usize).product();
            let expected = decode_tensor_to_f16(raw, info.ty, n).expect("decode gguf");
            assert_eq!(from_lma.get(name), Some(&expected), "mismatch for {name}");
        }
    }

    #[test]
    fn load_lma_keeps_q4_0_and_tq2_0_device_resident() {
        let dir = tempfile::tempdir().expect("tempdir");
        let gguf_path = dir.path().join("model.gguf");
        let lma_path = dir.path().join("model.lma");

        // Q4_0: 4 rows × 64 cols = 2 blocks/row → 4 × 36 bytes.
        let q4_payload: Vec<u8> = (0..4 * 2)
            .flat_map(|b| {
                let mut block = half::f16::from_f32((b as f32).mul_add(0.25, 0.5))
                    .to_le_bytes()
                    .to_vec();
                block.extend((0..16).map(|i| (b * 16 + i) as u8));
                block
            })
            .collect();
        // TQ2_0: 2 rows × 256 cols = 1 block/row → 2 × 66 bytes.
        let tq2_payload: Vec<u8> = (0..2)
            .flat_map(|b| {
                let mut block: Vec<u8> = (0..64).map(|i| (b * 64 + i) as u8).collect();
                block.extend(half::f16::from_f32(1.0 + b as f32).to_le_bytes());
                block
            })
            .collect();
        let norm = [0.5f32, 1.5, 2.5, 3.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        let gguf_bytes = build_gguf(&[
            ("blk.0.attn_q.weight", 2, &[64, 4], q4_payload.clone()),
            ("blk.0.ffn_gate.weight", 35, &[256, 2], tq2_payload.clone()),
            ("output_norm.weight", 0, &[4], norm),
        ]);
        std::fs::write(&gguf_path, &gguf_bytes).expect("write gguf");

        let config_json = r#"{"model_type": "gemma4_text", "vocab_size": 8, "hidden_size": 64, "intermediate_size": 256, "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 64, "max_position_embeddings": 64, "rms_norm_eps": 1e-6, "sliding_window": 16}"#;
        let gguf = GGUFModel::open(&gguf_path).expect("open gguf");
        compress_gguf_to_lma(&gguf, config_json, None, None, &lma_path, 3).expect("compress");

        let ctx = local_metal::context::MetalContext::new().expect("metal context");
        let loaded = load_lma(&lma_path, ctx.device()).expect("load lma");
        let w = &loaded.weights.buffers;

        let q4 = w.get("blk.0.attn_q.weight").expect("attn_q");
        assert_eq!(q4.ty(), GGUFType::Q4_0, "Q4_0 must stay quantized");
        assert_eq!(
            q4.device_bytes(),
            q4_payload.len(),
            "verbatim quantized upload"
        );
        assert_eq!((q4.rows(), q4.cols()), (4, 64));

        let tq2 = w.get("blk.0.ffn_gate.weight").expect("ffn_gate");
        assert_eq!(tq2.ty(), GGUFType::TQ2_0, "TQ2_0 must stay quantized");
        assert_eq!(tq2.device_bytes(), tq2_payload.len());

        let norm_w = w.get("output_norm.weight").expect("norm");
        assert_eq!(norm_w.ty(), GGUFType::F16, "norms expand to FP16");

        // Row gather from the shared device buffer matches a full decode.
        let full = decode_tensor_to_f16(&q4_payload, GGUFType::Q4_0, 4 * 64).expect("decode");
        for row in 0..4 {
            let got = q4.dequant_row(row).expect("row");
            assert_eq!(got, full[row * 64..(row + 1) * 64], "row {row}");
        }
    }

    #[test]
    fn bundle_records_companion_and_tokenizer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let main_path = dir.path().join("model.gguf");
        let companion_path = dir.path().join("mmproj.gguf");
        let lma_path = dir.path().join("model.lma");

        std::fs::write(
            &main_path,
            build_gguf(&[(
                "token_embd.weight",
                1,
                &[2, 2],
                f16_bytes(&[1.0, 2.0, 3.0, 4.0]),
            )]),
        )
        .expect("write main");
        std::fs::write(
            &companion_path,
            build_gguf(&[(
                "mm.input_projection.weight",
                1,
                &[2, 2],
                f16_bytes(&[10.0, 11.0, 12.0, 13.0]),
            )]),
        )
        .expect("write companion");

        let gguf = GGUFModel::open(&main_path).expect("open main");
        let companion = GGUFModel::open(&companion_path).expect("open companion");
        let tok = br#"{"fake":"tokenizer"}"#;
        compress_gguf_to_lma(&gguf, "{}", Some(&companion), Some(tok), &lma_path, 3)
            .expect("compress bundle");

        let mut archive = CompressedArchive::open(&lma_path).expect("open lma");
        let index_bytes = archive
            .decompress_vec(FrameKind::Metadata, 0)
            .expect("index");
        let index: LmaIndex = serde_json::from_slice(&index_bytes).expect("decode index");

        assert_eq!(index.tensors.len(), 1);
        assert_eq!(index.companion_tensors.len(), 1);
        assert_eq!(
            index.companion_tensors[0].name,
            "mm.input_projection.weight"
        );
        assert!(
            index.assistant_tensors.is_empty(),
            "MTP drafter tensors must not be bundled"
        );
        assert_ne!(index.tokenizer_frame, 0);

        let tok_back = archive
            .decompress_vec(FrameKind::Layer, index.tokenizer_frame)
            .expect("tokenizer frame");
        assert_eq!(tok_back, tok);

        let all_tensors = read_lma_tensors_f16(&lma_path).expect("read all tensors");
        assert!(all_tensors.contains_key("token_embd.weight"));
        assert!(all_tensors.contains_key("mm.input_projection.weight"));
    }

    #[test]
    fn rejects_wrong_format_tag() {
        let dir = tempfile::tempdir().expect("tempdir");
        let lma_path = dir.path().join("bad.lma");
        let plan = ArchiveOutputPlan::new(&lma_path, OutputPolicy::default());
        plan.prepare().expect("prepare");
        let mut writer = ArchiveWriter::new(plan.partial_path(), 1).expect("writer");
        writer
            .write_frame(
                FrameKind::Metadata,
                0,
                b"{\"format\":\"nope\",\"config_json\":\"\",\"tensors\":[]}",
            )
            .expect("write");
        writer.finish().expect("finish");
        plan.publish().expect("publish");

        let err = read_lma_tensors_f16(&lma_path);
        assert!(err.is_err(), "wrong format tag must be rejected");
    }

    #[test]
    fn chunked_payload_round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("chunked.lma");
        let payload: Vec<u8> = (0..1000u32).flat_map(u32::to_le_bytes).collect();

        let plan = ArchiveOutputPlan::new(&path, OutputPolicy::default());
        plan.prepare().expect("prepare");
        let mut writer = ArchiveWriter::new(plan.partial_path(), 3).expect("writer");
        let mut next = 1u32;
        let batch = [PendingTensor {
            name: "t".into(),
            ggml_type: 1,
            shape: vec![payload.len() as u64],
            payload: std::borrow::Cow::Borrowed(&payload),
        }];
        let mut records = Vec::new();
        write_tensor_batch(&mut writer, &mut next, 3, 64, &batch, &mut records).expect("write");
        writer.finish().expect("finish");
        plan.publish().expect("publish");

        let record = &records[0];
        assert!(
            !record.extra_frames.is_empty(),
            "4000-byte payload at 64-byte chunks must split"
        );
        let mut archive = CompressedArchive::open(&path).expect("open");
        let back = assemble_payload(&mut archive, record).expect("assemble");
        assert_eq!(
            back, payload,
            "chunked payload must reassemble byte-identically"
        );
    }
}
