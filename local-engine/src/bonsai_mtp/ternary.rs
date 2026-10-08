//! The trained head's source contract and its packing into artifact sections.
//!
//! Each matrix is either exactly ternary with one scale per 128 columns,
//! packed into the target's own `PTQ1_0` blocks so the target's kernels
//! multiply it, or int8 with one scale per row, for the few matrices too
//! sensitive for ternary. Both live in the target's Hadamard-rotated basis and
//! multiply the same rotated activations.
//!
//! The source contract is `model_mtp_ternary.safetensors`: for each matrix
//! EITHER `<name>.codes` (I8 `[rows, cols]`, values in {-1, 0, 1}) and
//! `<name>.scales` (F16 `[rows, cols / 128]`), meaning
//! `y[r] = Σ_c codes[r, c] · scales[r, c / 128] · (R x)[c]`, OR `<name>.int8`
//! (I8 `[rows, cols]`, values in [-127, 127]) and `<name>.row_scales` (F32
//! `[rows]`), meaning `y[r] = Σ_c int8[r, c] · row_scales[r] · (R x)[c]`, where
//! `R` is the target's forward signed Hadamard for width `cols`; plus the seven
//! BF16 zero-centered norms, folded to `1 + w`.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::MmapOptions;
use serde::Deserialize;

use super::Section;
use super::invalid;
use super::weights::{MATRIX_SPECS, MatrixSpec, NORM_SPECS, fold_norm};

/// The file `--export mtp-head=DIR` reads from `DIR`.
pub const MTP_TERNARY_SOURCE: &str = "model_mtp_ternary.safetensors";

/// Largest JSON header a head source may carry.
pub(super) const SAFETENSORS_MAX_HEADER: u64 = 1 << 20;

pub(super) const PTQ1_BLOCK_ELEMENTS: usize = 128;
pub(super) const PTQ1_BLOCK_BYTES: usize = 28;

/// Pack 128 ternary codes and an FP16 scale (raw bits) into one `PTQ1_0` block,
/// exactly as Prism's `quantize_row_ptq1_0_ref` does: base-3 digits `code + 1`,
/// five per byte for elements `m + 16n` (bytes 0..16) and `80 + m + 8n` (bytes
/// 16..24), four per byte for `120 + j + 2m` (bytes 24..26) shifted up one trit,
/// each byte the ceiling of `q · 256 / 243`, then `d` little-endian.
///
/// Codes outside {-1, 0, 1} are the caller's to reject.
pub(super) fn pack_ptq1_block(
    codes: &[i8; PTQ1_BLOCK_ELEMENTS],
    scale_bits: u16,
) -> [u8; PTQ1_BLOCK_BYTES] {
    let trit = |index: usize| (i16::from(codes[index]) + 1) as u16;
    // Ceiling division by 243 == 3^5 maps the base-3 value onto the byte the
    // decoder's multiply-and-shift reads trits back from.
    let byte = |q: u16| (q * 256).div_ceil(243) as u8;
    let mut block = [0_u8; PTQ1_BLOCK_BYTES];
    for (m, out) in block[..16].iter_mut().enumerate() {
        *out = byte((0..5).fold(0, |q, n| q * 3 + trit(m + n * 16)));
    }
    for (m, out) in block[16..24].iter_mut().enumerate() {
        *out = byte((0..5).fold(0, |q, n| q * 3 + trit(80 + m + n * 8)));
    }
    for (j, out) in block[24..26].iter_mut().enumerate() {
        *out = byte((0..4).fold(0, |q, m| q * 3 + trit(120 + j + m * 2)) * 3);
    }
    block[26..].copy_from_slice(&scale_bits.to_le_bytes());
    block
}

/// Pack a row-major `[rows, columns]` ternary matrix (`codes` as raw I8 bytes,
/// `scales` as little-endian F16 `[rows, columns / 128]`) into `PTQ1_0` rows.
pub(super) fn pack_ptq1_matrix(
    codes: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> crate::Result<Vec<u8>> {
    if !columns.is_multiple_of(PTQ1_BLOCK_ELEMENTS)
        || codes.len() != rows * columns
        || scales.len() != rows * columns / PTQ1_BLOCK_ELEMENTS * 2
    {
        return invalid("ternary MTP matrix has an unexpected size");
    }
    let blocks = rows * columns / PTQ1_BLOCK_ELEMENTS;
    let mut packed = Vec::with_capacity(blocks * PTQ1_BLOCK_BYTES);
    for (block, scale) in codes
        .as_chunks::<PTQ1_BLOCK_ELEMENTS>()
        .0
        .iter()
        .zip(scales.as_chunks::<2>().0)
    {
        let scale_bits = u16::from_le_bytes(*scale);
        if !half::f16::from_bits(scale_bits).is_finite() {
            return invalid("non-finite ternary MTP scale");
        }
        let block = block.map(u8::cast_signed);
        if block.iter().any(|code| !(-1..=1).contains(code)) {
            return invalid("ternary MTP code outside {-1, 0, 1}");
        }
        packed.extend_from_slice(&pack_ptq1_block(&block, scale_bits));
    }
    Ok(packed)
}

/// How one head matrix is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MatrixFormat {
    /// `PTQ1_0` blocks: one section, `<name>.ptq1`.
    Ptq1,
    /// Row-major int8 then one F32 scale per row: sections `<name>.int8` and
    /// `<name>.row_scales`.
    Int8,
}

pub(super) const PTQ1_SUFFIX: &str = ".ptq1";
pub(super) const INT8_SUFFIX: &str = ".int8";
pub(super) const ROW_SCALES_SUFFIX: &str = ".row_scales";

/// One format per head matrix, in `MATRIX_SPECS` order.
pub(super) type Formats = [MatrixFormat; MATRIX_SPECS.len()];

/// Section names and byte lengths of a head whose matrices are stored as
/// `formats` says, in artifact order: the matrices in `Weights` order (an int8
/// matrix's weights, then its row scales), then the seven folded F32 norms.
pub(super) fn head_spec(formats: &Formats) -> Vec<Section> {
    let mut spec = Vec::with_capacity(2 * MATRIX_SPECS.len() + NORM_SPECS.len());
    for (matrix, format) in MATRIX_SPECS.iter().zip(formats) {
        match format {
            MatrixFormat::Ptq1 => spec.push((
                format!("{}{PTQ1_SUFFIX}", matrix.section),
                matrix.rows * matrix.columns / PTQ1_BLOCK_ELEMENTS * PTQ1_BLOCK_BYTES,
            )),
            MatrixFormat::Int8 => {
                spec.push((
                    format!("{}{INT8_SUFFIX}", matrix.section),
                    matrix.rows * matrix.columns,
                ));
                spec.push((
                    format!("{}{ROW_SCALES_SUFFIX}", matrix.section),
                    matrix.rows * size_of::<f32>(),
                ));
            }
        }
    }
    for norm in &NORM_SPECS {
        spec.push((norm.section.to_owned(), norm.elements * size_of::<f32>()));
    }
    spec
}

/// The all-ternary head's sections.
#[cfg(test)]
pub(super) fn ternary_spec() -> Vec<Section> {
    head_spec(&[MatrixFormat::Ptq1; MATRIX_SPECS.len()])
}

/// The formats a head file's own section names select: each matrix's format
/// is the suffix of the section where it starts. Whether the names then match
/// `head_spec` of these formats exactly is the reader's check, not this one's.
pub(super) fn formats_from_names(names: &[&str]) -> Result<Formats, String> {
    let mut formats = [MatrixFormat::Ptq1; MATRIX_SPECS.len()];
    let mut at = 0;
    for (matrix, format) in MATRIX_SPECS.iter().zip(&mut formats) {
        let name = names
            .get(at)
            .ok_or_else(|| format!("holds no section for {}", matrix.section))?;
        (*format, at) = match name.strip_prefix(matrix.section) {
            Some(PTQ1_SUFFIX) => (MatrixFormat::Ptq1, at + 1),
            Some(INT8_SUFFIX) => (MatrixFormat::Int8, at + 2),
            _ => {
                return Err(format!(
                    "section {at} is {name}, this build binds {0}{PTQ1_SUFFIX} or {0}{INT8_SUFFIX}",
                    matrix.section
                ));
            }
        };
    }
    Ok(formats)
}

/// The pinned `(name, dtype, shape)` of a matrix's two source tensors.
fn matrix_tensors(
    matrix: &MatrixSpec,
    format: MatrixFormat,
) -> [(String, &'static str, Vec<usize>); 2] {
    match format {
        MatrixFormat::Ptq1 => [
            (
                format!("{}.codes", matrix.section),
                "I8",
                vec![matrix.rows, matrix.columns],
            ),
            (
                format!("{}.scales", matrix.section),
                "F16",
                vec![matrix.rows, matrix.columns / PTQ1_BLOCK_ELEMENTS],
            ),
        ],
        MatrixFormat::Int8 => [
            (
                format!("{}.int8", matrix.section),
                "I8",
                vec![matrix.rows, matrix.columns],
            ),
            (
                format!("{}.row_scales", matrix.section),
                "F32",
                vec![matrix.rows],
            ),
        ],
    }
}

/// Validate one int8 matrix: values in [-127, 127] (the symmetric range the
/// contract names; -128 has no positive twin) and finite row scales. Both are
/// stored verbatim: int8 bytes, then little-endian F32 scales.
pub(super) fn check_int8_matrix(weights: &[u8], row_scales: &[u8]) -> crate::Result<()> {
    if weights.iter().any(|&byte| byte.cast_signed() == i8::MIN) {
        return invalid("int8 MTP weight outside [-127, 127]");
    }
    if row_scales
        .as_chunks::<4>()
        .0
        .iter()
        .any(|bytes| !f32::from_le_bytes(*bytes).is_finite())
    {
        return invalid("non-finite int8 MTP row scale");
    }
    Ok(())
}

/// A source file packed into artifact sections: each matrix's format, and the
/// sections in the order `head_spec(&formats)` names them.
pub(super) struct HeadSections {
    pub(super) formats: Formats,
    pub(super) sections: Vec<Vec<u8>>,
}

/// Read `source` (the safetensors contract above) into artifact-order
/// sections. Each matrix is ternary or int8 according to which pair of tensors
/// the file holds for it; holding tensors of both pairs is refused.
#[allow(clippy::too_many_lines)]
pub(super) fn head_sections(source: &Path) -> crate::Result<HeadSections> {
    #[derive(Deserialize)]
    struct Entry {
        dtype: String,
        shape: Vec<usize>,
        data_offsets: [usize; 2],
    }
    let file = File::open(source)?;
    // SAFETY: read-only mapping of an input file this process never writes.
    #[allow(unsafe_code)]
    let map = unsafe { MmapOptions::new().map(&file)? };
    let header_len = map
        .first_chunk::<8>()
        .map(|bytes| u64::from_le_bytes(*bytes))
        .ok_or_else(|| crate::Error::InvalidFormat("MTP head source is too short".into()))?;
    if header_len == 0 || header_len > SAFETENSORS_MAX_HEADER || header_len + 8 > map.len() as u64 {
        return invalid("MTP head source safetensors header length is invalid");
    }
    let data_start = 8 + header_len as usize;
    let data = &map[data_start..];
    let mut entries: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&map[8..data_start])?;
    entries.remove("__metadata__");
    let mut formats = [MatrixFormat::Ptq1; MATRIX_SPECS.len()];
    for (matrix, format) in MATRIX_SPECS.iter().zip(&mut formats) {
        let holds = |format| {
            matrix_tensors(matrix, format)
                .iter()
                .any(|(name, _, _)| entries.contains_key(name))
        };
        *format = match (holds(MatrixFormat::Ptq1), holds(MatrixFormat::Int8)) {
            (true, true) => {
                return invalid(format!(
                    "MTP head source holds {} both as ternary and as int8",
                    matrix.section
                ));
            }
            (false, true) => MatrixFormat::Int8,
            // Holding neither is reported below as a missing ternary tensor.
            _ => MatrixFormat::Ptq1,
        };
    }
    let mut tensors = Vec::with_capacity(2 * MATRIX_SPECS.len() + NORM_SPECS.len());
    for (matrix, &format) in MATRIX_SPECS.iter().zip(&formats) {
        tensors.extend(matrix_tensors(matrix, format));
    }
    for norm in &NORM_SPECS {
        tensors.push((norm.section.to_owned(), "BF16", vec![norm.elements]));
    }
    if entries.len() != tensors.len() {
        return invalid(format!(
            "MTP head source must contain exactly {} tensors, found {}",
            tensors.len(),
            entries.len()
        ));
    }
    let mut ranges = HashMap::with_capacity(tensors.len());
    for (name, dtype, shape) in tensors {
        let value = entries
            .remove(&name)
            .ok_or_else(|| crate::Error::MissingTensor(name.clone()))?;
        let entry: Entry = serde_json::from_value(value)?;
        if entry.dtype != dtype || entry.shape != shape {
            return invalid(format!("MTP head tensor {name} must be {dtype} {shape:?}"));
        }
        let element = match dtype {
            "I8" => 1,
            "F32" => 4,
            _ => 2,
        };
        let [start, end] = entry.data_offsets;
        if end < start
            || end - start != shape.iter().product::<usize>() * element
            || end > data.len()
        {
            return invalid(format!("MTP head tensor {name} has an invalid byte range"));
        }
        ranges.insert(name, start..end);
    }
    let bytes = |name: String| -> &[u8] { &data[ranges[&name].clone()] };
    let mut sections = Vec::with_capacity(2 * MATRIX_SPECS.len() + NORM_SPECS.len());
    for (matrix, format) in MATRIX_SPECS.iter().zip(&formats) {
        match format {
            MatrixFormat::Ptq1 => sections.push(pack_ptq1_matrix(
                bytes(format!("{}.codes", matrix.section)),
                bytes(format!("{}.scales", matrix.section)),
                matrix.rows,
                matrix.columns,
            )?),
            MatrixFormat::Int8 => {
                let weights = bytes(format!("{}.int8", matrix.section));
                let row_scales = bytes(format!("{}.row_scales", matrix.section));
                check_int8_matrix(weights, row_scales)?;
                sections.push(weights.to_vec());
                sections.push(row_scales.to_vec());
            }
        }
    }
    for norm in &NORM_SPECS {
        sections.push(
            fold_norm(bytes(norm.section.to_owned()))?
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        );
    }
    Ok(HeadSections { formats, sections })
}
