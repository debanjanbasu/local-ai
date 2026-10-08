//! Exporting the trained head as the artifact an install ships.
//!
//! `local-ai bonsai --export mtp-head=<dir>` packs
//! `<dir>/model_mtp_ternary.safetensors` into `<dir>/mtp-head-ptq1-v1.bin`:
//! no GPU and no target checkpoint needed.

use std::path::{Path, PathBuf};

use super::format::{HeadFile, MTP_HEAD_ARTIFACT, write_artifact};
use super::ternary::{MTP_TERNARY_SOURCE, MatrixFormat, head_sections, head_spec};
use super::weights::MATRIX_SPECS;

/// What one exported head contains.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct MtpHeadArtifact {
    /// The artifact that was written.
    pub path: PathBuf,
    /// The safetensors it was packed from.
    pub source: PathBuf,
    /// Matrices stored as per-row int8; every other matrix is `PTQ1_0`.
    pub int8_matrices: Vec<String>,
    /// Sections in artifact order; the header names every one of them.
    pub sections: usize,
    /// Section bytes with the inter-section padding skipped: what a load hashes
    /// and uploads, and the artifact's content identity.
    pub payload_bytes: u64,
    /// Bytes on disk, header page and padding included.
    pub file_bytes: u64,
    pub payload_sha256: String,
}

impl MtpHeadArtifact {
    fn exported(
        source: &Path,
        int8_matrices: Vec<String>,
        sections: usize,
        head: HeadFile,
    ) -> crate::Result<Self> {
        Ok(Self {
            path: head.path,
            source: source.to_owned(),
            int8_matrices,
            sections,
            payload_bytes: widen(head.payload_bytes, "MTP head payload exceeds 64 bits")?,
            file_bytes: widen(head.file_bytes, "MTP head file exceeds 64 bits")?,
            payload_sha256: head.payload_sha256,
        })
    }
}

fn widen(bytes: usize, message: &'static str) -> crate::Result<u64> {
    u64::try_from(bytes).map_err(|_| crate::Error::InvalidFormat(message.into()))
}

/// Pack the trained head at `<directory>/model_mtp_ternary.safetensors` into
/// `<directory>/mtp-head-ptq1-v1.bin`, stored verbatim.
///
/// Ternary codes and scales are packed into the target's `PTQ1_0` blocks bit
/// for bit, int8 matrices and their row scales are stored as given, norms are
/// folded to `1 + w`, and the file records its section names (which say each
/// matrix's format) and payload digest, which every load re-derives and checks.
/// Install it in `models/bonsai2-27b-mtp/`.
pub fn export_head(directory: &Path) -> crate::Result<MtpHeadArtifact> {
    let source = directory.join(MTP_TERNARY_SOURCE);
    let packed = head_sections(&source)?;
    let spec = head_spec(&packed.formats);
    let int8_matrices = MATRIX_SPECS
        .iter()
        .zip(&packed.formats)
        .filter(|(_, format)| **format == MatrixFormat::Int8)
        .map(|(matrix, _)| matrix.section.to_owned())
        .collect();
    let count = packed.sections.len();
    let head = write_artifact(&directory.join(MTP_HEAD_ARTIFACT), &spec, &packed.sections)?;
    MtpHeadArtifact::exported(&source, int8_matrices, count, head)
}

/// Rewrite the head at `source` into `destination` with the matrices at
/// `int8` (indices into `MATRIX_SPECS`) requantized per row to int8 from their
/// dequantized `PTQ1_0` weights (`row_scale = absmax / 127`), every other
/// section copied. The result is a mixed head whose int8 matrices closely
/// approximate the ternary ones, for tests comparing the two formats' kernels.
#[cfg(test)]
pub fn requantize_head(source: &Path, destination: &Path, int8: &[usize]) -> crate::Result<()> {
    use super::format::read_head_with;
    use super::ternary::formats_from_names;

    let bind = |names: &[&str]| formats_from_names(names).map(|formats| head_spec(&formats));
    let head = read_head_with(source, &bind).map_err(crate::Error::InvalidFormat)?;
    let mut formats = formats_from_names(
        &head
            .spec
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
    )
    .map_err(crate::Error::InvalidFormat)?;
    let section = |index: usize| {
        let (offset, length) = head.sections[index];
        &head.map[offset..offset + length]
    };
    let mut sections = Vec::with_capacity(head.spec.len() + int8.len());
    let mut at = 0;
    for (index, (matrix, format)) in MATRIX_SPECS.iter().zip(&mut formats).enumerate() {
        let width = if *format == MatrixFormat::Int8 { 2 } else { 1 };
        if !int8.contains(&index) || *format == MatrixFormat::Int8 {
            sections.extend((at..at + width).map(|i| section(i).to_vec()));
            at += width;
            continue;
        }
        let packed = section(at);
        at += 1;
        let row_bytes = matrix.columns / 128 * 28;
        let mut weights = Vec::with_capacity(matrix.rows * matrix.columns);
        let mut scales = Vec::with_capacity(matrix.rows * 4);
        let mut row = vec![0.0_f32; matrix.columns];
        for packed_row in packed.chunks_exact(row_bytes) {
            local_metal::bonsai::decode_ptq1_row(packed_row, &mut row)?;
            let peak = row.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
            weights.extend(
                row.iter()
                    .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8 as u8),
            );
            scales.extend_from_slice(&scale.to_le_bytes());
        }
        sections.push(weights);
        sections.push(scales);
        *format = MatrixFormat::Int8;
    }
    sections.extend((at..head.spec.len()).map(|i| section(i).to_vec()));
    write_artifact(destination, &head_spec(&formats), &sections)?;
    Ok(())
}
