use crate::gguf::{GGUFModel, GGUFType};
use crate::layer::QuantWeight;
use crate::qat_recover;
use local_metal::buffer::MetalBuffer;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use std::collections::HashMap;

pub struct WeightMap {
    pub buffers: HashMap<String, QuantWeight>,
    /// The per-layer token-embedding table (`per_layer_token_embd.weight`), kept
    /// in its original quantized form on the CPU rather than dequantized into
    /// VRAM. For Gemma 4 E2B this single tensor is ~2.35 B parameters; only the
    /// current token's row is ever needed per forward step, so it is fetched and
    /// dequantized on demand. See [`QuantRowTable`].
    pub ple_table: Option<QuantRowTable>,
}

/// Whether quantized-at-rest weights are enabled (`LOCAL_AI_QWEIGHTS=0`
/// forces the legacy dequantize-everything-to-FP16 path).
#[must_use]
pub fn quant_weights_enabled() -> bool {
    std::env::var("LOCAL_AI_QWEIGHTS").ok().as_deref() != Some("0")
}

/// Whether a tensor should stay in its quantized GGUF block format in device
/// memory: 2-D matrices in a format with a device matvec/matmul kernel.
#[must_use]
pub const fn keep_quantized(ty: GGUFType, shape: &[u64]) -> bool {
    shape.len() == 2 && matches!(ty, GGUFType::Q4_0 | GGUFType::TQ2_0)
}

/// `(rows, cols)` for the engine's `[out, in]` weight convention: GGUF `ne0`
/// is the contiguous input dimension, `ne1` the output dimension.
fn weight_dims(shape: &[u64]) -> (u32, u32) {
    let cols = shape.first().copied().unwrap_or(1) as u32;
    let rows = shape.get(1).copied().unwrap_or(1) as u32;
    (rows, cols)
}

/// Build a device-resident [`QuantWeight`] from a raw GGUF tensor payload:
/// verbatim upload for formats with device kernels, FP16 expansion otherwise.
///
/// # Errors
///
/// Returns an error if the payload cannot be decoded or uploaded.
pub fn weight_from_raw(
    device: &ProtocolObject<dyn MTLDevice>,
    raw_bytes: &[u8],
    ty: GGUFType,
    shape: &[u64],
    name: &str,
    allow_quant: bool,
) -> crate::Result<QuantWeight> {
    let (rows, cols) = weight_dims(shape);
    if std::env::var_os("LOCAL_AI_WFMT").is_some() {
        eprintln!(
            "[wfmt] {name} ty={ty:?} rows={rows} cols={cols} kept={}",
            allow_quant && keep_quantized(ty, shape)
        );
    }
    if allow_quant && keep_quantized(ty, shape) {
        let buf = MetalBuffer::from_slice(device, raw_bytes)
            .map_err(|e| crate::Error::InvalidArgument(format!("MetalBuffer for {name}: {e}")))?;
        return QuantWeight::quantized(buf, ty, rows, cols);
    }
    let num_elements: usize = shape.iter().map(|&d| d as usize).product();
    let f16_data = decode_tensor_to_f16(raw_bytes, ty, num_elements)?;
    let buf = MetalBuffer::from_slice(device, &f16_data)
        .map_err(|e| crate::Error::InvalidArgument(format!("MetalBuffer for {name}: {e}")))?;
    Ok(QuantWeight::f16(buf, rows, cols))
}

/// Name of the per-layer token-embedding tensor that is streamed on demand.
pub const PLE_TABLE_NAME: &str = "per_layer_token_embd.weight";

/// A row-addressable quantized 2-D tensor held on the CPU.
///
/// Stored verbatim in its GGUF block format so that any single row can be
/// dequantized in isolation without materializing the whole tensor. This is the
/// mechanism that lets Gemma 4 E2B exploit its "embed-2B" structure: the giant
/// per-layer embedding table never enters VRAM; one row per decoded token does.
pub struct QuantRowTable {
    raw: Vec<u8>,
    ty: GGUFType,
    /// Elements per row (the contiguous, fastest-varying dimension).
    cols: usize,
    /// Number of rows (e.g. the vocabulary size).
    rows: usize,
    /// Bytes occupied by one quantized row.
    row_bytes: usize,
}

impl QuantRowTable {
    /// Build a row table from a verbatim quantized payload.
    ///
    /// `cols` is the row width (must be a whole number of quantization blocks)
    /// and `rows` the row count. The payload must be large enough to hold
    /// `rows * row_bytes`.
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is not block-aligned or the payload is short.
    pub fn new(raw: Vec<u8>, ty: GGUFType, cols: usize, rows: usize) -> crate::Result<Self> {
        let block = ty.block_size();
        if block == 0 || !cols.is_multiple_of(block) {
            return Err(crate::Error::InvalidFormat(format!(
                "QuantRowTable: cols {cols} not a multiple of {ty:?} block size {block}"
            )));
        }
        let row_bytes = ty.tensor_bytes(cols);
        if raw.len() < rows.saturating_mul(row_bytes) {
            return Err(crate::Error::InvalidFormat(format!(
                "QuantRowTable: payload {} < rows*row_bytes {}",
                raw.len(),
                rows * row_bytes
            )));
        }
        Ok(Self {
            raw,
            ty,
            cols,
            rows,
            row_bytes,
        })
    }

    /// Number of elements in one row.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes of the CPU-resident quantized payload.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.raw.len()
    }

    /// Dequantize a single row to FP16. Out-of-range rows clamp to the last row.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying block payload cannot be decoded.
    pub fn dequant_row(&self, row: usize) -> crate::Result<Vec<half::f16>> {
        let mut out = Vec::new();
        self.dequant_row_into(row, &mut out)?;
        Ok(out)
    }

    /// [`Self::dequant_row`] writing into a caller-owned `out` (cleared first) so
    /// a single scratch buffer can be reused across per-token row gathers.
    /// Produces identical contents to [`Self::dequant_row`].
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying block payload cannot be decoded.
    pub fn dequant_row_into(&self, row: usize, out: &mut Vec<half::f16>) -> crate::Result<()> {
        let row = row.min(self.rows.saturating_sub(1));
        let off = row * self.row_bytes;
        qat_recover::dequant_into(
            &self.raw[off..off + self.row_bytes],
            self.ty,
            self.cols,
            out,
        )
    }
}

/// Decode a raw tensor payload of GGUF type `ty` into FP16.
///
/// F32 is narrowed to F16, F16 is taken as-is, and every quantized type is
/// routed through [`qat_recover::dequant`]. This is the single source of truth
/// shared by the GGUF and `.lma` weight loaders.
///
/// # Errors
///
/// Returns an error if the quantized payload cannot be dequantized.
pub fn decode_tensor_to_f16(
    raw_bytes: &[u8],
    ty: GGUFType,
    num_elements: usize,
) -> crate::Result<Vec<half::f16>> {
    let f16_data = match ty {
        GGUFType::F32 => {
            let f32s: &[f32] = bytemuck::cast_slice(raw_bytes);
            f32s.iter().map(|&v| half::f16::from_f32(v)).collect()
        }
        GGUFType::F16 => bytemuck::cast_slice(raw_bytes).to_vec(),
        GGUFType::BF16 => {
            // bf16 is the top 16 bits of an f32; widen then narrow to f16.
            let u16s: &[u16] = bytemuck::cast_slice(raw_bytes);
            u16s.iter()
                .map(|&b| half::f16::from_f32(f32::from_bits(u32::from(b) << 16)))
                .collect()
        }
        _ => qat_recover::dequant(raw_bytes, ty, num_elements)?,
    };
    Ok(f16_data)
}

pub fn load_all_weights(
    gguf: &GGUFModel,
    device: &ProtocolObject<dyn MTLDevice>,
) -> crate::Result<WeightMap> {
    let allow_quant = quant_weights_enabled();
    let mut buffers = HashMap::new();
    let mut ple_table = None;
    for (name, info) in &gguf.tensors {
        let raw_bytes = gguf
            .tensor_data::<u8>(name)
            .ok_or_else(|| crate::Error::InvalidFormat(format!("tensor {name} has no data")))?;
        // Stream the giant per-layer embedding table on demand instead of
        // dequantizing all ~2.35 B params into VRAM (see `QuantRowTable`).
        if name == PLE_TABLE_NAME && info.shape.len() == 2 {
            let cols = info.shape[0] as usize;
            let rows = info.shape[1] as usize;
            ple_table = Some(QuantRowTable::new(raw_bytes.to_vec(), info.ty, cols, rows)?);
            continue;
        }
        let weight = weight_from_raw(device, raw_bytes, info.ty, &info.shape, name, allow_quant)?;
        buffers.insert(name.clone(), weight);
    }
    Ok(WeightMap { buffers, ple_table })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::float_cmp)]
    use super::*;

    /// One `Q4_0` block: f16 scale `d` followed by 16 packed nibble pairs.
    fn q4_0_block(d: f32, nibbles: [u8; 32]) -> Vec<u8> {
        let mut out = half::f16::from_f32(d).to_le_bytes().to_vec();
        for j in 0..16 {
            out.push((nibbles[j] & 0x0F) | (nibbles[j + 16] << 4));
        }
        out
    }

    #[test]
    fn dequant_row_matches_full_dequant_and_is_row_independent() {
        // Two rows of 32 Q4_0 values each, with distinct scales/nibbles.
        let row0 = q4_0_block(2.0, [8; 32]); // all (8-8)*2 = 0
        let mut nib1 = [8u8; 32];
        nib1[0] = 15; // (15-8)*0.5 = 3.5
        nib1[1] = 0; // (0-8)*0.5 = -4.0
        let row1 = q4_0_block(0.5, nib1);

        let mut raw = row0;
        raw.extend_from_slice(&row1);
        let table = QuantRowTable::new(raw, GGUFType::Q4_0, 32, 2).expect("table");

        assert_eq!(table.cols(), 32);
        let r0 = table.dequant_row(0).expect("row0");
        let r1 = table.dequant_row(1).expect("row1");

        // Row 0 is all zeros; row 1 has the two edited leading values.
        assert!(r0.iter().all(|v| v.to_f32() == 0.0));
        assert_eq!(r1[0].to_f32(), 3.5);
        assert_eq!(r1[1].to_f32(), -4.0);

        // Out-of-range rows clamp to the last row (no panic).
        assert_eq!(table.dequant_row(99).expect("clamp"), r1);
    }

    #[test]
    fn rejects_non_block_aligned_cols() {
        let raw = vec![0u8; 18];
        assert!(QuantRowTable::new(raw, GGUFType::Q4_0, 30, 1).is_err());
    }
}
