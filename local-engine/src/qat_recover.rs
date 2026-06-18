use crate::gguf::GGUFType;

/// Dequantize ggml `TQ2_0` (ternary 2-bit) tensors.
///
/// Layout per 256-element block (66 bytes): 64 bytes of packed 2-bit values
/// (`qs`) followed by an f16 scale `d`. Within `qs`, values are emitted in the
/// ggml order `for j in (0,32): for l in 0..4: for m in 0..32` where
/// `q = (qs[j+m] >> (2*l)) & 3`, and each weight is `d * (q - 1)`. Unsloth's
/// "negative scaler" QAT recovery is already baked into the sign/magnitude of
/// `d`, so the standard dequant recovers it.
pub fn dequant_tq2_0(data: &[u8], num_elements: usize) -> crate::Result<Vec<half::f16>> {
    let mut out = Vec::new();
    dequant_tq2_0_into(data, num_elements, &mut out)?;
    Ok(out)
}

/// `dequant_tq2_0` writing into a caller-owned `out` (cleared first) so a scratch
/// buffer can be reused across calls. Matches [`dequant_tq2_0`] exactly.
///
/// # Errors
///
/// Returns an error if the payload cannot be decoded.
pub fn dequant_tq2_0_into(
    data: &[u8],
    num_elements: usize,
    out: &mut Vec<half::f16>,
) -> crate::Result<()> {
    const BLOCK_BYTES: usize = 66;
    const QS_BYTES: usize = 64;
    const ELEMS: usize = 256;

    let blocks = data.len() / BLOCK_BYTES;
    out.clear();
    out.reserve(num_elements.min(blocks * ELEMS));
    for b in 0..blocks {
        let off = b * BLOCK_BYTES;
        let qs = &data[off..off + QS_BYTES];
        let d = half::f16::from_le_bytes([data[off + QS_BYTES], data[off + QS_BYTES + 1]]).to_f32();
        for j in (0..QS_BYTES).step_by(32) {
            for l in 0..4u8 {
                for m in 0..32 {
                    if out.len() >= num_elements {
                        return Ok(());
                    }
                    let q = (qs[j + m] >> (l * 2)) & 0x3;
                    out.push(half::f16::from_f32(d * (f32::from(q) - 1.0)));
                }
            }
        }
    }
    Ok(())
}

/// Dequantize ggml `Q4_0` (18-byte blocks: f16 `d` + 16 packed nibbles).
///
/// 32 values per block; `x = (nibble - 8) * d`. The 16 low nibbles form the
/// first half of the block, the 16 high nibbles the second half.
pub fn dequant_q4_0(data: &[u8], num_elements: usize) -> crate::Result<Vec<half::f16>> {
    let mut out = Vec::new();
    dequant_q4_0_into(data, num_elements, &mut out)?;
    Ok(out)
}

/// `dequant_q4_0` writing into a caller-owned `out` (cleared first) so a scratch
/// buffer can be reused across calls. Matches [`dequant_q4_0`] exactly.
///
/// # Errors
///
/// Returns an error if the payload cannot be decoded.
pub fn dequant_q4_0_into(
    data: &[u8],
    num_elements: usize,
    out: &mut Vec<half::f16>,
) -> crate::Result<()> {
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    let blocks = data.len() / BLOCK_BYTES;
    out.clear();
    out.reserve((blocks * QK).min(num_elements));
    for b in 0..blocks {
        let off = b * BLOCK_BYTES;
        let d = half::f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        let mut block = [0.0_f32; QK];
        for j in 0..QK / 2 {
            let q = data[off + 2 + j];
            block[j] = (f32::from(q & 0x0F) - 8.0) * d;
            block[j + QK / 2] = (f32::from(q >> 4) - 8.0) * d;
        }
        for &v in &block {
            if out.len() >= num_elements {
                return Ok(());
            }
            out.push(half::f16::from_f32(v));
        }
    }
    Ok(())
}

/// Dequantize ggml `Q8_0`: 34-byte blocks (f16 `d` + 32 signed int8 values).
/// Per ggml, `x = qs[j] * d`.
pub fn dequant_q8_0(data: &[u8], num_elements: usize) -> crate::Result<Vec<half::f16>> {
    let mut out = Vec::new();
    dequant_q8_0_into(data, num_elements, &mut out)?;
    Ok(out)
}

/// `dequant_q8_0` writing into a caller-owned `out` (cleared first) so a scratch
/// buffer can be reused across calls. Matches [`dequant_q8_0`] exactly.
///
/// # Errors
///
/// Returns an error if the payload cannot be decoded.
pub fn dequant_q8_0_into(
    data: &[u8],
    num_elements: usize,
    out: &mut Vec<half::f16>,
) -> crate::Result<()> {
    const BLOCK_BYTES: usize = 34;
    const QK: usize = 32;
    let blocks = data.len() / BLOCK_BYTES;
    out.clear();
    out.reserve((blocks * QK).min(num_elements));
    for b in 0..blocks {
        let off = b * BLOCK_BYTES;
        let d = half::f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        for j in 0..QK {
            if out.len() >= num_elements {
                return Ok(());
            }
            let q = i8::from_le_bytes([data[off + 2 + j]]);
            out.push(half::f16::from_f32(f32::from(q) * d));
        }
    }
    Ok(())
}

pub fn dequant(data: &[u8], ty: GGUFType, num_elements: usize) -> crate::Result<Vec<half::f16>> {
    let mut out = Vec::new();
    dequant_into(data, ty, num_elements, &mut out)?;
    Ok(out)
}

/// [`dequant`] writing into a caller-owned `out` (cleared first).
///
/// Lets a scratch buffer be reused across per-token / per-row calls instead of
/// allocating a fresh `Vec` each time. Produces byte-for-byte the same contents
/// as [`dequant`].
///
/// # Errors
///
/// Returns an error if `ty` is unsupported or the payload cannot be decoded.
pub fn dequant_into(
    data: &[u8],
    ty: GGUFType,
    num_elements: usize,
    out: &mut Vec<half::f16>,
) -> crate::Result<()> {
    match ty {
        GGUFType::F16 => {
            out.clear();
            out.extend_from_slice(bytemuck::cast_slice(data));
            Ok(())
        }
        GGUFType::BF16 => {
            out.clear();
            out.reserve(num_elements);
            for chunk in data.chunks_exact(2).take(num_elements) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::f16::from_f32(half::bf16::from_bits(bits).to_f32()));
            }
            Ok(())
        }
        GGUFType::Q4_0 => dequant_q4_0_into(data, num_elements, out),
        GGUFType::Q8_0 => dequant_q8_0_into(data, num_elements, out),
        GGUFType::TQ2_0 => dequant_tq2_0_into(data, num_elements, out),
        _ => Err(crate::Error::InvalidArgument(format!(
            "dequant not implemented for {ty:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::panic,
        clippy::float_cmp
    )]
    use super::*;

    /// Build one 66-byte `ggml` `TQ2_0` block: 64 `qs` bytes + f16 scale.
    fn tq2_0_block(qs: [u8; 64], d: f32) -> Vec<u8> {
        let mut buf = qs.to_vec();
        buf.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        buf
    }

    #[test]
    fn dequant_tq2_0_matches_ggml_layout() {
        // qs[0] packs four 2-bit fields: l0=3, l1=2, l2=1, l3=0  → 0b00_01_10_11 = 27.
        // Every weight is d*(q-1); d = 2.0.
        let mut qs = [0u8; 64];
        qs[0] = 0b0001_1011;
        let buf = tq2_0_block(qs, 2.0);

        let out = dequant_tq2_0(&buf, 256).expect("dequant");
        let f = |i: usize| half::f16::to_f32(out[i]);
        assert_eq!(out.len(), 256);
        // Output index = j_iter*128 + l*32 + m; qs[0] feeds m=0 at each l for j=0.
        assert!(
            (f(0) - 4.0).abs() < 0.01,
            "l0: q=3 -> 2*(3-1)=4, got {}",
            f(0)
        ); // idx 0
        assert!(
            (f(32) - 2.0).abs() < 0.01,
            "l1: q=2 -> 2*(2-1)=2, got {}",
            f(32)
        );
        assert!(
            (f(64) - 0.0).abs() < 0.01,
            "l2: q=1 -> 2*(1-1)=0, got {}",
            f(64)
        );
        assert!(
            (f(96) - (-2.0)).abs() < 0.01,
            "l3: q=0 -> 2*(0-1)=-2, got {}",
            f(96)
        );
        // qs[1]=0 -> q=0 everywhere -> d*(0-1) = -2.0.
        assert!((f(1) - (-2.0)).abs() < 0.01, "got {}", f(1));
    }

    #[test]
    fn dequant_tq2_0_truncates_to_num_elements() {
        let buf = tq2_0_block([0u8; 64], 1.0);
        let out = dequant_tq2_0(&buf, 10).expect("dequant");
        assert_eq!(out.len(), 10);
    }

    /// One `Q4_0` block: f16 scale `d` + 16 packed nibble pairs (lo, hi).
    fn q4_0_block(d: f32, nibbles: [u8; 32]) -> Vec<u8> {
        let mut out = half::f16::from_f32(d).to_le_bytes().to_vec();
        for j in 0..16 {
            out.push((nibbles[j] & 0x0F) | (nibbles[j + 16] << 4));
        }
        out
    }

    /// One `Q8_0` block: f16 scale `d` + 32 signed int8 values.
    fn q8_0_block(d: f32, qs: [i8; 32]) -> Vec<u8> {
        let mut out = half::f16::from_f32(d).to_le_bytes().to_vec();
        out.extend(qs.iter().map(|&q| q as u8));
        out
    }

    #[test]
    fn dequant_into_matches_allocating_for_every_type() {
        // A reused scratch buffer (pre-dirtied) must produce identical output to
        // the allocating `dequant` for each supported type.
        let mut scratch = vec![half::f16::from_f32(123.0); 7];

        let mut qs = [0u8; 64];
        qs[0] = 0b0001_1011;
        qs[3] = 0xA5;
        let tq = tq2_0_block(qs, 2.0);
        dequant_into(&tq, GGUFType::TQ2_0, 256, &mut scratch).expect("into");
        assert_eq!(scratch, dequant(&tq, GGUFType::TQ2_0, 256).expect("alloc"));

        let mut nib = [8u8; 32];
        nib[0] = 15;
        nib[1] = 0;
        nib[17] = 3;
        let q4 = q4_0_block(0.5, nib);
        dequant_into(&q4, GGUFType::Q4_0, 32, &mut scratch).expect("into");
        assert_eq!(scratch, dequant(&q4, GGUFType::Q4_0, 32).expect("alloc"));

        let mut qi = [1i8; 32];
        qi[0] = -7;
        qi[5] = 42;
        let q8 = q8_0_block(0.25, qi);
        dequant_into(&q8, GGUFType::Q8_0, 32, &mut scratch).expect("into");
        assert_eq!(scratch, dequant(&q8, GGUFType::Q8_0, 32).expect("alloc"));

        // F16 passthrough.
        let f16_raw: Vec<u8> = [1.0f32, -2.5, 3.25]
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        dequant_into(&f16_raw, GGUFType::F16, 3, &mut scratch).expect("into");
        assert_eq!(scratch, dequant(&f16_raw, GGUFType::F16, 3).expect("alloc"));

        // Truncation to num_elements behaves identically.
        dequant_into(&q8, GGUFType::Q8_0, 10, &mut scratch).expect("into");
        assert_eq!(scratch.len(), 10);
        assert_eq!(scratch, dequant(&q8, GGUFType::Q8_0, 10).expect("alloc"));
    }

    #[test]
    fn dequant_bf16_to_f16() {
        let values = [0.5_f32, -2.0, 3.25];
        let mut raw = Vec::new();
        for value in values {
            raw.extend_from_slice(&half::bf16::from_f32(value).to_le_bytes());
        }

        let out = dequant(&raw, GGUFType::BF16, values.len()).expect("dequant");
        assert_eq!(out.len(), values.len());
        for (got, want) in out.iter().zip(values) {
            assert!((got.to_f32() - want).abs() < 0.01);
        }
    }
}
