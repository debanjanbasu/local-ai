//! `TurboQuant` — data-oblivious, near-optimal KV-cache vector quantization.
//!
//! After Zandieh et al., "`TurboQuant`: Online Vector Quantization with
//! Near-optimal Distortion Rate" (arXiv:2504.19874).
//!
//! The algorithm has three ideas, all implemented here:
//!
//! 1. **Random rotation.** A randomized Hadamard transform `R = FWHT ∘ sign`
//!    rotates each vector. In high dimension this makes the coordinates of the
//!    (normalized) vector near-i.i.d. and concentrated (≈ `N(0, 1/d)`), so a
//!    single fixed scalar quantizer is near-optimal for *every* coordinate —
//!    with **no per-block scale/zero-point stored** (the usual 1–2 bit
//!    overhead). Only the vector norm is kept. Because `R` is orthonormal,
//!    `⟨q, x⟩ = ⟨R q, R x⟩`, so rotation is free for attention.
//!
//! 2. **MSE-optimal per-coordinate quantizer.** A Lloyd–Max codebook for the
//!    standard normal (computed once, data-obliviously) quantizes each rotated,
//!    norm-normalized coordinate. Storage per vector = `bits·d` code bits + one
//!    `f16` norm.
//!
//! This module is pure CPU/`f32` and Metal-free so the math can be unit-tested
//! in isolation; the live cache wires the encode/decode in.

use half::f16;

/// In-place normalized fast Walsh–Hadamard transform. `v.len()` must be a power
/// of two. Normalized (divided by `√n`) so the transform is orthonormal and is
/// its own inverse.
#[allow(clippy::many_single_char_names)]
pub fn fwht_normalized(v: &mut [f32]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two(), "FWHT length must be a power of two");
    let mut len = 1;
    while len < n {
        let mut i = 0;
        while i < n {
            for j in i..i + len {
                let a = v[j];
                let b = v[j + len];
                v[j] = a + b;
                v[j + len] = a - b;
            }
            i += len << 1;
        }
        len <<= 1;
    }
    let inv = 1.0 / (n as f32).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Deterministic `±1` sign flips for the randomized Hadamard rotation.
///
/// The rotation is `R(x) = FWHT(sign ⊙ x)` with inverse `R⁻¹(y) = sign ⊙
/// FWHT(y)` (both because the normalized FWHT and the sign flip are involutions).
#[derive(Debug, Clone)]
pub struct Rotation {
    signs: Vec<f32>,
}

impl Rotation {
    /// Build a rotation for vectors of length `dim` (must be a power of two)
    /// from a fixed `seed`, so encode and decode agree.
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut state = seed | 1;
        let signs = (0..dim)
            .map(|_| {
                // splitmix64
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                if z & 1 == 0 { 1.0 } else { -1.0 }
            })
            .collect();
        Self { signs }
    }

    /// Dimensionality this rotation operates on.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.signs.len()
    }

    /// The `±1` sign flips, one per coordinate.
    #[must_use]
    pub fn signs(&self) -> &[f32] {
        &self.signs
    }

    /// Apply `R(x) = FWHT(sign ⊙ x)` in place.
    pub fn apply(&self, v: &mut [f32]) {
        for (x, &s) in v.iter_mut().zip(&self.signs) {
            *x *= s;
        }
        fwht_normalized(v);
    }

    /// Apply the inverse `R⁻¹(y) = sign ⊙ FWHT(y)` in place.
    pub fn apply_inverse(&self, v: &mut [f32]) {
        fwht_normalized(v);
        for (x, &s) in v.iter_mut().zip(&self.signs) {
            *x *= s;
        }
    }
}

/// Symmetric Lloyd–Max reconstruction levels for the standard normal, computed
/// data-obliviously. Returns `2^bits` increasing levels.
#[must_use]
pub fn lloyd_max_gaussian(bits: u8) -> Vec<f32> {
    // Fine grid integration of N(0,1) over [-6, 6].
    const N: usize = 8192;
    const LO: f32 = -6.0;
    const HI: f32 = 6.0;
    let k = 1usize << bits;
    let dx = (HI - LO) / N as f32;
    let xs: Vec<f32> = (0..N).map(|i| (i as f32 + 0.5).mul_add(dx, LO)).collect();
    let pdf: Vec<f32> = xs
        .iter()
        .map(|&x| {
            (-0.5 * x * x).exp()
                * std::f32::consts::FRAC_1_SQRT_2
                * std::f32::consts::FRAC_2_SQRT_PI
                / 2.0
        })
        .collect();

    // Initialize centroids at Gaussian-spread positions.
    let mut levels: Vec<f32> = (0..k)
        .map(|i| {
            let t = (i as f32 + 0.5) / k as f32; // (0,1)
            // map through a rough inverse-cdf-ish spread
            6.0 * (t - 0.5)
        })
        .collect();

    for _ in 0..100 {
        let mut sum = vec![0.0f32; k];
        let mut wsum = vec![0.0f32; k];
        for (&x, &w) in xs.iter().zip(&pdf) {
            // nearest centroid
            let mut best = 0usize;
            let mut bd = f32::INFINITY;
            for (j, &c) in levels.iter().enumerate() {
                let d = (x - c).abs();
                if d < bd {
                    bd = d;
                    best = j;
                }
            }
            sum[best] = w.mul_add(x, sum[best]);
            wsum[best] += w;
        }
        for j in 0..k {
            if wsum[j] > 1e-12 {
                levels[j] = sum[j] / wsum[j];
            }
        }
    }
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    levels
}

/// A configured `TurboQuant` codec for vectors of a fixed dimension and bit-width.
pub struct TurboQuant {
    rotation: Rotation,
    /// Reconstruction levels scaled for the post-rotation coordinate std (`1/√d`).
    levels: Vec<f32>,
    bits: u8,
    dim: usize,
}

/// One quantized vector: packed codes plus the stored Euclidean norm.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedVector {
    pub codes: Vec<u8>,
    pub norm: f16,
}

impl TurboQuant {
    /// Build a codec for `dim`-length vectors at `bits` per coordinate.
    #[must_use]
    pub fn new(dim: usize, bits: u8, seed: u64) -> Self {
        let scale = 1.0 / (dim as f32).sqrt();
        let levels = lloyd_max_gaussian(bits)
            .into_iter()
            .map(|l| l * scale)
            .collect();
        Self {
            rotation: Rotation::new(dim, seed),
            levels,
            bits,
            dim,
        }
    }

    /// Bytes needed to store one vector's packed codes.
    #[must_use]
    pub const fn code_bytes(&self) -> usize {
        (self.dim * self.bits as usize).div_ceil(8)
    }

    /// Bits per coordinate.
    #[must_use]
    pub const fn bits(&self) -> u8 {
        self.bits
    }

    /// Reconstruction levels in rotated space (pre-scaled by `1/√dim`),
    /// `2^bits` increasing values — upload these to the GPU decode kernel.
    #[must_use]
    pub fn levels(&self) -> &[f32] {
        &self.levels
    }

    /// The rotation's `±1` sign flips — upload these to the GPU decode kernel.
    #[must_use]
    pub fn signs(&self) -> &[f32] {
        self.rotation.signs()
    }

    /// Quantize `x`: rotate, normalize, per-coordinate nearest-level, pack.
    #[must_use]
    pub fn encode(&self, x: &[f32]) -> QuantizedVector {
        let mut codes = vec![0u8; self.code_bytes()];
        let mut rotated = Vec::with_capacity(self.dim);
        let norm = self.encode_into(x, &mut rotated, &mut codes);
        QuantizedVector { codes, norm }
    }

    /// [`Self::encode`] writing packed codes directly into `dst_codes` and using
    /// `rotated` as a reusable scratch buffer (cleared first), returning the
    /// stored norm. No per-call allocation when both buffers are reused across
    /// heads/tokens. Produces byte-for-byte the same codes/norm as [`Self::encode`].
    ///
    /// `dst_codes` must be exactly [`Self::code_bytes`] long.
    pub fn encode_into(&self, x: &[f32], rotated: &mut Vec<f32>, dst_codes: &mut [u8]) -> f16 {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(dst_codes.len(), self.code_bytes());
        let norm = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        rotated.clear();
        rotated.extend_from_slice(x);
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for v in rotated.iter_mut() {
                *v *= inv;
            }
        }
        self.rotation.apply(rotated);

        // LSB-first bit packing matching `BitWriter`, straight into `dst_codes`.
        for b in dst_codes.iter_mut() {
            *b = 0;
        }
        let mut acc = 0u32;
        let mut nbits = 0u32;
        let mut idx = 0usize;
        for &v in rotated.iter() {
            acc |= self.nearest_level(v) << nbits;
            nbits += u32::from(self.bits);
            while nbits >= 8 {
                dst_codes[idx] = (acc & 0xFF) as u8;
                idx += 1;
                acc >>= 8;
                nbits -= 8;
            }
        }
        if nbits > 0 {
            dst_codes[idx] = (acc & 0xFF) as u8;
        }
        f16::from_f32(norm)
    }

    /// Reconstruct an `f32` vector from a [`QuantizedVector`].
    #[must_use]
    pub fn decode(&self, q: &QuantizedVector) -> Vec<f32> {
        let mut reader = BitReader::new(&q.codes);
        let mut y: Vec<f32> = (0..self.dim)
            .map(|_| self.levels[reader.pop(self.bits) as usize])
            .collect();
        self.rotation.apply_inverse(&mut y);
        let r = q.norm.to_f32();
        for v in &mut y {
            *v *= r;
        }
        y
    }

    fn nearest_level(&self, v: f32) -> u32 {
        let mut best = 0u32;
        let mut bd = f32::INFINITY;
        for (i, &l) in self.levels.iter().enumerate() {
            let d = (v - l).abs();
            if d < bd {
                bd = d;
                best = i as u32;
            }
        }
        best
    }
}

/// LSB-first bit reader matching the packing in [`TurboQuant::encode_into`].
struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    acc: u32,
    nbits: u32,
}

impl<'a> BitReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            acc: 0,
            nbits: 0,
        }
    }
    fn pop(&mut self, bits: u8) -> u32 {
        while self.nbits < u32::from(bits) {
            let byte = self.bytes.get(self.pos).copied().unwrap_or(0);
            self.acc |= u32::from(byte) << self.nbits;
            self.nbits += 8;
            self.pos += 1;
        }
        let mask = (1u32 << bits) - 1;
        let v = self.acc & mask;
        self.acc >>= bits;
        self.nbits -= u32::from(bits);
        v
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    fn seeded_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 31;
                ((z >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn fwht_is_orthonormal_involution() {
        let mut v = vec![1.0, -2.0, 3.0, 0.5, -1.0, 2.0, -0.5, 4.0];
        let orig = v.clone();
        let norm0: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
        fwht_normalized(&mut v);
        let norm1: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm0 - norm1).abs() < 1e-4, "FWHT must preserve norm");
        fwht_normalized(&mut v); // applying twice returns the original
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn rotation_inverse_recovers_input() {
        let r = Rotation::new(256, 42);
        let x = seeded_vec(256, 7);
        let mut y = x.clone();
        r.apply(&mut y);
        r.apply_inverse(&mut y);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn lloyd_max_levels_symmetric_increasing() {
        for bits in [2u8, 3, 4] {
            let l = lloyd_max_gaussian(bits);
            assert_eq!(l.len(), 1 << bits);
            for w in l.windows(2) {
                assert!(w[1] > w[0], "levels must be strictly increasing");
            }
            // symmetric about 0
            let n = l.len();
            for i in 0..n / 2 {
                assert!(
                    (l[i] + l[n - 1 - i]).abs() < 0.05,
                    "levels should be symmetric"
                );
            }
        }
    }

    #[test]
    fn encode_decode_low_relative_error_at_4bit() {
        let dim = 256;
        let tq = TurboQuant::new(dim, 4, 123);
        let x = seeded_vec(dim, 99);
        let q = tq.encode(&x);
        assert_eq!(q.codes.len(), tq.code_bytes());
        let x_hat = tq.decode(&q);
        let num: f32 = x.iter().zip(&x_hat).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = x.iter().map(|a| a * a).sum();
        let rel = (num / den).sqrt();
        // 4-bit TurboQuant is near-lossless: well under 10% relative error.
        assert!(rel < 0.10, "relative reconstruction error too high: {rel}");
    }

    #[test]
    fn encode_into_matches_allocating_encode() {
        // Across several bit widths, encode_into must produce byte-identical codes
        // and norm to the allocating encode, including with reused (dirty) scratch.
        let dim = 256;
        let mut rotated = vec![999.0f32; dim + 5];
        for bits in [2u8, 3, 4] {
            let tq = TurboQuant::new(dim, bits, 7);
            for seed in [1u64, 42, 1000] {
                let x = seeded_vec(dim, seed);
                let expected = tq.encode(&x);
                let mut codes = vec![0xAAu8; tq.code_bytes()];
                let norm = tq.encode_into(&x, &mut rotated, &mut codes);
                assert_eq!(
                    codes, expected.codes,
                    "codes mismatch bits={bits} seed={seed}"
                );
                assert_eq!(norm, expected.norm, "norm mismatch bits={bits} seed={seed}");
            }
        }
    }

    #[test]
    fn encode_into_zero_vector_matches() {
        let dim = 128;
        let tq = TurboQuant::new(dim, 4, 3);
        let x = vec![0.0f32; dim];
        let expected = tq.encode(&x);
        let mut rotated = Vec::new();
        let mut codes = vec![0u8; tq.code_bytes()];
        let norm = tq.encode_into(&x, &mut rotated, &mut codes);
        assert_eq!(codes, expected.codes);
        assert_eq!(norm, expected.norm);
    }

    #[test]
    fn inner_product_preserved() {
        let dim = 256;
        let tq = TurboQuant::new(dim, 4, 5);
        let k = seeded_vec(dim, 1);
        let qy = seeded_vec(dim, 2);
        let k_hat = tq.decode(&tq.encode(&k));
        let exact: f32 = qy.iter().zip(&k).map(|(a, b)| a * b).sum();
        let approx: f32 = qy.iter().zip(&k_hat).map(|(a, b)| a * b).sum();
        assert!(
            (exact - approx).abs() / exact.abs().max(1e-3) < 0.15,
            "{exact} vs {approx}"
        );
    }

    #[test]
    fn code_bytes_match_bitwidth() {
        assert_eq!(TurboQuant::new(256, 4, 0).code_bytes(), 128);
        assert_eq!(TurboQuant::new(256, 3, 0).code_bytes(), 96);
        assert_eq!(TurboQuant::new(256, 2, 0).code_bytes(), 64);
    }
}
