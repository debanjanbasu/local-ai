//! The iSTFTNet harmonic source on the CPU (`istftnet.py` `SineGen`,
//! `SourceModuleHnNSF` and `TorchSTFT.transform`), in the reference's own
//! F32 operation order so that the deterministic path matches it closely.
//!
//! F0 (at 80 frames per second) is upsampled to 24 kHz, turned into nine
//! harmonic sines with per-frame phase accumulation, gated by voicing (F0 >
//! 10 Hz), mixed with Gaussian noise, merged by a 9-to-1 linear layer and
//! `tanh`, and analysed by a 20-point STFT (hop 5, periodic Hann, centred
//! with reflection padding) into 11 magnitudes and 11 phases per frame.

// Multiplies and adds stay separate (no fused multiply-add) to follow the
// reference's F32 rounding.
#![expect(clippy::suboptimal_flops)]

const SAMPLE_RATE: f32 = 24_000.0;
/// Samples per F0 frame (generator upsampling 10 x 6 times the iSTFT hop 5).
const UPSAMPLE: usize = 300;
const HARMONICS: usize = 9;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;
const N_FFT: usize = 20;
const HOP: usize = 5;
const BINS: usize = N_FFT / 2 + 1;

/// Noise of the source module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Noise {
    /// No noise and no random initial phase: deterministic, for parity
    /// checks (the reference fixtures' "zero" mode).
    Zero,
    /// Gaussian noise from this seed (what listeners get; the reference uses
    /// `torch.randn`, so samples differ from it in the noise only).
    Seeded(u64),
}

/// xoshiro256** seeded by splitmix64, with Box-Muller normals.
struct Rng {
    s: [u64; 4],
    spare: Option<f32>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
            spare: None,
        }
    }

    const fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let theta = std::f64::consts::TAU * self.uniform();
        self.spare = Some((r * theta.sin()) as f32);
        (r * theta.cos()) as f32
    }
}

/// Harmonic source spectrogram `[22, 300 * f0.len() / 5 + 1]` (rows 0..11
/// magnitudes, 11..22 phases), flattened channel-major.
#[must_use]
pub fn harmonics(f0: &[f32], linear: ([f32; 9], f32), noise: Noise) -> Vec<f32> {
    let frames = f0.len();
    let samples = frames * UPSAMPLE;
    // rad_values = (f0 * h / sr) % 1, then linear downsampling by 300: every
    // 300-sample block holds one value, so the downsampled value is the
    // frame's own (the random initial phase of sample 0 is interpolated away).
    let mut phase = vec![0.0_f32; HARMONICS * frames];
    for h in 0..HARMONICS {
        let mut acc = 0.0_f64;
        for (i, &f) in f0.iter().enumerate() {
            let x = f * (h + 1) as f32 / SAMPLE_RATE;
            let rad = x - x.floor();
            // torch.cumsum on the CPU accumulates F32 in double.
            acc += f64::from(rad);
            phase[h * frames + i] = (acc as f32) * 2.0 * std::f32::consts::PI * UPSAMPLE as f32;
        }
    }
    let scale = 1.0_f32 / UPSAMPLE as f32;
    let mut rng = match noise {
        Noise::Zero => None,
        Noise::Seeded(seed) => Some(Rng::new(seed)),
    };
    let (weights, bias) = linear;
    let mut source = vec![0.0_f32; samples];
    for (n, out) in source.iter_mut().enumerate() {
        let src = (scale * (n as f32 + 0.5) - 0.5).max(0.0);
        let i0 = src as usize;
        let lambda1 = src - i0 as f32;
        let lambda0 = 1.0 - lambda1;
        let i1 = if i0 + 1 < frames { i0 + 1 } else { i0 };
        let f = f0[n / UPSAMPLE];
        let voiced = f > VOICED_THRESHOLD;
        let uv = if voiced { 1.0 } else { 0.0 };
        let noise_amp = uv * NOISE_STD + (1.0 - uv) * SINE_AMP / 3.0;
        let mut merged = bias;
        for h in 0..HARMONICS {
            let p = lambda0 * phase[h * frames + i0] + lambda1 * phase[h * frames + i1];
            let sine = p.sin() * SINE_AMP;
            let noise = rng.as_mut().map_or(0.0, |r| noise_amp * r.normal());
            merged += weights[h] * (sine * uv + noise);
        }
        *out = merged.tanh();
    }
    stft(&source)
}

/// `torch.stft(n_fft 20, hop 5, periodic Hann, center, reflect)` magnitudes
/// and phases, `[22, samples / 5 + 1]`.
fn stft(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let pad = N_FFT / 2;
    let at = |i: isize| -> f32 {
        // Reflection padding (no edge repeat).
        let len = n.cast_signed();
        let mut j = i - pad.cast_signed();
        if j < 0 {
            j = -j;
        }
        if j >= len {
            j = 2 * (len - 1) - j;
        }
        x[j as usize]
    };
    let frames = n / HOP + 1;
    let window: Vec<f32> = (0..N_FFT)
        .map(|m| 0.5 - 0.5 * (std::f32::consts::TAU * m as f32 / N_FFT as f32).cos())
        .collect();
    let mut cos_table = [[0.0_f32; N_FFT]; BINS];
    let mut sin_table = [[0.0_f32; N_FFT]; BINS];
    for k in 0..BINS {
        for m in 0..N_FFT {
            let angle = std::f64::consts::TAU * ((k * m) % N_FFT) as f64 / N_FFT as f64;
            cos_table[k][m] = angle.cos() as f32;
            sin_table[k][m] = angle.sin() as f32;
        }
    }
    let mut out = vec![0.0_f32; 2 * BINS * frames];
    let mut frame = [0.0_f32; N_FFT];
    for f in 0..frames {
        for (m, value) in frame.iter_mut().enumerate() {
            *value = at((f * HOP + m).cast_signed()) * window[m];
        }
        for k in 0..BINS {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            for m in 0..N_FFT {
                re += frame[m] * cos_table[k][m];
                im -= frame[m] * sin_table[k][m];
            }
            out[k * frames + f] = re.hypot(im);
            out[(BINS + k) * frames + f] = im.atan2(re);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_of_a_bin_centred_sine_peaks_in_its_bin() {
        let x: Vec<f32> = (0..400)
            .map(|n| (std::f32::consts::TAU * 3.0 * n as f32 / 20.0).sin())
            .collect();
        let spec = stft(&x);
        let frames = 400 / HOP + 1;
        let mid = frames / 2;
        let peak = (0..BINS)
            .max_by(|a, b| spec[a * frames + mid].total_cmp(&spec[b * frames + mid]))
            .unwrap_or(0);
        assert_eq!(peak, 3);
    }

    #[test]
    fn unvoiced_zero_noise_source_is_constant() {
        let spec = harmonics(&[0.0; 4], ([0.5; 9], 0.25), Noise::Zero);
        let frames = 4 * UPSAMPLE / HOP + 1;
        assert_eq!(spec.len(), 22 * frames);
        let dc = spec[frames / 2];
        // Constant tanh(0.25) through a Hann window: DC magnitude = 10 * tanh(0.25).
        assert!((dc - 10.0 * 0.25_f32.tanh()).abs() < 1e-4, "{dc}");
    }

    #[test]
    fn seeded_noise_is_reproducible() {
        let a = harmonics(&[120.0; 3], ([0.1; 9], 0.0), Noise::Seeded(7));
        let b = harmonics(&[120.0; 3], ([0.1; 9], 0.0), Noise::Seeded(7));
        let c = harmonics(&[120.0; 3], ([0.1; 9], 0.0), Noise::Seeded(8));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
