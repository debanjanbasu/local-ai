//! Log-mel features exactly as transformers' `Gemma4AudioFeatureExtractor`
//! (5.19.0) computes them for one clip at a time.
//!
//! 16 kHz mono input, truncated to 30 s (`max_length` 480,000 samples) and
//! right-padded with zeros to a multiple of 128 samples; 160 zeros are
//! prepended ("semicausal" padding) and 321-sample windows are taken every
//! 160 samples. Each frame's first 320 samples are multiplied by a periodic
//! Hann window (F32 values, F32 products, as numpy does), zero-padded to a
//! 512-point real FFT (F64), and the magnitudes go through 128 HTK-scale
//! triangular mel filters over 0–8 kHz (no normalization); the feature is
//! `ln(mel + 0.001)` rounded to F32. A frame is valid only if its whole
//! window lies inside the real (unpadded) audio; invalid frames are zero.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

/// Sample rate the model expects.
pub const SAMPLE_RATE: u32 = 16_000;
/// Mel bins per frame.
pub const MEL_BINS: usize = 128;
/// Longest clip, in samples (30 s); longer input is refused by the caller.
pub const MAX_SAMPLES: usize = 480_000;

const FRAME: usize = 320;
const HOP: usize = 160;
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
const PAD_MULTIPLE: usize = 128;
const MEL_FLOOR: f64 = 1e-3;
const MAX_FREQUENCY: f64 = 8_000.0;

/// The log-mel features of one clip.
pub struct LogMel {
    /// `[frames, 128]` row-major; frames at or past `valid` are zero.
    pub features: Vec<f32>,
    pub frames: usize,
    /// Leading frames whose analysis window is real audio.
    pub valid: usize,
}

/// Frame count and valid-frame count for a clip of `samples` samples
/// (`samples <= MAX_SAMPLES`).
#[must_use]
pub const fn frame_counts(samples: usize) -> (usize, usize) {
    let padded = samples.next_multiple_of(PAD_MULTIPLE) + FRAME / 2;
    let frames = if padded > FRAME {
        (padded - FRAME - 1) / HOP + 1
    } else {
        0
    };
    // Frame i is valid iff its last sample, i*HOP + FRAME (in padded
    // coordinates) - FRAME/2, is a real sample: i*HOP + HOP < samples.
    let valid = if samples > HOP {
        (samples - HOP - 1) / HOP + 1
    } else {
        0
    };
    (frames, if valid < frames { valid } else { frames })
}

/// Audio soft tokens of a clip with `valid` valid frames: the two stride-2
/// subsampling convolutions keep every second frame mask entry twice.
#[must_use]
pub const fn audio_tokens(valid: usize) -> usize {
    valid.div_ceil(2).div_ceil(2)
}

fn hertz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hertz(mel: f64) -> f64 {
    700.0 * (10_f64.powf(mel / 2595.0) - 1.0)
}

/// `numpy.linspace(first, last, count)`: `i * step + first`, rounded
/// separately as numpy does (no fused multiply-add).
#[expect(clippy::suboptimal_flops)]
fn linspace(first: f64, last: f64, count: usize) -> Vec<f64> {
    let increment = (last - first) / (count - 1) as f64;
    (0..count)
        .map(|i| {
            if i == count - 1 {
                last
            } else {
                i as f64 * increment + first
            }
        })
        .collect()
}

/// transformers' `mel_filter_bank(257, 128, 0, 8000, 16000, norm=None,
/// mel_scale="htk")`, `[257][128]`.
fn mel_filters() -> Vec<f64> {
    let mels = linspace(hertz_to_mel(0.0), hertz_to_mel(MAX_FREQUENCY), MEL_BINS + 2);
    let filter: Vec<f64> = mels.iter().map(|m| mel_to_hertz(*m)).collect();
    let fft_freqs = linspace(0.0, f64::from(SAMPLE_RATE / 2), BINS);
    let mut out = vec![0.0; BINS * MEL_BINS];
    for (bin, freq) in fft_freqs.iter().enumerate() {
        for m in 0..MEL_BINS {
            let down = -(filter[m] - freq) / (filter[m + 1] - filter[m]);
            let up = (filter[m + 2] - freq) / (filter[m + 2] - filter[m + 1]);
            out[bin * MEL_BINS + m] = down.min(up).max(0.0);
        }
    }
    out
}

/// Periodic Hann window of 320 samples as F32: `np.hanning(321)[:-1]`,
/// i.e. `0.5 + 0.5 cos(pi n / 320)` for `n = -320, -318, ...`, in numpy's
/// operation order.
#[expect(clippy::suboptimal_flops)]
fn window() -> Vec<f32> {
    (0..FRAME)
        .map(|k| {
            let n = 2.0 * k as f64 - FRAME as f64;
            (0.5 + 0.5 * (std::f64::consts::PI * n / FRAME as f64).cos()) as f32
        })
        .collect()
}

/// Reusable feature extractor (window, filter bank and FFT plan).
pub struct FeatureExtractor {
    window: Vec<f32>,
    filters: Vec<f64>,
    fft: Arc<dyn RealToComplex<f64>>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            window: window(),
            filters: mel_filters(),
            fft: RealFftPlanner::<f64>::new().plan_fft_forward(FFT),
        }
    }

    /// Features of one 16 kHz mono clip of at most [`MAX_SAMPLES`] samples
    /// (longer input is truncated, as the reference does).
    #[must_use]
    pub fn extract(&self, samples: &[f32]) -> LogMel {
        let samples = &samples[..samples.len().min(MAX_SAMPLES)];
        let (frames, valid) = frame_counts(samples.len());
        let mut features = vec![0.0_f32; frames * MEL_BINS];
        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut magnitude = [0.0_f64; BINS];
        for frame in 0..valid {
            // Padded coordinates: sample j of the frame is samples[frame*HOP + j - HOP].
            for (j, value) in input.iter_mut().enumerate() {
                *value = if j < FRAME {
                    let index = (frame * HOP + j).checked_sub(FRAME / 2);
                    let sample = index.and_then(|i| samples.get(i)).copied().unwrap_or(0.0);
                    f64::from(sample * self.window[j])
                } else {
                    0.0
                };
            }
            if self.fft.process(&mut input, &mut spectrum).is_err() {
                continue;
            }
            for (m, c) in magnitude.iter_mut().zip(&spectrum) {
                *m = c.re.hypot(c.im);
            }
            let row = &mut features[frame * MEL_BINS..(frame + 1) * MEL_BINS];
            for (m, out) in row.iter_mut().enumerate() {
                let mel: f64 = magnitude
                    .iter()
                    .enumerate()
                    .map(|(bin, mag)| mag * self.filters[bin * MEL_BINS + m])
                    .sum();
                *out = (mel + MEL_FLOOR).ln() as f32;
            }
        }
        LogMel {
            features,
            frames,
            valid,
        }
    }
}
