//! Whisper's log-mel front end.
//!
//! As `whisper.audio.log_mel_spectrogram` computes it: a centered STFT (periodic Hann window of 400 samples, hop
//! 160, reflect padding), the power of bins 0..=200 with the last frame
//! dropped, the 128-band filterbank of the original package, `log10` clamped
//! at 1e-10, a floor 8 below the global maximum, then `(x + 4) / 4`.
//!
//! (The audio-embedding work may want the same features; this file has no
//! dependency on the rest of the speech model.)

use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const FFT_BINS: usize = N_FFT / 2 + 1;
/// Mel frames in one 30-second window.
pub const WINDOW_FRAMES: usize = 3000;
/// Samples in one 30-second window at 16 kHz.
pub const WINDOW_SAMPLES: usize = WINDOW_FRAMES * HOP;

/// A log-mel spectrogram, `[n_mels][frames]` like the reference.
pub struct LogMel {
    pub n_mels: usize,
    pub frames: usize,
    pub data: Vec<f32>,
}

impl LogMel {
    /// Frames `seek .. seek + size` as one encoder window, time-major
    /// `[3000][n_mels]`, zero beyond `size` (the reference's `pad_or_trim`
    /// pads the log-mel with zeros).
    #[must_use]
    pub fn window(&self, seek: usize, size: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; WINDOW_FRAMES * self.n_mels];
        let size = size
            .min(WINDOW_FRAMES)
            .min(self.frames.saturating_sub(seek));
        for m in 0..self.n_mels {
            let row = &self.data[m * self.frames..(m + 1) * self.frames];
            for t in 0..size {
                out[t * self.n_mels + m] = row[seek + t];
            }
        }
        out
    }
}

/// The filterbank with each band's non-zero bin range, for a sparse product.
pub struct MelFilters {
    n_mels: usize,
    weights: Vec<f64>,
    ranges: Vec<(usize, usize)>,
    fft: Arc<dyn Fft<f64>>,
    window: Vec<f64>,
}

impl MelFilters {
    /// `filters` is `[n_mels][201]` F32, row-major.
    #[must_use]
    pub fn new(filters: &[f32], n_mels: usize) -> Self {
        let weights: Vec<f64> = filters.iter().map(|v| f64::from(*v)).collect();
        let ranges = (0..n_mels)
            .map(|m| {
                let row = &weights[m * FFT_BINS..(m + 1) * FFT_BINS];
                let first = row.iter().position(|v| *v != 0.0).unwrap_or(0);
                let last = row.iter().rposition(|v| *v != 0.0).map_or(0, |i| i + 1);
                (first, last.max(first))
            })
            .collect();
        let window = (0..N_FFT)
            .map(|n| (-0.5_f64).mul_add((2.0 * PI * n as f64 / N_FFT as f64).cos(), 0.5))
            .collect();
        Self {
            n_mels,
            weights,
            ranges,
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
            window,
        }
    }

    /// The log-mel of `audio` (16 kHz) followed by `padding` zero samples.
    #[must_use]
    pub fn log_mel(&self, audio: &[f32], padding: usize) -> LogMel {
        let length = audio.len() + padding;
        let sample = |i: usize| -> f64 { audio.get(i).map_or(0.0, |v| f64::from(*v)) };
        // Reflect padding by N_FFT / 2 on both sides (torch.stft center=True).
        let half = N_FFT / 2;
        let padded = |i: usize| -> f64 {
            // i indexes the padded signal [half + length + half].
            if i < half {
                sample(half - i)
            } else if i - half < length {
                sample(i - half)
            } else {
                let over = i - half - length;
                sample(length.saturating_sub(2 + over))
            }
        };
        let frames = length / HOP;
        let mut data = vec![0.0_f32; self.n_mels * frames];
        let mut buffer = vec![Complex::new(0.0, 0.0); N_FFT];
        let mut power = vec![0.0_f64; FFT_BINS];
        let mut scratch = vec![Complex::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];
        let mut max = f32::NEG_INFINITY;
        for f in 0..frames {
            let start = f * HOP;
            for (n, slot) in buffer.iter_mut().enumerate() {
                *slot = Complex::new(padded(start + n) * self.window[n], 0.0);
            }
            self.fft.process_with_scratch(&mut buffer, &mut scratch);
            for (bin, p) in power.iter_mut().enumerate() {
                *p = buffer[bin].norm_sqr();
            }
            for m in 0..self.n_mels {
                let (first, last) = self.ranges[m];
                let row = &self.weights[m * FFT_BINS..(m + 1) * FFT_BINS];
                let energy: f64 = (first..last).map(|bin| row[bin] * power[bin]).sum();
                let value = (energy as f32).max(1e-10).log10();
                max = max.max(value);
                data[m * frames + f] = value;
            }
        }
        let floor = max - 8.0;
        for v in &mut data {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
        LogMel {
            n_mels: self.n_mels,
            frames,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::suboptimal_flops)]
    use super::*;

    fn triangle_filters(n_mels: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; n_mels * FFT_BINS];
        for m in 0..n_mels {
            let center = 1 + m * (FFT_BINS - 2) / n_mels;
            for d in 0..3 {
                if center + d >= 1 {
                    out[m * FFT_BINS + center + d - 1] = [0.5, 1.0, 0.5][d];
                }
            }
        }
        out
    }

    #[test]
    fn frames_and_normalization_follow_the_reference() {
        let filters = MelFilters::new(&triangle_filters(8), 8);
        let audio: Vec<f32> = (0..4000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let mel = filters.log_mel(&audio, 1600);
        assert_eq!(mel.frames, (4000 + 1600) / HOP);
        let max = mel.data.iter().copied().fold(f32::MIN, f32::max);
        let min = mel.data.iter().copied().fold(f32::MAX, f32::min);
        assert!(max - min <= 2.0 + 1e-6, "dynamic range is 8 decades / 4");
    }

    #[test]
    fn a_pure_tone_peaks_in_its_band() {
        // 1 kHz at 16 kHz is bin 25 of the 400-point FFT.
        let mut filters = vec![0.0_f32; 2 * FFT_BINS];
        filters[25] = 1.0;
        filters[FFT_BINS + 100] = 1.0;
        let bank = MelFilters::new(&filters, 2);
        let audio: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin())
            .collect();
        let mel = bank.log_mel(&audio, 0);
        let mid = mel.frames / 2;
        assert!(mel.data[mid] > mel.data[mel.frames + mid] + 1.0);
    }

    #[test]
    fn windows_are_time_major_and_zero_padded() {
        let mel = LogMel {
            n_mels: 2,
            frames: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let w = mel.window(1, 5);
        assert_eq!(&w[..6], &[2.0, 5.0, 3.0, 6.0, 0.0, 0.0]);
        assert_eq!(w.len(), WINDOW_FRAMES * 2);
    }
}
