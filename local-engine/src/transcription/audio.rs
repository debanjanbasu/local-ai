//! Audio files to the model's input: 16 kHz mono, quantized to 16-bit PCM.
//!
//! The reference loads audio with `ffmpeg -ac 1 -ar 16000 -f s16le` and
//! divides by 32768, so this module decodes with Symphonia (pure Rust: FLAC,
//! MP3, WAV, AAC and ALAC in MP4/M4A, Vorbis in Ogg and WebM/Matroska),
//! averages the channels, resamples other rates with a Kaiser-windowed sinc
//! filter, and rounds to 16-bit steps. 16 kHz input (`LibriSpeech`, FLEURS)
//! therefore reaches the model exactly as it reaches the reference; other
//! rates match it to the resampler's precision. Opus (the usual codec of Ogg
//! and `WebM` voice notes) has no maintained pure-Rust decoder and is refused.
//!
//! (Shared-looking with the audio-embedding work; kept self-contained here.)

use std::io::Cursor;

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

pub const SAMPLE_RATE: u32 = 16_000;

/// Why a file could not become model input.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("unsupported audio: {0}")]
    Unsupported(String),
    #[error("malformed audio: {0}")]
    Malformed(String),
    #[error("the audio is longer than the {0}-second limit")]
    TooLong(u64),
    #[error("the audio holds no samples")]
    Empty,
}

fn symphonia_error(error: &symphonia::core::errors::Error) -> AudioError {
    use symphonia::core::errors::Error;
    match error {
        Error::Unsupported(what) => AudioError::Unsupported(format!(
            "{what} (decodable: FLAC, MP3, WAV, AAC/ALAC in MP4 or M4A, Vorbis in Ogg or WebM; \
             not Opus)"
        )),
        other => AudioError::Malformed(other.to_string()),
    }
}

/// Polyphase-free windowed-sinc resampler to 16 kHz over a streaming input.
struct Resampler {
    /// Input samples per output sample.
    step: f64,
    /// Filter half-width in input samples.
    half_width: f64,
    /// The windowed-sinc kernel at distances `0, 1/RESOLUTION, …,
    /// half_width` (input samples), interpolated linearly.
    table: Vec<f64>,
    input: Vec<f64>,
    /// Absolute index of `input[0]`.
    base: u64,
    /// Next output sample index.
    next: u64,
}

/// Kernel table points per input sample.
const RESOLUTION: f64 = 512.0;
const ZERO_CROSSINGS: f64 = 16.0;
const KAISER_BETA: f64 = 9.0;
/// Pass band as a fraction of the lower Nyquist rate (`FFmpeg`'s default).
const CUTOFF: f64 = 0.97;

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let y = x * x / 4.0;
    for k in 1..64 {
        term *= y / f64::from(k * k);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

impl Resampler {
    fn new(rate: u32) -> Self {
        let step = f64::from(rate) / f64::from(SAMPLE_RATE);
        let cutoff = CUTOFF * (1.0 / step).min(1.0);
        let half_width = ZERO_CROSSINGS / cutoff;
        let norm = bessel_i0(KAISER_BETA);
        let points = (half_width * RESOLUTION).ceil() as usize + 2;
        let table = (0..points)
            .map(|i| {
                let distance = i as f64 / RESOLUTION;
                let x = distance / half_width;
                if x >= 1.0 {
                    return 0.0;
                }
                let window = bessel_i0(KAISER_BETA * x.mul_add(-x, 1.0).max(0.0).sqrt()) / norm;
                let arg = std::f64::consts::PI * cutoff * distance;
                let sinc = if arg.abs() < 1e-12 {
                    1.0
                } else {
                    arg.sin() / arg
                };
                cutoff * sinc * window
            })
            .collect();
        Self {
            step,
            half_width,
            table,
            input: Vec::new(),
            base: 0,
            next: 0,
        }
    }

    fn kernel(&self, distance: f64) -> f64 {
        let position = distance.abs() * RESOLUTION;
        let index = position as usize;
        if index + 1 >= self.table.len() {
            return 0.0;
        }
        let frac = position - index as f64;
        (self.table[index + 1] - self.table[index]).mul_add(frac, self.table[index])
    }

    /// Outputs computable from the buffered input; `flush` treats the input
    /// as ended (zeros beyond it).
    fn drain(&mut self, out: &mut Vec<f32>, flush: bool) {
        let available = self.base + self.input.len() as u64;
        loop {
            let center = self.next as f64 * self.step;
            let last_needed = (center + self.half_width).floor() as u64;
            if !flush && last_needed >= available {
                break;
            }
            if flush && center >= available as f64 {
                break;
            }
            let first = (center - self.half_width).ceil().max(0.0) as u64;
            let mut acc = 0.0;
            for index in first.max(self.base)..=last_needed.min(available.saturating_sub(1)) {
                let value = self.input[(index - self.base) as usize];
                acc = value.mul_add(self.kernel(center - index as f64), acc);
            }
            out.push(acc as f32);
            self.next += 1;
        }
        // Drop input no future output needs.
        let keep_from = ((self.next as f64)
            .mul_add(self.step, -self.half_width)
            .floor()
            .max(0.0) as u64)
            .min(available);
        if keep_from > self.base {
            self.input.drain(..(keep_from - self.base) as usize);
            self.base = keep_from;
        }
    }
}

/// `FFmpeg`'s float to s16 conversion (round half to even, clip), back to
/// float as the reference divides by 32768.
fn quantize(value: f32) -> f32 {
    (value * 32768.0).round_ties_even().clamp(-32768.0, 32767.0) / 32768.0
}

/// Decode a whole audio file to 16 kHz mono samples. `extension` (from the
/// upload's file name) helps the container probe; at most `max_seconds` of
/// audio are accepted.
#[allow(clippy::too_many_lines)]
pub fn decode(
    bytes: Vec<u8>,
    extension: Option<&str>,
    max_seconds: u64,
) -> Result<Vec<f32>, AudioError> {
    let mut hint = Hint::new();
    if let Some(extension) = extension {
        hint.with_extension(extension);
    }
    let stream = MediaSourceStream::new(
        Box::new(Cursor::new(bytes)),
        MediaSourceStreamOptions::default(),
    );
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|error| symphonia_error(&error))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| AudioError::Unsupported("the file has no audio track".into()))?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| AudioError::Unsupported("the audio track has no codec parameters".into()))?
        .clone();
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|error| symphonia_error(&error))?;
    let limit = max_seconds * u64::from(SAMPLE_RATE);
    let mut out = Vec::new();
    let mut resampler: Option<Resampler> = None;
    let mut rate = 0_u32;
    let mut interleaved: Vec<f32> = Vec::new();
    let mut mono: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(symphonia::core::errors::Error::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(error) => return Err(symphonia_error(&error)),
        };
        if packet.track_id != track_id {
            continue;
        }
        let pcm = match decoder.decode(&packet) {
            Ok(pcm) => pcm,
            // A damaged packet is skipped, as players and FFmpeg do.
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(error) => return Err(symphonia_error(&error)),
        };
        let spec = pcm.spec();
        let channels = spec.channels().count().max(1);
        if rate == 0 {
            rate = spec.rate();
            if rate == 0 {
                return Err(AudioError::Malformed("sample rate 0".into()));
            }
            if rate != SAMPLE_RATE {
                resampler = Some(Resampler::new(rate));
            }
        } else if spec.rate() != rate {
            return Err(AudioError::Unsupported(
                "the sample rate changes mid-stream".into(),
            ));
        }
        interleaved.clear();
        pcm.copy_to_vec_interleaved(&mut interleaved);
        mono.clear();
        mono.extend(
            interleaved
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32),
        );
        match resampler.as_mut() {
            Some(resampler) => {
                resampler.input.extend(mono.iter().map(|v| f64::from(*v)));
                resampler.drain(&mut out, false);
            }
            None => out.extend_from_slice(&mono),
        }
        if out.len() as u64 > limit {
            return Err(AudioError::TooLong(max_seconds));
        }
    }
    if let Some(resampler) = resampler.as_mut() {
        resampler.drain(&mut out, true);
        // Exactly round(duration * 16 kHz) samples.
        let expected = ((resampler.base + resampler.input.len() as u64) as f64 / resampler.step)
            .round() as usize;
        out.truncate(expected);
    }
    if out.len() as u64 > limit {
        return Err(AudioError::TooLong(max_seconds));
    }
    if out.is_empty() {
        return Err(AudioError::Empty);
    }
    for v in &mut out {
        *v = quantize(*v);
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn wav(samples: &[i16], rate: u32, channels: u16) -> Vec<u8> {
        let data_len = (samples.len() * 2) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16_u32.to_le_bytes());
        out.extend_from_slice(&1_u16.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * u32::from(channels) * 2).to_le_bytes());
        out.extend_from_slice(&(channels * 2).to_le_bytes());
        out.extend_from_slice(&16_u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    #[test]
    fn sixteen_khz_pcm_is_exact() {
        let samples: Vec<i16> = (0..1600).map(|i| ((i * 37) % 2000 - 1000) as i16).collect();
        let got = decode(wav(&samples, 16_000, 1), Some("wav"), 60).expect("decode");
        assert_eq!(got.len(), samples.len());
        for (g, s) in got.iter().zip(&samples) {
            assert_eq!(*g, f32::from(*s) / 32768.0);
        }
    }

    #[test]
    fn stereo_is_averaged_and_rates_are_converted() {
        // 48 kHz stereo, 440 Hz in both channels at different gains.
        let frames = 48_000;
        let mut samples = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let v = (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 48_000.0).sin();
            samples.push((v * 8000.0) as i16);
            samples.push((v * 4000.0) as i16);
        }
        let got = decode(wav(&samples, 48_000, 2), None, 60).expect("decode");
        assert_eq!(got.len(), 16_000);
        // Compare with the ideal 16 kHz tone at the mean gain away from edges.
        let mut worst = 0.0_f64;
        for (n, g) in got.iter().enumerate().skip(200).take(15_000) {
            let ideal =
                (2.0 * std::f64::consts::PI * 440.0 * n as f64 / 16_000.0).sin() * 6000.0 / 32768.0;
            worst = worst.max((f64::from(*g) - ideal).abs());
        }
        assert!(worst < 2e-3, "worst error {worst}");
    }

    #[test]
    fn limits_and_garbage_are_refused() {
        let samples = vec![0_i16; 16_000 * 3];
        assert!(matches!(
            decode(wav(&samples, 16_000, 1), Some("wav"), 2),
            Err(AudioError::TooLong(2))
        ));
        assert!(decode(b"not audio at all".to_vec(), Some("mp3"), 60).is_err());
        assert!(matches!(
            decode(wav(&[], 16_000, 1), Some("wav"), 60),
            Err(AudioError::Empty)
        ));
    }
}
