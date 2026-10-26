#![allow(clippy::expect_used, clippy::float_cmp)]

use super::*;

#[test]
fn frame_and_token_counts_follow_the_reference() {
    // 10 s: 999 frames, all valid, and 250 tokens, as the
    // reference processor reports for AudioCaps clips.
    assert_eq!(frame_counts(160_000), (999, 999));
    assert_eq!(audio_tokens(999), 250);
    // Padding to a multiple of 128 samples adds frames that are masked.
    assert_eq!(frame_counts(800), (5, 4));
    assert_eq!(audio_tokens(4), 1);
    assert_eq!(frame_counts(160), (1, 0));
    assert_eq!(audio_tokens(0), 0);
    assert_eq!(MAX_AUDIO_TOKENS, 750);
}

#[test]
fn clips_are_validated() {
    assert!(validate_clip(&[0.0; 161]).is_ok());
    assert!(validate_clip(&[0.0; 160]).is_err());
    assert!(matches!(
        validate_clip(&vec![0.0; MAX_SAMPLES + 1]),
        Err(Error::ContextOverflow(_))
    ));
    let mut clip = vec![0.0; 1000];
    clip[3] = f32::NAN;
    assert!(validate_clip(&clip).is_err());
}

#[test]
fn silence_is_the_log_mel_floor_on_valid_frames_only() {
    let mel = FeatureExtractor::new().extract(&[0.0; 800]);
    assert_eq!((mel.frames, mel.valid), (5, 4));
    let floor = (1e-3_f64).ln() as f32;
    assert!(mel.features[..4 * MEL_BINS].iter().all(|v| *v == floor));
    assert!(mel.features[4 * MEL_BINS..].iter().all(|v| *v == 0.0));
}

#[test]
fn a_tone_peaks_in_its_mel_bin() {
    let tone: Vec<f32> = (0..16_000)
        .map(|i| {
            (0.5 * (2.0 * std::f64::consts::PI * 1000.0 * f64::from(i) / 16_000.0).sin()) as f32
        })
        .collect();
    let mel = FeatureExtractor::new().extract(&tone);
    let row = &mel.features[50 * MEL_BINS..51 * MEL_BINS];
    let peak = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("bins");
    // 1 kHz is mel 1000 on the HTK scale; bins are 2840/129 mel apart.
    let expected =
        (1000.0 / (2595.0 * (1.0_f64 + 8000.0 / 700.0).log10() / 129.0)).round() as usize - 1;
    assert!(
        peak.abs_diff(expected) <= 1,
        "peak {peak}, expected {expected}"
    );
}

#[test]
fn downmix_and_resampling_produce_16_khz_mono() {
    let stereo: Vec<f32> = (0..44_100).flat_map(|i| [i as f32 * 1e-5, 0.0]).collect();
    let mono = to_mono_16k(&stereo, 2, 44_100).expect("resample");
    assert!(mono.len().abs_diff(16_000) <= 1, "{}", mono.len());
    let same = to_mono_16k(&[0.5, -0.5, 1.0, 0.0], 2, 16_000).expect("downmix");
    assert_eq!(same, vec![0.0, 0.5]);
    assert!(to_mono_16k(&[0.0], 0, 16_000).is_err());
}

#[test]
fn decoding_rejects_garbage_and_reads_wav() {
    assert!(decode_audio(b"not audio at all".to_vec()).is_err());
    // 16-bit PCM WAV, 8 kHz stereo, 0.1 s.
    let frames = 800_u32;
    let mut wav = Vec::new();
    let data_len = frames * 4;
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&8000_u32.to_le_bytes());
    wav.extend_from_slice(&(8000_u32 * 4).to_le_bytes());
    wav.extend_from_slice(&4_u16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for _ in 0..frames {
        wav.extend_from_slice(&16_384_i16.to_le_bytes());
        wav.extend_from_slice(&0_i16.to_le_bytes());
    }
    let samples = decode_audio(wav).expect("wav");
    assert!(samples.len().abs_diff(1600) <= 1, "{}", samples.len());
    // Mean of 0.5 and 0.0 away from the resampler's edges.
    assert!((samples[800] - 0.25).abs() < 1e-3, "{}", samples[800]);
}

/// A rate outside [`SAMPLE_RATES`] is refused before decoding buffers
/// `MAX_DECODE_SECONDS` of it or the resampler is sized for it.
#[test]
fn sample_rates_outside_the_range_are_refused() {
    let wav = |rate: u32| {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36_u32 + 128).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&rate.wrapping_mul(2).to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&128_u32.to_le_bytes());
        wav.extend_from_slice(&[0; 128]);
        wav
    };
    for rate in [200_000_000, 384_001, 3_999] {
        let error = decode_audio(wav(rate)).expect_err("refused");
        assert!(error.to_string().contains("sample rate"), "{rate}: {error}");
        assert!(to_mono_16k(&[0.0; 4], 1, rate).is_err(), "{rate}");
    }
    decode_audio(wav(8_000)).expect("accepted");
}

/// The signal every file in `tests/fixtures/audio-formats/` encodes: 0.5 s
/// of `0.4 sin(2π·440 t) + 0.2 sin(2π·1250 t)` at 16 kHz mono, quantized to
/// 16 bits (`ffmpeg -f lavfi -i aevalsrc=…:s=16000:d=0.5`, then one encoder
/// each; the Vorbis files are its stereo upmix).
fn format_tone() -> Vec<f32> {
    (0..8000)
        .map(|i| {
            let t = f64::from(i) / 16_000.0;
            let tau = 2.0 * std::f64::consts::PI;
            let value = 0.2f64.mul_add((tau * 1250.0 * t).sin(), 0.4 * (tau * 440.0 * t).sin());
            ((value * 32768.0).round().clamp(-32768.0, 32767.0) / 32768.0) as f32
        })
        .collect()
}

/// The best normalized cross-correlation of `decoded` with `reference` over
/// shifts of up to `max_lag` samples either way (lossy codecs add priming
/// and padding).
fn best_correlation(decoded: &[f32], reference: &[f32], max_lag: usize) -> f64 {
    let energy = |x: &[f32]| x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
    let mut best = f64::MIN;
    for lag in 0..=2 * max_lag {
        let (a, b) = if lag <= max_lag {
            (decoded.get(max_lag - lag..).unwrap_or_default(), reference)
        } else {
            (decoded, reference.get(lag - max_lag..).unwrap_or_default())
        };
        let n = a.len().min(b.len());
        if n < reference.len() / 2 {
            continue;
        }
        let (a, b) = (&a[..n], &b[..n]);
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        best = best.max(dot / (energy(a) * energy(b)).sqrt().max(f64::MIN_POSITIVE));
    }
    best
}

/// Every container and codec the decoder is built with decodes to the
/// encoded signal: lossless ones to the sample, lossy ones to a high
/// correlation with at most codec delay. The same files go through the
/// transcription decoder, which shares the codec set. Opus is refused.
#[test]
fn every_supported_container_and_codec_decodes_the_signal() {
    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/audio-formats");
    let tone = format_tone();
    let lossless = ["tone.flac", "tone-alac.m4a", "tone-flac.mkv"];
    let lossy = [
        "tone.mp3",
        "tone.ogg",
        "tone-aac.m4a",
        "tone-aac.mp4",
        "tone-vorbis.webm",
    ];
    for name in lossless.iter().chain(&lossy) {
        let bytes = std::fs::read(format!("{DIR}/{name}")).expect(name);
        let extension = name.rsplit('.').next();
        let embedding = decode_audio(bytes.clone()).expect(name);
        let transcription = crate::transcription::audio::decode(bytes, extension, 60).expect(name);
        for decoded in [&embedding, &transcription] {
            if lossless.contains(name) {
                assert_eq!(decoded.len(), tone.len(), "{name}");
                let worst = decoded
                    .iter()
                    .zip(&tone)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                assert!(worst <= 2.0 / 32768.0, "{name}: max error {worst}");
            } else {
                assert!(
                    decoded.len().abs_diff(tone.len()) <= 2048,
                    "{name}: {} samples",
                    decoded.len()
                );
                let correlation = best_correlation(decoded, &tone, 2048);
                assert!(correlation >= 0.99, "{name}: correlation {correlation}");
            }
        }
    }
    let opus = std::fs::read(format!("{DIR}/tone-opus.ogg")).expect("opus");
    assert!(decode_audio(opus.clone()).is_err());
    assert!(crate::transcription::audio::decode(opus, Some("ogg"), 60).is_err());
}

#[derive(serde::Deserialize)]
struct FixtureRow {
    name: String,
    samples: usize,
    frames: usize,
    valid_frames: usize,
    audio_tokens: usize,
    embedding_f32le_b64: String,
    wave_f32le_b64: Option<String>,
    features_f32le_b64: Option<String>,
}

fn f32s_from_base64(text: &str) -> Vec<f32> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut bytes = Vec::with_capacity(text.len() * 3 / 4);
    let (mut buffer, mut bits) = (0_u32, 0);
    for byte in text.bytes().filter(|b| *b != b'=') {
        let value = ALPHABET.iter().position(|a| *a == byte).expect("base64") as u32;
        buffer = (buffer << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push((buffer >> bits) as u8);
        }
    }
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The reference fixture: sentence-transformers 6.1.0 / transformers 5.19.0
/// FP32 on Kaggle (`eg2-audioref`): three `AudioCaps` crops (0.05, 0.25 and
/// 1 s; waveform, log-mel features, embedding) and the embeddings of a 2 s
/// 1 kHz tone and 1 s of silence, whose waveforms are generated here.
fn fixture() -> Vec<(FixtureRow, Vec<f32>)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/embeddinggemma2-audio-reference.jsonl"
    );
    std::fs::read_to_string(path)
        .expect("fixture")
        .lines()
        .map(|line| {
            let row: FixtureRow = serde_json::from_str(line).expect("row");
            let wave = row.wave_f32le_b64.as_deref().map_or_else(
                || match row.name.as_str() {
                    "silence" => vec![0.0; row.samples],
                    _ => (0..row.samples)
                        .map(|i| {
                            let t = i as f64 / 16_000.0;
                            (0.5 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin()) as f32
                        })
                        .collect(),
                },
                f32s_from_base64,
            );
            assert_eq!(wave.len(), row.samples);
            (row, wave)
        })
        .collect()
}

/// Log-mel features match the reference feature extractor to F32 rounding.
#[test]
fn log_mel_features_match_the_reference_fixture() {
    let extractor = FeatureExtractor::new();
    for (row, wave) in fixture() {
        let mel = extractor.extract(&wave);
        assert_eq!(
            (mel.frames, mel.valid),
            (row.frames, row.valid_frames),
            "{}",
            row.name
        );
        assert_eq!(audio_tokens(mel.valid), row.audio_tokens, "{}", row.name);
        if let Some(features) = &row.features_f32le_b64 {
            let reference = f32s_from_base64(features);
            assert_eq!(reference.len(), mel.features.len());
            let worst = mel
                .features
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(worst < 2e-6, "{}: max |difference| {worst}", row.name);
        }
    }
}

/// Native audio embeddings within the parity gate (mean cosine ≥ 0.9995,
/// min ≥ 0.999) of the reference.
#[test]
#[ignore = "requires the installed embedding model and audio artifact"]
fn native_audio_embeddings_match_the_reference_fixture() {
    let rows = fixture();
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let dir = super::super::installed_model(&Path::new(root).join("models")).expect("installed");
    let mut model = EmbeddingModel::open(&dir).expect("open");
    // The audio artifact loads on first use (no `load_audio`).
    assert!(model.supports_audio() && !model.has_audio());
    let clips: Vec<&[f32]> = rows.iter().map(|(_, wave)| wave.as_slice()).collect();
    let embeddings = model.embed_audio(&clips, &mut || Ok(())).expect("embed");
    assert!(model.has_audio());
    let cosines: Vec<f32> = embeddings
        .iter()
        .zip(&rows)
        .map(|(native, (row, _))| {
            let reference = f32s_from_base64(&row.embedding_f32le_b64);
            assert_eq!(reference.len(), super::super::EMBEDDING_DIMENSIONS);
            native.iter().zip(&reference).map(|(a, b)| a * b).sum()
        })
        .collect();
    let mean = cosines.iter().sum::<f32>() / cosines.len() as f32;
    let min = cosines.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(mean >= 0.9995 && min >= 0.999, "cosines {cosines:?}");
}
