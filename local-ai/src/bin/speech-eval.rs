//! Parity, G2P and throughput checks for the native Kokoro-82M synthesizer
//! (see `docs/SPEECH.md`).
//!
//! ```text
//! speech-eval synth KOKORO_REF_DIR OUT_DIR     (GPU)
//! speech-eval compare KOKORO_REF_DIR OUT_DIR   (CPU)
//! speech-eval g2p KOKORO_REF_DIR
//! speech-eval bench [--chunks N]
//! ```
//!
//! The model loads from `models/kokoro-82m` under the working directory.
//! `KOKORO_REF_DIR` is the extracted reference archive of the Kaggle job
//! (`tools/kaggle_kokoro_job.py --mode reference`):
//!
//! - `synth` then `compare` (split so the GPU lock is held only for
//!   synthesis): for every fixture row, synthesizes the reference phonemes with
//!   the reference voice and speed in the deterministic mode (no source
//!   noise, as the reference's `zero` waveforms) and in the seeded mode, and
//!   the row's text through the native G2P and chunking (as the reference's
//!   `pipeline` waveforms). Reports durations, waveform SNR and log-mel
//!   distances against the reference, intermediate activations for the rows
//!   that have them, and writes every native and reference waveform as
//!   16-bit WAV under `OUT_DIR` (`native-zero/`, `native-seed/`,
//!   `native-text/`, `reference-zero/`, `reference-seed/`,
//!   `reference-text/`, with `texts.json`) for the ASR job.
//! - `g2p`: native G2P (lexicon only, like `misaki` without a fallback)
//!   against `misaki` on `g2p.jsonl`.
//! - `bench`: real-time factor of fixed sentences (best of three).

use std::collections::BTreeMap;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use local_engine::speech::{
    Noise, SAMPLE_RATE, SpeechModel, Taps, encode, installed_model,
};
use serde::Deserialize;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
struct Manifest {
    rows: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    index: usize,
    text: String,
    voice: String,
    speed: f32,
    phonemes: String,
    input_ids: Vec<u32>,
    pred_dur: Vec<u32>,
}

#[derive(Deserialize)]
struct G2pRow {
    source: String,
    lang: String,
    text: String,
    phonemes: String,
}

fn open() -> Result<SpeechModel> {
    let dir = installed_model(Path::new("models"))
        .ok_or("models/kokoro-82m is not installed (see docs/SPEECH.md)")?;
    let started = Instant::now();
    let model = SpeechModel::open(&dir)?;
    eprintln!("loaded {} in {:.2?}", dir.display(), started.elapsed());
    Ok(model)
}

fn f32_file(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

/// Minimal `.npy` reader (little-endian F32, C or Fortran order) inside an
/// uncompressed `.npz`.
fn npz(path: &Path) -> Result<BTreeMap<String, (Vec<usize>, Vec<f32>)>> {
    let mut out = BTreeMap::new();
    let bytes = std::fs::read(path)?;
    let mut at = 0;
    while at + 30 <= bytes.len() && bytes[at..at + 4] == [0x50, 0x4b, 0x03, 0x04] {
        let u16_at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]) as usize;
        let u32_at = |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
        let mut size = u32_at(at + 18);
        let name_len = u16_at(at + 26);
        let extra_len = u16_at(at + 28);
        let name = String::from_utf8_lossy(&bytes[at + 30..at + 30 + name_len]).to_string();
        let data_start = at + 30 + name_len + extra_len;
        if size == 0xFFFF_FFFF {
            // ZIP64 extra field: original then compressed size (u64 each).
            let extra = &bytes[at + 30 + name_len..data_start];
            size = u64::from_le_bytes(extra[12..20].try_into()?) as usize;
        }
        let data = &bytes[data_start..data_start + size];
        let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
        let header = String::from_utf8_lossy(&data[10..10 + header_len]).to_string();
        let fortran = header.contains("'fortran_order': True");
        let shape_text = header
            .split("'shape': (")
            .nth(1)
            .and_then(|s| s.split(')').next())
            .unwrap_or("");
        let shape: Vec<usize> = shape_text
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let mut values: Vec<f32> = data[10 + header_len..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if fortran && shape.len() == 3 && shape[0] == 1 {
            // [1, a, b] in column-major -> row-major.
            let (a, b) = (shape[1], shape[2]);
            let mut c = vec![0.0; a * b];
            for i in 0..a {
                for j in 0..b {
                    c[i * b + j] = values[j * a + i];
                }
            }
            values = c;
        }
        out.insert(name.trim_end_matches(".npy").to_owned(), (shape, values));
        at = data_start + size;
    }
    Ok(out)
}

fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, encode::wav(samples, SAMPLE_RATE))?;
    Ok(())
}

/// SNR of `native` against `reference` over their common length, dB.
fn snr(reference: &[f32], native: &[f32]) -> f64 {
    let n = reference.len().min(native.len());
    let (mut signal, mut noise) = (0.0_f64, 0.0_f64);
    for i in 0..n {
        signal += f64::from(reference[i]).powi(2);
        noise += f64::from(reference[i] - native[i]).powi(2);
    }
    10.0 * (signal / noise.max(1e-20)).log10()
}

/// 80-bin log-mel power spectrogram (n_fft 1024, hop 256, Hann, 0-12 kHz,
/// HTK mel scale), dB.
fn log_mel(x: &[f32]) -> Vec<Vec<f64>> {
    const N: usize = 1024;
    const HOP: usize = 256;
    const MELS: usize = 80;
    let sr = f64::from(SAMPLE_RATE);
    let mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (lo, hi) = (mel(0.0), mel(sr / 2.0));
    let points: Vec<f64> = (0..MELS + 2)
        .map(|i| hz(lo + (hi - lo) * i as f64 / (MELS + 1) as f64))
        .collect();
    let bins = N / 2 + 1;
    let freq = |k: usize| k as f64 * sr / N as f64;
    let window: Vec<f64> = (0..N)
        .map(|n| 0.5 - 0.5 * (std::f64::consts::TAU * n as f64 / N as f64).cos())
        .collect();
    let mut cos = vec![0.0; bins * N];
    let mut sin = vec![0.0; bins * N];
    for k in 0..bins {
        for n in 0..N {
            let a = std::f64::consts::TAU * ((k * n) % N) as f64 / N as f64;
            cos[k * N + n] = a.cos();
            sin[k * N + n] = a.sin();
        }
    }
    let frames = x.len().saturating_sub(N) / HOP + 1;
    let mut out = Vec::with_capacity(frames);
    let mut frame = vec![0.0_f64; N];
    for f in 0..frames {
        for n in 0..N {
            frame[n] = f64::from(*x.get(f * HOP + n).unwrap_or(&0.0)) * window[n];
        }
        let power: Vec<f64> = (0..bins)
            .map(|k| {
                let (mut re, mut im) = (0.0, 0.0);
                for n in 0..N {
                    re += frame[n] * cos[k * N + n];
                    im -= frame[n] * sin[k * N + n];
                }
                re * re + im * im
            })
            .collect();
        let row: Vec<f64> = (0..MELS)
            .map(|m| {
                let (a, b, c) = (points[m], points[m + 1], points[m + 2]);
                let mut sum = 0.0;
                for (k, p) in power.iter().enumerate() {
                    let fk = freq(k);
                    let w = if fk > a && fk <= b {
                        (fk - a) / (b - a)
                    } else if fk > b && fk < c {
                        (c - fk) / (c - b)
                    } else {
                        0.0
                    };
                    sum += w * p;
                }
                10.0 * (sum + 1e-10).log10()
            })
            .collect();
        out.push(row);
    }
    out
}

/// Mean absolute log-mel difference (dB, over cells within 80 dB of the
/// reference's peak) and cosine of the linear mel power.
fn mel_distance(reference: &[f32], native: &[f32]) -> (f64, f64) {
    let n = reference.len().min(native.len());
    let a = log_mel(&reference[..n]);
    let b = log_mel(&native[..n]);
    let peak = a.iter().flatten().copied().fold(f64::MIN, f64::max);
    let (mut diff, mut count, mut dot, mut na, mut nb) = (0.0, 0_usize, 0.0, 0.0, 0.0);
    for (ra, rb) in a.iter().zip(&b) {
        for (x, y) in ra.iter().zip(rb) {
            let floor = peak - 80.0;
            if *x > floor || *y > floor {
                diff += (x.max(floor) - y.max(floor)).abs();
                count += 1;
            }
            let (px, py) = (10f64.powf(x / 10.0), 10f64.powf(y / 10.0));
            dot += px * py;
            na += px * px;
            nb += py * py;
        }
    }
    (diff / count.max(1) as f64, dot / (na.sqrt() * nb.sqrt()).max(1e-30))
}

fn rel_err(reference: &[f32], native: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (r, n) in reference.iter().zip(native) {
        num += f64::from(r - n).powi(2);
        den += f64::from(*r).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn write_f32(path: &Path, samples: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes)?;
    Ok(())
}

/// GPU half of `parity`: synthesize every fixture row three ways and write
/// raw F32 outputs, intermediate activations and durations under `out`.
fn synth(dir: &Path, out: &Path) -> Result<()> {
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("fixtures/manifest.json"))?)?;
    let model = open()?;
    let mut durations = BTreeMap::new();
    let started = Instant::now();
    for row in &manifest.rows {
        let key = format!("{:03}", row.index);
        let mut taps = Taps::new();
        let want_taps = dir.join(format!("fixtures/intermediates/{key}.npz")).is_file();
        let zero = model.synthesize_phonemes(
            &row.phonemes,
            &row.voice,
            row.speed,
            Noise::Zero,
            &mut || Ok(()),
            want_taps.then_some(&mut taps),
        )?;
        write_f32(&out.join(format!("native-zero/{key}.f32")), &zero.samples)?;
        for (name, values) in &taps {
            write_f32(&out.join(format!("taps/{key}/{name}.f32")), values)?;
        }
        let seeded = model.synthesize_phonemes(
            &row.phonemes,
            &row.voice,
            row.speed,
            Noise::Seeded(row.index as u64),
            &mut || Ok(()),
            None,
        )?;
        write_f32(&out.join(format!("native-seed/{key}.f32")), &seeded.samples)?;
        let mut text_audio = Vec::new();
        model.synthesize(
            &row.text,
            &row.voice,
            row.speed,
            Noise::Seeded(row.index as u64),
            &mut || Ok(()),
            &mut |samples| {
                text_audio.extend_from_slice(samples);
                Ok(())
            },
        )?;
        write_f32(&out.join(format!("native-text/{key}.f32")), &text_audio)?;
        durations.insert(key, zero.durations);
    }
    eprintln!("synthesized {} rows in {:.2?}", manifest.rows.len(), started.elapsed());
    std::fs::write(out.join("durations.json"), serde_json::to_vec(&durations)?)?;
    Ok(())
}

/// CPU half of `parity`: compare `synth`'s outputs with the reference and
/// write 16-bit WAVs of both for the ASR job.
#[allow(clippy::too_many_lines)]
fn compare(dir: &Path, out: &Path) -> Result<()> {
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("fixtures/manifest.json"))?)?;
    let durations: BTreeMap<String, Vec<u32>> =
        serde_json::from_slice(&std::fs::read(out.join("durations.json"))?)?;
    let dirs = models_g2p()?;
    let mut texts = BTreeMap::new();
    let (mut same_dur, mut same_ids, mut g2p_same) = (0, 0, 0);
    let mut zero_stats = Vec::new();
    let mut seed_stats = Vec::new();
    let mut text_stats = Vec::new();
    for row in &manifest.rows {
        let key = format!("{:03}", row.index);
        texts.insert(key.clone(), row.text.clone());
        let ids = dirs.ids(&row.phonemes);
        same_ids += usize::from(ids == row.input_ids);
        let dur = durations.get(&key).ok_or("missing durations")?;
        let dur_same = *dur == row.pred_dur;
        same_dur += usize::from(dur_same);
        if !dur_same {
            let diffs: Vec<String> = dur
                .iter()
                .zip(&row.pred_dur)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| format!("{i}:{a}/{b}"))
                .collect();
            println!("  {key} durations differ at {}", diffs.join(" "));
        }
        let mut stats = Vec::new();
        for (mode, reference_dir) in [("zero", "zero"), ("seed", "seed"), ("text", "pipeline")] {
            let native = f32_file(&out.join(format!("native-{mode}/{key}.f32")))?;
            let reference = f32_file(&dir.join(format!("fixtures/{reference_dir}/{key}.f32")))?;
            let (mad, cos) = mel_distance(&reference, &native);
            let s = snr(&reference, &native);
            stats.push((mad, cos, s, native.len(), reference.len()));
            write_wav(&out.join(format!("wav/native-{mode}/{key}.wav")), &native)?;
            write_wav(&out.join(format!("wav/reference-{mode}/{key}.wav")), &reference)?;
        }
        zero_stats.push((stats[0].0, stats[0].1, stats[0].2, dur_same));
        seed_stats.push((stats[1].0, stats[1].1));
        text_stats.push((stats[2].0, stats[2].1));
        let native_ps = dirs.chunks(&row.text, &row.voice);
        let same = native_ps == [row.phonemes.clone()];
        g2p_same += usize::from(same);
        if !same {
            println!("  {key} g2p differs:\n    ref    {}\n    native {}", row.phonemes, native_ps.join(" | "));
        }
        let taps_dir = out.join(format!("taps/{key}"));
        if taps_dir.is_dir() {
            let reference_taps = npz(&dir.join(format!("fixtures/intermediates/{key}.npz")))?;
            for entry in std::fs::read_dir(&taps_dir)? {
                let path = entry?.path();
                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_owned();
                let native = f32_file(&path)?;
                let Some((shape, values)) = reference_taps.get(&name) else {
                    continue;
                };
                // Token-major reference tensors [1, T, C] -> [C, T].
                let token_major = matches!(
                    name.as_str(),
                    "bert" | "bert_encoder" | "predictor.text_encoder" | "predictor.lstm"
                        | "predictor.duration_proj" | "predictor.shared"
                );
                let values = if token_major && shape.len() == 3 {
                    let (t, c) = (shape[1], shape[2]);
                    let mut v = vec![0.0; t * c];
                    for i in 0..t {
                        for j in 0..c {
                            v[j * t + i] = values[i * c + j];
                        }
                    }
                    v
                } else {
                    values.clone()
                };
                if values.len() == native.len() {
                    println!("  {key} {name:32} rel_err {:.2e}", rel_err(&values, &native));
                } else {
                    println!("  {key} {name:32} size {} vs {}", native.len(), values.len());
                }
            }
        }
        println!(
            "{key} dur {} | zero MAD {:.3} dB cos {:.5} SNR {:.1} dB (len {} / {}) | seed MAD {:.3} cos {:.5} | text MAD {:.3} cos {:.5}",
            if dur_same { "same" } else { "DIFF" },
            stats[0].0, stats[0].1, stats[0].2, stats[0].3, stats[0].4,
            stats[1].0, stats[1].1, stats[2].0, stats[2].1,
        );
    }
    std::fs::write(out.join("wav/texts.json"), serde_json::to_vec_pretty(&texts)?)?;
    let n = manifest.rows.len();
    let summary = |name: &str, v: &[(f64, f64)]| {
        let mean_mad = v.iter().map(|x| x.0).sum::<f64>() / v.len().max(1) as f64;
        let max_mad = v.iter().map(|x| x.0).fold(0.0, f64::max);
        let mean_cos = v.iter().map(|x| x.1).sum::<f64>() / v.len().max(1) as f64;
        let min_cos = v.iter().map(|x| x.1).fold(1.0, f64::min);
        println!("{name}: rows {}, log-mel MAD mean {mean_mad:.3} dB (max {max_mad:.3}), mel cosine mean {mean_cos:.5} (min {min_cos:.5})", v.len());
    };
    println!("rows {n}; ids identical {same_ids}; durations identical {same_dur}; native G2P identical {g2p_same}");
    let zero_same: Vec<(f64, f64)> = zero_stats.iter().filter(|x| x.3).map(|x| (x.0, x.1)).collect();
    let mut snrs: Vec<f64> = zero_stats.iter().filter(|x| x.3).map(|x| x.2).collect();
    snrs.sort_by(f64::total_cmp);
    summary("zero (identical durations)", &zero_same);
    if !snrs.is_empty() {
        println!("zero waveform SNR: median {:.1} dB, min {:.1} dB", snrs[snrs.len() / 2], snrs[0]);
    }
    summary("seed", &seed_stats);
    summary("text", &text_stats);
    Ok(())
}

/// CPU-only G2P helpers for `compare`.
struct G2pOnly {
    american: local_engine::speech::G2p,
    british: local_engine::speech::G2p,
    vocab: std::collections::HashMap<char, u32>,
}

impl G2pOnly {
    fn ids(&self, phonemes: &str) -> Vec<u32> {
        let mut ids = vec![0];
        ids.extend(phonemes.chars().filter_map(|c| self.vocab.get(&c).copied()));
        ids.push(0);
        ids
    }

    fn chunks(&self, text: &str, voice: &str) -> Vec<String> {
        let g2p = if voice.starts_with('b') { &self.british } else { &self.american };
        local_engine::speech::chunk_text(g2p, text, 510)
            .into_iter()
            .map(|c| c.phonemes)
            .collect()
    }
}

fn models_g2p() -> Result<G2pOnly> {
    let dir = installed_model(Path::new("models"))
        .ok_or("models/kokoro-82m is not installed (see docs/SPEECH.md)")?;
    let (american, british) = local_engine::speech::open_g2p(&dir)?;
    let vocab = local_engine::speech::open_vocab(&dir)?;
    Ok(G2pOnly { american, british, vocab })
}

fn g2p(dir: &Path) -> Result<()> {
    let mut text = String::new();
    std::fs::File::open(dir.join("g2p.jsonl"))?.read_to_string(&mut text)?;
    let rows: Vec<G2pRow> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let dir_models = installed_model(Path::new("models"))
        .ok_or("models/kokoro-82m is not installed (see docs/SPEECH.md)")?;
    let (american, british) = local_engine::speech::open_g2p(&dir_models)?;
    let mut stats: BTreeMap<(String, String), (usize, usize, usize, usize)> = BTreeMap::new();
    let mut shown = 0;
    for row in &rows {
        let g2p = if row.lang == "b" { &british } else { &american };
        let native = g2p.phonemize(&row.text, false);
        let native = native.trim_end();
        let reference = row.phonemes.trim_end();
        let entry = stats.entry((row.source.clone(), row.lang.clone())).or_default();
        entry.0 += 1;
        entry.1 += usize::from(native == reference);
        let (rw, nw): (Vec<&str>, Vec<&str>) = (
            reference.split_whitespace().collect(),
            native.split_whitespace().collect(),
        );
        if rw.len() == nw.len() {
            entry.2 += rw.len();
            entry.3 += rw.iter().zip(&nw).filter(|(a, b)| a == b).count();
        }
        if native != reference && shown < 40 {
            shown += 1;
            println!("[{}] {}\n  ref    {reference}\n  native {native}", row.lang, row.text);
        }
    }
    for ((source, lang), (n, same, words, same_words)) in stats {
        println!(
            "{source} {lang}: {same}/{n} sentences identical; words identical {same_words}/{words} ({:.2}%) in sentences with equal word counts",
            100.0 * same_words as f64 / words.max(1) as f64
        );
    }
    Ok(())
}

fn bench(chunks: usize) -> Result<()> {
    let model = open()?;
    let text = "Software engineers often debate the merits of different programming styles, but \
                most agree that clear and simple code is easier to maintain over time.";
    let text = vec![text; chunks].join("\n");
    for voice in ["af_heart", "bm_george"] {
        let mut best = f64::MAX;
        let mut audio = 0.0;
        for _ in 0..3 {
            let started = Instant::now();
            let usage = model.synthesize(&text, voice, 1.0, Noise::Seeded(0), &mut || Ok(()), &mut |_| Ok(()))?;
            best = best.min(started.elapsed().as_secs_f64());
            audio = usage.samples as f64 / f64::from(SAMPLE_RATE);
        }
        println!("{voice}: {audio:.2} s audio in {best:.3} s, RTF {:.4} ({:.0}x real time)", best / audio, audio / best);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("synth") if args.len() == 3 => synth(Path::new(&args[1]), &PathBuf::from(&args[2])),
        Some("compare") if args.len() == 3 => compare(Path::new(&args[1]), &PathBuf::from(&args[2])),
        Some("g2p") if args.len() == 2 => g2p(Path::new(&args[1])),
        Some("bench") => {
            let chunks = args
                .iter()
                .position(|a| a == "--chunks")
                .and_then(|i| args.get(i + 1))
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            bench(chunks)
        }
        _ => Err("usage: speech-eval synth|compare KOKORO_REF_DIR OUT_DIR | g2p KOKORO_REF_DIR | bench [--chunks N]".into()),
    }
}
