//! Reference checks and throughput for the native Whisper large-v3 model
//! (see `docs/TRANSCRIPTION.md`).
//!
//! ```text
//! transcription-eval parity DIR
//! transcription-eval transcribe MANIFEST.json OUT.jsonl
//! transcription-eval bench AUDIO
//! ```
//!
//! The model loads from `models/whisper-large-v3` under the working
//! directory. `DIR` is the reference kernel's output with `audio.tar`
//! extracted into it:
//!
//! - `parity`: for each case of `parity/parity.json` (openai-whisper
//!   20250625, FP32 CPU), compares the log-mel window and encoder output
//!   (first case, from `parity/*.npy`), the language probabilities, the greedy
//!   decode (token for token) and the teacher-forced logits along the
//!   reference tokens (argmax and top-10 agreement, logit differences).
//! - `transcribe`: runs `transcribe()` on each `{"id", "path", "task",
//!   "language"?, "prompt"?}` of the manifest at temperature 0 (no fallback,
//!   like the reference runs) and writes one JSON line per file with the
//!   text, the segments' tokens and the wall time.
//! - `bench`: real-time factor of one file (best of three), with the encoder
//!   and per-token decoder times.

use std::path::{Path, PathBuf};
use std::time::Instant;

use local_engine::transcription::{
    Pace, Task, Temperature, TranscribeOptions, TranscriptionModel, audio, installed_model,
};
use serde::Deserialize;
use serde_json::{Value, json};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const PADDING: usize = 480_000;

fn open() -> Result<TranscriptionModel> {
    let dir = installed_model(Path::new("models"))
        .ok_or("models/whisper-large-v3 is not installed (see docs/TRANSCRIPTION.md)")?;
    let started = Instant::now();
    let model = TranscriptionModel::open(&dir)?;
    eprintln!("loaded {} in {:.2?}", dir.display(), started.elapsed());
    Ok(model)
}

fn load(path: &Path) -> Result<Vec<f32>> {
    let extension = path.extension().and_then(|e| e.to_str()).map(str::to_owned);
    Ok(audio::decode(
        std::fs::read(path)?,
        extension.as_deref(),
        7200,
    )?)
}

/// A C-order little-endian F32 `.npy` array: shape and values.
fn npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let raw = std::fs::read(path)?;
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not an .npy file", path.display()).into());
    }
    let (length, start) = if raw[6] == 1 {
        (usize::from(u16::from_le_bytes([raw[8], raw[9]])), 10)
    } else {
        (
            u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&raw[start..start + length])?;
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        return Err(format!("{}: unsupported array {header}", path.display()).into());
    }
    let shape_text = header
        .split("'shape': (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .ok_or("npy header without shape")?;
    let shape: Vec<usize> = shape_text
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse())
        .collect::<std::result::Result<_, _>>()?;
    let values = raw[start + length..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok((shape, values))
}

#[derive(Deserialize)]
struct Step {
    top_ids: Vec<u32>,
    top_logits: Vec<f32>,
    logsumexp: f32,
    next: Option<u32>,
    next_logit: Option<f32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    language: String,
    task: String,
    sequence: Vec<u32>,
    sample_begin: usize,
    decoded_tokens: Vec<u32>,
    steps: Vec<Step>,
    lang_top5: Vec<(String, f64)>,
    mel_frames: usize,
}

fn audio_path(dir: &Path, name: &str) -> Option<PathBuf> {
    if let Some(rest) = name.strip_prefix("ls-") {
        let uid = rest.strip_suffix("-prompt").unwrap_or(rest);
        return Some(dir.join("librispeech").join(format!("{uid}.flac")));
    }
    let rest = name.strip_prefix("fleurs-")?.strip_suffix("-translate")?;
    let (lang, file) = rest.split_once('-')?;
    Some(dir.join("fleurs").join(lang).join(file))
}

fn max_abs(a: &[f32], b: &[f32]) -> (f32, f64) {
    let mut worst = 0.0_f32;
    let mut sum = 0.0_f64;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        worst = worst.max(d);
        sum += f64::from(d);
    }
    (worst, sum / a.len().max(1) as f64)
}

fn logsumexp(row: &[f32]) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|v| f64::from(v - max).exp()).sum();
    max + sum.ln() as f32
}

#[allow(clippy::too_many_lines)]
fn parity(dir: &Path) -> Result<()> {
    let cases: Vec<Case> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("parity/parity.json"))?)?;
    let mut model = open()?;
    let busy = || false;
    let mut pause = || Ok(());
    let mut pace = Pace {
        busy: &busy,
        pause: &mut pause,
    };
    let (mut steps, mut argmax_equal, mut top10_equal) = (0_usize, 0_usize, 0_usize);
    let (mut worst_logit, mut worst_lse) = (0.0_f32, 0.0_f32);
    let mut decodes_equal = 0;
    for case in &cases {
        let samples = if case.name.starts_with("silence") {
            vec![0.0_f32; 80_000]
        } else {
            let path = audio_path(dir, &case.name).ok_or("unknown case")?;
            load(&path)?
        };
        let mel = model.log_mel(&samples, PADDING);
        let content = mel.frames - 3000;
        if content != case.mel_frames {
            eprintln!(
                "{}: {content} content frames, reference {}",
                case.name, case.mel_frames
            );
        }
        let window = mel.window(0, content.min(3000));
        let mel_npy = dir.join(format!("parity/{}.mel.npy", case.name));
        let mut entry = json!({"name":case.name});
        if mel_npy.is_file() {
            let (shape, reference) = npy(&mel_npy)?;
            // Reference is [n_mels][3000]; ours is time-major.
            let n_mels = shape[0];
            let mut ours = vec![0.0_f32; reference.len()];
            for t in 0..3000 {
                for m in 0..n_mels {
                    ours[m * 3000 + t] = window[t * n_mels + m];
                }
            }
            let (worst, mean) = max_abs(&ours, &reference);
            entry["mel_max_abs"] = json!(worst);
            entry["mel_mean_abs"] = json!(mean);
        }
        let encoded = model.encoder_output(&window, &mut pace)?;
        let enc_npy = dir.join(format!("parity/{}.encoder.npy", case.name));
        if enc_npy.is_file() {
            let (_, reference) = npy(&enc_npy)?;
            let (worst, mean) = max_abs(&encoded, &reference);
            let scale = reference.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            entry["encoder_max_abs"] = json!(worst);
            entry["encoder_mean_abs"] = json!(mean);
            entry["encoder_ref_max"] = json!(scale);
        }
        let languages = model.window_languages(&mut pace)?;
        entry["lang_top"] = json!(languages.iter().take(3).collect::<Vec<_>>());
        entry["lang_ref_top"] = json!(case.lang_top5.iter().take(3).collect::<Vec<_>>());
        let prompt: Vec<u32> = if case.sample_begin > 4 {
            case.sequence[1..case.sample_begin - 3].to_vec()
        } else {
            Vec::new()
        };
        let task = if case.task == "translate" {
            Task::Translate
        } else {
            Task::Transcribe
        };
        let decoded = model.decode_window(task, &case.language, &prompt, 0.0, &mut pace)?;
        let equal = decoded == case.decoded_tokens;
        decodes_equal += usize::from(equal);
        entry["decode_equal"] = json!(equal);
        if !equal {
            let first = decoded
                .iter()
                .zip(&case.decoded_tokens)
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| decoded.len().min(case.decoded_tokens.len()));
            entry["decode_first_difference"] = json!(first);
            entry["decoded_text"] = json!(model.decode_text(&decoded));
            entry["reference_text"] = json!(model.decode_text(&case.decoded_tokens));
        }
        let logits = model.sequence_logits(&case.sequence, &mut pace)?;
        let begin = case.sample_begin - 1;
        let (mut case_worst, mut case_argmax) = (0.0_f32, 0);
        for (i, step) in case.steps.iter().enumerate() {
            let row = &logits[begin + i];
            let mut order: Vec<u32> = (0..row.len() as u32).collect();
            order.select_nth_unstable_by(10, |a, b| row[*b as usize].total_cmp(&row[*a as usize]));
            let mut top: Vec<u32> = order[..10].to_vec();
            top.sort_by(|a, b| row[*b as usize].total_cmp(&row[*a as usize]));
            steps += 1;
            if top[0] == step.top_ids[0] {
                argmax_equal += 1;
                case_argmax += 1;
            }
            let mut a = top.clone();
            let mut b = step.top_ids.clone();
            a.sort_unstable();
            b.sort_unstable();
            top10_equal += usize::from(a == b);
            for (id, value) in step.top_ids.iter().zip(&step.top_logits) {
                case_worst = case_worst.max((row[*id as usize] - value).abs());
            }
            if let (Some(next), Some(value)) = (step.next, step.next_logit) {
                case_worst = case_worst.max((row[next as usize] - value).abs());
            }
            worst_lse = worst_lse.max((logsumexp(row) - step.logsumexp).abs());
        }
        worst_logit = worst_logit.max(case_worst);
        entry["steps"] = json!(case.steps.len());
        entry["argmax_equal"] = json!(case_argmax);
        entry["logit_max_abs"] = json!(case_worst);
        println!("{entry}");
    }
    println!(
        "{}",
        json!({
            "cases":cases.len(),
            "greedy_decodes_identical":decodes_equal,
            "teacher_forced_steps":steps,
            "argmax_agreement":argmax_equal,
            "top10_set_agreement":top10_equal,
            "max_abs_logit_difference":worst_logit,
            "max_abs_logsumexp_difference":worst_lse,
        })
    );
    Ok(())
}

#[derive(Deserialize)]
struct Item {
    id: String,
    path: PathBuf,
    task: String,
    language: Option<String>,
    prompt: Option<String>,
}

fn transcribe(manifest: &Path, out: &Path) -> Result<()> {
    let items: Vec<Item> = serde_json::from_str(&std::fs::read_to_string(manifest)?)?;
    let base = manifest.parent().unwrap_or_else(|| Path::new("."));
    let mut model = open()?;
    let busy = || false;
    let mut pause = || Ok(());
    let mut pace = Pace {
        busy: &busy,
        pause: &mut pause,
    };
    let mut lines = String::new();
    let (mut audio_seconds, mut wall) = (0.0_f64, 0.0_f64);
    for item in &items {
        let samples = load(&base.join(&item.path))?;
        let options = TranscribeOptions {
            task: if item.task == "translate" {
                Task::Translate
            } else {
                Task::Transcribe
            },
            language: item.language.clone(),
            prompt: item.prompt.clone(),
            temperature: Temperature::Fixed(0.0),
            condition_on_previous_text: true,
            seed: 1,
        };
        let started = Instant::now();
        let transcript = model.transcribe(&samples, &options, &mut pace, &mut |_| Ok(()))?;
        let seconds = started.elapsed().as_secs_f64();
        audio_seconds += transcript.duration;
        wall += seconds;
        let segments: Vec<Value> = transcript
            .segments
            .iter()
            .map(|s| {
                json!({"seek":s.seek,"start":s.start,"end":s.end,"text":s.text,"tokens":s.tokens,
                            "avg_logprob":s.avg_logprob,"no_speech_prob":s.no_speech_prob,
                            "temperature":s.temperature})
            })
            .collect();
        let line = json!({"id":item.id,"text":transcript.text,"language":transcript.language,
                          "segments":segments,"seconds":seconds,"duration":transcript.duration});
        eprintln!("{} {:.2}s {}", item.id, seconds, transcript.text);
        lines.push_str(&line.to_string());
        lines.push('\n');
        std::fs::write(out, &lines)?;
    }
    eprintln!(
        "{} files, {audio_seconds:.1} s of audio in {wall:.1} s: real-time factor {:.4}",
        items.len(),
        wall / audio_seconds.max(1e-9)
    );
    Ok(())
}

fn bench(path: &Path) -> Result<()> {
    let samples = load(path)?;
    let mut model = open()?;
    let busy = || false;
    let mut pause = || Ok(());
    let mut pace = Pace {
        busy: &busy,
        pause: &mut pause,
    };
    let duration = samples.len() as f64 / 16_000.0;
    let mel = model.log_mel(&samples, PADDING);
    let window = mel.window(0, (mel.frames - 3000).min(3000));
    let mut encoder = f64::MAX;
    for _ in 0..3 {
        let started = Instant::now();
        model.encoder_output(&window, &mut pace)?;
        encoder = encoder.min(started.elapsed().as_secs_f64());
    }
    let started = Instant::now();
    let tokens = model.decode_window(Task::Transcribe, "en", &[], 0.0, &mut pace)?;
    let decode = started.elapsed().as_secs_f64();
    let options = TranscribeOptions {
        language: Some("en".into()),
        temperature: Temperature::Fixed(0.0),
        ..TranscribeOptions::default()
    };
    let mut best = f64::MAX;
    let mut text = String::new();
    for _ in 0..3 {
        let started = Instant::now();
        let transcript = model.transcribe(&samples, &options, &mut pace, &mut |_| Ok(()))?;
        best = best.min(started.elapsed().as_secs_f64());
        text = transcript.text;
    }
    println!(
        "{}",
        json!({
            "file":path.display().to_string(),
            "audio_seconds":duration,
            "encoder_window_seconds":encoder,
            "first_window_decode_seconds":decode,
            "first_window_tokens":tokens.len(),
            "decode_ms_per_token":decode * 1000.0 / (tokens.len() + 1) as f64,
            "transcribe_seconds":best,
            "real_time_factor":best / duration,
            "text":text,
        })
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [mode, dir] if mode == "parity" => parity(Path::new(dir)),
        [mode, manifest, out] if mode == "transcribe" => {
            transcribe(Path::new(manifest), Path::new(out))
        }
        [mode, file] if mode == "bench" => bench(Path::new(file)),
        _ => Err(
            "usage: transcription-eval parity DIR | transcribe MANIFEST OUT | bench AUDIO".into(),
        ),
    }
}
