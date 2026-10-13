//! Parity, retrieval and throughput checks for the native `EmbeddingGemma` 2
//! audio encoder (see `docs/EMBEDDINGS.md`).
//!
//! ```text
//! embedding-audio-eval features REF_DIR [AUDIOCAPS_DIR]
//! embedding-audio-eval parity REF_DIR
//! embedding-audio-eval retrieval AUDIOCAPS_DIR
//! embedding-audio-eval bench [--clips N] [--seconds S]
//! ```
//!
//! The model loads from `models/embeddinggemma2-text` (text artifact plus
//! the audio add-on) under the working directory. `REF_DIR` holds the
//! reference kernel's fixtures (sentence-transformers 6.1.0, transformers
//! 5.19.0, FP32) as `index.json` plus little-endian F32 files: per entry the
//! 16 kHz waveform as fed (`wave`), the log-mel features (`feat`, `[frames,
//! 128]`), the `embed_audio` soft tokens (`soft`, `[tokens, 512]`), the
//! normalized 768-d embedding (`emb`, inline) and, for the resampling probe,
//! a 44.1 kHz stereo waveform (`wave44k1`).
//!
//! - `features`: frame counts and the largest absolute log-mel difference on
//!   valid frames; with `AUDIOCAPS_DIR`, also whether Symphonia's FLAC decode
//!   of each source clip equals the reference's soundfile decode bit for bit.
//! - `parity`: cosine of native soft tokens and embeddings to the reference
//!   (mean and minimum), packing independence, and the 44.1 kHz stereo probe
//!   (native downmix + resampler vs librosa).
//! - `retrieval`: `AudioCaps` test (the Kaggle mirror's `audio/test/*.flac`
//!   and `test_texts.json`): R@1/5/10 text→audio with query-prompted
//!   captions and audio→text with document-prompted captions (the reference
//!   protocol), plus every caption form. With `AUDIO_EVAL_DUMP=DIR` the raw
//!   embeddings are written there too.
//! - `bench`: clips per second for packed batches of synthetic clips.

use std::path::{Path, PathBuf};
use std::time::Instant;

use local_engine::embedding::{
    EmbeddingModel, FeatureExtractor, MEL_BINS, decode_audio, installed_model, to_mono_16k,
};
use serde::Deserialize;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
struct Entry {
    name: String,
    clip: serde_json::Value,
    crop: Option<usize>,
    frames: usize,
    valid_frames: usize,
    audio_tokens: usize,
    wave: String,
    feat: String,
    soft: String,
    emb: Vec<f32>,
    wave44k1: Option<String>,
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

fn entries(dir: &Path) -> Result<Vec<Entry>> {
    Ok(serde_json::from_slice(&std::fs::read(
        dir.join("index.json"),
    )?)?)
}

fn open() -> Result<EmbeddingModel> {
    let dir = installed_model(Path::new("models"))
        .ok_or("models/embeddinggemma2-text is not installed (see docs/EMBEDDINGS.md)")?;
    let started = Instant::now();
    let mut model = EmbeddingModel::open(&dir)?;
    model.load_audio(&dir)?;
    eprintln!("loaded {} (+audio) in {:.2?}", dir.display(), started.elapsed());
    Ok(model)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt())
}

fn summary(values: &[f32]) -> (f64, f32, usize) {
    let mean = values.iter().map(|c| f64::from(*c)).sum::<f64>() / values.len() as f64;
    let (worst, min) = values
        .iter()
        .enumerate()
        .fold((0, f32::INFINITY), |acc, (i, c)| if *c < acc.1 { (i, *c) } else { acc });
    (mean, min, worst)
}

fn features(dir: &Path, audiocaps: Option<&Path>) -> Result<()> {
    let extractor = FeatureExtractor::new();
    let (mut worst, mut worst_name, mut counts_ok) = (0.0_f32, String::new(), 0);
    let (mut decoded, mut identical) = (0, 0);
    let list = entries(dir)?;
    for entry in &list {
        let wave = read_f32(&dir.join(&entry.wave))?;
        let reference = read_f32(&dir.join(&entry.feat))?;
        let native = extractor.extract(&wave);
        if (native.frames, native.valid) == (entry.frames, entry.valid_frames)
            && reference.len() == native.features.len()
        {
            counts_ok += 1;
        } else {
            println!(
                "{}: native frames {}/{} vs reference {}/{}",
                entry.name, native.valid, native.frames, entry.valid_frames, entry.frames
            );
            continue;
        }
        let diff = native
            .features
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        if diff > worst {
            (worst, worst_name) = (diff, entry.name.clone());
        }
        if let Some(file) = &entry.wave44k1 {
            let stereo = read_f32(&dir.join(file))?;
            let probe = extractor.extract(&to_mono_16k(&stereo, 2, 44_100)?);
            let n = probe.features.len().min(reference.len());
            let mean = probe.features[..n]
                .iter()
                .zip(&reference[..n])
                .map(|(a, b)| f64::from((a - b).abs()))
                .sum::<f64>()
                / n as f64;
            let top = (0..n)
                .filter(|i| i % MEL_BINS >= 120)
                .map(|i| f64::from((probe.features[i] - reference[i]).abs()))
                .sum::<f64>()
                / (n / MEL_BINS * 8) as f64;
            println!(
                "resampling probe {}: mean |log-mel difference| native vs librosa {mean:.4} (top 8 bins {top:.4})",
                entry.name
            );
        }
        if let (Some(root), Some(key), None, None) =
            (audiocaps, entry.clip.as_str(), entry.crop, &entry.wave44k1)
        {
            let path = root.join("audio/test").join(format!("{key}.flac"));
            let samples = decode_audio(std::fs::read(path)?)?;
            decoded += 1;
            identical += usize::from(samples == wave);
        }
    }
    println!(
        "frame counts identical: {counts_ok}/{}; max |log-mel difference| over all frames: {worst:.3e} ({worst_name}); {} frames x {MEL_BINS} bins compared",
        list.len(),
        list.iter().map(|e| e.frames).sum::<usize>()
    );
    if audiocaps.is_some() {
        println!("FLAC decodes bitwise identical to the reference waveform: {identical}/{decoded}");
    }
    Ok(())
}

fn parity(dir: &Path) -> Result<()> {
    let list = entries(dir)?;
    let waves = list
        .iter()
        .map(|e| read_f32(&dir.join(&e.wave)))
        .collect::<Result<Vec<_>>>()?;
    let mut model = open()?;
    let clips: Vec<&[f32]> = waves.iter().map(Vec::as_slice).collect();
    let soft = model.audio_soft_tokens(&clips, &mut || Ok(()))?;
    let mut soft_min = Vec::new();
    let mut soft_mean = Vec::new();
    for (entry, native) in list.iter().zip(&soft) {
        let reference = read_f32(&dir.join(&entry.soft))?;
        if reference.len() != native.len() || native.len() != entry.audio_tokens * 512 {
            return Err(format!(
                "{}: {} native soft values vs {} reference ({} tokens)",
                entry.name,
                native.len(),
                reference.len(),
                entry.audio_tokens
            )
            .into());
        }
        let rows: Vec<f32> = native
            .as_chunks::<512>()
            .0
            .iter()
            .zip(reference.as_chunks::<512>().0)
            .map(|(a, b)| cosine(a, b))
            .collect();
        let (mean, min, _) = summary(&rows);
        soft_mean.push(mean as f32);
        soft_min.push(min);
    }
    let (mean, _, _) = summary(&soft_mean);
    let (_, min, worst) = summary(&soft_min);
    println!(
        "soft tokens: mean per-clip cosine {mean:.6}, min token cosine {min:.6} ({})",
        list[worst].name
    );
    let started = Instant::now();
    let embeddings = model.embed_audio(&clips, &mut || Ok(()))?;
    let elapsed = started.elapsed();
    let cosines: Vec<f32> = embeddings
        .iter()
        .zip(&list)
        .map(|(native, entry)| dot(native, &entry.emb))
        .collect();
    for (entry, c) in list.iter().zip(&cosines) {
        println!("  {:<28} tokens {:>3}  cosine {c:.6}", entry.name, entry.audio_tokens);
    }
    let (mean, min, worst) = summary(&cosines);
    println!(
        "embedding cosine to reference over {} clips: mean {mean:.6}, min {min:.6} ({})",
        cosines.len(),
        list[worst].name
    );
    let seconds: f64 = waves.iter().map(|w| w.len() as f64 / 16_000.0).sum();
    println!(
        "{seconds:.1} s of audio in {elapsed:.2?} ({:.1} clips/s, {:.0}x real time)",
        clips.len() as f64 / elapsed.as_secs_f64(),
        seconds / elapsed.as_secs_f64()
    );
    let (mut identical, mut alone_min) = (0, f32::INFINITY);
    for (clip, packed) in clips.iter().zip(&embeddings) {
        let single = model.embed_audio(std::slice::from_ref(clip), &mut || Ok(()))?;
        alone_min = alone_min.min(dot(&single[0], packed));
        identical += usize::from(single[0] == *packed);
    }
    println!(
        "packing independence over {} clips: {identical} bitwise identical, min cosine {alone_min:.7}",
        clips.len()
    );
    for entry in &list {
        if let Some(file) = &entry.wave44k1 {
            let stereo = read_f32(&dir.join(file))?;
            let mono = to_mono_16k(&stereo, 2, 44_100)?;
            let native = model.embed_audio(&[mono.as_slice()], &mut || Ok(()))?;
            println!(
                "44.1 kHz stereo probe {}: native downmix+resample vs librosa reference cosine {:.6} ({} vs {} samples)",
                entry.name,
                dot(&native[0], &entry.emb),
                mono.len(),
                read_f32(&dir.join(&entry.wave))?.len()
            );
        }
    }
    Ok(())
}

const CAPTION_FORMS: [(&str, &str); 3] = [
    ("q", "task: search result | query: "),
    ("d", "title: none | text: "),
    ("raw", ""),
];

fn recall(scores: &[Vec<f32>], relevant: &[Vec<usize>]) -> [f64; 3] {
    let mut hits = [0.0; 3];
    for (row, rel) in scores.iter().zip(relevant) {
        let mut order: Vec<usize> = (0..row.len()).collect();
        order.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
        for (k, hit) in [1, 5, 10].iter().zip(hits.iter_mut()) {
            if order[..*k].iter().any(|i| rel.contains(i)) {
                *hit += 1.0;
            }
        }
    }
    hits.map(|h| 100.0 * h / scores.len() as f64)
}

fn similarity(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    a.iter().map(|x| b.iter().map(|y| dot(x, y)).collect()).collect()
}

fn retrieval(root: &Path) -> Result<()> {
    let texts: std::collections::BTreeMap<String, Vec<String>> =
        serde_json::from_slice(&std::fs::read(root.join("test_texts.json"))?)?;
    let mut keys: Vec<&String> = texts.keys().collect();
    keys.sort_by_key(|k| k.parse::<u64>().unwrap_or(u64::MAX));
    let mut clips = Vec::new();
    let mut captions = Vec::new();
    let mut caption_clip = Vec::new();
    let started = Instant::now();
    for key in keys {
        let path: PathBuf = root.join("audio/test").join(format!("{key}.flac"));
        if !path.exists() {
            continue;
        }
        for caption in &texts[key] {
            captions.push(caption.trim().to_owned());
            caption_clip.push(clips.len());
        }
        clips.push(decode_audio(std::fs::read(path)?)?);
    }
    println!(
        "AudioCaps test: {} clips, {} captions (decoded in {:.2?})",
        clips.len(),
        captions.len(),
        started.elapsed()
    );
    let mut model = open()?;
    let started = Instant::now();
    let mut media = Vec::with_capacity(clips.len());
    for chunk in clips.chunks(32) {
        let refs: Vec<&[f32]> = chunk.iter().map(Vec::as_slice).collect();
        media.extend(model.embed_audio(&refs, &mut || Ok(()))?);
    }
    let elapsed = started.elapsed();
    let seconds: f64 = clips.iter().map(|c| c.len() as f64 / 16_000.0).sum();
    println!(
        "embedded {} clips ({seconds:.0} s of audio) in {elapsed:.2?}: {:.1} clips/s, {:.0}x real time",
        clips.len(),
        clips.len() as f64 / elapsed.as_secs_f64(),
        seconds / elapsed.as_secs_f64()
    );
    let mut forms = Vec::new();
    for (name, prefix) in CAPTION_FORMS {
        let formatted: Vec<String> = captions.iter().map(|c| format!("{prefix}{c}")).collect();
        let refs: Vec<&str> = formatted.iter().map(String::as_str).collect();
        forms.push((name, model.embed_texts(&refs, &mut || Ok(()))?));
    }
    if let Some(dump) = std::env::var_os("AUDIO_EVAL_DUMP") {
        // Optional: raw F32 embeddings for offline analysis.
        let dump = PathBuf::from(dump);
        std::fs::create_dir_all(&dump)?;
        let write = |name: &str, rows: &[Vec<f32>]| -> Result<()> {
            let bytes: Vec<u8> = rows.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
            Ok(std::fs::write(dump.join(name), bytes)?)
        };
        write("media.f32", &media)?;
        for (name, embeddings) in &forms {
            write(&format!("captions-{name}.f32"), embeddings)?;
        }
        std::fs::write(dump.join("caption_clip.json"), serde_json::to_vec(&caption_clip)?)?;
    }
    let t2m_relevant: Vec<Vec<usize>> = caption_clip.iter().map(|c| vec![*c]).collect();
    let mut m2t_relevant = vec![Vec::new(); clips.len()];
    for (caption, clip) in caption_clip.iter().enumerate() {
        m2t_relevant[*clip].push(caption);
    }
    let fmt = |r: [f64; 3]| {
        format!(
            "{:.1}/{:.1}/{:.1} (mean {:.1})",
            r[0],
            r[1],
            r[2],
            (r[0] + r[1] + r[2]) / 3.0
        )
    };
    let mut t2m = Vec::new();
    let mut m2t = Vec::new();
    for (name, embeddings) in &forms {
        t2m.push((*name, recall(&similarity(embeddings, &media), &t2m_relevant)));
        m2t.push((*name, recall(&similarity(&media, embeddings), &m2t_relevant)));
    }
    for (name, r) in &t2m {
        println!("text->audio, captions {name:<3}: R@1/5/10 {}", fmt(*r));
    }
    for (name, r) in &m2t {
        println!("audio->text, captions {name:<3}: R@1/5/10 {}", fmt(*r));
    }
    println!(
        "reference protocol: t2m (query prompt) {}, m2t (document prompt) {}",
        fmt(t2m[0].1),
        fmt(m2t[1].1)
    );
    Ok(())
}

fn bench(args: &[String]) -> Result<()> {
    let value = |flag: &str, default: usize| -> Result<usize> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map_or(Ok(default), |v| v.parse().map_err(Into::into))
    };
    let count = value("--clips", 32)?;
    let seconds = value("--seconds", 10)?;
    let mut state = 0x2545_f491_u32;
    let clips: Vec<Vec<f32>> = (0..count)
        .map(|_| {
            (0..seconds * 16_000)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (f64::from(state) / f64::from(u32::MAX) - 0.5) as f32 * 0.2
                })
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = clips.iter().map(Vec::as_slice).collect();
    let mut model = open()?;
    model.embed_audio(&refs[..1], &mut || Ok(()))?;
    for (label, batch) in [("1 clip", &refs[..1]), ("batch", &refs[..])] {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let started = Instant::now();
            model.embed_audio(batch, &mut || Ok(()))?;
            best = best.min(started.elapsed().as_secs_f64());
        }
        println!(
            "{label}: {} x {seconds} s clips in {:.1} ms (best of 3): {:.1} clips/s, {:.0}x real time",
            batch.len(),
            best * 1e3,
            batch.len() as f64 / best,
            (batch.len() * seconds) as f64 / best
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("features") if args.len() >= 3 => {
            features(Path::new(&args[2]), args.get(3).map(Path::new))
        }
        Some("parity") if args.len() == 3 => parity(Path::new(&args[2])),
        Some("retrieval") if args.len() == 3 => retrieval(Path::new(&args[2])),
        Some("bench") => bench(&args[2..]),
        _ => Err("usage: embedding-audio-eval features REF_DIR [AUDIOCAPS_DIR] | parity REF_DIR | \
                  retrieval AUDIOCAPS_DIR | bench [--clips N] [--seconds S]"
            .into()),
    }
}
