//! Parity and throughput checks for Bonsai image input (see
//! `docs/BONSAI.md`, "Image input").
//!
//! ```text
//! bonsai-vision-eval tower OUT_DIR
//! bonsai-vision-eval llama OUT_DIR
//! bonsai-vision-eval video OUT_DIR CLIP...
//! bonsai-vision-eval bench IMAGE_FILE [low|auto|high]
//! ```
//!
//! `OUT_DIR` is the output of the private Kaggle reference job
//! `bonsai-vision-ref` (see docs/BONSAI.md, "Image input"): `images/`, `hf/`
//! (transformers `Qwen2VLImageProcessorFast` pixel values and the FP32
//! Qwen3.5 vision tower's outputs and intermediates), `llama/` (llama.cpp
//! mtmd token layouts, image embeddings, logits and greedy continuations
//! from Prism's fork) and
//! `log.json`. The model and projector load from `models/bonsai2-27b-ptq1/`.
//!
//! - `tower`: preprocessing (pixel values must be identical) and the tower
//!   (cosine and largest error of every reported intermediate and of the
//!   output) for every transformers case.
//! - `llama`: per prompt, the token layout, the image embeddings against
//!   llama.cpp's, then the decoder end to end: next-token logits after the
//!   prompt and after each of llama.cpp's first four greedy tokens (KL,
//!   top-1, cosine), with the native tower's embeddings and with llama.cpp's
//!   own, and the native greedy continuation against llama.cpp's.
//! - `video`: per clip, the native decode (probe, sampled indices and
//!   frames), preprocessing (patch rows), tower output and prompt layout
//!   (expanded text and M-RoPE positions) under `OUT_DIR/<clip>/`, for the
//!   transformers comparison (`Qwen3VLVideoProcessor`, the Qwen3.5 tower and
//!   `get_rope_index`; see docs/BONSAI.md, "Video input").
//! - `bench`: tower latency of one image at a detail.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use local_engine::MtpMode;
use local_engine::bonsai::{BonsaiPackage, WIDTH};
use local_engine::bonsai_model::BonsaiEngine;
use local_engine::bonsai_native::KvOptions;
use local_engine::bonsai_ngram::NgramSettings;
use local_engine::bonsai_tokenizer::BonsaiTokenizer;
use local_engine::bonsai_vision::video::{decode_sampled, timestamp_text};
use local_engine::bonsai_vision::{
    IMAGE_PLACEHOLDER, ImageDetail, VIDEO_PAD, VISION_END, VISION_PROJECTOR_FILE, VISION_START,
    VisionEncoder, Visual, expand_prompt, prepare_decoded, prepare_image, prepare_video,
    rotate_rows,
};
use local_engine::embedding::vision::preprocess::decode_image;
use serde_json::Value;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const MODEL: &str = "models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf";

fn projector() -> PathBuf {
    Path::new(MODEL).with_file_name(VISION_PROJECTOR_FILE)
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect())
}

/// Cosine and largest absolute difference.
#[allow(clippy::suboptimal_flops)]
fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut max) = (0.0, 0.0, 0.0, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
        max = max.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), max)
}

/// Smallest per-row cosine over rows of `width`.
fn min_row_cosine(a: &[f32], b: &[f32], width: usize) -> f64 {
    a.chunks(width)
        .zip(b.chunks(width))
        .map(|(x, y)| compare(x, y).0)
        .fold(f64::INFINITY, f64::min)
}

fn encoder(signs: &[f32]) -> Result<VisionEncoder> {
    let started = Instant::now();
    let encoder = VisionEncoder::open(&projector(), signs)?;
    eprintln!(
        "projector verified and loaded in {:.2}s",
        started.elapsed().as_secs_f64()
    );
    Ok(encoder)
}

fn package_signs() -> Result<Vec<f32>> {
    Ok(BonsaiPackage::open(MODEL)?
        .hadamard()
        .signs(WIDTH as u32)?
        .to_vec())
}

fn tower(out: &Path) -> Result<()> {
    let log: Value = serde_json::from_slice(&std::fs::read(out.join("log.json"))?)?;
    let encoder = encoder(&package_signs()?)?;
    for case in log["hf_cases"].as_array().ok_or("no hf_cases")? {
        let Some(tag) = case["tag"].as_str() else {
            continue;
        };
        if case.get("error").is_some() {
            println!("{tag}: reference failed");
            continue;
        }
        let image = case["image"].as_str().ok_or("image")?;
        let max_pixels = case["max_pixels"].as_u64().ok_or("max_pixels")? as usize;
        let bytes = std::fs::read(out.join("images").join(image))?;
        let decoded = decode_image(&bytes)?;
        let prepared = prepare_decoded(&decoded, max_pixels)?;
        let grid = case["grid_thw"].as_array().ok_or("grid")?;
        let (gh, gw) = (
            grid[1].as_u64().ok_or("grid")? as usize,
            grid[2].as_u64().ok_or("grid")? as usize,
        );
        if prepared.grid() != (gh, gw) {
            println!(
                "{tag}: grid {:?} differs from the reference {:?}",
                prepared.grid(),
                (gh, gw)
            );
            continue;
        }
        // Reference rows are 1536: (channel, temporal, 16, 16).
        let reference = read_f32(&out.join("hf").join(format!("{tag}.pixels.f32")))?;
        let native = prepared.patch_rows();
        let mut mismatched = 0usize;
        let mut max_diff = 0.0_f32;
        for (row, reference) in native.chunks(768).zip(reference.chunks(1536)) {
            for channel in 0..3 {
                for t in 0..2 {
                    let want = &reference[channel * 512 + t * 256..channel * 512 + t * 256 + 256];
                    let got = &row[channel * 256..channel * 256 + 256];
                    for (a, b) in got.iter().zip(want) {
                        if a.to_bits() != b.to_bits() {
                            mismatched += 1;
                            max_diff = max_diff.max((a - b).abs());
                        }
                    }
                }
            }
        }
        let mut probes = Vec::new();
        let started = Instant::now();
        let embedding = encoder.encode_raw(&prepared, None, &mut |name, values| {
            let path = out.join("hf").join(format!("{tag}.{name}.f32"));
            if path.is_file() {
                probes.push((name.to_owned(), values.to_vec(), path));
            }
        })?;
        let seconds = started.elapsed().as_secs_f64();
        let reference = read_f32(&out.join("hf").join(format!("{tag}.embd.f32")))?;
        let (cosine, max) = compare(&embedding, &reference);
        println!(
            "{tag}: grid {gh}x{gw}, {} tokens, pixels {} of {} differ (max {max_diff:e}); \
             output cosine {cosine:.7} (min row {:.6}), max |diff| {max:.4e}; {seconds:.2}s",
            prepared.tokens(),
            mismatched,
            native.len() * 2,
            min_row_cosine(&embedding, &reference, WIDTH),
        );
        for (name, values, path) in probes {
            let reference = read_f32(&path)?;
            let (cosine, max) = compare(&values, &reference);
            println!("  {name}: cosine {cosine:.7}, max |diff| {max:.4e}");
        }
    }
    Ok(())
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = logits
        .iter()
        .map(|&x| (f64::from(x) - f64::from(max)).exp())
        .collect::<Vec<_>>();
    let sum = exps.iter().sum::<f64>();
    exps.into_iter().map(|e| e / sum).collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

/// KL(reference || native), top-1 agreement and cosine of two logit rows.
fn logit_stats(reference: &[f32], native: &[f32]) -> (f64, bool, f64) {
    let (p, q) = (softmax(reference), softmax(native));
    let kl = p
        .iter()
        .zip(&q)
        .filter(|(p, _)| **p > 0.0)
        .map(|(p, q)| p * (p / q.max(1e-300)).ln())
        .sum::<f64>();
    (
        kl,
        argmax(reference) == argmax(native),
        compare(reference, native).0,
    )
}

struct LlamaImage {
    tokens: usize,
    nx: usize,
    ny: usize,
}

fn parse_layout(text: &str) -> (Vec<Vec<u32>>, Vec<LlamaImage>) {
    let mut texts = Vec::new();
    let mut images = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("T") => texts.push(parts.filter_map(|p| p.parse().ok()).collect()),
            Some("I") => {
                let v = parts
                    .filter_map(|p| p.parse::<usize>().ok())
                    .collect::<Vec<_>>();
                images.push(LlamaImage {
                    tokens: v[0],
                    nx: v[1],
                    ny: v[2],
                });
            }
            _ => {}
        }
    }
    (texts, images)
}

#[allow(clippy::too_many_lines)]
fn llama(out: &Path) -> Result<()> {
    let log: Value = serde_json::from_slice(&std::fs::read(out.join("log.json"))?)?;
    let mut engine = BonsaiEngine::open_with_options(
        Path::new(MODEL),
        Some(16_384),
        128,
        None,
        &MtpMode::Off(None),
        NgramSettings::default(),
        KvOptions::default(),
    )?;
    let signs = engine.input_signs()?.to_vec();
    let encoder = encoder(&signs)?;
    for case in log["llama_cases"].as_array().ok_or("no llama_cases")? {
        let name = case["name"].as_str().ok_or("name")?;
        if case["rc"].as_i64() != Some(0) {
            println!("{name}: reference failed (rc {})", case["rc"]);
            continue;
        }
        let dir = out.join("llama");
        let prompt = std::fs::read_to_string(dir.join(format!("{name}.prompt.txt")))?
            .replace("<__media__>", IMAGE_PLACEHOLDER);
        let ids = engine.tokenizer().encode(&prompt)?;
        let (texts, layout) = parse_layout(&std::fs::read_to_string(
            dir.join(format!("{name}.tokens.txt")),
        )?);
        let files = case["images"]
            .as_array()
            .ok_or("images")?
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        let prepared = files
            .iter()
            .map(|file| {
                prepare_image(
                    &std::fs::read(out.join("images").join(file))?,
                    ImageDetail::Auto,
                )
                .map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?;
        // Our text tokens between images, against llama.cpp's text chunks.
        let native_texts = ids
            .split(|&id| id == local_engine::bonsai_vision::IMAGE_PAD)
            .map(<[u32]>::to_vec)
            .collect::<Vec<_>>();
        println!("{name}: text chunks identical: {}", native_texts == texts);
        let mut native_rows = Vec::new();
        let mut llama_rows = Vec::new();
        let mut same_grids = true;
        for (index, (image, reference)) in prepared.iter().zip(&layout).enumerate() {
            let rows = encoder.encode_raw(image, None, &mut |_, _| {})?;
            let grid_matches = (image.cols, image.rows) == (reference.nx, reference.ny)
                && image.tokens() == reference.tokens;
            same_grids &= grid_matches;
            let reference_rows = read_f32(&dir.join(format!("{name}.embd{index}.f32")))?;
            if grid_matches {
                let (cosine, max) = compare(&rows, &reference_rows);
                println!(
                    "  image {index}: {}x{} tokens, embedding cosine vs llama.cpp {cosine:.6} \
                     (min row {:.5}), max |diff| {max:.4e}",
                    image.rows,
                    image.cols,
                    min_row_cosine(&rows, &reference_rows, WIDTH)
                );
            } else {
                println!(
                    "  image {index}: native grid {}x{} ({} tokens), llama.cpp {}x{} ({} tokens)",
                    image.rows,
                    image.cols,
                    image.tokens(),
                    reference.ny,
                    reference.nx,
                    reference.tokens
                );
            }
            native_rows.push(rows);
            llama_rows.push(reference_rows);
        }
        let generated = std::fs::read_to_string(dir.join(format!("{name}.gen.txt")))?;
        let reference_tokens = generated
            .lines()
            .skip(1)
            .filter_map(|line| line.split_whitespace().next()?.parse::<u32>().ok())
            .collect::<Vec<_>>();
        let reference_logits = (0..5)
            .map(|step| read_f32(&dir.join(format!("{name}.logits{step}.f32"))))
            .take_while(Result::is_ok)
            .collect::<Result<Vec<_>>>()?;
        let continuation =
            &reference_tokens[..(reference_logits.len() - 1).min(reference_tokens.len())];
        let mut variants: Vec<(&str, Vec<Vec<f32>>)> = vec![("native tower", native_rows)];
        if same_grids && !prepared.is_empty() {
            variants.push(("llama.cpp embeddings", llama_rows));
        }
        for (label, embeddings) in variants {
            let pairs = prepared
                .iter()
                .zip(embeddings)
                .map(|(image, mut rows)| {
                    rotate_rows(&mut rows, &signs);
                    Visual::Image(image, Arc::<[f32]>::from(rows))
                })
                .collect::<Vec<_>>();
            let (expanded, images) = expand_prompt(&ids, encoder.digest(), &pairs)?;
            let images = (!images.is_empty()).then(|| Arc::new(images));
            let logits = engine.teacher_forced_logits(&expanded, images.clone(), continuation)?;
            let stats = reference_logits
                .iter()
                .zip(&logits)
                .map(|(r, n)| logit_stats(r, n))
                .collect::<Vec<_>>();
            println!(
                "  {label}: prompt {} tokens; KL {}; top-1 {}/{}; cosine {}",
                expanded.len(),
                stats
                    .iter()
                    .map(|s| format!("{:.2e}", s.0))
                    .collect::<Vec<_>>()
                    .join(" "),
                stats.iter().filter(|s| s.1).count(),
                stats.len(),
                stats
                    .iter()
                    .map(|s| format!("{:.5}", s.2))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            // Native greedy continuation.
            let mut tokens = Vec::new();
            let mut forced = Vec::new();
            for _ in 0..reference_tokens.len() {
                let logits = engine.teacher_forced_logits(&expanded, images.clone(), &forced)?;
                let next = argmax(logits.last().ok_or("logits")?) as u32;
                tokens.push(next);
                if next == 248_046 {
                    break;
                }
                forced.push(next);
            }
            let agree = tokens
                .iter()
                .zip(&reference_tokens)
                .take_while(|(a, b)| a == b)
                .count();
            println!(
                "    greedy: {agree}/{} tokens agree; native {:?}; llama.cpp {:?}",
                reference_tokens.len(),
                engine.tokenizer().decode(&tokens, true)?,
                engine.tokenizer().decode(&reference_tokens, true)?,
            );
        }
    }
    Ok(())
}

fn bench(file: &Path, detail: ImageDetail) -> Result<()> {
    let encoder = encoder(&package_signs()?)?;
    let bytes = std::fs::read(file)?;
    let started = Instant::now();
    let image = prepare_image(&bytes, detail)?;
    let prepare = started.elapsed().as_secs_f64();
    for run in 0..3 {
        let started = Instant::now();
        let rows = encoder.encode_raw(&image, None, &mut |_, _| {})?;
        println!(
            "run {run}: {}x{} merged ({} tokens, {} patches): preprocess {prepare:.3}s, tower {:.3}s, \
             output norm {:.3}",
            image.rows,
            image.cols,
            image.tokens(),
            image.patches(),
            started.elapsed().as_secs_f64(),
            rows.iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt()
        );
    }
    Ok(())
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Dump each clip's native video preprocessing, tower output and prompt
/// layout under `out/<stem>/` for the transformers comparison.
#[allow(clippy::too_many_lines)]
fn video(out: &Path, clips: &[String]) -> Result<()> {
    let package = BonsaiPackage::open(MODEL)?;
    let tokenizer = BonsaiTokenizer::from_package(&package)?;
    let encoder = encoder(package.hadamard().signs(WIDTH as u32)?)?;
    for clip in clips {
        let path = Path::new(clip);
        let stem = path.file_stem().ok_or("clip name")?.to_string_lossy();
        let dir = out.join(stem.as_ref());
        std::fs::create_dir_all(&dir)?;
        let bytes = std::fs::read(path)?;
        let started = Instant::now();
        let (probe, indices, frames) = decode_sampled(&bytes)?;
        let decoded = started.elapsed().as_secs_f64();
        let mut raw = Vec::new();
        for frame in &frames {
            raw.extend_from_slice(&frame.pixels);
        }
        // The FFmpeg command-line path (swscale-exact, torchcodec's frames)
        // as the decode reference.
        let stream = local_engine::embedding::vision::video::probe(path)?;
        let ffmpeg =
            local_engine::embedding::vision::video::decode_frames(path, &stream, &indices)?;
        let differ = frames
            .iter()
            .zip(&ffmpeg)
            .map(|(a, b)| {
                a.pixels
                    .iter()
                    .zip(&b.pixels)
                    .filter(|(x, y)| x != y)
                    .count()
            })
            .sum::<usize>();
        println!(
            "{}: decode vs FFmpeg ({:?}, {:?}, {} frames, {:?} fps): {differ} of {} values differ",
            path.display(),
            stream.pixel_format,
            stream.color_space,
            stream.timing.frames,
            stream.timing.fps,
            raw.len(),
        );
        std::fs::write(dir.join("frames.u8"), raw)?;
        let started = Instant::now();
        let video = prepare_video(&bytes)?;
        let prepared = started.elapsed().as_secs_f64();
        let first = video.groups.first().ok_or("no groups")?;
        let mut pixels = Vec::new();
        let mut embedding = Vec::new();
        let started = Instant::now();
        for group in &video.groups {
            pixels.extend(group.patch_rows());
            embedding.extend(encoder.encode_raw(group, None, &mut |_, _| {})?);
        }
        let tower = started.elapsed().as_secs_f64();
        write_f32(&dir.join("pixels.f32"), &pixels)?;
        write_f32(&dir.join("embd.f32"), &embedding)?;
        // A prompt: text, the template's video placeholder, text.
        let mut ids = tokenizer.encode("Describe:")?;
        ids.extend([VISION_START, VIDEO_PAD, VISION_END]);
        ids.extend(tokenizer.encode(" Done.")?);
        let groups = video
            .groups
            .iter()
            .zip(&video.timestamps)
            .map(|(group, &seconds)| {
                Ok((
                    tokenizer.encode(&timestamp_text(seconds))?,
                    group,
                    vec![0.0; group.tokens() * WIDTH].into(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let (expanded, images) = expand_prompt(&ids, encoder.digest(), &[Visual::Video(groups)])?;
        let mut text = String::new();
        for (position, &id) in expanded.iter().enumerate() {
            if images.covers(position) {
                text.push_str("<|video_pad|>");
            } else {
                text.push_str(&tokenizer.decode(&[id], false)?);
            }
        }
        let rope = (0..expanded.len())
            .map(|position| images.rope(position))
            .collect::<Vec<_>>();
        let record = serde_json::json!({
            "clip": clip,
            "total_frames": probe.total_frames,
            "fps": probe.fps,
            "width": probe.width,
            "height": probe.height,
            "indices": indices,
            "frames": frames.len(),
            "resized": [first.image.height, first.image.width],
            "grid_thw": [video.groups.len(), first.rows * 2, first.cols * 2],
            "timestamps": video.timestamps,
            "tokens": video.tokens(),
            "prompt_text": text,
            "rope": rope,
            "video_token": (0..expanded.len()).map(|position| images.covers(position)).collect::<Vec<_>>(),
            "video_spans": images.spans().iter().map(|span| span.start).collect::<Vec<_>>(),
            "seconds": {"decode": decoded, "prepare": prepared, "tower": tower},
        });
        std::fs::write(dir.join("native.json"), serde_json::to_vec_pretty(&record)?)?;
        println!(
            "{stem}: {} frames at {:.3} fps, sampled {}, {} groups of {}x{} merged ({} tokens); \
             decode {decoded:.2}s, prepare {prepared:.2}s, tower {tower:.2}s",
            probe.total_frames,
            probe.fps,
            indices.len(),
            video.groups.len(),
            first.rows,
            first.cols,
            video.tokens(),
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        Some("tower") if args.len() == 2 => tower(Path::new(&args[1])),
        Some("llama") if args.len() == 2 => llama(Path::new(&args[1])),
        Some("video") if args.len() >= 3 => video(Path::new(&args[1]), &args[2..]),
        Some("bench") if args.len() >= 2 => bench(
            Path::new(&args[1]),
            args.get(2)
                .map_or(Ok(ImageDetail::Auto), |d| ImageDetail::parse(d))?,
        ),
        _ => Err(
            "usage: bonsai-vision-eval tower|llama OUT_DIR | video OUT_DIR CLIP... | bench IMAGE \
             [low|auto|high]"
                .into(),
        ),
    }
}
