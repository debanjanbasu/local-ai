//! Preprocessing parity, embedding parity, retrieval and throughput checks
//! for the native `EmbeddingGemma` 2 image and video embeddings (see
//! `docs/EMBEDDINGS.md`).
//!
//! ```text
//! embedding-vision-eval preprocess FIXTURES/index.json
//! embedding-vision-eval parity FIXTURES/index.json
//! embedding-vision-eval retrieval FLICKR_DIR
//! embedding-vision-eval bench IMAGE_FILE [--images N]
//! ```
//!
//! The model loads from `models/embeddinggemma2-text` (text and vision
//! artifacts) under the working directory. `index.json` lists the reference
//! fixtures generated on Kaggle (transformers 5.19.0, sentence-transformers
//! 6.1.0, FP32) and unpacked to raw files next to it:
//!
//! - images: `{"name", "file", "raw", "raw_shape" [3, h, w], "pv", "columns",
//!   "rows", "ids", "emb"}`; `raw` is the decoded `uint8` CHW image the
//!   processor received, `pv` the processor's patches as `uint8` (the values
//!   are exactly `u8 * f32(1/255)`), `ids` the reference token ids and `emb`
//!   the normalized reference embedding;
//! - videos: `{"name", "file", "raw", "raw_shape" [t, 3, h, w], "indices",
//!   "pv", "columns", "rows", "ids", "emb", "metadata"}`.
//!
//! `preprocess` compares the native decoder (`raw`), the native resize and
//! patching from the reference's decoded pixels (`pv`, must be identical) and
//! the whole native pipeline from the file; `parity` the embeddings of both
//! the reference patches and the native pipeline against `emb` (cosine).
//! `retrieval` reads `FLICKR_DIR/captions.json` (`files`, `captions`,
//! `cap2img`, `t2m_prompt`, `m2t_prompt`) and reports R@1/5/10 both ways.

use std::path::{Path, PathBuf};
use std::time::Instant;

use local_engine::embedding::vision::preprocess::{
    self, ImagePatches, RgbImage, VideoTiming, decode_image, patchify,
};
use local_engine::embedding::vision::{
    IMAGE_SOFT_TOKENS, VIDEO_FPS, VIDEO_MAX_FRAMES, VIDEO_SOFT_TOKENS, video,
};
use local_engine::embedding::{EmbeddingModel, MediaPart, installed_model};
use serde::Deserialize;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
struct Index {
    images: Vec<ImageRef>,
    videos: Vec<VideoRef>,
}

#[derive(Deserialize)]
struct ImageRef {
    name: String,
    file: String,
    raw: String,
    raw_shape: Vec<usize>,
    pv: String,
    columns: usize,
    rows: usize,
    ids: Vec<u32>,
    emb: Vec<f32>,
}

#[derive(Deserialize)]
struct VideoRef {
    name: String,
    file: String,
    raw: String,
    raw_shape: Vec<usize>,
    indices: Vec<usize>,
    pv: String,
    columns: usize,
    rows: usize,
    ids: Vec<u32>,
    emb: Vec<f32>,
    frames: usize,
    fps: f64,
    duration: f64,
}

#[derive(Deserialize)]
#[expect(clippy::struct_field_names)]
struct Captions {
    files: Vec<String>,
    captions: Vec<String>,
    cap2img: Vec<usize>,
    t2m_prompt: String,
    m2t_prompt: String,
}

fn open() -> Result<EmbeddingModel> {
    let dir = installed_model(Path::new("models"))?;
    let started = Instant::now();
    let model = EmbeddingModel::open(&dir)?;
    eprintln!("loaded {} in {:.2?}", dir.display(), started.elapsed());
    Ok(model)
}

fn read_index(path: &str) -> Result<(Index, PathBuf)> {
    let index: Index = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let base = Path::new(path)
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    Ok((index, base))
}

/// CHW planes -> interleaved RGB rows.
fn rgb_from_chw(bytes: &[u8], height: usize, width: usize) -> Result<RgbImage> {
    let plane = height * width;
    if bytes.len() != plane * 3 {
        return Err("raw image size mismatch".into());
    }
    let mut pixels = Vec::with_capacity(plane * 3);
    for i in 0..plane {
        pixels.extend([bytes[i], bytes[plane + i], bytes[2 * plane + i]]);
    }
    Ok(RgbImage::new(width, height, pixels)?)
}

/// Reference patches (`uint8` values) as native input.
fn patches_from_u8(bytes: &[u8], columns: usize, rows: usize) -> ImagePatches {
    ImagePatches {
        columns,
        rows,
        values: bytes
            .iter()
            .take(columns * rows * preprocess::PATCH_VALUES)
            .map(|&v| f32::from(v) * preprocess::RESCALE)
            .collect(),
    }
}

fn to_u8(values: &[f32]) -> Vec<u8> {
    values.iter().map(|v| (v * 255.0).round() as u8).collect()
}

fn diff(a: &[u8], b: &[u8]) -> (usize, u8) {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x != y)
        .fold((0, 0), |(n, m), (x, y)| (n + 1, m.max(x.abs_diff(*y))))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[expect(clippy::too_many_lines)]
fn preprocess_gate(path: &str) -> Result<()> {
    let (index, base) = read_index(path)?;
    let (mut resize_ok, mut decode_ok, mut pipeline_ok) = (0, 0, 0);
    for image in &index.images {
        let [_, height, width] = image.raw_shape[..] else {
            return Err("raw_shape must be [3, h, w]".into());
        };
        let raw = std::fs::read(base.join(&image.raw))?;
        let pv = std::fs::read(base.join(&image.pv))?;
        let valid = image.columns * image.rows * preprocess::PATCH_VALUES;
        let reference = rgb_from_chw(&raw, height, width)?;
        let from_raw = patchify(&reference, IMAGE_SOFT_TOKENS)?;
        let grid_ok = (from_raw.columns, from_raw.rows) == (image.columns, image.rows);
        let (n, max) = diff(&to_u8(&from_raw.values), &pv[..valid.min(pv.len())]);
        let soft = from_raw.soft_tokens();
        let ids_ok = image.ids.len() == soft + 4;
        if grid_ok && n == 0 && ids_ok {
            resize_ok += 1;
        }
        let decoded = decode_image(&std::fs::read(base.join(&image.file))?)?;
        let (dn, dmax) = if (decoded.width, decoded.height) == (width, height) {
            diff(&decoded.pixels, &reference.pixels)
        } else {
            (usize::MAX, 255)
        };
        if dn == 0 {
            decode_ok += 1;
        }
        let native = patchify(&decoded, IMAGE_SOFT_TOKENS)?;
        let (pn, pmax) = if native.values.len() == valid {
            diff(&to_u8(&native.values), &pv[..valid])
        } else {
            (usize::MAX, 255)
        };
        if pn == 0 {
            pipeline_ok += 1;
        }
        println!(
            "image {:<40} grid {}x{} resize {} ({n} diff, max {max}) decode {} ({dn} diff, max {dmax}) pipeline {pn} diff max {pmax}",
            image.name,
            image.columns,
            image.rows,
            if grid_ok && n == 0 && ids_ok {
                "exact"
            } else {
                "DIFF"
            },
            if dn == 0 { "exact" } else { "differs" },
        );
    }
    println!(
        "images: resize+patch from reference pixels identical {resize_ok}/{n}, decoder identical {decode_ok}/{n}, whole pipeline identical {pipeline_ok}/{n}",
        n = index.images.len()
    );
    let (mut sample_ok, mut frames_ok, mut video_resize_ok) = (0, 0, 0);
    for v in &index.videos {
        let [frames, _, height, width] = v.raw_shape[..] else {
            return Err("video raw_shape must be [t, 3, h, w]".into());
        };
        let timing = VideoTiming {
            frames: v.frames,
            fps: Some(v.fps),
            duration: Some(v.duration),
        };
        let picked = preprocess::sample_frame_indices(timing, VIDEO_FPS, VIDEO_MAX_FRAMES);
        if picked == v.indices {
            sample_ok += 1;
        }
        let file = base.join(&v.file);
        let probed = video::probe(&file);
        let raw = std::fs::read(base.join(&v.raw))?;
        let pv = std::fs::read(base.join(&v.pv))?;
        let plane = 3 * height * width;
        let per_frame_pv = 1260 * preprocess::PATCH_VALUES;
        let valid = v.columns * v.rows * preprocess::PATCH_VALUES;
        let mut resize_diff = 0;
        let mut decode_diff = (0, 0);
        let decoded = probed
            .as_ref()
            .map_err(ToString::to_string)
            .and_then(|stream| {
                if (stream.width, stream.height) == (width, height) {
                    video::decode_frames(&file, stream, &v.indices)
                        .map_err(|error| error.to_string())
                } else {
                    Err("frame size differs".into())
                }
            });
        for f in 0..frames {
            let reference = rgb_from_chw(&raw[f * plane..(f + 1) * plane], height, width)?;
            let patches = patchify(&reference, VIDEO_SOFT_TOKENS)?;
            resize_diff += diff(
                &to_u8(&patches.values),
                &pv[f * per_frame_pv..f * per_frame_pv + valid],
            )
            .0;
            if let Ok(decoded) = &decoded {
                let (n, m) = diff(&decoded[f].pixels, &reference.pixels);
                decode_diff = (decode_diff.0 + n, decode_diff.1.max(m));
            }
        }
        if resize_diff == 0 {
            video_resize_ok += 1;
        }
        if decoded.is_ok() && decode_diff.0 == 0 {
            frames_ok += 1;
        }
        println!(
            "video {:<24} frames {} sample {} probe {:?} (reference {} frames {:.6} fps {:.6} s) resize diff {resize_diff} decode {}",
            v.name,
            frames,
            if picked == v.indices { "exact" } else { "DIFF" },
            probed
                .as_ref()
                .map(|s| s.timing)
                .map_err(ToString::to_string),
            v.frames,
            v.fps,
            v.duration,
            match &decoded {
                Ok(_) => format!("{} diff, max {}", decode_diff.0, decode_diff.1),
                Err(error) => error.clone(),
            }
        );
    }
    println!(
        "videos: frame indices identical {sample_ok}/{n}, resize+patch identical {video_resize_ok}/{n}, ffmpeg frames identical {frames_ok}/{n}",
        n = index.videos.len()
    );
    Ok(())
}

fn summary(label: &str, cosines: &[f32]) {
    if cosines.is_empty() {
        return;
    }
    let mean = cosines.iter().map(|c| f64::from(*c)).sum::<f64>() / cosines.len() as f64;
    let min = cosines.iter().copied().fold(f32::MAX, f32::min);
    println!(
        "{label}: n {} mean cosine {mean:.6} min {min:.6}",
        cosines.len()
    );
}

fn parity(path: &str) -> Result<()> {
    let (index, base) = read_index(path)?;
    let mut model = open()?;
    let mut pace = || Ok(());
    let (mut from_ref, mut from_file) = (Vec::new(), Vec::new());
    let started = Instant::now();
    for image in &index.images {
        let pv = std::fs::read(base.join(&image.pv))?;
        let reference = patches_from_u8(&pv, image.columns, image.rows);
        let layout = model.multimodal_layout(&[MediaPart::Image(&reference)])?;
        let ids: Vec<u32> = layout.iter().map(|slot| slot.unwrap_or(258_880)).collect();
        let a = model.embed_image(&reference, &mut pace)?;
        let native =
            local_engine::embedding::prepare_image(&std::fs::read(base.join(&image.file))?)?;
        let b = model.embed_image(&native, &mut pace)?;
        let (ca, cb) = (dot(&a, &image.emb), dot(&b, &image.emb));
        println!(
            "image {:<40} ids {} cos(reference patches) {ca:.6} cos(native pipeline) {cb:.6}",
            image.name,
            if ids == image.ids { "same" } else { "DIFF" }
        );
        from_ref.push(ca);
        from_file.push(cb);
    }
    eprintln!("images done in {:.1?}", started.elapsed());
    summary("images, reference patches", &from_ref);
    summary("images, native pipeline from the file", &from_file);
    let (mut v_ref, mut v_file) = (Vec::new(), Vec::new());
    for v in &index.videos {
        let pv = std::fs::read(base.join(&v.pv))?;
        let per_frame = 1260 * preprocess::PATCH_VALUES;
        let frames: Vec<ImagePatches> = (0..v.indices.len())
            .map(|f| patches_from_u8(&pv[f * per_frame..(f + 1) * per_frame], v.columns, v.rows))
            .collect();
        let layout = model.multimodal_layout(&[MediaPart::Video(&frames)])?;
        let started = Instant::now();
        let a = model.embed_video(&frames, &mut pace)?;
        let elapsed = started.elapsed();
        let native = video::decode_video_file(&base.join(&v.file));
        let cb = match &native {
            Ok(native) => dot(&model.embed_video(native, &mut pace)?, &v.emb),
            Err(error) => {
                eprintln!("{}: {error}", v.name);
                f32::NAN
            }
        };
        let ca = dot(&a, &v.emb);
        println!(
            "video {:<24} frames {} tokens {} (reference {}) cos(reference patches) {ca:.6} cos(native ffmpeg pipeline) {cb:.6} ({elapsed:.1?})",
            v.name,
            frames.len(),
            layout.len(),
            v.ids.len()
        );
        v_ref.push(ca);
        if cb.is_finite() {
            v_file.push(cb);
        }
    }
    summary("videos, reference patches", &v_ref);
    summary("videos, native ffmpeg pipeline", &v_file);
    Ok(())
}

fn recall(scores: &[Vec<f32>], relevant: &dyn Fn(usize, usize) -> bool) -> [f64; 3] {
    let mut hits = [0_usize; 3];
    for (q, row) in scores.iter().enumerate() {
        let mut order: Vec<usize> = (0..row.len()).collect();
        order.sort_by(|a, b| row[*b].total_cmp(&row[*a]).then(a.cmp(b)));
        for (slot, k) in [1, 5, 10].into_iter().enumerate() {
            if order[..k].iter().any(|&d| relevant(q, d)) {
                hits[slot] += 1;
            }
        }
    }
    hits.map(|h| 100.0 * h as f64 / scores.len() as f64)
}

fn retrieval(dir: &str) -> Result<()> {
    let dir = Path::new(dir);
    let captions: Captions =
        serde_json::from_str(&std::fs::read_to_string(dir.join("captions.json"))?)?;
    let mut model = open()?;
    let mut pace = || Ok(());
    let started = Instant::now();
    let mut images = Vec::with_capacity(captions.files.len());
    for chunk in captions.files.chunks(8) {
        let prepared = chunk
            .iter()
            .map(|file| {
                Ok(local_engine::embedding::prepare_image(&std::fs::read(
                    dir.join(file),
                )?)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let inputs: Vec<Vec<MediaPart<'_>>> =
            prepared.iter().map(|p| vec![MediaPart::Image(p)]).collect();
        images.extend(model.embed_multimodal(&inputs, &mut pace)?);
    }
    let image_seconds = started.elapsed().as_secs_f64();
    let embed_texts = |model: &mut EmbeddingModel, prompt: &str| -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = captions
            .captions
            .iter()
            .map(|c| format!("{prompt}{c}"))
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        Ok(model.embed_texts(&refs, &mut || Ok(()))?)
    };
    let queries = embed_texts(&mut model, &captions.t2m_prompt)?;
    let docs = embed_texts(&mut model, &captions.m2t_prompt)?;
    let t2m: Vec<Vec<f32>> = queries
        .iter()
        .map(|q| images.iter().map(|i| dot(q, i)).collect())
        .collect();
    let m2t: Vec<Vec<f32>> = images
        .iter()
        .map(|i| docs.iter().map(|d| dot(i, d)).collect())
        .collect();
    let a = recall(&t2m, &|q, d| captions.cap2img[q] == d);
    let b = recall(&m2t, &|q, d| captions.cap2img[d] == q);
    let mean = |r: [f64; 3]| r.iter().sum::<f64>() / 3.0;
    println!(
        "Flickr30k 1K ({} images, {} captions; images embedded at {:.2} images/s)",
        images.len(),
        captions.captions.len(),
        images.len() as f64 / image_seconds
    );
    println!(
        "text->image R@1/5/10 {:.1} / {:.1} / {:.1} (mean {:.2})",
        a[0],
        a[1],
        a[2],
        mean(a)
    );
    println!(
        "image->text R@1/5/10 {:.1} / {:.1} / {:.1} (mean {:.2})",
        b[0],
        b[1],
        b[2],
        mean(b)
    );
    Ok(())
}

fn bench(file: &str, images: usize) -> Result<()> {
    let mut model = open()?;
    let prepared = local_engine::embedding::prepare_image(&std::fs::read(file)?)?;
    let mut pace = || Ok(());
    model.embed_image(&prepared, &mut pace)?;
    let inputs: Vec<Vec<MediaPart<'_>>> = (0..images)
        .map(|_| vec![MediaPart::Image(&prepared)])
        .collect();
    let mut best = f64::MAX;
    for _ in 0..3 {
        let started = Instant::now();
        model.embed_multimodal(&inputs, &mut pace)?;
        best = best.min(started.elapsed().as_secs_f64());
    }
    let started = Instant::now();
    for _ in 0..images {
        local_engine::embedding::prepare_image(&std::fs::read(file)?)?;
    }
    let prep = started.elapsed().as_secs_f64() / images as f64;
    println!(
        "{images} images x {} soft tokens: best {:.1} ms ({:.2} images/s); decode+preprocess {:.1} ms/image (CPU)",
        prepared.soft_tokens(),
        best * 1e3,
        images as f64 / best,
        prep * 1e3
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("preprocess") if args.len() == 2 => preprocess_gate(&args[1]),
        Some("parity") if args.len() == 2 => parity(&args[1]),
        Some("retrieval") if args.len() == 2 => retrieval(&args[1]),
        Some("bench") if args.len() >= 2 => {
            let images = match args.get(2..4) {
                Some([flag, n]) if flag == "--images" => n.parse()?,
                _ => 8,
            };
            bench(&args[1], images)
        }
        _ => Err("usage: embedding-vision-eval preprocess|parity INDEX.json | retrieval FLICKR_DIR | bench IMAGE [--images N]".into()),
    }
}
