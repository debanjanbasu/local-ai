//! Native image and video embeddings: the `EmbeddingGemma` 2 vision encoder.
//!
//! Images and video frames are preprocessed exactly as the reference
//! processors do ([`preprocess`]), encoded by the 16-layer `gemma4_vision`
//! tower on Metal (width 768, 12 heads x 64, gated GELU-tanh MLP 3072, a
//! learned 2-D position table, axial `RoPE` θ 100, bidirectional attention
//! within each image or frame), 3x3 average-pooled (F32, x sqrt(768)) to one
//! soft token per 3x3 patches (280 per image, 140 per video frame by default)
//! and projected by `embed_vision` (`RMSNorm` without scale, 768→512). The
//! soft tokens replace the `<|image|>` / `<|video|>` placeholders of a
//! `<bos> … <eos>` sequence (each image or frame framed by `<|image>` …
//! `<image|>`) that runs through the shared text encoder, so an image, a video
//! and a text land in one 768-d space.
//!
//! The encoder is optional: it loads (lazily, on the first media input) when
//! the install directory also holds the vision artifact written by
//! `tools/convert_embeddinggemma2_vision.py`. Like the text path it keeps the
//! checkpoint's BF16 matrices and runs F32 activations.

mod artifact;
pub mod preprocess;
pub mod video;

use std::path::Path;

use local_metal::batch::CommandBatch;
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::embedding::{Bf16Rows, EmbeddingKernels};
use local_metal::embedding_vision::VisionKernels;
use local_metal::shaders::ShaderLibrary;

use super::{EmbeddingModel, HIDDEN};
use crate::Error;
use artifact::VisionArtifact;
pub use preprocess::{
    IMAGE_SOFT_TOKENS, ImagePatches, PATCH_SIZE, POOLING, RgbImage, SUPPORTED_SOFT_TOKENS,
    VIDEO_FPS, VIDEO_MAX_FRAMES, VIDEO_SOFT_TOKENS, VideoTiming,
};

/// Vision artifact file name inside the embedding install directory.
pub const VISION_ARTIFACT: &str = "embeddinggemma2-vision-bf16.bin";
/// Longest multimodal input in tokens (the model's context), `<bos>`,
/// `<eos>`, the image/frame delimiters and every soft token included.
pub const MAX_MULTIMODAL_TOKENS: usize = 8192;

const WIDTH: usize = 768;
const MLP: usize = 3072;
const LAYERS: usize = 16;
const HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const POSITIONS: usize = 10_240;
const ROPE_THETA: f32 = 100.0;
const TEXT_HIDDEN: usize = HIDDEN;
const EPSILON: f32 = 1e-6;
const BOI: u32 = 255_999;
const EOI: u32 = 258_882;
const BOS: u32 = 2;
const EOS: u32 = 1;
/// Patches per vision pass: one 280-token image or two 140-token frames.
/// A larger image (budget 560 or 1120) takes a pass of its own.
const PASS_PATCHES: usize = 2520;

/// Whether `dir` (an embedding install directory) holds the vision artifact.
#[must_use]
pub fn vision_installed(dir: &Path) -> bool {
    dir.join(VISION_ARTIFACT).is_file()
}

/// One part of a multimodal input, in order. Text is embedded as given (the
/// model card uses no prompt for media).
#[derive(Clone, Copy, Debug)]
pub enum MediaPart<'a> {
    Text(&'a str),
    /// One preprocessed image ([`preprocess::patchify`]).
    Image(&'a ImagePatches),
    /// Preprocessed, already sampled frames of one video.
    Video(&'a [ImagePatches]),
}

/// Decode and preprocess an encoded image file for the default budget of
/// [`IMAGE_SOFT_TOKENS`].
///
/// # Errors
///
/// As [`preprocess::decode_image`] and [`preprocess::patchify`].
pub fn prepare_image(bytes: &[u8]) -> crate::Result<ImagePatches> {
    preprocess::patchify(&preprocess::decode_image(bytes)?, IMAGE_SOFT_TOKENS)
}

/// Sample and preprocess decoded video frames as the reference does.
///
/// `frames` are all frames of the stream, in order. Sampling takes
/// [`VIDEO_FPS`] frames per second when `timing` has a rate and duration
/// (otherwise every frame), at most [`VIDEO_MAX_FRAMES`], at
/// [`VIDEO_SOFT_TOKENS`] tokens per frame.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for no frames, frames of different sizes, or a
/// frame count that disagrees with `timing`.
pub fn prepare_video(
    frames: &[RgbImage],
    timing: Option<VideoTiming>,
) -> crate::Result<Vec<ImagePatches>> {
    let timing = timing.unwrap_or(VideoTiming {
        frames: frames.len(),
        fps: None,
        duration: None,
    });
    if frames.is_empty() || timing.frames != frames.len() {
        return Err(Error::InvalidArgument(format!(
            "video has {} decoded frames, its timing says {}",
            frames.len(),
            timing.frames
        )));
    }
    if frames
        .iter()
        .any(|f| (f.width, f.height) != (frames[0].width, frames[0].height))
    {
        return Err(Error::InvalidArgument("video frames differ in size".into()));
    }
    preprocess::sample_frame_indices(timing, VIDEO_FPS, VIDEO_MAX_FRAMES)
        .into_iter()
        .map(|i| preprocess::patchify(&frames[i], VIDEO_SOFT_TOKENS))
        .collect()
}

struct Layer {
    q: usize,
    k: usize,
    v: usize,
    o: usize,
    gate: usize,
    up: usize,
    down: usize,
    input_norm: usize,
    post_attention_norm: usize,
    pre_ffn_norm: usize,
    post_ffn_norm: usize,
    q_norm: usize,
    k_norm: usize,
}

struct Workspace {
    rows: usize,
    hidden: MetalBuffer,
    normed: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    attention: MetalBuffer,
    branch: MetalBuffer,
    gate: MetalBuffer,
    up: MetalBuffer,
    positions: MetalBuffer,
    rope_positions: MetalBuffer,
    members: MetalBuffer,
}

impl Workspace {
    fn new(context: &MetalContext, rows: usize) -> crate::Result<Self> {
        let buffer = |width: usize| MetalBuffer::empty(context.device(), rows * width * 4);
        Ok(Self {
            rows,
            hidden: buffer(WIDTH)?,
            normed: buffer(WIDTH)?,
            q: buffer(WIDTH)?,
            k: buffer(WIDTH)?,
            v: buffer(WIDTH)?,
            attention: buffer(WIDTH)?,
            branch: buffer(WIDTH)?,
            gate: buffer(MLP)?,
            up: buffer(MLP)?,
            positions: buffer(2)?,
            rope_positions: buffer(2 * HEADS)?,
            members: buffer(1)?,
        })
    }
}

/// The loaded vision encoder (owned by [`EmbeddingModel`]).
pub(super) struct VisionTower {
    kernels: VisionKernels,
    artifact: VisionArtifact,
    weights: MetalBuffer,
    input_proj: usize,
    table: usize,
    embed_vision: usize,
    layers: Vec<Layer>,
    workspace: Option<Workspace>,
}

impl VisionTower {
    fn open(path: &Path, context: &MetalContext) -> crate::Result<Self> {
        let artifact = VisionArtifact::open(path)?;
        let shaders = ShaderLibrary::new(context.device())?;
        let kernels = VisionKernels::new(context, &shaders)?;
        let weights = MetalBuffer::from_slice(context.device(), artifact.payload())?;
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = |name: &str| format!("layers.{i}.{name}");
            let matrix = |name: &str, rows, cols| artifact.bf16(&p(name), rows, cols);
            let norm = |name: &str, n| artifact.f32(&p(name), n);
            layers.push(Layer {
                q: matrix("self_attn.q_proj", WIDTH, WIDTH)?,
                k: matrix("self_attn.k_proj", WIDTH, WIDTH)?,
                v: matrix("self_attn.v_proj", WIDTH, WIDTH)?,
                o: matrix("self_attn.o_proj", WIDTH, WIDTH)?,
                gate: matrix("mlp.gate_proj", MLP, WIDTH)?,
                up: matrix("mlp.up_proj", MLP, WIDTH)?,
                down: matrix("mlp.down_proj", WIDTH, MLP)?,
                input_norm: norm("input_layernorm", WIDTH)?,
                post_attention_norm: norm("post_attention_layernorm", WIDTH)?,
                pre_ffn_norm: norm("pre_feedforward_layernorm", WIDTH)?,
                post_ffn_norm: norm("post_feedforward_layernorm", WIDTH)?,
                q_norm: norm("self_attn.q_norm", HEAD_DIM)?,
                k_norm: norm("self_attn.k_norm", HEAD_DIM)?,
            });
        }
        Ok(Self {
            kernels,
            input_proj: artifact.bf16("patch_embedder.input_proj", WIDTH, WIDTH)?,
            table: artifact.bf16(
                "patch_embedder.position_embedding_table",
                2 * POSITIONS,
                WIDTH,
            )?,
            embed_vision: artifact.bf16("embed_vision", TEXT_HIDDEN, WIDTH)?,
            artifact,
            weights,
            layers,
            workspace: None,
        })
    }

    const fn matrix(&self, offset: usize, rows: usize, columns: usize) -> Bf16Rows<'_> {
        Bf16Rows {
            weights: &self.weights,
            offset,
            rows: rows as u32,
            columns: columns as u32,
        }
    }

    /// Soft tokens (`[Σ soft tokens, 512]`, frame order) of `frames`.
    fn encode(
        &mut self,
        context: &MetalContext,
        shared: &EmbeddingKernels,
        frames: &[&ImagePatches],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<f32>> {
        let mut out = Vec::new();
        let mut start = 0;
        while start < frames.len() {
            let mut end = start + 1;
            let mut patches = frames[start].patch_count();
            while end < frames.len() && patches + frames[end].patch_count() <= PASS_PATCHES {
                patches += frames[end].patch_count();
                end += 1;
            }
            if self.workspace.as_ref().is_none_or(|ws| ws.rows < patches) {
                self.workspace = None;
                self.workspace = Some(Workspace::new(context, patches.max(PASS_PATCHES))?);
            }
            out.extend(self.pass(context, shared, &frames[start..end], pace)?);
            start = end;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_lines)]
    fn pass(
        &self,
        context: &MetalContext,
        k: &EmbeddingKernels,
        frames: &[&ImagePatches],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<f32>> {
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| Error::Generation("vision workspace missing".into()))?;
        let tokens: usize = frames.iter().map(|f| f.patch_count()).sum();
        let pooled: usize = frames.iter().map(|f| f.soft_tokens()).sum();
        let mut input = Vec::with_capacity(tokens * WIDTH);
        let mut positions = Vec::with_capacity(tokens * 2);
        let mut rope = Vec::with_capacity(tokens * 2 * HEADS);
        let mut members = Vec::with_capacity(pooled * 9);
        let mut first = 0_u32;
        for frame in frames {
            if frame.values.len() != frame.patch_count() * preprocess::PATCH_VALUES
                || frame.columns % POOLING != 0
                || frame.rows % POOLING != 0
                || frame.columns.max(frame.rows) > POSITIONS
            {
                return Err(Error::InvalidArgument("malformed image patches".into()));
            }
            // The reference feeds 2 * (pixel - 0.5) to the input projection.
            input.extend(frame.values.iter().map(|&v| (v - 0.5) * 2.0));
            for index in 0..frame.patch_count() {
                let (x, y) = frame.position(index);
                positions.extend([x as u32, y as u32]);
                for _ in 0..HEADS {
                    rope.extend([x as u32, y as u32]);
                }
            }
            let (pooled_columns, pooled_rows) = (frame.columns / POOLING, frame.rows / POOLING);
            for oy in 0..pooled_rows {
                for ox in 0..pooled_columns {
                    // Members ascend in patch order within each pooled token.
                    for dy in 0..POOLING {
                        for dx in 0..POOLING {
                            let index = (oy * POOLING + dy) * frame.columns + ox * POOLING + dx;
                            members.push(first + index as u32);
                        }
                    }
                }
            }
            first += frame.patch_count() as u32;
        }
        ws.normed.copy_from_bytes(bytemuck::cast_slice(&input), 0);
        ws.positions
            .copy_from_bytes(bytemuck::cast_slice(&positions), 0);
        ws.rope_positions
            .copy_from_bytes(bytemuck::cast_slice(&rope), 0);
        ws.members
            .copy_from_bytes(bytemuck::cast_slice(&members), 0);
        let count = tokens as u32;
        let width = WIDTH as u32;
        let weights = &self.weights;
        let mut batch = CommandBatch::new(context)?;
        k.bf16_gemm(
            &mut batch,
            self.matrix(self.input_proj, WIDTH, WIDTH),
            &ws.normed,
            &ws.hidden,
            count,
        )?;
        self.kernels.add_positions(
            &mut batch,
            &ws.hidden,
            (weights, self.table),
            POSITIONS as u32,
            &ws.positions,
            count,
            width,
        )?;
        let heads = HEADS as u32;
        let head_rows = count * heads;
        for layer in &self.layers {
            k.rms(
                &mut batch,
                &ws.hidden,
                Some((weights, layer.input_norm)),
                &ws.normed,
                count,
                width,
                1.0,
                EPSILON,
            )?;
            for (matrix, out) in [(layer.q, &ws.q), (layer.k, &ws.k), (layer.v, &ws.v)] {
                k.bf16_gemm(
                    &mut batch,
                    self.matrix(matrix, WIDTH, WIDTH),
                    &ws.normed,
                    out,
                    count,
                )?;
            }
            let d = HEAD_DIM as u32;
            k.rms(
                &mut batch,
                &ws.q,
                Some((weights, layer.q_norm)),
                &ws.q,
                head_rows,
                d,
                1.0,
                EPSILON,
            )?;
            k.rms(
                &mut batch,
                &ws.k,
                Some((weights, layer.k_norm)),
                &ws.k,
                head_rows,
                d,
                1.0,
                EPSILON,
            )?;
            k.rms(&mut batch, &ws.v, None, &ws.v, head_rows, d, 1.0, EPSILON)?;
            // Axial RoPE: the first 32 dims of each head rotate by x, the last
            // 32 by y, each as rotate-half over 32 dims with θ = 100.
            let halves = head_rows * 2;
            k.rope(
                &mut batch,
                &ws.q,
                &ws.rope_positions,
                halves,
                1,
                d / 2,
                ROPE_THETA,
            )?;
            k.rope(
                &mut batch,
                &ws.k,
                &ws.rope_positions,
                halves,
                1,
                d / 2,
                ROPE_THETA,
            )?;
            let mut start = 0_u32;
            for frame in frames {
                let n = frame.patch_count() as u32;
                self.kernels.attention(
                    &mut batch,
                    &ws.q,
                    &ws.k,
                    &ws.v,
                    &ws.attention,
                    start,
                    n,
                    heads,
                )?;
                start += n;
            }
            k.bf16_gemm(
                &mut batch,
                self.matrix(layer.o, WIDTH, WIDTH),
                &ws.attention,
                &ws.branch,
                count,
            )?;
            k.rms_add(
                &mut batch,
                &ws.hidden,
                &ws.branch,
                (weights, layer.post_attention_norm),
                count,
                width,
                1.0,
                EPSILON,
            )?;
            k.rms(
                &mut batch,
                &ws.hidden,
                Some((weights, layer.pre_ffn_norm)),
                &ws.normed,
                count,
                width,
                1.0,
                EPSILON,
            )?;
            k.bf16_gemm(
                &mut batch,
                self.matrix(layer.gate, MLP, WIDTH),
                &ws.normed,
                &ws.gate,
                count,
            )?;
            k.bf16_gemm(
                &mut batch,
                self.matrix(layer.up, MLP, WIDTH),
                &ws.normed,
                &ws.up,
                count,
            )?;
            let mlp = MLP as u32;
            k.gelu_mul(&mut batch, &ws.gate, &ws.up, &ws.gate, count, mlp, mlp, 0)?;
            k.bf16_gemm(
                &mut batch,
                self.matrix(layer.down, WIDTH, MLP),
                &ws.gate,
                &ws.branch,
                count,
            )?;
            k.rms_add(
                &mut batch,
                &ws.hidden,
                &ws.branch,
                (weights, layer.post_ffn_norm),
                count,
                width,
                1.0,
                EPSILON,
            )?;
            batch.commit_and_renew(context)?;
            pace()?;
        }
        let pooled_count = pooled as u32;
        self.kernels.pool(
            &mut batch,
            &ws.hidden,
            &ws.members,
            &ws.q,
            pooled_count,
            width,
            (WIDTH as f32).sqrt(),
        )?;
        k.rms(
            &mut batch,
            &ws.q,
            None,
            &ws.q,
            pooled_count,
            width,
            1.0,
            EPSILON,
        )?;
        k.bf16_gemm(
            &mut batch,
            self.matrix(self.embed_vision, TEXT_HIDDEN, WIDTH),
            &ws.q,
            &ws.k,
            pooled_count,
        )?;
        batch.commit_and_wait()?;
        Ok(ws.k.as_slice::<f32>()[..pooled * TEXT_HIDDEN].to_vec())
    }
}

impl EmbeddingModel {
    /// Whether the vision artifact is installed next to the text artifact
    /// (media inputs are refused otherwise).
    #[must_use]
    pub fn supports_vision(&self) -> bool {
        self.vision.is_some() || vision_installed(&self.dir)
    }

    /// SHA-256 of the vision artifact payload, once loaded.
    #[must_use]
    pub fn vision_payload_sha256(&self) -> Option<&str> {
        self.vision
            .as_ref()
            .map(|v| v.artifact.payload_sha256.as_str())
    }

    /// Embed one preprocessed image (full width, L2-normalized).
    ///
    /// # Errors
    ///
    /// As [`Self::embed_multimodal`].
    pub fn embed_image(
        &mut self,
        image: &ImagePatches,
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<f32>> {
        self.single(&[MediaPart::Image(image)], pace)
    }

    /// Embed the sampled, preprocessed frames of one video
    /// ([`prepare_video`]).
    ///
    /// # Errors
    ///
    /// As [`Self::embed_multimodal`].
    pub fn embed_video(
        &mut self,
        frames: &[ImagePatches],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<f32>> {
        self.single(&[MediaPart::Video(frames)], pace)
    }

    fn single(
        &mut self,
        parts: &[MediaPart<'_>],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<f32>> {
        self.embed_multimodal(&[parts.to_vec()], pace)?
            .pop()
            .ok_or_else(|| Error::Generation("no embedding produced".into()))
    }

    /// Token sequence of one multimodal input, with `None` marking soft-token
    /// slots (filled in frame order), as the reference processor lays it out:
    /// `<bos>`, then each part in order (text tokens; per image or video
    /// frame `<|image>`, one slot per soft token, `<image|>`), then `<eos>`.
    ///
    /// # Errors
    ///
    /// Tokenizer failures, media placeholder ids inside text, empty videos,
    /// or more than [`MAX_MULTIMODAL_TOKENS`] tokens.
    pub fn multimodal_layout(&self, parts: &[MediaPart<'_>]) -> crate::Result<Vec<Option<u32>>> {
        let mut layout = vec![Some(BOS)];
        let mut text = String::new();
        let flush = |text: &mut String, layout: &mut Vec<Option<u32>>| -> crate::Result<()> {
            if !text.is_empty() {
                let encoding = self
                    .tokenizer
                    .encode(text.as_str(), false)
                    .map_err(|error| Error::Tokenizer(error.to_string()))?;
                if encoding
                    .get_ids()
                    .iter()
                    .any(|id| super::MEDIA_PLACEHOLDERS.contains(id))
                {
                    return Err(Error::InvalidArgument(
                        "text part contains an image, audio or video placeholder token".into(),
                    ));
                }
                layout.extend(encoding.get_ids().iter().map(|&id| Some(id)));
                text.clear();
            }
            Ok(())
        };
        let frame = |patches: &ImagePatches, layout: &mut Vec<Option<u32>>| {
            layout.push(Some(BOI));
            layout.extend(std::iter::repeat_n(None, patches.soft_tokens()));
            layout.push(Some(EOI));
        };
        for part in parts {
            match part {
                MediaPart::Text(t) => text.push_str(t),
                MediaPart::Image(image) => {
                    flush(&mut text, &mut layout)?;
                    frame(image, &mut layout);
                }
                MediaPart::Video(frames) => {
                    if frames.is_empty() {
                        return Err(Error::InvalidArgument("video has no frames".into()));
                    }
                    flush(&mut text, &mut layout)?;
                    for f in *frames {
                        frame(f, &mut layout);
                    }
                }
            }
            if layout.len() > MAX_MULTIMODAL_TOKENS {
                break;
            }
        }
        flush(&mut text, &mut layout)?;
        layout.push(Some(EOS));
        if layout.len() > MAX_MULTIMODAL_TOKENS {
            return Err(Error::ContextOverflow(format!(
                "multimodal input has {} tokens; the limit is {MAX_MULTIMODAL_TOKENS}",
                layout.len()
            )));
        }
        Ok(layout)
    }

    /// Embed multimodal inputs (each a sequence of text, image and video
    /// parts; a text-only input matches [`Self::embed_texts`]); one
    /// full-width L2-normalized vector per input, in order.
    ///
    /// Every image and frame runs through the vision encoder (in passes of up
    /// to 2,520 patches, one command buffer per layer, `pace` after each),
    /// then the sequences through the text encoder (packed up to 1,024
    /// tokens per pass; a longer input takes its own).
    ///
    /// # Errors
    ///
    /// [`Self::multimodal_layout`] errors, a missing vision artifact (for
    /// media parts), a `pace` error, or a Metal failure.
    pub fn embed_multimodal(
        &mut self,
        inputs: &[Vec<MediaPart<'_>>],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        let layouts = inputs
            .iter()
            .map(|parts| self.multimodal_layout(parts))
            .collect::<crate::Result<Vec<_>>>()?;
        let frames: Vec<&ImagePatches> = inputs
            .iter()
            .flatten()
            .flat_map(|part| match part {
                MediaPart::Text(_) => &[][..],
                MediaPart::Image(image) => std::slice::from_ref(*image),
                MediaPart::Video(frames) => frames,
            })
            .collect();
        let soft = if frames.is_empty() {
            Vec::new()
        } else {
            let context = &self.context;
            let shared = &self.kernels;
            if self.vision.is_none() {
                let path = self.dir.join(VISION_ARTIFACT);
                if !path.is_file() {
                    return Err(Error::InvalidArgument(format!(
                        "image and video embeddings need {} (see docs/EMBEDDINGS.md)",
                        path.display()
                    )));
                }
                self.vision = Some(VisionTower::open(&path, context)?);
            }
            let vision = self
                .vision
                .as_mut()
                .ok_or_else(|| Error::Generation("vision encoder missing".into()))?;
            vision.encode(context, shared, &frames, pace)?
        };
        let mut out = Vec::with_capacity(inputs.len());
        let mut soft_rows = soft.as_chunks::<TEXT_HIDDEN>().0.iter();
        let mut start = 0;
        while start < layouts.len() {
            let mut end = start + 1;
            let mut tokens = layouts[start].len();
            while end < layouts.len() && tokens + layouts[end].len() <= super::PASS_TOKENS {
                tokens += layouts[end].len();
                end += 1;
            }
            let mut x0 = Vec::with_capacity(tokens * HIDDEN);
            for slot in layouts[start..end].iter().flatten() {
                match slot {
                    Some(id) => self.push_token_embedding(&mut x0, *id),
                    None => x0.extend_from_slice(soft_rows.next().ok_or_else(|| {
                        Error::Generation("fewer soft tokens than placeholders".into())
                    })?),
                }
            }
            let lens: Vec<usize> = layouts[start..end].iter().map(Vec::len).collect();
            out.extend(self.encode_embedded(&x0, &lens, pace)?);
            start = end;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
