//! Native `EmbeddingGemma` 2 text embeddings on Metal.
//!
//! Google's `EmbeddingGemma` 2 maps text (and, in the full model, images,
//! audio and video) into one 768-dimensional Matryoshka space. This module
//! runs its **text path** natively: a 24-layer bidirectional Gemma-4-style
//! encoder (hidden 512; five sliding-window layers per global one; per-layer
//! embeddings), mean pooling over every token including `<bos>`/`<eos>` and
//! the prompt, the 512→768 projection and L2 normalization.
//!
//! The model is optional and opt-in by installation: it exists only when
//! `models/embeddinggemma2-text/` holds the BF16 artifact written by
//! `tools/convert_embeddinggemma2.py` and its `tokenizer.json`
//! (see `docs/EMBEDDINGS.md`). Every matrix, the token table included, keeps
//! the checkpoint's BF16 values; activations, accumulation and softmax are
//! F32 (the model card rules out F16 activations), so the output matches the
//! FP32 reference to rounding.
//!
//! Inputs are embedded as given. Retrieval quality depends on the model
//! card's prompts, which [`EmbeddingPrompt`] formats: queries as
//! `task: {task} | query: {text}`, documents as `title: {title} | text: {text}`.
//!
//! Images and video (the optional vision encoder in [`vision`]) and audio
//! (the optional audio encoder, [`EmbeddingModel::embed_audio`]) become soft
//! tokens of one multimodal sequence ([`EmbeddingModel::embed_multimodal`])
//! that runs through this same text encoder.
//!
//! Work is split into short command buffers (a few layers each) with a
//! caller-supplied pacing hook between them, so a server can keep embedding
//! work from delaying concurrent decoding (see [`EmbeddingModel::embed_ids`]).

mod artifact;
mod audio;
mod multimodal;
pub mod vision;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use local_metal::batch::CommandBatch;
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::embedding::{AttentionShape, Bf16Rows, EmbeddingKernels};
use local_metal::shaders::ShaderLibrary;
use tokenizers::Tokenizer;

use crate::Error;
use artifact::Artifact;
pub use audio::{
    AUDIO_ARTIFACT, FeatureExtractor, LogMel, MAX_AUDIO_TOKENS, MAX_SAMPLES, MEL_BINS, SAMPLE_RATE,
    audio_tokens, decode_audio, frame_counts, to_mono_16k, validate_clip,
};
pub use multimodal::{MAX_MULTIMODAL_TOKENS, MediaPart};
pub use vision::{prepare_image, prepare_video};

/// Installation directory name inside a models directory (the one that holds
/// the Bonsai checkpoint's own folder, e.g. `models/`).
pub const EMBEDDING_MODEL_DIR: &str = "embeddinggemma2-text";
/// Artifact file name inside [`EMBEDDING_MODEL_DIR`].
pub const EMBEDDING_ARTIFACT: &str = "embeddinggemma2-text-bf16.bin";
/// Model id reported by the API.
pub const EMBEDDING_MODEL_ID: &str = "embeddinggemma-2";
/// Full embedding width.
pub const EMBEDDING_DIMENSIONS: usize = EMBED_DIM;
/// Matryoshka widths the model was trained for (truncate, then renormalize).
pub const MATRYOSHKA_DIMENSIONS: [usize; 4] = [768, 512, 256, 128];
/// Longest input, in tokens including `<bos>` and `<eos>`. The model's own
/// context is 8,192; attention here is quadratic and unblocked, so inputs are
/// capped at the workspace size.
pub const MAX_INPUT_TOKENS: usize = WORKSPACE_TOKENS;

const HIDDEN: usize = 512;
const INTERMEDIATE: usize = 2048;
const LAYERS: usize = 24;
const Q_HEADS: usize = 4;
const VOCAB: usize = 262_144;
const EMBED_DIM: usize = 768;
const SLIDING_WINDOW: u32 = 512;
const EPSILON: f32 = 1e-6;
/// Placeholder ids of image, audio and video soft tokens: the text path has
/// no embedding for them (the reference model refuses them without media).
const MEDIA_PLACEHOLDERS: [u32; 3] = [258_880, 258_881, 258_884];
/// Rows of the GPU workspace: the longest pass.
const WORKSPACE_TOKENS: usize = 2048;
/// Inputs are packed into passes of up to this many tokens; a longer input
/// (up to [`WORKSPACE_TOKENS`]) takes a pass of its own.
const PASS_TOKENS: usize = 1024;
/// Token-layers encoded per command buffer before the pacing hook runs
/// (whole layers, at least one). About 5 ms of GPU time at the ~4.8 us per
/// token-layer measured on an M-series GPU, so a decode step waiting behind
/// one embedding submission is delayed by about that much.
const SUBMIT_TOKEN_LAYERS: usize = 1024;

const fn is_global(layer: usize) -> bool {
    layer % 6 == 5
}

/// Head dim, KV heads, `RoPE` theta and attention window of one layer.
const fn layer_shape(layer: usize) -> (usize, usize, f32, u32) {
    if is_global(layer) {
        (512, 1, 1_000_000.0, 0)
    } else {
        (256, 2, 10_000.0, SLIDING_WINDOW)
    }
}

/// The model card's prompt formats.
#[derive(Clone, Copy, Debug)]
pub enum EmbeddingPrompt<'a> {
    /// `task: {task} | query: ` with a card task such as `search result`,
    /// `question answering`, `fact checking`, `code retrieval`,
    /// `classification`, `clustering` or `sentence similarity`.
    Query { task: &'a str },
    /// `title: {title or "none"} | text: `.
    Document { title: Option<&'a str> },
}

impl EmbeddingPrompt<'_> {
    #[must_use]
    pub fn format(self, text: &str) -> String {
        match self {
            Self::Query { task } => format!("task: {task} | query: {text}"),
            Self::Document { title } => {
                let title = title
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
                    .unwrap_or("none");
                format!("title: {title} | text: {text}")
            }
        }
    }
}

/// Truncate a full embedding to its leading `dimensions` and L2-normalize
/// (Matryoshka). `dimensions` larger than the vector keeps it whole.
#[must_use]
pub fn truncate_normalized(embedding: &[f32], dimensions: usize) -> Vec<f32> {
    let mut out = embedding[..dimensions.min(embedding.len())].to_vec();
    normalize(&mut out);
    out
}

fn normalize(values: &mut [f32]) {
    let norm = values
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        let inv = (1.0 / norm) as f32;
        for v in values {
            *v *= inv;
        }
    }
}

/// The installation directory inside `models_dir`
/// (`models_dir/embeddinggemma2-text`), if the artifact and tokenizer are there.
#[must_use]
pub fn installed_model(models_dir: &Path) -> Option<PathBuf> {
    let dir = models_dir.join(EMBEDDING_MODEL_DIR);
    (dir.join(EMBEDDING_ARTIFACT).is_file() && dir.join("tokenizer.json").is_file()).then_some(dir)
}

struct LayerWeights {
    q: usize,
    k: usize,
    v: usize,
    o: usize,
    gate: usize,
    up: usize,
    down: usize,
    ple_gate: usize,
    ple_projection: usize,
    input_norm: usize,
    post_attention_norm: usize,
    pre_ffn_norm: usize,
    post_ffn_norm: usize,
    post_ple_norm: usize,
    q_norm: usize,
    k_norm: usize,
    scalar: f32,
}

struct Workspace {
    rows: usize,
    input: MetalBuffer,
    hidden: MetalBuffer,
    normed: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    attention: MetalBuffer,
    branch: MetalBuffer,
    gate: MetalBuffer,
    up: MetalBuffer,
    ple: MetalBuffer,
    ple_gate: MetalBuffer,
    positions: MetalBuffer,
    bounds: MetalBuffer,
}

impl Workspace {
    fn new(context: &MetalContext, tokens: usize) -> crate::Result<Self> {
        let rows = |width: usize| MetalBuffer::empty(context.device(), tokens * width * 4);
        Ok(Self {
            rows: tokens,
            input: rows(HIDDEN)?,
            hidden: rows(HIDDEN)?,
            normed: rows(HIDDEN)?,
            q: rows(Q_HEADS * 512)?,
            k: rows(HIDDEN)?,
            v: rows(HIDDEN)?,
            attention: rows(Q_HEADS * 512)?,
            branch: rows(HIDDEN)?,
            gate: rows(INTERMEDIATE)?,
            up: rows(INTERMEDIATE)?,
            ple: rows(HIDDEN)?,
            ple_gate: rows(HIDDEN)?,
            positions: rows(1)?,
            bounds: rows(2)?,
        })
    }
}

/// The loaded text encoder. Not internally synchronized: one embedding call
/// at a time (wrap it in a mutex to share it).
pub struct EmbeddingModel {
    context: MetalContext,
    kernels: EmbeddingKernels,
    tokenizer: Arc<Tokenizer>,
    artifact: Artifact,
    /// Everything after the token table, as one GPU buffer; offsets below are
    /// relative to it.
    weights: MetalBuffer,
    table: usize,
    ple_projection: usize,
    ple_norm: usize,
    final_norm: usize,
    layers: Vec<LayerWeights>,
    projection: Vec<f32>,
    workspace: Workspace,
    /// Install directory (the optional vision and audio artifacts load from
    /// it on first use).
    dir: PathBuf,
    vision: Option<vision::VisionTower>,
    /// The optional audio encoder (loaded on first use or by
    /// [`Self::load_audio`]).
    audio: Option<audio::AudioTower>,
}

// SAFETY: Metal devices, queues, pipelines and buffers may be transferred
// between threads. The model owns every object exclusively and is `!Sync`
// (its Metal objects are), so no encoder or workspace is used concurrently;
// sharing goes through a mutex.
#[allow(unsafe_code)]
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for EmbeddingModel {}

impl EmbeddingModel {
    /// Load and verify an installation directory (artifact + tokenizer).
    ///
    /// # Errors
    ///
    /// Fails when a file is missing, the artifact does not verify, or Metal
    /// cannot be set up.
    pub fn open(dir: &Path) -> crate::Result<Self> {
        let artifact = Artifact::open(&dir.join(EMBEDDING_ARTIFACT))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|error| Error::Tokenizer(format!("embedding tokenizer: {error}")))?;
        let context = MetalContext::new()?;
        let shaders = ShaderLibrary::new(context.device())?;
        let kernels = EmbeddingKernels::new(&context, &shaders)?;
        let gpu_start = artifact.table_end()?.next_multiple_of(256);
        let weights = MetalBuffer::from_slice(context.device(), &artifact.payload()[gpu_start..])?;
        let gpu = |offset: usize| -> crate::Result<usize> {
            offset.checked_sub(gpu_start).ok_or_else(|| {
                Error::InvalidFormat("embedding artifact: tensor inside the token table".into())
            })
        };
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let (dim, kv, _, _) = layer_shape(i);
            let p = |name: &str| format!("layers.{i}.{name}");
            let norm = |name: &str, n: usize| artifact.f32_offset(&p(name), n).and_then(gpu);
            layers.push(LayerWeights {
                q: artifact
                    .bf16(&p("self_attn.q_proj"), Q_HEADS * dim, HIDDEN)
                    .and_then(gpu)?,
                k: artifact
                    .bf16(&p("self_attn.k_proj"), kv * dim, HIDDEN)
                    .and_then(gpu)?,
                v: artifact
                    .bf16(&p("self_attn.v_proj"), kv * dim, HIDDEN)
                    .and_then(gpu)?,
                o: artifact
                    .bf16(&p("self_attn.o_proj"), HIDDEN, Q_HEADS * dim)
                    .and_then(gpu)?,
                gate: artifact
                    .bf16(&p("mlp.gate_proj"), INTERMEDIATE, HIDDEN)
                    .and_then(gpu)?,
                up: artifact
                    .bf16(&p("mlp.up_proj"), INTERMEDIATE, HIDDEN)
                    .and_then(gpu)?,
                down: artifact
                    .bf16(&p("mlp.down_proj"), HIDDEN, INTERMEDIATE)
                    .and_then(gpu)?,
                ple_gate: artifact
                    .bf16(&p("ple_block.per_layer_input_gate"), HIDDEN, HIDDEN)
                    .and_then(gpu)?,
                ple_projection: artifact
                    .bf16(&p("ple_block.per_layer_projection"), HIDDEN, HIDDEN)
                    .and_then(gpu)?,
                input_norm: norm("input_layernorm", HIDDEN)?,
                post_attention_norm: norm("post_attention_layernorm", HIDDEN)?,
                pre_ffn_norm: norm("pre_feedforward_layernorm", HIDDEN)?,
                post_ffn_norm: norm("post_feedforward_layernorm", HIDDEN)?,
                post_ple_norm: norm("ple_block.post_per_layer_input_norm", HIDDEN)?,
                q_norm: norm("self_attn.q_norm", dim)?,
                k_norm: norm("self_attn.k_norm", dim)?,
                scalar: artifact.f32_values(&p("layer_scalar"), 1)?[0],
            });
        }
        let ple_projection = artifact
            .bf16("ple.per_layer_model_projection", LAYERS * HIDDEN, HIDDEN)
            .and_then(gpu)?;
        let ple_norm = gpu(artifact.f32_offset("ple.per_layer_projection_norm", HIDDEN)?)?;
        let final_norm = gpu(artifact.f32_offset("norm", HIDDEN)?)?;
        let projection = artifact.f32_values("embedding_projection", EMBED_DIM * HIDDEN)?;
        let table = artifact.bf16("embed_tokens", VOCAB, HIDDEN)?;
        let workspace = Workspace::new(&context, WORKSPACE_TOKENS)?;
        Ok(Self {
            context,
            kernels,
            tokenizer: Arc::new(tokenizer),
            artifact,
            weights,
            table,
            ple_projection,
            ple_norm,
            final_norm,
            layers,
            projection,
            workspace,
            dir: dir.to_path_buf(),
            vision: None,
            audio: None,
        })
    }

    /// SHA-256 of the verified artifact payload.
    #[must_use]
    pub fn payload_sha256(&self) -> &str {
        &self.artifact.payload_sha256
    }

    /// Token ids of `text` as the reference tokenizer produces them,
    /// `<bos>` and `<eos>` included.
    ///
    /// # Errors
    ///
    /// Fails if the tokenizer rejects the text.
    pub fn tokenize(&self, text: &str) -> crate::Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| Error::Tokenizer(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Check one input's ids: non-empty, within [`MAX_INPUT_TOKENS`], known
    /// ids, and no media placeholders.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] describing the first problem.
    pub fn validate_ids(ids: &[u32]) -> crate::Result<()> {
        if ids.is_empty() {
            return Err(Error::InvalidArgument(
                "embedding input has no tokens".into(),
            ));
        }
        if ids.len() > MAX_INPUT_TOKENS {
            return Err(Error::ContextOverflow(format!(
                "embedding input has {} tokens; the limit is {MAX_INPUT_TOKENS}",
                ids.len()
            )));
        }
        if let Some(id) = ids.iter().find(|id| **id as usize >= VOCAB) {
            return Err(Error::InvalidArgument(format!(
                "token id {id} is out of range"
            )));
        }
        if ids.iter().any(|id| MEDIA_PLACEHOLDERS.contains(id)) {
            return Err(Error::InvalidArgument(
                "embedding input contains an image, audio or video placeholder token".into(),
            ));
        }
        Ok(())
    }

    /// Embed texts (tokenized here), full width, L2-normalized.
    ///
    /// # Errors
    ///
    /// As [`Self::tokenize`], [`Self::validate_ids`] and [`Self::embed_ids`].
    pub fn embed_texts(
        &mut self,
        texts: &[&str],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        let ids = texts
            .iter()
            .map(|text| self.tokenize(text))
            .collect::<crate::Result<Vec<_>>>()?;
        self.embed_ids(&ids, pace)
    }

    /// Embed token-id sequences (each framed by `<bos>` … `<eos>` as
    /// [`Self::tokenize`] returns them); one full-width L2-normalized vector
    /// per input, in order.
    ///
    /// Inputs are packed into passes of up to 1,024 tokens (a longer input
    /// takes a pass of its own). Each pass is submitted in command buffers of
    /// about 1,024 token-layers (whole layers); after each one completes,
    /// `pace` runs and may block (to yield the GPU) or fail (to cancel).
    ///
    /// # Errors
    ///
    /// Invalid ids, a `pace` error, or a Metal failure.
    pub fn embed_ids(
        &mut self,
        inputs: &[Vec<u32>],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        for ids in inputs {
            Self::validate_ids(ids)?;
        }
        let mut out = Vec::with_capacity(inputs.len());
        let mut start = 0;
        while start < inputs.len() {
            let mut end = start + 1;
            let mut tokens = inputs[start].len();
            while end < inputs.len() && tokens + inputs[end].len() <= PASS_TOKENS {
                tokens += inputs[end].len();
                end += 1;
            }
            out.extend(self.pass(&inputs[start..end], pace)?);
            start = end;
        }
        Ok(out)
    }

    const fn matrix(&self, offset: usize, rows: usize, columns: usize) -> Bf16Rows<'_> {
        Bf16Rows {
            weights: &self.weights,
            offset,
            rows: rows as u32,
            columns: columns as u32,
        }
    }

    /// Append the input embedding of token `id` (the mapped BF16 table row,
    /// x sqrt(512)) to `x0`; `id` must be below the vocabulary size.
    fn push_token_embedding(&self, x0: &mut Vec<f32>, id: u32) {
        let scale = (HIDDEN as f32).sqrt();
        let row = &self.artifact.payload()[self.table + id as usize * HIDDEN * 2..][..HIDDEN * 2];
        x0.extend(
            row.as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16) * scale),
        );
    }

    fn pass(
        &mut self,
        inputs: &[Vec<u32>],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        let tokens: usize = inputs.iter().map(Vec::len).sum();
        // Token embeddings on the CPU from the mapped BF16 table.
        let mut x0 = Vec::with_capacity(tokens * HIDDEN);
        for &id in inputs.iter().flatten() {
            self.push_token_embedding(&mut x0, id);
        }
        let lens: Vec<usize> = inputs.iter().map(Vec::len).collect();
        self.encode_embedded(&x0, &lens, pace)
    }

    /// Run the encoder over packed input embeddings `x0` (`[Σ lens, 512]`,
    /// sequences of `lens` tokens back to back); one normalized embedding per
    /// sequence. The workspace grows to fit a longer pass.
    #[allow(clippy::too_many_lines)]
    fn encode_embedded(
        &mut self,
        x0: &[f32],
        lens: &[usize],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        let tokens: usize = lens.iter().sum();
        if tokens == 0 || x0.len() != tokens * HIDDEN || lens.contains(&0) {
            return Err(Error::InvalidArgument("malformed embedding pass".into()));
        }
        if tokens > self.workspace.rows {
            self.workspace = Workspace::new(&self.context, tokens)?;
        }
        let count = tokens as u32;
        let scale = (HIDDEN as f32).sqrt();
        let mut positions = Vec::with_capacity(tokens);
        let mut bounds = Vec::with_capacity(tokens * 2);
        let mut first = 0_u32;
        for &len in lens {
            let end = first + len as u32;
            for position in 0..len {
                positions.push(position as u32);
                bounds.extend([first, end]);
            }
            first = end;
        }
        let ws = &self.workspace;
        ws.input.copy_from_bytes(bytemuck::cast_slice(x0), 0);
        ws.hidden.copy_from_bytes(bytemuck::cast_slice(x0), 0);
        ws.positions
            .copy_from_bytes(bytemuck::cast_slice(&positions), 0);
        ws.bounds.copy_from_bytes(bytemuck::cast_slice(&bounds), 0);
        let kernels = &self.kernels;
        let layers_per_submit = (SUBMIT_TOKEN_LAYERS / tokens).max(1);
        let mut batch = CommandBatch::new(&self.context)?;
        let weights = &self.weights;
        let hidden = HIDDEN as u32;
        for (i, layer) in self.layers.iter().enumerate() {
            let (dim, kv, theta, window) = layer_shape(i);
            let (d, kvh) = (dim as u32, kv as u32);
            kernels.rms(
                &mut batch,
                &ws.hidden,
                Some((weights, layer.input_norm)),
                &ws.normed,
                count,
                hidden,
                1.0,
                EPSILON,
            )?;
            kernels.bf16_gemm(
                &mut batch,
                self.matrix(layer.q, Q_HEADS * dim, HIDDEN),
                &ws.normed,
                &ws.q,
                count,
            )?;
            kernels.bf16_gemm(
                &mut batch,
                self.matrix(layer.k, kv * dim, HIDDEN),
                &ws.normed,
                &ws.k,
                count,
            )?;
            kernels.bf16_gemm(
                &mut batch,
                self.matrix(layer.v, kv * dim, HIDDEN),
                &ws.normed,
                &ws.v,
                count,
            )?;
            kernels.rms(
                &mut batch,
                &ws.q,
                Some((weights, layer.q_norm)),
                &ws.q,
                count * Q_HEADS as u32,
                d,
                1.0,
                EPSILON,
            )?;
            kernels.rms(
                &mut batch,
                &ws.k,
                Some((weights, layer.k_norm)),
                &ws.k,
                count * kvh,
                d,
                1.0,
                EPSILON,
            )?;
            kernels.rms(&mut batch, &ws.v, None, &ws.v, count * kvh, d, 1.0, EPSILON)?;
            kernels.rope(
                &mut batch,
                &ws.q,
                &ws.positions,
                count,
                Q_HEADS as u32,
                d,
                theta,
            )?;
            kernels.rope(&mut batch, &ws.k, &ws.positions, count, kvh, d, theta)?;
            let shape = AttentionShape {
                tokens: count,
                q_heads: Q_HEADS as u32,
                kv_heads: kvh,
                head_dim: d,
                window,
            };
            kernels.attention(
                &mut batch,
                shape,
                &ws.q,
                &ws.k,
                &ws.v,
                &ws.bounds,
                &ws.attention,
            )?;
            kernels.bf16_gemm(
                &mut batch,
                self.matrix(layer.o, HIDDEN, Q_HEADS * dim),
                &ws.attention,
                &ws.branch,
                count,
            )?;
            kernels.rms_add(
                &mut batch,
                &ws.hidden,
                &ws.branch,
                (weights, layer.post_attention_norm),
                count,
                hidden,
                1.0,
                EPSILON,
            )?;
            // Gated GELU feed-forward.
            kernels.rms(
                &mut batch,
                &ws.hidden,
                Some((weights, layer.pre_ffn_norm)),
                &ws.normed,
                count,
                hidden,
                1.0,
                EPSILON,
            )?;
            kernels.bf16_gemm(
                &mut batch,
                self.matrix(layer.gate, INTERMEDIATE, HIDDEN),
                &ws.normed,
                &ws.gate,
                count,
            )?;
            kernels.bf16_gemm(
                &mut batch,
                self.matrix(layer.up, INTERMEDIATE, HIDDEN),
                &ws.normed,
                &ws.up,
                count,
            )?;
            let inter = INTERMEDIATE as u32;
            kernels.gelu_mul(
                &mut batch, &ws.gate, &ws.up, &ws.gate, count, inter, inter, 0,
            )?;
            kernels.bf16_gemm(
                &mut batch,
                self.matrix(layer.down, HIDDEN, INTERMEDIATE),
                &ws.gate,
                &ws.branch,
                count,
            )?;
            kernels.rms_add(
                &mut batch,
                &ws.hidden,
                &ws.branch,
                (weights, layer.post_ffn_norm),
                count,
                hidden,
                1.0,
                EPSILON,
            )?;
            // Per-layer embedding: this layer's slice of the PLE projection of
            // the input embeddings, scaled by 1/sqrt(512) and RMS-normalized.
            let slice = self.ple_projection + i * HIDDEN * HIDDEN * 2;
            kernels.bf16_gemm(
                &mut batch,
                self.matrix(slice, HIDDEN, HIDDEN),
                &ws.input,
                &ws.ple,
                count,
            )?;
            kernels.rms(
                &mut batch,
                &ws.ple,
                Some((weights, self.ple_norm)),
                &ws.ple,
                count,
                hidden,
                1.0 / scale,
                EPSILON,
            )?;
            kernels.bf16_gemm(
                &mut batch,
                self.matrix(layer.ple_gate, HIDDEN, HIDDEN),
                &ws.hidden,
                &ws.ple_gate,
                count,
            )?;
            kernels.gelu_mul(
                &mut batch,
                &ws.ple_gate,
                &ws.ple,
                &ws.ple_gate,
                count,
                hidden,
                hidden,
                0,
            )?;
            kernels.bf16_gemm(
                &mut batch,
                self.matrix(layer.ple_projection, HIDDEN, HIDDEN),
                &ws.ple_gate,
                &ws.branch,
                count,
            )?;
            kernels.rms_add(
                &mut batch,
                &ws.hidden,
                &ws.branch,
                (weights, layer.post_ple_norm),
                count,
                hidden,
                layer.scalar,
                EPSILON,
            )?;
            if (i + 1) % layers_per_submit == 0 {
                batch.commit_and_renew(&self.context)?;
                pace()?;
            }
        }
        kernels.rms(
            &mut batch,
            &ws.hidden,
            Some((weights, self.final_norm)),
            &ws.normed,
            count,
            hidden,
            1.0,
            EPSILON,
        )?;
        batch.commit_and_wait()?;
        let normed = &ws.normed.as_slice::<f32>()[..tokens * HIDDEN];
        let mut out = Vec::with_capacity(lens.len());
        let mut row = 0;
        for &len in lens {
            let mut mean = vec![0.0_f64; HIDDEN];
            for token in row..row + len {
                for (m, v) in mean
                    .iter_mut()
                    .zip(&normed[token * HIDDEN..(token + 1) * HIDDEN])
                {
                    *m += f64::from(*v);
                }
            }
            row += len;
            let n = len as f64;
            let mean: Vec<f32> = mean.iter().map(|m| (m / n) as f32).collect();
            let mut embedding: Vec<f32> = self
                .projection
                .as_chunks::<HIDDEN>()
                .0
                .iter()
                .map(|row| row.iter().zip(&mean).map(|(w, m)| w * m).sum())
                .collect();
            normalize(&mut embedding);
            out.push(embedding);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
