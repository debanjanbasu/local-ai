pub mod archive;
pub mod cli;
pub mod continuous_batching;
mod gemma4_audio;
mod gemma4_vision;
pub mod gguf;
pub mod kv_cache;
pub mod layer;
pub mod lma;
pub mod multimodal;
pub mod pipeline;
pub mod qat_recover;
pub mod request_queue;
pub mod sampler;
pub mod tokenizer;
pub mod turboquant;

mod error;
pub use error::Error;

pub type Result<T> = std::result::Result<T, Error>;

use std::path::PathBuf;

pub use multimodal::{
    DecodedRgbImage, DecodedVideoFrame, MediaInput, MultimodalPrompt, MultimodalSupport, PcmAudio,
};

use crate::pipeline::Pipeline;

/// Default output budget for the no-knobs path. `usize::MAX` means "use the
/// largest generation budget that fits the effective model context"; the
/// pipeline clamps this before allocating output buffers.
pub const DEFAULT_MAX_OUTPUT_TOKENS: usize = usize::MAX;
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_TOP_P: f32 = 0.95;

#[derive(Debug, Clone, Default)]
pub struct EngineConfig {
    pub model_dir: PathBuf,
    /// Requested context window in positions. `0` = auto: target the model's
    /// full `max_position_embeddings`. Explicit values are capped at the model
    /// maximum, and either way the device memory budget makes the final call.
    pub max_context_length: usize,
}

#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            max_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
        }
    }
}

pub struct Engine {
    pipeline: Pipeline,
}

impl Engine {
    pub fn new(engine_config: &EngineConfig) -> Result<Self> {
        if engine_config.model_dir.as_os_str().is_empty() {
            return Err(Error::Io("model directory path is empty".into()));
        }
        let pipeline =
            Pipeline::new_qat(&engine_config.model_dir, engine_config.max_context_length)?;
        Ok(Self { pipeline })
    }

    pub fn generate(&mut self, prompt: &str, params: &GenerateParams) -> Result<String> {
        self.pipeline.generate(prompt, params)
    }

    /// Generate a chat reply: the prompt is wrapped in the model's turn
    /// template (raw continuation if the model has no chat tokens).
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn generate_chat(&mut self, prompt: &str, params: &GenerateParams) -> Result<String> {
        self.pipeline.generate_chat(prompt, params)
    }

    /// Generate from a structured multimodal prompt.
    ///
    /// Text-only prompts reuse the normal chat path. Image/video/audio prompts are
    /// accepted only when the installed `.lma` includes the corresponding
    /// Gemma 4 modality tensors; otherwise this returns an actionable error
    /// instead of silently dropping media inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed model bundle lacks required modality
    /// tensors, or if generation fails.
    pub fn generate_multimodal_chat(
        &mut self,
        prompt: &MultimodalPrompt,
        params: &GenerateParams,
    ) -> Result<String> {
        self.pipeline.generate_multimodal_chat(prompt, params)
    }

    #[must_use]
    pub const fn multimodal_support(&self) -> MultimodalSupport {
        self.pipeline.multimodal_support()
    }

    /// Reset conversation state (token position and KV cache) so the next
    /// [`generate`](Self::generate) call starts a fresh sequence.
    pub fn reset(&mut self) {
        self.pipeline.reset();
    }

    #[must_use]
    pub const fn config(&self) -> &local_core::config::Gemma4QATConfig {
        self.pipeline.config()
    }

    /// Run the continuous-batching server loop on this engine's pipeline.
    ///
    /// Allocates `n_lanes` decode lanes, each with `lane_capacity` tokens of KV
    /// (clamped to the model's effective context; `0` = use the full effective
    /// context), then drives [`pipeline::Pipeline::serve_batched`]: it pulls
    /// [`pipeline::ServeRequest`]s from `rx`, packs them into lanes, decodes all
    /// active lanes together, and replies per request as each finishes.
    ///
    /// Intended to run on the single thread that owns this `Engine`; connection
    /// threads submit requests over the channel and block on each request's
    /// reply.
    ///
    /// # Errors
    ///
    /// Returns an error if the lane state cannot be allocated or on any
    /// prefill / decode failure that is not isolated to a single request.
    pub fn serve_batched(
        &mut self,
        n_lanes: usize,
        lane_capacity: usize,
        rx: &std::sync::mpsc::Receiver<pipeline::ServeRequest>,
    ) -> Result<()> {
        let max_ctx = self.pipeline.max_effective_context();
        let capacity = if lane_capacity == 0 {
            max_ctx
        } else {
            lane_capacity.min(max_ctx)
        };
        let mut state = self.pipeline.new_batched_decode_state(n_lanes, capacity)?;
        self.pipeline.serve_batched(&mut state, rx)
    }
}
