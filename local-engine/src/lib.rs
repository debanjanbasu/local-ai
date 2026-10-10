#![doc = include_str!("../README.md")]

mod api;
pub mod runtime;

pub mod bonsai;
#[doc(hidden)]
#[allow(clippy::must_use_candidate)]
pub mod bonsai_model;
mod bonsai_mtp;
#[doc(hidden)]
pub mod bonsai_native;
#[doc(hidden)]
#[allow(clippy::must_use_candidate, clippy::too_long_first_doc_paragraph)]
pub mod bonsai_ngram;
#[doc(hidden)]
#[allow(clippy::too_long_first_doc_paragraph)]
pub mod bonsai_tokenizer;
pub mod judgment;
mod prompt_cache;
#[doc(hidden)]
#[allow(clippy::must_use_candidate)]
pub mod resources;
mod sampler;
mod tools;

mod error;
pub use api::{
    CancelHandle, ChatMessage, ChatOutput, ChatRequest, CompletionRequest, Engine, EngineHandle,
    EngineInfo, Event, EventStream, Plan, Sampling, Signal, Stats,
};
pub use error::Error;
pub use tools::{ToolCall, ToolDefinition};

pub type Result<T> = std::result::Result<T, Error>;

pub use bonsai_mtp::{
    DEFAULT_BONSAI_MTP_ARTIFACT, DEFAULT_MTP_DEPTH, DEFAULT_MTP_ENABLED, MAX_MTP_DEPTH,
    MTP_HEAD_ARTIFACT, MtpHeadArtifact, MtpMode, MtpResolution, MtpSettings, export_head,
};
pub use runtime::{
    DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MIN_P, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P,
    EVENT_BUFFER, GenerateParams, GenerationStats, MtpStats, NgramStats, PrefillProgress,
};
