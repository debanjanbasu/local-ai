#![doc = include_str!("../README.md")]

pub mod activity;
mod api;
pub mod runtime;
pub mod speech;

pub mod bonsai;
#[doc(hidden)]
pub mod bonsai_dflash;
#[doc(hidden)]
#[expect(clippy::must_use_candidate)]
pub mod bonsai_model;
mod bonsai_mtp;
#[doc(hidden)]
pub mod bonsai_native;
#[doc(hidden)]
#[expect(clippy::must_use_candidate, clippy::too_long_first_doc_paragraph)]
pub mod bonsai_ngram;
#[doc(hidden)]
#[expect(clippy::too_long_first_doc_paragraph)]
pub mod bonsai_tokenizer;
pub mod bonsai_vision;
pub mod compaction;
pub mod embedding;
pub mod judgment;
pub mod moderation;
mod prompt_cache;
#[doc(hidden)]
#[expect(clippy::must_use_candidate)]
pub mod resources;
mod sampler;
mod structured;
mod tools;
pub mod transcription;

mod error;
pub use api::{
    CancelHandle, ChatImage, ChatMessage, ChatOutput, ChatRequest, Engine, EngineHandle,
    EngineInfo, Event, EventRelay, EventStream, ImageDetail, PendingDecision, Plan, Sampling,
    Signal, Stats, ToolOutput,
};
pub use bonsai_tokenizer::ReasoningEffort;
pub use error::Error;
pub use structured::ResponseFormat;
pub use tools::grammar::{LARK_KEYWORD, check_lark_grammar};
pub use tools::{ToolCall, ToolCallDelta, ToolChoice, ToolDefinition};

pub type Result<T> = std::result::Result<T, Error>;

pub use bonsai_dflash::{
    DEFAULT_BONSAI_DFLASH_ARTIFACT, DFLASH_HEAD_ARTIFACT, DflashHeadArtifact, export_dflash_head,
};
pub use bonsai_mtp::{
    DEFAULT_BONSAI_MTP_ARTIFACT, DEFAULT_MTP_DEPTH, DEFAULT_MTP_ENABLED, MAX_MTP_DEPTH,
    MTP_HEAD_ARTIFACT, MtpHeadArtifact, MtpMode, MtpResolution, MtpSettings, export_head,
};
pub use runtime::{
    DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MIN_P, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P,
    EVENT_BUFFER, GenerateParams, GenerationStats, MAX_TOP_LOGPROBS, MtpStats, NgramStats,
    PrefillProgress,
};
pub use sampler::{TokenLogprob, TopLogprob};
