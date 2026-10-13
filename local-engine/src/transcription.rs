//! Native speech recognition and translation on Metal.
//!
//! Whisper large-v3 (`OpenAI`, MIT, 1.55 B parameters) transcribes 99 languages and
//! translates them into English with one encoder-decoder model. This module
//! runs it natively: the log-mel front end on the CPU ([`mel`]), the encoder
//! and decoder on Metal with the checkpoint's F16 matrices and F32
//! activations, and the original package's decoding and long-form
//! transcription logic (greedy or sampled decoding with timestamp rules, the
//! temperature fallback, the no-speech skip and the 30-second sliding window).
//!
//! The model is optional and opt-in by installation: it exists only when
//! `models/whisper-large-v3/` holds the artifact written by
//! `tools/convert_whisper.py` and its `tokenizer.json` (see `docs/TRANSCRIPTION.md`).
//! Audio files are decoded by [`audio`].

mod artifact;
pub mod audio;
mod decode;
pub mod mel;
mod model;
pub mod tokens;

use std::path::{Path, PathBuf};

use crate::Error;
use artifact::Artifact;
use decode::{Rng, WindowOptions, Work};
use mel::{LogMel, MelFilters, WINDOW_FRAMES, WINDOW_SAMPLES};
use model::{Bound, WhisperGpu};
use tokens::WhisperTokenizer;

/// Installation directory name inside a models directory.
pub const TRANSCRIPTION_MODEL_DIR: &str = "whisper-large-v3";
/// Artifact file name inside [`TRANSCRIPTION_MODEL_DIR`].
pub const TRANSCRIPTION_ARTIFACT: &str = "whisper-large-v3-f16.bin";
/// Model id reported by the API.
pub const TRANSCRIPTION_MODEL_ID: &str = "whisper-large-v3";
/// Input sample rate.
pub const SAMPLE_RATE: u32 = audio::SAMPLE_RATE;

const N_MELS: usize = 128;
const CTX_AUDIO: usize = 1500;
const CTX_TEXT: usize = 448;
const STATE: usize = 1280;
const HEADS: usize = 20;
const LAYERS: usize = 32;
const VOCAB: usize = 51_866;
const FFN: usize = 5120;

/// The installation directory inside `models_dir`, if the artifact and the
/// tokenizer are there.
#[must_use]
pub fn installed_model(models_dir: &Path) -> Option<PathBuf> {
    let dir = models_dir.join(TRANSCRIPTION_MODEL_DIR);
    (dir.join(TRANSCRIPTION_ARTIFACT).is_file() && dir.join("tokenizer.json").is_file())
        .then_some(dir)
}

/// Transcribe in the spoken language, or translate into English.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Transcribe,
    Translate,
}

/// Decoding temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Temperature {
    /// Greedy, then 0.2, 0.4 … 1.0 while a window's output is too
    /// repetitive (compression ratio > 2.4) or too unlikely (average
    /// log-probability < −1), as the reference does by default.
    Fallback,
    /// One temperature (0 is greedy, without fallback).
    Fixed(f32),
}

/// What to transcribe and how.
#[derive(Clone, Debug)]
pub struct TranscribeOptions {
    pub task: Task,
    /// ISO code of the spoken language; detected from the first 30 seconds
    /// when `None`.
    pub language: Option<String>,
    /// Text that precedes the audio (style, spelling, vocabulary).
    pub prompt: Option<String>,
    pub temperature: Temperature,
    /// Feed each window the text of the previous ones (the reference's
    /// default).
    pub condition_on_previous_text: bool,
    /// Seed for sampling at temperatures above 0.
    pub seed: u64,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            task: Task::Transcribe,
            language: None,
            prompt: None,
            temperature: Temperature::Fallback,
            condition_on_previous_text: true,
            seed: 0,
        }
    }
}

/// One timed piece of the transcript, as the reference reports it.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub id: usize,
    /// Mel frame (10 ms) where its window started.
    pub seek: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// Text and timestamp tokens.
    pub tokens: Vec<u32>,
    /// Log-probability of each token in `tokens`.
    pub logprobs: Vec<f32>,
    pub temperature: f32,
    pub avg_logprob: f64,
    pub compression_ratio: f64,
    pub no_speech_prob: f64,
}

/// Work done for a transcript.
#[derive(Clone, Copy, Debug, Default)]
pub struct Usage {
    /// 30-second encoder passes (language detection and fallbacks included).
    pub audio_windows: usize,
    /// Prompt and start-of-transcript tokens fed to the decoder.
    pub prompt_tokens: usize,
    /// Tokens sampled (end-of-text and fallback attempts included).
    pub output_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct Transcript {
    /// The text as the reference returns it (usually with a leading space).
    pub text: String,
    /// ISO code of the language (given or detected).
    pub language: String,
    /// Seconds of audio.
    pub duration: f64,
    pub segments: Vec<Segment>,
    pub usage: Usage,
}

/// Pacing between GPU submissions.
///
/// `busy` reports whether higher-priority work (Bonsai decoding) is running,
/// which makes submissions short; `pause` runs after each submission and may
/// block (to yield the GPU) or fail (to cancel).
pub struct Pace<'a> {
    pub busy: &'a dyn Fn() -> bool,
    pub pause: &'a mut dyn FnMut() -> crate::Result<()>,
}

/// The loaded model. Not internally synchronized: one call at a time.
pub struct TranscriptionModel {
    gpu: WhisperGpu,
    artifact: Artifact,
    tokenizer: WhisperTokenizer,
    filters: MelFilters,
}

impl TranscriptionModel {
    /// Load and verify an installation directory (artifact + tokenizer).
    ///
    /// # Errors
    ///
    /// Fails when a file is missing, the artifact does not verify, or Metal
    /// cannot be set up.
    pub fn open(dir: &Path) -> crate::Result<Self> {
        let artifact = Artifact::open(&dir.join(TRANSCRIPTION_ARTIFACT))?;
        let tokenizer =
            WhisperTokenizer::open(&dir.join("tokenizer.json"), artifact.tokens.clone())?;
        let filters = artifact.f32_values("mel_filters", N_MELS, mel::FFT_BINS)?;
        let gpu = WhisperGpu::new(&artifact)?;
        Ok(Self {
            gpu,
            filters: MelFilters::new(&filters, N_MELS),
            artifact,
            tokenizer,
        })
    }

    /// SHA-256 of the verified artifact payload.
    #[must_use]
    pub fn payload_sha256(&self) -> &str {
        &self.artifact.payload_sha256
    }

    /// The log-mel of `audio` followed by `padding` zero samples.
    #[must_use]
    pub fn log_mel(&self, audio: &[f32], padding: usize) -> LogMel {
        self.filters.log_mel(audio, padding)
    }

    /// Text tokens of `text` (no special tokens).
    ///
    /// # Errors
    ///
    /// If the tokenizer fails.
    pub fn encode_text(&self, text: &str) -> crate::Result<Vec<u32>> {
        self.tokenizer.encode(text)
    }

    /// Text of token ids (special and timestamp tokens skipped).
    #[must_use]
    pub fn decode_text(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids)
    }

    /// UTF-8 bytes of one token (empty for special and timestamp tokens).
    #[must_use]
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.tokenizer.token_bytes(id)
    }

    /// Every regular token's UTF-8 bytes, indexed by id (ids at or above
    /// [`Self::eot`] have none); shareable without holding the model.
    #[must_use]
    pub fn token_byte_table(&self) -> std::sync::Arc<Vec<Vec<u8>>> {
        self.tokenizer.byte_table()
    }

    /// The end-of-text token; every id at or above it is special.
    #[must_use]
    pub const fn eot(&self) -> u32 {
        self.tokenizer.special.eot
    }

    fn bound(&mut self) -> Bound<'_> {
        Bound {
            gpu: &mut self.gpu,
            payload: self.artifact.payload(),
        }
    }

    /// Encode one window (`[3000][128]` time-major) and return the encoder
    /// output `[1500][1280]` (for reference checks).
    ///
    /// # Errors
    ///
    /// Metal failures, a wrongly sized window, or a `pace` error.
    pub fn encoder_output(
        &mut self,
        window: &[f32],
        pace: &mut Pace<'_>,
    ) -> crate::Result<Vec<f32>> {
        use decode::Forward as _;
        self.bound().encode(window, pace)?;
        Ok(self.gpu.encoder_output())
    }

    /// Logits of every position of `tokens` against the last encoded window
    /// (teacher forcing, for reference checks).
    ///
    /// # Errors
    ///
    /// No encoded window, too many tokens, or Metal failures.
    pub fn sequence_logits(
        &mut self,
        tokens: &[u32],
        pace: &mut Pace<'_>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        use decode::Forward as _;
        let rows: Vec<usize> = (0..tokens.len()).collect();
        self.bound().prefill(tokens, &rows, pace)
    }

    /// Language probabilities of the last encoded window, most likely first.
    ///
    /// # Errors
    ///
    /// No encoded window or Metal failures.
    pub fn window_languages(&mut self, pace: &mut Pace<'_>) -> crate::Result<Vec<(String, f64)>> {
        let tokenizer = &self.tokenizer;
        let mut bound = Bound {
            gpu: &mut self.gpu,
            payload: self.artifact.payload(),
        };
        decode::language_probabilities(&mut bound, tokenizer, pace)
    }

    /// Greedy-or-sampled decode of the last encoded window with timestamps
    /// (one `DecodingTask`); the sampled tokens without end-of-text.
    ///
    /// # Errors
    ///
    /// Unknown language, no encoded window, or Metal failures.
    pub fn decode_window(
        &mut self,
        task: Task,
        language: &str,
        prompt: &[u32],
        temperature: f32,
        pace: &mut Pace<'_>,
    ) -> crate::Result<Vec<u32>> {
        let language = self
            .tokenizer
            .special
            .language(language)
            .ok_or_else(|| Error::InvalidArgument(format!("unknown language {language:?}")))?;
        let tokenizer = &self.tokenizer;
        let mut bound = Bound {
            gpu: &mut self.gpu,
            payload: self.artifact.payload(),
        };
        let mut work = Work::default();
        Ok(decode::decode_window(
            &mut bound,
            tokenizer,
            task,
            language,
            prompt,
            temperature,
            &mut Rng::new(1),
            pace,
            &mut work,
        )?
        .tokens)
    }

    /// Transcribe or translate 16 kHz mono `audio`; `on_segments` sees the
    /// segments of each 30-second window as soon as they are final (and may
    /// fail to cancel).
    ///
    /// # Errors
    ///
    /// An unknown language, a tokenizer or Metal failure, or a `pace` or
    /// `on_segments` error.
    pub fn transcribe(
        &mut self,
        audio: &[f32],
        options: &TranscribeOptions,
        pace: &mut Pace<'_>,
        on_segments: &mut dyn FnMut(&[Segment]) -> crate::Result<()>,
    ) -> crate::Result<Transcript> {
        let mel = self.filters.log_mel(audio, WINDOW_SAMPLES);
        let tokenizer = &self.tokenizer;
        let mut bound = Bound {
            gpu: &mut self.gpu,
            payload: self.artifact.payload(),
        };
        let mut work = Work::default();
        let detected = options.language.is_none();
        let language = if let Some(language) = &options.language {
            tokens::language_code(language)
                .ok_or_else(|| {
                    Error::InvalidArgument(format!("Whisper does not know language {language:?}"))
                })?
                .to_owned()
        } else {
            use decode::Forward as _;
            bound.encode(&mel.window(0, WINDOW_FRAMES), pace)?;
            work.windows += 1;
            decode::language_probabilities(&mut bound, tokenizer, pace)?
                .into_iter()
                .next()
                .map(|(code, _)| code)
                .ok_or_else(|| Error::Generation("no language probabilities".into()))?
        };
        let language_token = tokenizer
            .special
            .language(&language)
            .ok_or_else(|| Error::InvalidArgument(format!("unknown language {language:?}")))?;
        let initial_prompt = options
            .prompt
            .as_ref()
            .map(|prompt| tokenizer.encode(&format!(" {}", prompt.trim())))
            .transpose()?
            .unwrap_or_default();
        let window_options = WindowOptions {
            task: options.task,
            language: language_token,
            temperature: options.temperature,
            condition_on_previous_text: options.condition_on_previous_text,
        };
        let mut rng = Rng::new(options.seed);
        let (segments, text_tokens) = decode::transcribe(
            &mut bound,
            tokenizer,
            &mel,
            detected,
            &window_options,
            &initial_prompt,
            &mut rng,
            pace,
            &mut work,
            on_segments,
        )?;
        Ok(Transcript {
            text: tokenizer.decode(&text_tokens),
            language,
            duration: audio.len() as f64 / f64::from(SAMPLE_RATE),
            segments,
            usage: Usage {
                audio_windows: work.windows,
                prompt_tokens: work.prompt_tokens,
                output_tokens: work.sampled_tokens,
            },
        })
    }
}

#[cfg(test)]
mod tests;
