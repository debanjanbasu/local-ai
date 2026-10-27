//! Native text-to-speech with Kokoro-82M (hexgrad, Apache-2.0) on Metal.
//!
//! Kokoro is a `StyleTTS` 2 / `iSTFTNet` model: PL-BERT and a prosody predictor
//! turn phonemes and a 256-value voice style into durations, pitch and
//! energy; a decoder and an iSTFTNet generator turn those into 24 kHz audio.
//! Every weight keeps the checkpoint's F32 values; [`model`] runs it with
//! the kernels of `shaders/kokoro.metal`. English text is phonemized by a
//! port of `misaki` ([`g2p`]) reading its own lexicons; no GPL code (espeak-ng)
//! is linked, so words outside the lexicons use a heuristic fallback.
//!
//! The model is optional and opt-in by installation: it exists only when
//! `models/kokoro-82m/` holds the artifact written by
//! `tools/convert_kokoro.py` (see `docs/SPEECH.md`).
//!
//! Text is split like Kokoro's own pipeline: at newlines, then into chunks
//! of at most 510 phonemes (preferring sentence, then clause, then comma
//! boundaries). Each chunk is synthesized separately and handed to the
//! caller as soon as it is ready, so long inputs can stream.

mod artifact;
pub mod encode;
pub mod g2p;
mod model;
mod numbers;
mod source;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::Error;
use artifact::{Artifact, STYLE_WIDTH, VOICE_ROWS};
pub use g2p::{G2p, Lexicon, Token};
pub use model::{Synthesis, Taps};
pub use source::Noise;

/// Installation directory name inside a models directory.
pub const SPEECH_MODEL_DIR: &str = "kokoro-82m";
/// Artifact file name inside [`SPEECH_MODEL_DIR`].
pub const SPEECH_ARTIFACT: &str = "kokoro-82m-f32.bin";
/// Model id reported by the API.
pub const SPEECH_MODEL_ID: &str = "kokoro-82m";
/// Output sample rate.
pub const SAMPLE_RATE: u32 = 24_000;
/// Longest accepted input, in characters (the pinned API's limit).
pub const MAX_INPUT_CHARS: usize = 4096;
/// Phonemes per synthesized chunk (Kokoro's context of 512 ids minus the two
/// boundary ids).
pub const MAX_CHUNK_PHONEMES: usize = 510;
/// Accepted speed range (the pinned API's).
pub const SPEED_RANGE: (f32, f32) = (0.25, 4.0);

/// The pinned API's built-in voice names, each mapped to a distinct
/// Kokoro voice. These are Kokoro's voices, not `OpenAI`'s: the names are
/// accepted so that clients written against the API work unchanged.
pub const VOICE_ALIASES: [(&str, &str); 13] = [
    ("alloy", "af_alloy"),
    ("ash", "am_michael"),
    ("ballad", "bm_george"),
    ("coral", "af_heart"),
    ("echo", "am_echo"),
    ("fable", "bm_fable"),
    ("onyx", "am_onyx"),
    ("nova", "af_nova"),
    ("sage", "af_sarah"),
    ("shimmer", "af_bella"),
    ("verse", "am_puck"),
    ("marin", "bf_emma"),
    ("cedar", "am_fenrir"),
];

/// The installation directory inside `models_dir`.
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`] naming the artifact when it is missing:
/// it is required.
pub fn installed_model(models_dir: &Path) -> crate::Result<PathBuf> {
    let dir = models_dir.join(SPEECH_MODEL_DIR);
    crate::resources::require_files("Kokoro-82M", &dir, &[SPEECH_ARTIFACT])?;
    Ok(dir)
}

/// The American and British G2P of an installation, without loading the
/// network (CPU only).
///
/// # Errors
///
/// A missing or unverifiable artifact.
pub fn open_g2p(dir: &Path) -> crate::Result<(G2p, G2p)> {
    let artifact = Artifact::open(&dir.join(SPEECH_ARTIFACT))?;
    g2p_pair(&artifact)
}

fn g2p_pair(artifact: &Artifact) -> crate::Result<(G2p, G2p)> {
    let us = Arc::new(Lexicon::new(
        artifact.blob("us_gold")?,
        artifact.blob("us_silver")?,
        false,
    )?);
    let gb = Arc::new(Lexicon::new(
        artifact.blob("gb_gold")?,
        artifact.blob("gb_silver")?,
        true,
    )?);
    Ok((G2p::new(Arc::clone(&us)), G2p::with_alternate(gb, us)))
}

/// The phoneme vocabulary of an installation (CPU only).
///
/// # Errors
///
/// A missing or unverifiable artifact.
pub fn open_vocab(dir: &Path) -> crate::Result<HashMap<char, u32>> {
    let artifact = Artifact::open(&dir.join(SPEECH_ARTIFACT))?;
    Ok(vocab_of(&artifact))
}

fn vocab_of(artifact: &Artifact) -> HashMap<char, u32> {
    let mut vocab = HashMap::new();
    for (symbol, id) in &artifact.config.vocab {
        let mut chars = symbol.chars();
        if let (Some(c), None) = (chars.next(), chars.next()) {
            vocab.insert(c, *id);
        }
    }
    vocab
}

/// Split and phonemize text with `g2p` like Kokoro's pipeline
/// (`KPipeline.__call__` / `en_tokenize`): at newlines, then into chunks of
/// at most `limit` phonemes (at most [`MAX_CHUNK_PHONEMES`]).
#[must_use]
pub fn chunk_text(g2p: &G2p, text: &str, limit: usize) -> Vec<Chunk> {
    let limit = limit.clamp(1, MAX_CHUNK_PHONEMES);
    let mut out = Vec::new();
    for line in text.trim().split(['\n', '\r']) {
        if line.trim().is_empty() {
            continue;
        }
        let tokens = g2p.tokens(line, true);
        out.extend(chunk_tokens(&tokens, limit));
    }
    out.retain(|c| !c.phonemes.is_empty());
    out
}

/// One chunk of text and its phonemes, ready to synthesize.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub text: String,
    pub phonemes: String,
}

/// The loaded synthesizer. Not internally synchronized: one call at a time
/// (wrap it in a mutex to share it).
pub struct SpeechModel {
    network: model::Network,
    american: Arc<G2p>,
    british: Arc<G2p>,
    voices: BTreeMap<String, Vec<f32>>,
    vocab: HashMap<char, u32>,
}

impl SpeechModel {
    /// Load and verify an installation directory.
    ///
    /// # Errors
    ///
    /// A missing or unverifiable artifact, or a Metal failure.
    pub fn open(dir: &Path) -> crate::Result<Self> {
        let artifact = Artifact::open(&dir.join(SPEECH_ARTIFACT))?;
        let (american, british) = g2p_pair(&artifact)?;
        let mut voices = BTreeMap::new();
        for name in artifact.voice_names() {
            if let Some(pack) = artifact.voice(name) {
                voices.insert(name.to_owned(), pack);
            }
        }
        let vocab = vocab_of(&artifact);
        Ok(Self {
            network: model::Network::open(artifact)?,
            american: Arc::new(american),
            british: Arc::new(british),
            voices,
            vocab,
        })
    }

    /// Voice resolution and phonemization without the network, for request
    /// validation while a synthesis holds the model.
    #[must_use]
    pub fn catalog(&self) -> Catalog {
        Catalog {
            voices: self.voices.keys().cloned().collect(),
            american: Arc::clone(&self.american),
            british: Arc::clone(&self.british),
        }
    }

    /// SHA-256 of the verified artifact payload.
    #[must_use]
    pub fn payload_sha256(&self) -> &str {
        &self.network.artifact().payload_sha256
    }

    /// Kokoro voice ids, sorted.
    pub fn voices(&self) -> impl Iterator<Item = &str> {
        self.voices.keys().map(String::as_str)
    }

    /// The Kokoro voice for a request's voice: a Kokoro id as is, or one of
    /// the pinned API's names through [`VOICE_ALIASES`].
    #[must_use]
    pub fn resolve_voice(&self, name: &str) -> Option<&str> {
        let id = VOICE_ALIASES
            .iter()
            .find(|(alias, _)| *alias == name)
            .map_or(name, |(_, id)| id);
        self.voices.get_key_value(id).map(|(k, _)| k.as_str())
    }

    /// The G2P for a voice's accent (`b*` voices are British).
    #[must_use]
    pub fn g2p_for(&self, voice: &str) -> &G2p {
        if voice.starts_with('b') {
            &self.british
        } else {
            &self.american
        }
    }

    /// Split and phonemize text for `voice`'s accent like Kokoro's pipeline
    /// (`KPipeline.__call__` / `en_tokenize`); chunks hold at most
    /// `limit` phonemes (at most [`MAX_CHUNK_PHONEMES`]).
    #[must_use]
    pub fn chunks(&self, text: &str, voice: &str, limit: usize) -> Vec<Chunk> {
        chunk_text(self.g2p_for(voice), text, limit)
    }

    /// Kokoro ids of a phoneme string, framed by the boundary id 0 (unknown
    /// symbols are dropped, as the reference does).
    #[must_use]
    pub fn ids(&self, phonemes: &str) -> Vec<u32> {
        let mut ids = vec![0];
        ids.extend(phonemes.chars().filter_map(|c| self.vocab.get(&c).copied()));
        ids.push(0);
        ids
    }

    /// The style vector for `voice` and a chunk of `phonemes` characters
    /// (Kokoro indexes the pack by phoneme count).
    fn style(&self, voice: &str, phonemes: usize) -> crate::Result<&[f32]> {
        let pack = self
            .voices
            .get(voice)
            .ok_or_else(|| Error::InvalidArgument(format!("unknown voice {voice:?}")))?;
        let row = phonemes.clamp(1, VOICE_ROWS) - 1;
        Ok(&pack[row * STYLE_WIDTH..(row + 1) * STYLE_WIDTH])
    }

    /// Synthesize one phoneme string (at most [`MAX_CHUNK_PHONEMES`]
    /// characters) with a Kokoro voice id. `pace` runs between GPU
    /// submissions and may block (to yield) or fail (to cancel); `taps`
    /// collects intermediate activations for parity checks.
    ///
    /// # Errors
    ///
    /// Invalid input, a `pace` error, or a Metal failure.
    pub fn synthesize_phonemes(
        &self,
        phonemes: &str,
        voice: &str,
        speed: f32,
        noise: Noise,
        pace: &mut dyn FnMut() -> crate::Result<()>,
        taps: Option<&mut Taps>,
    ) -> crate::Result<Synthesis> {
        let count = phonemes.chars().count();
        if count == 0 || count > MAX_CHUNK_PHONEMES {
            return Err(Error::InvalidArgument(format!(
                "a speech chunk holds 1 to {MAX_CHUNK_PHONEMES} phonemes, not {count}"
            )));
        }
        let style = self.style(voice, count)?;
        let ids = self.ids(phonemes);
        if ids.len() < 3 {
            return Err(Error::InvalidArgument(
                "the phonemes hold no Kokoro symbol".into(),
            ));
        }
        self.network
            .synthesize(&ids, style, speed, noise, pace, taps)
    }

    /// Synthesize text with `voice` (a Kokoro id or a [`VOICE_ALIASES`]
    /// name), handing each chunk's samples to `sink` as soon as it is ready.
    /// A chunk too long to synthesize at this speed is split in half at a
    /// word boundary and retried. Returns the phoneme and sample counts.
    ///
    /// # Errors
    ///
    /// Invalid input, a `pace` or `sink` error, or a Metal failure.
    pub fn synthesize(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
        noise: Noise,
        pace: &mut dyn FnMut() -> crate::Result<()>,
        sink: &mut dyn FnMut(&[f32]) -> crate::Result<()>,
    ) -> crate::Result<SpeechUsage> {
        if text.chars().count() > MAX_INPUT_CHARS {
            return Err(Error::InvalidArgument(format!(
                "input holds more than {MAX_INPUT_CHARS} characters"
            )));
        }
        if !(SPEED_RANGE.0..=SPEED_RANGE.1).contains(&speed) {
            return Err(Error::InvalidArgument(format!(
                "speed must be between {} and {}",
                SPEED_RANGE.0, SPEED_RANGE.1
            )));
        }
        let voice = self
            .resolve_voice(voice)
            .ok_or_else(|| Error::InvalidArgument(format!("unknown voice {voice:?}")))?
            .to_owned();
        let chunks = self.chunks(text, &voice, MAX_CHUNK_PHONEMES);
        if chunks.is_empty() {
            return Err(Error::InvalidArgument(
                "the input has nothing to pronounce".into(),
            ));
        }
        let mut usage = SpeechUsage::default();
        let mut pending: Vec<String> = chunks.into_iter().rev().map(|c| c.phonemes).collect();
        let mut index = 0_u64;
        while let Some(phonemes) = pending.pop() {
            let chunk_noise = match noise {
                Noise::Zero => Noise::Zero,
                Noise::Seeded(base) => Noise::Seeded(base.wrapping_add(index)),
            };
            index += 1;
            match self.synthesize_phonemes(&phonemes, &voice, speed, chunk_noise, pace, None) {
                Ok(out) => {
                    usage.phonemes += phonemes.chars().count();
                    usage.samples += out.samples.len();
                    sink(&out.samples)?;
                }
                Err(Error::ContextOverflow(_)) if phonemes.contains(' ') => {
                    let (a, b) = split_half(&phonemes);
                    pending.push(b);
                    pending.push(a);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(usage)
    }
}

/// Voices and G2P of a loaded model, shareable across threads.
#[derive(Clone)]
pub struct Catalog {
    voices: Vec<String>,
    american: Arc<G2p>,
    british: Arc<G2p>,
}

impl Catalog {
    /// Kokoro voice ids, sorted.
    #[must_use]
    pub fn voices(&self) -> &[String] {
        &self.voices
    }

    /// As [`SpeechModel::resolve_voice`].
    #[must_use]
    pub fn resolve_voice(&self, name: &str) -> Option<&str> {
        let id = VOICE_ALIASES
            .iter()
            .find(|(alias, _)| *alias == name)
            .map_or(name, |(_, id)| id);
        self.voices.iter().find(|v| *v == id).map(String::as_str)
    }

    /// As [`SpeechModel::chunks`].
    #[must_use]
    pub fn chunks(&self, text: &str, voice: &str, limit: usize) -> Vec<Chunk> {
        let g2p = if voice.starts_with('b') {
            &self.british
        } else {
            &self.american
        };
        chunk_text(g2p, text, limit)
    }
}

/// Phonemes and samples produced by [`SpeechModel::synthesize`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeechUsage {
    pub phonemes: usize,
    pub samples: usize,
}

fn split_half(phonemes: &str) -> (String, String) {
    let middle = phonemes.len() / 2;
    let spaces: Vec<usize> = phonemes.match_indices(' ').map(|(i, _)| i).collect();
    let at = spaces
        .iter()
        .min_by_key(|i| i.abs_diff(middle))
        .copied()
        .unwrap_or(middle);
    // `at` is a space's byte index (or the middle, which split_at_checked
    // refuses if it is not a character boundary).
    let (a, b) = phonemes.split_at_checked(at).unwrap_or((phonemes, ""));
    (a.trim().to_owned(), b.trim().to_owned())
}

fn tokens_to_ps(tokens: &[Token]) -> String {
    let mut ps = String::new();
    for t in tokens {
        ps.push_str(t.phonemes.as_deref().unwrap_or(""));
        if t.whitespace {
            ps.push(' ');
        }
    }
    ps.trim().to_owned()
}

fn tokens_to_text(tokens: &[Token]) -> String {
    let mut text = String::new();
    for t in tokens {
        text.push_str(&t.text);
        if t.whitespace {
            text.push(' ');
        }
    }
    text.trim().to_owned()
}

/// `KPipeline.waterfall_last`: where to cut `tokens` so the rest fits.
fn waterfall_last(tokens: &[Token], next_count: usize, limit: usize) -> usize {
    for set in ["!.?…", ":;", ",—"] {
        let Some(mut z) = tokens.iter().rposition(|t| {
            t.phonemes
                .as_deref()
                .is_some_and(|p| p.chars().count() == 1 && set.contains(p))
        }) else {
            continue;
        };
        z += 1;
        if z < tokens.len() && matches!(tokens[z].phonemes.as_deref(), Some(")" | "”")) {
            z += 1;
        }
        if next_count.saturating_sub(tokens_to_ps(&tokens[..z]).chars().count()) <= limit {
            return z;
        }
    }
    tokens.len()
}

/// `KPipeline.en_tokenize`: group tokens into chunks of at most `limit`
/// phonemes.
fn chunk_tokens(tokens: &[Token], limit: usize) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut tks: Vec<Token> = Vec::new();
    let mut pcount = 0_usize;
    for t in tokens {
        let mut t = t.clone();
        t.phonemes = Some(t.phonemes.unwrap_or_default());
        let mut next_ps = t.phonemes.clone().unwrap_or_default();
        if t.whitespace {
            next_ps.push(' ');
        }
        let next_pcount = pcount + next_ps.trim_end().chars().count();
        if next_pcount > limit && !tks.is_empty() {
            let z = waterfall_last(&tks, next_pcount, limit).max(1);
            out.push(Chunk {
                text: tokens_to_text(&tks[..z]),
                phonemes: tokens_to_ps(&tks[..z]),
            });
            tks.drain(..z);
            pcount = tokens_to_ps(&tks).chars().count();
            if tks.is_empty() {
                next_ps = next_ps.trim_start().to_owned();
            }
        }
        // A single token longer than the limit is cut (the reference truncates).
        if next_ps.chars().count() > limit {
            let cut: String = next_ps.chars().take(limit).collect();
            t.phonemes = Some(cut.trim_end().to_owned());
            next_ps = cut;
        }
        tks.push(t);
        pcount += next_ps.chars().count();
    }
    if !tks.is_empty() {
        out.push(Chunk {
            text: tokens_to_text(&tks),
            phonemes: tokens_to_ps(&tks),
        });
    }
    for chunk in &mut out {
        if chunk.phonemes.chars().count() > limit {
            chunk.phonemes = chunk.phonemes.chars().take(limit).collect();
        }
    }
    out
}

#[cfg(test)]
mod tests;
