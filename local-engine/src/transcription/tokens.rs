//! Whisper's multilingual tokenizer: special-token ids from the artifact,
//! byte-level BPE text from the checkpoint's own `tokenizer.json`.
//!
//! Text is encoded with the HF `tokenizers` crate (the same GPT-2 split and
//! merges as the reference's tiktoken encoding) and decoded byte for byte:
//! token strings map back to bytes through GPT-2's byte-to-unicode table and
//! the bytes become UTF-8 with replacement characters, as tiktoken's
//! `decode(..., errors="replace")` does.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::Error;

/// Whisper's languages in token order (`<|en|>` first): ISO code and the
/// lower-case English name the API reports.
pub const LANGUAGES: [(&str, &str); 100] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];

/// The ISO code of a language given as a code or an English name
/// (case-insensitive), if Whisper knows it.
#[must_use]
pub fn language_code(language: &str) -> Option<&'static str> {
    let wanted = language.trim().to_ascii_lowercase();
    LANGUAGES
        .iter()
        .find(|(code, name)| *code == wanted || *name == wanted)
        .map(|(code, _)| *code)
}

/// The English name of a language code.
#[must_use]
pub fn language_name(code: &str) -> Option<&'static str> {
    LANGUAGES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, name)| *name)
}

/// Special-token ids, from the converter's header (the pinned
/// `generation_config.json` and `tokenizer.json`).
#[derive(Deserialize, Clone, Debug)]
pub(crate) struct SpecialTokens {
    pub sot: u32,
    pub eot: u32,
    pub sot_prev: u32,
    pub no_timestamps: u32,
    pub transcribe: u32,
    pub translate: u32,
    pub sot_lm: u32,
    pub no_speech: u32,
    pub timestamp_begin: u32,
    pub languages: BTreeMap<String, u32>,
    pub suppress_tokens: Vec<u32>,
    pub begin_suppress_tokens: Vec<u32>,
    #[allow(dead_code)]
    pub alignment_heads: Vec<[u32; 2]>,
    pub max_initial_timestamp_index: u32,
}

impl SpecialTokens {
    pub(crate) fn validate(&self) -> crate::Result<()> {
        let invalid = |m: &str| Error::InvalidFormat(format!("speech artifact tokens: {m}"));
        if self.languages.len() != LANGUAGES.len() {
            return Err(invalid("language count is not 100"));
        }
        for (index, (code, _)) in LANGUAGES.iter().enumerate() {
            if self.languages.get(*code) != Some(&(self.sot + 1 + index as u32)) {
                return Err(invalid(&format!("language {code} has an unexpected id")));
            }
        }
        let vocab = super::VOCAB as u32;
        if self.timestamp_begin + 1501 != vocab
            || [
                self.sot,
                self.eot,
                self.sot_prev,
                self.no_timestamps,
                self.transcribe,
                self.translate,
                self.sot_lm,
                self.no_speech,
            ]
            .iter()
            .any(|id| *id < self.eot || *id >= self.timestamp_begin)
            || self.suppress_tokens.iter().any(|id| *id >= vocab)
            || self.begin_suppress_tokens.iter().any(|id| *id >= vocab)
        {
            return Err(invalid("special token ids are inconsistent"));
        }
        Ok(())
    }

    /// The language token of an ISO code.
    pub(crate) fn language(&self, code: &str) -> Option<u32> {
        self.languages.get(code).copied()
    }

    /// The reference's `suppress_tokens="-1"`: the non-speech symbols and
    /// every special token but end-of-text and the timestamps.
    pub(crate) fn suppressed(&self) -> Vec<u32> {
        let mut all = self.suppress_tokens.clone();
        all.extend([
            self.transcribe,
            self.translate,
            self.sot,
            self.sot_prev,
            self.sot_lm,
            self.no_speech,
        ]);
        all.sort_unstable();
        all.dedup();
        all
    }
}

/// GPT-2's reversible byte <-> printable-character table, as a map from the
/// character back to its byte.
fn unicode_to_byte() -> HashMap<char, u8> {
    let mut printable: Vec<u32> = (u32::from(b'!')..=u32::from(b'~'))
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars = printable.clone();
    let mut extra = 0;
    for byte in 0..256_u32 {
        if !printable.contains(&byte) {
            printable.push(byte);
            chars.push(256 + extra);
            extra += 1;
        }
    }
    printable
        .into_iter()
        .zip(chars)
        .filter_map(|(byte, ch)| Some((char::from_u32(ch)?, u8::try_from(byte).ok()?)))
        .collect()
}

pub(crate) struct WhisperTokenizer {
    /// `None` only in tests that never encode.
    tokenizer: Option<Tokenizer>,
    /// Bytes of every regular token id below end-of-text.
    bytes: Arc<Vec<Vec<u8>>>,
    pub(crate) special: SpecialTokens,
}

impl WhisperTokenizer {
    pub(crate) fn open(path: &Path, special: SpecialTokens) -> crate::Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|error| Error::Tokenizer(format!("speech tokenizer: {error}")))?;
        let table = unicode_to_byte();
        let mut bytes = Vec::with_capacity(special.eot as usize);
        for id in 0..special.eot {
            let token = tokenizer
                .id_to_token(id)
                .ok_or_else(|| Error::Tokenizer(format!("speech tokenizer lacks id {id}")))?;
            let raw = token
                .chars()
                .map(|ch| table.get(&ch).copied())
                .collect::<Option<Vec<u8>>>()
                .ok_or_else(|| {
                    Error::Tokenizer(format!("speech token {id} is not byte-level BPE"))
                })?;
            bytes.push(raw);
        }
        Ok(Self {
            tokenizer: Some(tokenizer),
            bytes: Arc::new(bytes),
            special,
        })
    }

    /// A decode-only tokenizer over given token bytes.
    #[cfg(test)]
    pub(crate) fn from_bytes(bytes: Vec<Vec<u8>>, special: SpecialTokens) -> Self {
        Self {
            tokenizer: None,
            bytes: Arc::new(bytes),
            special,
        }
    }

    /// Text tokens of `text`, no special tokens added.
    pub(crate) fn encode(&self, text: &str) -> crate::Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::Tokenizer("no encoder loaded".into()))?
            .encode(text, false)
            .map_err(|error| Error::Tokenizer(error.to_string()))?
            .get_ids()
            .to_vec())
    }

    /// Every regular token's bytes, indexed by id.
    pub(crate) fn byte_table(&self) -> Arc<Vec<Vec<u8>>> {
        Arc::clone(&self.bytes)
    }

    /// Bytes of a text token (empty for special and timestamp tokens).
    pub(crate) fn token_bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    /// Text of the regular tokens among `ids` (special and timestamp tokens
    /// are skipped), invalid UTF-8 replaced.
    pub(crate) fn decode(&self, ids: &[u32]) -> String {
        let raw: Vec<u8> = ids
            .iter()
            .flat_map(|id| self.token_bytes(*id).iter().copied())
            .collect();
        String::from_utf8_lossy(&raw).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_resolve_by_code_or_name() {
        assert_eq!(language_code("EN"), Some("en"));
        assert_eq!(language_code("German"), Some("de"));
        assert_eq!(language_code("haitian creole"), Some("ht"));
        assert_eq!(language_code("xx"), None);
        assert_eq!(language_name("yue"), Some("cantonese"));
        let mut codes: Vec<_> = LANGUAGES.iter().map(|(c, _)| *c).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), 100);
    }

    #[test]
    fn byte_table_is_a_bijection() {
        let table = unicode_to_byte();
        assert_eq!(table.len(), 256);
        assert_eq!(table.get(&'Ġ'), Some(&b' '));
        assert_eq!(table.get(&'A'), Some(&b'A'));
        let mut bytes: Vec<u8> = table.values().copied().collect();
        bytes.sort_unstable();
        bytes.dedup();
        assert_eq!(bytes.len(), 256);
    }
}
