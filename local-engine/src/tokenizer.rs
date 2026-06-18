//! Text tokenization backed by the Hugging Face `tokenizers` library.
//!
//! Loads a Gemma `tokenizer.json` and provides prompt encoding (with optional
//! BOS) and detokenization of generated token IDs back to text.

use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;

/// A loaded tokenizer plus its special-token IDs.
pub struct Tokenizer {
    inner: HfTokenizer,
    bos_id: Option<u32>,
    eos_ids: Vec<u32>,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Tokenizer`](crate::Error::Tokenizer) if the file cannot
    /// be read or parsed.
    pub fn from_file(path: &Path) -> crate::Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| crate::Error::Tokenizer(format!("{}: {e}", path.display())))?;
        Ok(Self::from_inner(inner))
    }

    /// Load a tokenizer from in-memory `tokenizer.json` bytes (e.g. embedded
    /// in a single-file `.lma` bundle).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Tokenizer`](crate::Error::Tokenizer) if the bytes
    /// cannot be parsed.
    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        let inner = HfTokenizer::from_bytes(bytes)
            .map_err(|e| crate::Error::Tokenizer(format!("embedded tokenizer.json: {e}")))?;
        Ok(Self::from_inner(inner))
    }

    fn from_inner(inner: HfTokenizer) -> Self {
        let bos_id = inner.token_to_id("<bos>");
        // Gemma stops on `<eos>` and the chat turn terminator — named
        // `<end_of_turn>` on Gemma 2/3 and `<turn|>` on Gemma 4.
        let eos_ids = ["<eos>", "<end_of_turn>", "<turn|>"]
            .iter()
            .filter_map(|t| inner.token_to_id(t))
            .collect();
        Self {
            inner,
            bos_id,
            eos_ids,
        }
    }

    /// Load a tokenizer from a model directory (expects `tokenizer.json`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Tokenizer`](crate::Error::Tokenizer) if the file is
    /// missing or invalid.
    pub fn from_model_dir(dir: &Path) -> crate::Result<Self> {
        Self::from_file(&dir.join("tokenizer.json"))
    }

    /// Encode `text` into token IDs, optionally prepending the BOS token.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Tokenizer`](crate::Error::Tokenizer) if encoding fails.
    pub fn encode(&self, text: &str, add_bos: bool) -> crate::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| crate::Error::Tokenizer(e.to_string()))?;
        let ids = encoding.get_ids();
        if add_bos && let Some(bos) = self.bos_id {
            let mut out = Vec::with_capacity(ids.len() + 1);
            out.push(bos);
            out.extend_from_slice(ids);
            Ok(out)
        } else {
            Ok(ids.to_vec())
        }
    }

    /// Decode token IDs back to text, skipping special tokens.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Tokenizer`](crate::Error::Tokenizer) if decoding fails.
    pub fn decode(&self, ids: &[u32]) -> crate::Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| crate::Error::Tokenizer(e.to_string()))
    }

    /// Beginning-of-sequence token ID, if defined.
    #[must_use]
    pub const fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    /// End-of-sequence token IDs (includes the chat turn terminator).
    #[must_use]
    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// Wrap a user message in the model's chat template, detected from the
    /// turn-token names in the vocabulary (`<|turn>`/`<turn|>` on Gemma 4,
    /// `<start_of_turn>`/`<end_of_turn>` on earlier Gemma). Returns `None`
    /// when the vocabulary has no turn tokens (base, non-chat models).
    #[must_use]
    pub fn chat_prompt(&self, user: &str) -> Option<String> {
        let (open, close) = if self.inner.token_to_id("<|turn>").is_some() {
            ("<|turn>", "<turn|>")
        } else if self.inner.token_to_id("<start_of_turn>").is_some() {
            ("<start_of_turn>", "<end_of_turn>")
        } else {
            return None;
        };
        Some(format!("{open}user\n{user}{close}\n{open}model\n"))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

    /// Build a tiny in-memory tokenizer saved to a temp `tokenizer.json` so the
    /// wrapper's load/encode/decode/special-token paths are exercised without
    /// depending on the multi-megabyte production vocabulary.
    fn write_tiny_tokenizer(dir: &Path) {
        let mut vocab = ahash::AHashMap::new();
        for (i, tok) in ["<bos>", "<eos>", "hello", "world"].iter().enumerate() {
            vocab.insert((*tok).to_string(), i as u32);
        }
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<eos>".to_string())
            .build()
            .expect("build wordlevel");
        let mut tk = HfTokenizer::new(model);
        tk.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        tk.add_special_tokens(&[
            AddedToken::from("<bos>", true),
            AddedToken::from("<eos>", true),
        ]);
        tk.save(dir.join("tokenizer.json"), false)
            .expect("save tokenizer");
    }

    #[test]
    fn round_trip_and_special_tokens() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_tiny_tokenizer(dir.path());

        let tok = Tokenizer::from_model_dir(dir.path()).expect("load");
        assert_eq!(tok.bos_id(), Some(0));
        assert_eq!(tok.eos_ids(), &[1]);

        let no_bos = tok.encode("hello world", false).expect("encode");
        assert_eq!(no_bos, vec![2, 3]);

        let with_bos = tok.encode("hello world", true).expect("encode");
        assert_eq!(with_bos, vec![0, 2, 3]);

        let text = tok.decode(&[2, 3]).expect("decode");
        assert_eq!(text, "hello world");
    }
}
