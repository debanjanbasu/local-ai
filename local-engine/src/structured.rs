//! Schema-constrained generation: OpenAI-style `response_format` enforced on
//! every target token selection by [llguidance](https://github.com/guidance-ai/llguidance).
//!
//! A [`ResponseFormat`] other than [`ResponseFormat::Text`] is compiled
//! against the checkpoint's own token bytes before any prompt work, so an
//! invalid or unsupported schema fails the request up front instead of
//! producing unconstrained text. The compiled [`Grammar`] then rides in the
//! request's [`Sampler`](crate::sampler::Sampler): each selection masks the
//! logits to the tokens the grammar allows (end-of-sequence only once the
//! document is complete) and advances the matcher by the one token selected.
//!
//! Compilation is strict: unsupported keywords, unknown formats, `oneOf`
//! whose branches cannot be proven exclusive, and `$ref`s outside the
//! document (`#/...`) are errors, never warnings.
//! Nothing is fetched: llguidance is built without its `referencing` feature.

use std::sync::{Arc, OnceLock};

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{SimpleVob, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};
use serde_json::{Value, json};
use tokenizers::Tokenizer;

/// The shape a generation's final answer must take (`OpenAI` `response_format`).
///
/// For a chat request with `thinking`, reasoning stays unconstrained and the
/// format applies from the token after the closing `</think>`; without
/// thinking, and for raw completions, it applies from the first generated
/// token. Requests with tools cannot also carry a format.
///
/// Generation ends at end-of-sequence only once the answer is a complete
/// document the schema accepts. A token limit or cancellation still stops it
/// early, with incomplete JSON: see
/// [`GenerationStats::response_format_complete`](crate::GenerationStats::response_format_complete).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ResponseFormat {
    /// Unconstrained text (the default).
    #[default]
    Text,
    /// Any JSON object: `OpenAI` `{"type": "json_object"}`, the schema
    /// `{"type": "object"}`.
    JsonObject,
    /// A document valid under this JSON Schema: the `schema` member of `OpenAI`
    /// `{"type": "json_schema", "json_schema": {...}}`. Must be a JSON object.
    /// The constraint is enforced whatever `OpenAI`'s `strict` flag says.
    JsonSchema(Value),
}

impl ResponseFormat {
    /// Whether this format leaves generation unconstrained.
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text)
    }

    /// The JSON Schema to compile, or `None` for unconstrained text.
    fn schema(&self) -> crate::Result<Option<Value>> {
        match self {
            Self::Text => Ok(None),
            Self::JsonObject => Ok(Some(json!({"type": "object"}))),
            Self::JsonSchema(schema) => {
                let Some(object) = schema.as_object() else {
                    return Err(invalid("JSON schema must be a JSON object"));
                };
                // llguidance reads compile options, including `lenient` (which
                // turns unsupported keywords into warnings), from this key.
                if object.contains_key("x-guidance") {
                    return Err(invalid("JSON schema key \"x-guidance\" is not supported"));
                }
                Ok(Some(schema.clone()))
            }
        }
    }
}

fn invalid(message: impl std::fmt::Display) -> crate::Error {
    crate::Error::InvalidArgument(format!("invalid response_format: {message}"))
}

/// One tokenizer's grammar compiler, built on first use and shared by every
/// clone of the tokenizer: a plain request never pays for it.
#[derive(Default)]
pub struct GrammarCompiler {
    factory: OnceLock<Result<Arc<ParserFactory>, String>>,
}

impl GrammarCompiler {
    fn factory(
        &self,
        tokenizer: &Arc<Tokenizer>,
        eos_ids: &[u32],
    ) -> crate::Result<&Arc<ParserFactory>> {
        self.factory
            .get_or_init(|| build_factory(tokenizer, eos_ids))
            .as_ref()
            .map_err(|error| crate::Error::Tokenizer(error.clone()))
    }

    /// Compile `format` for one request. `reasoning_end` is the token after
    /// which the constraint starts, or `None` to constrain from the first token.
    pub fn compile(
        &self,
        tokenizer: &Arc<Tokenizer>,
        eos_ids: &[u32],
        format: &ResponseFormat,
        reasoning_end: Option<u32>,
    ) -> crate::Result<Option<Grammar>> {
        let Some(schema) = format.schema()? else {
            return Ok(None);
        };
        let factory = self.factory(tokenizer, eos_ids)?;
        let parser = factory
            .create_parser(TopLevelGrammar::from_json_schema(schema))
            .map_err(invalid)?;
        let mut matcher = Matcher::new(Ok(parser));
        if let Some(error) = matcher.get_error() {
            return Err(invalid(error));
        }
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() {
            return Err(invalid(warnings.join("; ")));
        }
        // The first mask proves the grammar can start and warms its caches.
        matcher.compute_mask().map_err(invalid)?;
        Ok(Some(Grammar {
            matcher,
            reasoning_end,
            failure: None,
        }))
    }
}

/// The checkpoint's exact token bytes, read by llguidance from the very
/// tokenizer generation decodes with, serialized: byte-level BPE symbols map
/// back to raw bytes, and special tokens get llguidance's `0xFF` marker so no
/// grammar can produce them as text.
fn build_factory(
    tokenizer: &Arc<Tokenizer>,
    eos_ids: &[u32],
) -> Result<Arc<ParserFactory>, String> {
    let Some(&eos) = eos_ids.first() else {
        return Err("a constrained tokenizer needs an end-of-sequence token".into());
    };
    let serialized = tokenizer
        .to_string(false)
        .map_err(|error| format!("serializing the tokenizer: {error}"))?;
    let definition: Value = serde_json::from_str(&serialized)
        .map_err(|error| format!("reading the serialized tokenizer: {error}"))?;
    let mut words = llguidance::token_bytes_from_tokenizer_json(&definition)
        .map_err(|error| format!("reading token bytes: {error}"))?;
    let vocab = tokenizer.get_vocab_size(true);
    if words.len() > vocab {
        return Err(format!(
            "token bytes cover {} ids but the vocabulary has {vocab}",
            words.len()
        ));
    }
    words.resize(vocab, Vec::new());
    if eos_ids.iter().any(|&id| id as usize >= vocab) {
        return Err("end-of-sequence token is outside the vocabulary".into());
    }
    let info = TokRxInfo::new(vocab as u32, eos);
    let trie = TokTrie::from(&info, &words).with_eos_tokens(eos_ids);
    let env: TokEnv = Arc::new(BonsaiTokEnv {
        tokenizer: Arc::clone(tokenizer),
        trie,
    });
    let mut factory = ParserFactory::new_simple(&env).map_err(|error| error.to_string())?;
    factory.quiet();
    factory.limits_mut().verbose_errors = false;
    Ok(Arc::new(factory))
}

/// llguidance's view of the checkpoint tokenizer: the trie of exact token
/// bytes, and canonical tokenization for the bytes a grammar forces.
struct BonsaiTokEnv {
    tokenizer: Arc<Tokenizer>,
    trie: TokTrie,
}

impl TokenizerEnv for BonsaiTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.trie.tokenize_with_greedy_fallback(bytes, |text| {
            // Forced text that spells a special token must stay ordinary bytes.
            match self.tokenizer.encode(text, false) {
                Ok(encoding)
                    if !encoding
                        .get_ids()
                        .iter()
                        .any(|&id| self.trie.is_special_token(id)) =>
                {
                    encoding.get_ids().to_vec()
                }
                _ => self.trie.greedy_tokenize(text.as_bytes()),
            }
        })
    }
}

/// One request's compiled response format and its progress through it.
#[derive(Clone)]
pub struct Grammar {
    matcher: Matcher,
    /// While reasoning: the token that ends it and starts the constraint.
    reasoning_end: Option<u32>,
    /// Why the grammar stopped accepting tokens, once it has.
    failure: Option<String>,
}

impl Grammar {
    /// Whether the next selection must be masked.
    pub const fn masking(&self) -> bool {
        self.reasoning_end.is_none() && self.failure.is_none()
    }

    /// Whether the constrained answer has begun.
    pub const fn answering(&self) -> bool {
        self.reasoning_end.is_none()
    }

    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    pub fn fail(&mut self, message: String) {
        self.failure.get_or_insert(message);
    }

    /// The tokens the grammar allows next; only end-of-sequence once the
    /// document is complete and cannot continue.
    pub fn mask(&mut self) -> Result<SimpleVob, String> {
        self.matcher
            .compute_mask_or_eos()
            .map_err(|error| error.to_string())
    }

    /// Record the one token a target selection chose under `mask` (the mask
    /// it was chosen from, `None` if unmasked). The reasoning delimiter
    /// starts the constraint; an answer token advances the matcher. Returns
    /// whether the grammar accepted it; otherwise the failure is recorded.
    pub fn select(&mut self, token: u32, is_eos: bool, mask: Option<&SimpleVob>) -> bool {
        if self.failure.is_some() {
            return false;
        }
        let outcome = match (self.reasoning_end, mask) {
            (Some(end), _) => {
                if token == end {
                    self.reasoning_end = None;
                }
                Ok(())
            }
            (None, None) => Err("an answer token was selected without the grammar mask".into()),
            (None, Some(mask)) if !mask.is_allowed(token) => Err(format!(
                "token {token} was selected outside the grammar mask"
            )),
            // End-of-sequence was only allowed because the grammar accepts.
            (None, Some(_)) if is_eos => Ok(()),
            (None, Some(_)) => self
                .matcher
                .consume_token(token)
                .map_err(|error| error.to_string()),
        };
        match outcome {
            Ok(()) => true,
            Err(error) => {
                self.fail(error);
                false
            }
        }
    }

    #[cfg(test)]
    pub fn trigger_error_for_tests(&mut self) {
        let _ = self.matcher.test_trigger_lexer_error();
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
