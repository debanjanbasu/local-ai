use std::collections::HashSet;
use std::fmt::Write as _;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::bonsai::BonsaiPackage;
use crate::structured::{Grammar, GrammarCompiler, ResponseFormat};

pub const DEFAULT_REASONING_INSTRUCTION: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

const CHAT_TEMPLATE_SHA256: &str =
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
const QWEN35_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const THINK_END: u32 = 248_069;

/// The checkpoint tokenizer.
///
/// Cloning is cheap and shares the one immutable vocabulary and merge table, so
/// a clone can count prompt tokens on another thread without rereading the GGUF.
#[derive(Clone)]
pub struct BonsaiTokenizer {
    inner: Arc<Tokenizer>,
    eos_ids: Vec<u32>,
    /// Response-format compiler over these token bytes, built on first use.
    grammar: Arc<GrammarCompiler>,
}

/// Everything the tokenizer needs from the checkpoint, in the shape
/// `--export index` emits under `tokenizer` (`schema_version` 1). Field names
/// mirror the GGUF `tokenizer.ggml.*` / `tokenizer.chat_template` keys so the
/// object is a plain copy of the checkpoint metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerDefinition {
    pub schema_version: u32,
    pub model: String,
    pub pre: String,
    pub add_bos_token: bool,
    pub tokens: Vec<String>,
    pub token_type: Vec<i32>,
    pub merges: Vec<String>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub padding_token_id: u32,
    pub chat_template: String,
}

impl TokenizerDefinition {
    pub const SCHEMA_VERSION: u32 = 1;

    #[doc(hidden)]
    pub fn from_package(package: &BonsaiPackage) -> crate::Result<Self> {
        Ok(Self {
            schema_version: Self::SCHEMA_VERSION,
            model: package.metadata_string("tokenizer.ggml.model")?.to_owned(),
            pre: package.metadata_string("tokenizer.ggml.pre")?.to_owned(),
            add_bos_token: package.metadata_bool("tokenizer.ggml.add_bos_token")?,
            tokens: package.metadata_strings("tokenizer.ggml.tokens")?,
            token_type: package.metadata_i32_array("tokenizer.ggml.token_type")?,
            merges: package.metadata_strings("tokenizer.ggml.merges")?,
            bos_token_id: package.metadata_u32("tokenizer.ggml.bos_token_id")?,
            eos_token_id: package.metadata_u32("tokenizer.ggml.eos_token_id")?,
            padding_token_id: package.metadata_u32("tokenizer.ggml.padding_token_id")?,
            chat_template: package
                .metadata_string("tokenizer.chat_template")?
                .to_owned(),
        })
    }
}

#[derive(Clone, Copy)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
    pub reasoning_content: Option<&'a str>,
}

impl BonsaiTokenizer {
    #[doc(hidden)]
    pub fn from_package(package: &BonsaiPackage) -> crate::Result<Self> {
        Self::from_definition(TokenizerDefinition::from_package(package)?)
    }

    /// Build from checkpoint metadata, whether read from the GGUF or from an
    /// exported index. Every Bonsai-specific invariant is checked here so
    /// both sources pass through identical validation.
    pub(crate) fn from_definition(definition: TokenizerDefinition) -> crate::Result<Self> {
        let TokenizerDefinition {
            schema_version,
            model,
            pre,
            add_bos_token,
            tokens,
            token_type: types,
            merges,
            bos_token_id: bos,
            eos_token_id: eos,
            padding_token_id: pad,
            chat_template: template,
        } = definition;
        if schema_version != TokenizerDefinition::SCHEMA_VERSION {
            return invalid(format!(
                "unsupported tokenizer schema_version {schema_version}"
            ));
        }
        require(&model, "gpt2", "model")?;
        require(&pre, "qwen35", "pre-tokenizer")?;
        if add_bos_token {
            return invalid("tokenizer.ggml.add_bos_token must be false");
        }
        if tokens.len() != 248_320 || types.len() != tokens.len() || merges.len() != 247_587 {
            return invalid(format!(
                "unexpected tokenizer lengths: tokens={}, types={}, merges={}",
                tokens.len(),
                types.len(),
                merges.len()
            ));
        }

        let mut vocab = serde_json::Map::with_capacity(tokens.len());
        let mut names = HashSet::with_capacity(tokens.len());
        let mut added = Vec::new();
        for (id, (token, token_type)) in tokens.iter().zip(&types).enumerate() {
            if !(1..=6).contains(token_type) {
                return invalid(format!("invalid token type {token_type} at id {id}"));
            }
            if !names.insert(token.as_str()) {
                return invalid(format!("duplicate vocabulary token {token:?}"));
            }
            vocab.insert(token.clone(), Value::from(id as u64));
            if matches!(token_type, 3 | 4) {
                added.push(json!({
                    "id": id, "content": token, "single_word": false,
                    "lstrip": false, "rstrip": false, "normalized": false, "special": true
                }));
            }
        }
        validate_merges(&merges, &names)?;

        for (name, id) in [("bos", bos), ("eos", eos), ("padding", pad)] {
            if id as usize >= tokens.len() {
                return invalid(format!("{name} token id {id} is outside vocabulary"));
            }
        }
        if bos != 248_044
            || pad != 248_044
            || eos != 248_046
            || tokens[bos as usize] != "<|endoftext|>"
            || tokens[eos as usize] != "<|im_end|>"
            || tokens[THINK_END as usize] != "</think>"
        {
            return invalid("unexpected Bonsai special token ids or spellings");
        }

        let hash = Sha256::digest(template.as_bytes()).iter().fold(
            String::with_capacity(64),
            |mut hex, byte| {
                let _ = write!(hex, "{byte:02x}");
                hex
            },
        );
        if hash != CHAT_TEMPLATE_SHA256 {
            return invalid(format!("unsupported tokenizer chat template sha256 {hash}"));
        }

        let definition = tokenizer_json(vocab, added, merges);
        let bytes = serde_json::to_vec(&definition)?;
        let inner = Tokenizer::from_bytes(&bytes)
            .map_err(|error| crate::Error::Tokenizer(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
            eos_ids: vec![eos],
            grammar: Arc::default(),
        })
    }

    /// Compile `format` for one request before any of its prompt work.
    /// With `reasoning`, the constraint starts after the closing `</think>`.
    /// `None` for [`ResponseFormat::Text`].
    pub(crate) fn compile_format(
        &self,
        format: &ResponseFormat,
        reasoning: bool,
    ) -> crate::Result<Option<Grammar>> {
        self.compile_format_ending(format, reasoning.then_some(THINK_END))
    }

    /// [`Self::compile_format`] with an explicit reasoning delimiter.
    pub(crate) fn compile_format_ending(
        &self,
        format: &ResponseFormat,
        reasoning_end: Option<u32>,
    ) -> crate::Result<Option<Grammar>> {
        self.grammar
            .compile(&self.inner, &self.eos_ids, format, reasoning_end)
    }

    #[doc(hidden)]
    pub fn encode(&self, text: &str) -> crate::Result<Vec<u32>> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| crate::Error::Tokenizer(error.to_string()))
    }

    /// Encode `text` once (exactly as [`Self::encode`]) and resolve each UTF-8
    /// byte endpoint in `byte_ends` (exclusive, strictly increasing) to the
    /// index of the token that ends exactly there.
    ///
    /// An endpoint is rejected, never rounded, when it is zero, past the text,
    /// inside a UTF-8 character, or inside a token (a merge spans it). When
    /// byte-level BPE splits one character into several tokens, they share the
    /// character's span, and the endpoint after the character resolves to the
    /// last of them.
    #[doc(hidden)]
    pub fn encode_with_token_ends(
        &self,
        text: &str,
        byte_ends: &[usize],
    ) -> crate::Result<(Vec<u32>, Vec<usize>)> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| crate::Error::Tokenizer(error.to_string()))?;
        let ends = token_end_indices(text, encoding.get_offsets(), byte_ends)
            .map_err(crate::Error::InvalidArgument)?;
        Ok((encoding.get_ids().to_vec(), ends))
    }

    #[doc(hidden)]
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> crate::Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|error| crate::Error::Tokenizer(error.to_string()))
    }

    /// A capped thought with no closing marker is not a completed final answer.
    pub(crate) fn final_answer(
        &self,
        ids: &[u32],
        thinking: bool,
    ) -> crate::Result<Option<String>> {
        let Some(start) = answer_start(ids, thinking) else {
            return Ok(None);
        };
        self.decode(&ids[start..], true).map(Some)
    }

    /// Decode one more token incrementally over state the caller owns, so a
    /// generation can outlive a borrow of the tokenizer. The closing thought
    /// marker stays visible instead of joining reasoning and the final answer
    /// into an indistinguishable string.
    pub(crate) fn stream_step(
        &self,
        state: &mut StreamDecodeState,
        id: u32,
    ) -> crate::Result<Option<String>> {
        tokenizers::step_decode_stream(
            &**self.inner,
            vec![id],
            false,
            &mut state.ids,
            &mut state.prefix,
            &mut state.prefix_index,
        )
        .map_err(|error| crate::Error::Tokenizer(error.to_string()))
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub(crate) fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    #[doc(hidden)]
    pub fn chat_prompt(user: &str, thinking: bool) -> crate::Result<String> {
        nonempty(user, "message content")?;
        Self::chat_messages(
            &[ChatMessage {
                role: "user",
                content: user,
                reasoning_content: None,
            }],
            thinking,
        )
    }

    #[doc(hidden)]
    pub fn chat_messages(messages: &[ChatMessage<'_>], thinking: bool) -> crate::Result<String> {
        let turns = messages
            .iter()
            .map(|message| crate::tools::Turn {
                role: message.role,
                content: message.content,
                reasoning_content: message.reasoning_content,
                tool_calls: &[],
                tool_call_id: None,
            })
            .collect::<Vec<_>>();
        crate::tools::render(&turns, thinking, &crate::tools::ToolSet::default())
    }
}

fn nonempty<'a>(text: &'a str, field: &str) -> crate::Result<&'a str> {
    let text = text.trim();
    if text.is_empty() {
        Err(crate::Error::InvalidArgument(format!("{field} is empty")))
    } else {
        Ok(text)
    }
}

/// Map exclusive UTF-8 byte endpoints to ending token indices, given each
/// token's `[start, end)` byte span in `text` from a single encode.
fn token_end_indices(
    text: &str,
    offsets: &[(usize, usize)],
    byte_ends: &[usize],
) -> Result<Vec<usize>, String> {
    if let Some(&[before, after]) = byte_ends
        .array_windows::<2>()
        .find(|[before, after]| before >= after)
    {
        return Err(format!(
            "token_end_offsets must be strictly increasing: {before} then {after}"
        ));
    }
    // Spans of a split character coincide, so neighbours may overlap, but
    // starts and ends never move backwards.
    let ordered = offsets.iter().all(|&(start, end)| start <= end)
        && offsets
            .array_windows::<2>()
            .all(|[a, b]| a.0 <= b.0 && a.1 <= b.1);
    if !ordered || offsets.last().is_some_and(|&(_, end)| end > text.len()) {
        return Err("tokenizer returned non-monotonic byte offsets".into());
    }
    byte_ends
        .iter()
        .map(|&byte| {
            if byte == 0 || byte > text.len() {
                return Err(format!(
                    "byte offset {byte} is outside 1..={} (text bytes)",
                    text.len()
                ));
            }
            if !text.is_char_boundary(byte) {
                return Err(format!("byte offset {byte} is inside a UTF-8 character"));
            }
            // Tokens wholly before the endpoint; the last of them must end on
            // it and the next must not start before it.
            let count = offsets.partition_point(|&(_, end)| end <= byte);
            let next = offsets.get(count);
            match (
                count.checked_sub(1).map(|index| (index, offsets[index])),
                next,
            ) {
                (_, Some(&(start, end))) if start < byte => Err(format!(
                    "byte offset {byte} is inside token {count} (bytes {start}..{end}); \
                     it is not a token boundary"
                )),
                (Some((index, (_, end))), _) if end == byte => Ok(index),
                _ => Err(format!("byte offset {byte} does not end a token")),
            }
        })
        .collect()
}

pub(crate) fn answer_start(ids: &[u32], thinking: bool) -> Option<usize> {
    if thinking {
        ids.iter()
            .position(|&id| id == THINK_END)
            .map(|index| index + 1)
    } else {
        Some(0)
    }
}

fn tokenizer_json(
    vocab: serde_json::Map<String, Value>,
    added: Vec<Value>,
    merges: Vec<String>,
) -> Value {
    let mut definition = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": null,
        // GGUF's qwen35 BPE preserves input bytes; NFC changes token IDs for
        // decomposed accents and was disproved against the pinned fork.
        "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN35_PATTERN}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "model": {"type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": false,
            "byte_fallback": false, "ignore_merges": false, "vocab": null, "merges": null}
    });
    definition["added_tokens"] = Value::Array(added);
    definition["model"]["vocab"] = Value::Object(vocab);
    definition["model"]["merges"] = Value::Array(merges.into_iter().map(Value::String).collect());
    definition
}

fn validate_merges(merges: &[String], vocab: &HashSet<&str>) -> crate::Result<()> {
    let mut seen = HashSet::with_capacity(merges.len());
    for (rank, merge) in merges.iter().enumerate() {
        let Some((left, right)) = merge.split_once(' ') else {
            return invalid(format!("malformed BPE merge at rank {rank}: {merge:?}"));
        };
        if left.is_empty() || right.is_empty() || right.contains(' ') {
            return invalid(format!("malformed BPE merge at rank {rank}: {merge:?}"));
        }
        if !vocab.contains(left)
            || !vocab.contains(right)
            || !vocab.contains(format!("{left}{right}").as_str())
        {
            return invalid(format!(
                "BPE merge at rank {rank} has an unknown endpoint or result: {merge:?}"
            ));
        }
        if !seen.insert(merge) {
            return invalid(format!("duplicate BPE merge {merge:?}"));
        }
    }
    Ok(())
}

fn require(actual: &str, expected: &str, field: &str) -> crate::Result<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid(format!(
            "unsupported {field} {actual:?}; expected {expected:?}"
        ))
    }
}

fn invalid<T>(message: impl Into<String>) -> crate::Result<T> {
    Err(crate::Error::InvalidFormat(message.into()))
}

/// Incremental detokenizer state for [`BonsaiTokenizer::stream_step`]: the
/// same three fields `tokenizers`' own `DecodeStream` keeps.
#[derive(Default)]
pub(crate) struct StreamDecodeState {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
impl BonsaiTokenizer {
    /// The real pre-tokenizer/byte-level pipeline over a tiny vocabulary: the
    /// 256 byte symbols, merges `a b` and `ab c`, and one special token `<|x|>`.
    pub(crate) fn tiny_for_tests() -> Self {
        Self::tiny_with_specials_for_tests(&["<|x|>"], &[])
    }

    /// [`Self::tiny_for_tests`] with `specials` (ids 258 on, in order) in
    /// place of `<|x|>`, and the specials named in `eos` as end-of-sequence.
    pub(crate) fn tiny_with_specials_for_tests(specials: &[&str], eos: &[&str]) -> Self {
        let mut symbols: Vec<char> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .collect();
        symbols.sort_unstable();
        let mut vocab: Vec<String> = symbols.iter().map(char::to_string).collect();
        vocab.extend(["ab".into(), "abc".into()]);
        let first = vocab.len();
        vocab.extend(specials.iter().map(|&special| special.to_owned()));
        let map = vocab
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), serde_json::Value::from(id)))
            .collect();
        let added = specials
            .iter()
            .enumerate()
            .map(|(index, special)| {
                serde_json::json!({
                    "id": first + index, "content": special, "single_word": false,
                    "lstrip": false, "rstrip": false, "normalized": false, "special": true
                })
            })
            .collect();
        let definition = tokenizer_json(map, added, vec!["a b".into(), "ab c".into()]);
        let bytes = serde_json::to_vec(&definition).expect("json");
        let eos_ids = eos
            .iter()
            .map(|name| {
                let index = specials.iter().position(|special| special == name);
                (first + index.expect("eos names a special")) as u32
            })
            .collect();
        Self {
            inner: Arc::new(Tokenizer::from_bytes(&bytes).expect("tiny tokenizer")),
            eos_ids,
            grammar: Arc::default(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{
        BonsaiTokenizer, ChatMessage, DEFAULT_REASONING_INSTRUCTION, THINK_END,
        TokenizerDefinition, answer_start, require, validate_merges,
    };
    use crate::bonsai::BonsaiPackage;
    use std::collections::HashSet;

    fn message<'a>(
        role: &'a str,
        content: &'a str,
        reasoning_content: Option<&'a str>,
    ) -> ChatMessage<'a> {
        ChatMessage {
            role,
            content,
            reasoning_content,
        }
    }

    #[test]
    fn final_answer_requires_the_thought_boundary_only_in_thinking_mode() {
        assert_eq!(answer_start(&[7, 11, 13], false), Some(0));
        assert_eq!(answer_start(&[7, 11, 13], true), None);
        assert_eq!(answer_start(&[7, 11, THINK_END, 13], true), Some(3));
        assert_eq!(answer_start(&[THINK_END], true), Some(1));
        assert_eq!(answer_start(&[], true), None);
    }

    fn tiny_tokenizer() -> BonsaiTokenizer {
        BonsaiTokenizer::tiny_for_tests()
    }

    #[test]
    fn clones_share_one_tokenizer() {
        let tokenizer = tiny_tokenizer();
        let clone = tokenizer.clone();
        assert!(std::sync::Arc::ptr_eq(&tokenizer.inner, &clone.inner));
        assert_eq!(clone.encode("abc<|x|>").expect("encode"), vec![257, 258]);
    }

    #[test]
    fn byte_endpoints_resolve_to_exact_ending_tokens_in_one_encode() {
        let tokenizer = tiny_tokenizer();
        // Bytes: "abc"=0..3, "\n"=3..4, "é"=4..6, "😀"=6..10, "<|x|>"=10..15, "c"=15..16.
        let text = "abc\né😀<|x|>c";
        let ids = tokenizer.encode(text).expect("encode");
        // abc, \n, é (2 byte tokens), 😀 (4 byte tokens), <|x|>, c.
        assert_eq!(ids.len(), 10);
        let (same, ends) = tokenizer
            .encode_with_token_ends(text, &[3, 4, 6, 10, 15, 16])
            .expect("boundaries");
        assert_eq!(same, ids);
        // A split character's endpoint is its last byte token.
        assert_eq!(ends, vec![0, 1, 3, 7, 8, 9]);
        assert_eq!(ids[8] as usize, 256 + 2);

        for (bad, why) in [
            (&[2][..], "inside token"),    // "ab|c" is merged into one token
            (&[5][..], "UTF-8 character"), // inside é
            (&[8][..], "UTF-8 character"), // inside 😀
            (&[12][..], "inside token"),   // inside the special token
            (&[0][..], "outside"),         // no token ends at 0
            (&[17][..], "outside"),        // past the text
            (&[4, 3][..], "strictly increasing"),
            (&[4, 4][..], "strictly increasing"),
        ] {
            let error = tokenizer
                .encode_with_token_ends(text, bad)
                .expect_err("rejected")
                .to_string();
            assert!(error.contains(why), "{bad:?}: {error}");
        }
        assert!(tokenizer.encode_with_token_ends("", &[1]).is_err());
        assert_eq!(
            tokenizer.encode_with_token_ends(text, &[]).expect("none").1,
            Vec::<usize>::new()
        );
    }

    #[test]
    fn rejects_bad_family_and_bad_merges() {
        assert!(require("llama", "gpt2", "model").is_err());
        let vocab = HashSet::from(["a", "b", "ab"]);
        assert!(validate_merges(&["a b".into()], &vocab).is_ok());
        assert!(validate_merges(&["a c".into()], &vocab).is_err());
        assert!(validate_merges(&["a b".into(), "a b".into()], &vocab).is_err());
    }

    #[test]
    fn renders_exact_multiturn_template_without_rewriting_content() {
        let prompt = BonsaiTokenizer::chat_messages(
            &[
                message("system", "  Be exact.  ", None),
                message("user", " First ", None),
                message(
                    "assistant",
                    "  code = \"</think>\"; Answer  ",
                    Some("  prior reasoning  "),
                ),
                message("user", " Second ", None),
            ],
            true,
        )
        .expect("prompt");
        assert_eq!(
            prompt,
            format!(
                "<|im_start|>system\n{DEFAULT_REASONING_INSTRUCTION}\n\nBe exact.<|im_end|>\n<|im_start|>user\nFirst<|im_end|>\n<|im_start|>assistant\n<think>\nprior reasoning\n</think>\n\ncode = \"</think>\"; Answer<|im_end|>\n<|im_start|>user\nSecond<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        );
    }

    #[test]
    fn renders_empty_system_history_and_generation_suffixes() {
        let history = [
            message("system", " \n ", None),
            message("user", "", None),
            message("assistant", "", Some("")),
            message("user", " final ", None),
        ];
        assert_eq!(
            BonsaiTokenizer::chat_messages(&history, false).expect("nonthinking prompt"),
            "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n<|im_start|>user\nfinal<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            BonsaiTokenizer::chat_messages(&history, true).expect("thinking prompt"),
            format!(
                "<|im_start|>system\n{DEFAULT_REASONING_INSTRUCTION}<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n<|im_start|>user\nfinal<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        );
    }

    #[test]
    fn validates_messages_and_single_user_empty_prompt() {
        assert!(BonsaiTokenizer::chat_messages(&[], true).is_err());
        assert!(BonsaiTokenizer::chat_messages(&[message("assistant", "x", None)], true).is_err());
        assert!(BonsaiTokenizer::chat_messages(&[message("user", " ", None)], true).is_err());
        assert!(BonsaiTokenizer::chat_prompt(" ", true).is_err());
        assert!(
            BonsaiTokenizer::chat_messages(
                &[message("tool", "x", None), message("user", "y", None),],
                true
            )
            .is_err()
        );
        assert!(
            BonsaiTokenizer::chat_messages(
                &[
                    message("user", "first", None),
                    message("system", "late", None),
                ],
                false
            )
            .is_err()
        );
    }

    /// Independent token IDs captured from Prism's /tokenize endpoint at
    /// 9a9394a895b96003ca842a6041cb28ac49a108f7 with the pinned GGUF.
    const PRISM_TOKENIZATIONS: &[(&str, &[u32])] = &[
        ("plain ASCII", &[20139, 38016]),
        ("中文 café e\u{301} 😀", &[99986, 50203, 378, 52033, 87209]),
        ("é e\u{301}", &[933, 378, 52033]),
        ("I'm we'd they'll", &[40, 2688, 567, 4035, 781, 3172]),
        (
            " \t\n\n  12345\r\n",
            &[6800, 271, 220, 220, 16, 17, 18, 19, 20, 317],
        ),
        (
            "বাংলা हिन्दी العربية",
            &[
                188_623, 150_521, 151_185, 190_488, 150_127, 177_453, 171_405,
            ],
        ),
        (
            "a\u{301}\u{327} Ⅳ²𝟜",
            &[64, 52033, 136, 100, 220, 68086, 96, 28495, 54362, 253, 250],
        ),
    ];

    fn assert_prism_token_ids(tokenizer: &BonsaiTokenizer) {
        assert_eq!(tokenizer.vocab_size(), 248_320);
        assert_eq!(tokenizer.eos_ids(), &[248_046]);
        for (text, expected) in PRISM_TOKENIZATIONS {
            let ids = tokenizer.encode(text).expect("encode");
            assert_eq!(&ids, expected, "{text:?}");
            assert_eq!(tokenizer.decode(&ids, false).expect("decode"), *text);
            let (same, ends) = tokenizer
                .encode_with_token_ends(text, &[text.len()])
                .expect("final boundary");
            assert_eq!((same, ends), (ids.clone(), vec![ids.len() - 1]), "{text:?}");
        }
        // "plain" + " ASCII" are two tokens; the space starts the second.
        let (_, ends) = tokenizer
            .encode_with_token_ends("plain ASCII", &[5, 11])
            .expect("word boundary");
        assert_eq!(ends, vec![0, 1]);
        assert!(
            tokenizer
                .encode_with_token_ends("plain ASCII", &[6])
                .is_err()
        );
    }

    /// The `tokenizer` object from `--export index` must build a tokenizer
    /// identical to the GGUF-embedded one. The index is exported in-process from
    /// the pinned model, exactly as `bonsai --export index` does.
    #[test]
    #[ignore = "requires the pinned model"]
    fn exported_tokenizer_object_reproduces_prism_token_ids() {
        let path = std::path::Path::new(crate::bonsai::DEFAULT_BONSAI_GGUF);
        let index = BonsaiPackage::open(path)
            .expect("valid package")
            .export_index(path)
            .expect("index");
        let definition: TokenizerDefinition =
            serde_json::from_value(index["tokenizer"].clone()).expect("tokenizer object");
        let tokenizer = BonsaiTokenizer::from_definition(definition).expect("sidecar tokenizer");
        assert_prism_token_ids(&tokenizer);
    }

    #[test]
    #[ignore = "requires the pinned model"]
    fn real_gguf_roundtrips_and_renders_prompts() {
        let path = crate::bonsai::DEFAULT_BONSAI_GGUF;
        let package = BonsaiPackage::open(path).expect("valid package");
        let tokenizer = BonsaiTokenizer::from_package(&package).expect("embedded tokenizer");
        assert_prism_token_ids(&tokenizer);
        assert_eq!(
            tokenizer
                .final_answer(&[40, THINK_END, 271, 21], true)
                .expect("answer"),
            Some("\n\n6".into())
        );
        assert_eq!(
            tokenizer.final_answer(&[40, 21], true).expect("incomplete"),
            None
        );
        assert_eq!(
            BonsaiTokenizer::chat_prompt("  hello  ", false).expect("no thinking"),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            BonsaiTokenizer::chat_prompt(" hello ", true).expect("thinking"),
            format!(
                "<|im_start|>system\n{DEFAULT_REASONING_INSTRUCTION}<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        );
        assert!(BonsaiTokenizer::chat_prompt("  ", true).is_err());
    }
}
