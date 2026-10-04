use std::collections::HashSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::bonsai::BonsaiPackage;

pub const DEFAULT_REASONING_INSTRUCTION: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

const CHAT_TEMPLATE_SHA256: &str =
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
const QWEN35_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const THINK_END: u32 = 248_069;

pub struct BonsaiTokenizer {
    inner: Tokenizer,
    eos_ids: Vec<u32>,
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
            inner,
            eos_ids: vec![eos],
        })
    }

    #[doc(hidden)]
    pub fn encode(&self, text: &str) -> crate::Result<Vec<u32>> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| crate::Error::Tokenizer(error.to_string()))
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

    pub(crate) fn stream_decoder(&self) -> impl FnMut(u32) -> crate::Result<Option<String>> + '_ {
        // Keep the closing thought marker visible instead of joining reasoning
        // and the final answer into an indistinguishable string.
        let mut decoder = self.inner.decode_stream(false);
        move |id| {
            decoder
                .step(id)
                .map_err(|error| crate::Error::Tokenizer(error.to_string()))
        }
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
        if messages.is_empty() {
            return Err(crate::Error::InvalidArgument("no messages provided".into()));
        }
        if messages.len() == 1
            && messages[0].role == "user"
            && messages[0].content.trim().is_empty()
        {
            return Err(crate::Error::InvalidArgument(
                "message content is empty".into(),
            ));
        }
        let mut rendered = String::new();
        let mut has_user = false;
        if thinking {
            rendered.push_str("<|im_start|>system\n");
            rendered.push_str(DEFAULT_REASONING_INSTRUCTION);
            if messages[0].role == "system" {
                let system = messages[0].content.trim();
                if !system.is_empty() {
                    rendered.push_str("\n\n");
                    rendered.push_str(system);
                }
            }
            rendered.push_str("<|im_end|>\n");
        } else if messages[0].role == "system" {
            let system = messages[0].content.trim();
            if !system.is_empty() {
                rendered.push_str("<|im_start|>system\n");
                rendered.push_str(system);
                rendered.push_str("<|im_end|>\n");
            }
        }
        for (index, message) in messages.iter().enumerate() {
            let content = message.content.trim();
            match message.role {
                "system" if index == 0 => {}
                "system" => {
                    return Err(crate::Error::InvalidArgument(
                        "system message must be first".into(),
                    ));
                }
                "user" => {
                    has_user = true;
                    rendered.push_str("<|im_start|>user\n");
                    rendered.push_str(content);
                    rendered.push_str("<|im_end|>\n");
                }
                "assistant" => {
                    rendered.push_str("<|im_start|>assistant\n<think>\n");
                    rendered.push_str(message.reasoning_content.unwrap_or_default().trim());
                    rendered.push_str("\n</think>\n\n");
                    rendered.push_str(content);
                    rendered.push_str("<|im_end|>\n");
                }
                _ => {
                    return Err(crate::Error::InvalidArgument(format!(
                        "unsupported message role {:?}",
                        message.role
                    )));
                }
            }
        }
        if !has_user {
            return Err(crate::Error::InvalidArgument(
                "no user query found in messages".into(),
            ));
        }
        rendered.push_str("<|im_start|>assistant\n");
        if thinking {
            rendered.push_str("<think>\n");
        } else {
            rendered.push_str("<think>\n\n</think>\n\n");
        }
        Ok(rendered)
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

fn answer_start(ids: &[u32], thinking: bool) -> Option<usize> {
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
        }
    }

    /// The `tokenizer` object from `--export index` must build a tokenizer
    /// identical to the GGUF-embedded one.
    #[test]
    #[ignore = "set BONSAI_INDEX to an absolute path to the JSON written by `bonsai --export index > index.json`; requires --test-threads=1 (parallel runs spuriously purge volatile prompt-cache state)"]
    fn exported_tokenizer_object_reproduces_prism_token_ids() {
        let path = std::env::var("BONSAI_INDEX").expect("BONSAI_INDEX is required");
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).expect("index bytes")).expect("index");
        let definition: TokenizerDefinition =
            serde_json::from_value(index["tokenizer"].clone()).expect("tokenizer object");
        let tokenizer = BonsaiTokenizer::from_definition(definition).expect("sidecar tokenizer");
        assert_prism_token_ids(&tokenizer);
    }

    #[test]
    #[ignore = "set BONSAI_GGUF to an absolute models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf; requires --test-threads=1 (parallel runs spuriously purge volatile prompt-cache state)"]
    fn real_gguf_roundtrips_and_renders_prompts() {
        let path = std::env::var("BONSAI_GGUF").expect("BONSAI_GGUF is required");
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
