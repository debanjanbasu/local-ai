pub mod gemma4_qat;

pub use gemma4_qat::{
    Gemma4MultimodalConfig, Gemma4QATConfig, LayerAttentionType, LayerQuantGroup, QuantType,
    RopeConfig, RopeParams, RopeType,
};

#[derive(Debug)]
pub enum ModelConfig {
    Gemma4QAT(Gemma4QATConfig),
}

impl ModelConfig {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let value: serde_json::Value = serde_json::from_str(json)?;
        let mt = value.get("model_type").and_then(serde_json::Value::as_str);
        // A `gemma4` multimodal config nests the text fields under `text_config`;
        // a bare `gemma4_text` / QAT config carries them at the top level.
        let inner = if mt == Some("gemma4") {
            value
                .get("text_config")
                .cloned()
                .unwrap_or_else(|| value.clone())
        } else {
            value.clone()
        };
        let mut cfg: Gemma4QATConfig = serde_json::from_value(inner)?;
        if mt == Some("gemma4") {
            cfg.multimodal = serde_json::from_value(value)?;
        }
        Ok(Self::Gemma4QAT(cfg))
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::panic,
        clippy::float_cmp,
        clippy::unreadable_literal,
        clippy::bool_assert_comparison
    )]
    use super::*;

    #[test]
    fn detect_gemma4_qat_model_type() {
        let json = r#"{"model_type": "gemma4_text", "vocab_size": 262144, "hidden_size": 1536, "intermediate_size": 6144, "num_hidden_layers": 35, "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 256, "max_position_embeddings": 131072, "rms_norm_eps": 1e-6, "sliding_window": 512}"#;
        let mc = ModelConfig::from_json(json).expect("parse");
        match mc {
            ModelConfig::Gemma4QAT(_) => {}
        }
    }

    #[test]
    fn detect_gemma4_with_text_config() {
        let json = r#"{
  "model_type": "gemma4",
  "text_config": {
    "model_type": "gemma4_text",
    "vocab_size": 262144,
    "hidden_size": 1536,
    "intermediate_size": 6144,
    "num_hidden_layers": 35,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-6,
    "sliding_window": 512,
    "layer_types": ["sliding_attention","sliding_attention","full_attention"],
    "num_kv_shared_layers": 20,
    "hidden_size_per_layer_input": 256,
    "rope_parameters": {
      "sliding_attention": {"rope_theta": 10000.0},
      "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}
    }
  },
  "vision_config": {},
  "audio_config": {}
}"#;
        let mc = ModelConfig::from_json(json).expect("parse");
        match &mc {
            ModelConfig::Gemma4QAT(cfg) => {
                assert_eq!(cfg.hidden_size, 1536);
                assert_eq!(cfg.global_head_dim, 512);
                assert_eq!(cfg.is_full_attention(2), true);
                assert!(cfg.multimodal.has_vision_config());
                assert!(cfg.multimodal.has_audio_config());
            }
        }
    }

    #[test]
    fn preserves_gemma4_multimodal_token_ids() {
        let json = r#"{
  "model_type": "gemma4",
  "boi_token_id": 255999,
  "boa_token_id": 256000,
  "image_token_id": 258880,
  "audio_token_id": 258881,
  "eoi_token_id": 258882,
  "eoa_token_id": 258883,
  "vision_soft_tokens_per_image": 280,
  "text_config": {
    "model_type": "gemma4_text",
    "vocab_size": 262144,
    "hidden_size": 1536,
    "intermediate_size": 6144,
    "num_hidden_layers": 35,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-6,
    "sliding_window": 512
  },
  "vision_config": {"model_type": "gemma4_vision"},
  "audio_config": {"model_type": "gemma4_audio"}
}"#;
        let mc = ModelConfig::from_json(json).expect("parse");
        let ModelConfig::Gemma4QAT(cfg) = mc;
        assert_eq!(cfg.multimodal.image_token_id, Some(258_880));
        assert_eq!(cfg.multimodal.audio_token_id, Some(258_881));
        assert_eq!(cfg.multimodal.vision_soft_tokens_per_image, Some(280));
        assert!(cfg.multimodal.declares_multimodal());
    }
}
