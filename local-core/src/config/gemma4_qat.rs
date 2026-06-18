use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Gemma4MultimodalConfig {
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,
    #[serde(default)]
    pub audio_config: Option<serde_json::Value>,
    #[serde(default)]
    pub boi_token_id: Option<u32>,
    #[serde(default)]
    pub boa_token_id: Option<u32>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub audio_token_id: Option<u32>,
    #[serde(default)]
    pub eoi_token_id: Option<u32>,
    #[serde(default)]
    pub eoa_token_id: Option<u32>,
    #[serde(default)]
    pub video_token_id: Option<u32>,
    #[serde(default)]
    pub vision_soft_tokens_per_image: Option<u32>,
    #[serde(default)]
    pub video_soft_tokens_per_frame: Option<u32>,
    #[serde(default)]
    pub video_frames_per_video: Option<u32>,
}

impl Gemma4MultimodalConfig {
    #[must_use]
    pub const fn has_vision_config(&self) -> bool {
        self.vision_config.is_some()
    }

    #[must_use]
    pub const fn has_audio_config(&self) -> bool {
        self.audio_config.is_some()
    }

    #[must_use]
    pub const fn has_video_config(&self) -> bool {
        self.video_token_id.is_some()
    }

    #[must_use]
    pub const fn declares_multimodal(&self) -> bool {
        self.has_vision_config()
            || self.has_audio_config()
            || self.image_token_id.is_some()
            || self.audio_token_id.is_some()
            || self.video_token_id.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum QuantType {
    #[default]
    Fp16,
    Tq2_0,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayerQuantGroup {
    pub weight_type: QuantType,
    #[serde(default)]
    pub layer_indices: Vec<usize>,
    pub block_size: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerAttentionType {
    SlidingAttention,
    FullAttention,
}

impl<'de> serde::Deserialize<'de> for LayerAttentionType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "sliding_attention" => Ok(Self::SlidingAttention),
            "full_attention" => Ok(Self::FullAttention),
            _ => Err(serde::de::Error::unknown_variant(
                &s,
                &["sliding_attention", "full_attention"],
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeType {
    Default,
    Proportional,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: Option<RopeType>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeConfig {
    #[serde(default)]
    pub sliding_attention: Option<RopeParams>,
    #[serde(default)]
    pub full_attention: Option<RopeParams>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4QATConfig {
    #[serde(skip)]
    pub multimodal: Gemma4MultimodalConfig,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default)]
    pub global_head_dim: usize,
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: String,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    #[serde(default)]
    pub layer_types: Vec<LayerAttentionType>,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default = "default_true")]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub hidden_size_per_layer_input: usize,
    #[serde(default)]
    pub quant_groups: Vec<LayerQuantGroup>,
    #[serde(default)]
    pub use_dynamic_recovery: bool,
    #[serde(default)]
    pub rope_parameters: Option<RopeConfig>,
}

const fn default_true() -> bool {
    true
}
fn default_hidden_activation() -> String {
    "gelu_pytorch_tanh".into()
}

impl Gemma4QATConfig {
    #[must_use]
    pub fn quant_type_for_layer(&self, layer_idx: usize) -> QuantType {
        self.quant_groups
            .iter()
            .find(|g| g.layer_indices.contains(&layer_idx))
            .map_or(QuantType::Fp16, |g| g.weight_type)
    }

    #[must_use]
    pub fn layer_attention_type(&self, layer_idx: usize) -> LayerAttentionType {
        self.layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerAttentionType::SlidingAttention)
    }

    #[must_use]
    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        matches!(
            self.layer_attention_type(layer_idx),
            LayerAttentionType::FullAttention
        )
    }

    #[must_use]
    pub fn attn_head_dim(&self, layer_idx: usize) -> usize {
        if self.is_full_attention(layer_idx) && self.global_head_dim > 0 {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }

    #[must_use]
    pub fn rope_theta(&self, layer_idx: usize) -> f64 {
        if let Some(cfg) = &self.rope_parameters {
            if self.is_full_attention(layer_idx) {
                if let Some(fp) = &cfg.full_attention {
                    return fp.rope_theta;
                }
            } else if let Some(sp) = &cfg.sliding_attention {
                return sp.rope_theta;
            }
        }
        10000.0
    }

    #[must_use]
    pub fn partial_rotary_factor(&self, layer_idx: usize) -> f64 {
        if self.is_full_attention(layer_idx)
            && let Some(fp) = self
                .rope_parameters
                .as_ref()
                .and_then(|c| c.full_attention.as_ref())
        {
            return fp.partial_rotary_factor.unwrap_or(1.0);
        }
        1.0
    }

    /// Index of the first KV-shared layer (layers at or after this reuse an
    /// earlier layer's KV cache instead of computing their own).
    #[must_use]
    pub const fn first_kv_shared_layer(&self) -> usize {
        self.num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers)
    }

    /// Whether layer `layer_idx` reuses another layer's KV cache.
    #[must_use]
    pub const fn is_kv_shared(&self, layer_idx: usize) -> bool {
        self.num_kv_shared_layers > 0 && layer_idx >= self.first_kv_shared_layer()
    }

    /// For a KV-shared layer, the source layer whose KV cache it reuses: the
    /// last non-shared layer with the same attention type. `None` for non-shared
    /// layers.
    #[must_use]
    pub fn kv_source_layer(&self, layer_idx: usize) -> Option<usize> {
        if !self.is_kv_shared(layer_idx) {
            return None;
        }
        let want = self.layer_attention_type(layer_idx);
        (0..self.first_kv_shared_layer())
            .rev()
            .find(|&j| self.layer_attention_type(j) == want)
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

    fn e2b_like_config() -> Gemma4QATConfig {
        // 35 layers, pattern [sliding×4, full] repeating → full at idx%5==4.
        let layer_types: Vec<&str> = (0..35)
            .map(|i| {
                if i % 5 == 4 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect();
        let json = serde_json::json!({
            "vocab_size": 262144, "hidden_size": 1536, "intermediate_size": 6144,
            "num_hidden_layers": 35, "num_attention_heads": 8, "num_key_value_heads": 1,
            "head_dim": 256, "global_head_dim": 512, "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6, "sliding_window": 512, "num_kv_shared_layers": 20,
            "layer_types": layer_types,
        });
        serde_json::from_value(json).expect("config")
    }

    #[test]
    fn kv_sharing_maps_shared_layers_to_same_type_source() {
        let cfg = e2b_like_config();
        assert_eq!(cfg.first_kv_shared_layer(), 15);
        // Layers 0-14 own their KV.
        for i in 0..15 {
            assert!(!cfg.is_kv_shared(i), "layer {i} should own KV");
            assert_eq!(cfg.kv_source_layer(i), None);
        }
        // Shared full layers (19,24,29,34) reuse layer 14 (last full < 15).
        for i in [19, 24, 29, 34] {
            assert!(cfg.is_kv_shared(i));
            assert_eq!(cfg.kv_source_layer(i), Some(14), "full layer {i}");
        }
        // Shared SWA layers reuse layer 13 (last sliding < 15).
        for i in [15, 16, 17, 18, 20, 23, 33] {
            assert!(cfg.is_kv_shared(i));
            assert_eq!(cfg.kv_source_layer(i), Some(13), "swa layer {i}");
        }
    }

    #[test]
    fn parse_e2b_qat_no_groups_defaults_to_fp16() {
        let json = r#"{
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
  "num_kv_shared_layers": 20,
  "hidden_size_per_layer_input": 256,
  "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
  "rope_parameters": {
    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}
  }
}"#;
        let cfg: Gemma4QATConfig = serde_json::from_str(json).expect("parse");
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 35);
        assert_eq!(cfg.global_head_dim, 512);
        assert_eq!(cfg.num_kv_shared_layers, 20);
        assert_eq!(cfg.hidden_size_per_layer_input, 256);
        assert!(cfg.quant_groups.is_empty());
        assert_eq!(cfg.is_full_attention(4), true);
        assert_eq!(cfg.is_full_attention(0), false);
        assert_eq!(cfg.attn_head_dim(4), 512);
        assert_eq!(cfg.attn_head_dim(0), 256);
        assert_eq!(cfg.rope_theta(4), 1000000.0);
        assert_eq!(cfg.rope_theta(0), 10000.0);
        assert!((cfg.partial_rotary_factor(4) - 0.25).abs() < 1e-6);
        assert!(!cfg.use_dynamic_recovery);
    }

    #[test]
    fn quant_type_defaults_fp16() {
        let cfg = Gemma4QATConfig {
            multimodal: Gemma4MultimodalConfig::default(),
            vocab_size: 100,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 16,
            global_head_dim: 0,
            hidden_activation: "gelu_pytorch_tanh".into(),
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-6,
            sliding_window: 128,
            layer_types: vec![],
            num_kv_shared_layers: 0,
            use_double_wide_mlp: false,
            hidden_size_per_layer_input: 0,
            quant_groups: vec![LayerQuantGroup {
                weight_type: QuantType::Tq2_0,
                layer_indices: vec![0, 1],
                block_size: Some(16),
            }],
            use_dynamic_recovery: true,
            rope_parameters: None,
        };
        assert_eq!(cfg.quant_type_for_layer(0), QuantType::Tq2_0);
        assert_eq!(cfg.quant_type_for_layer(1), QuantType::Tq2_0);
        assert_eq!(cfg.quant_type_for_layer(2), QuantType::Fp16);
    }
}
