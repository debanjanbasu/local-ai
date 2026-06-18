#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RecipeFamily {
    Gemma4,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BackendTarget {
    Metal,
}
#[derive(Debug, Clone, Copy)]
pub struct QuantRecipe {
    pub family: RecipeFamily,
    pub backend: BackendTarget,
}

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationProvenance {
    Imatrix { n_tokens: u64, corpus_hash: String },
    UniformFallback,
    UniformPlaceholder,
}

impl CalibrationProvenance {
    #[must_use]
    pub const fn from_imatrix(_imatrix: bool) -> Self {
        Self::UniformFallback
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantRecipeStamp {
    pub family: RecipeFamily,
    pub backend_target: BackendTarget,
}

impl From<QuantRecipe> for QuantRecipeStamp {
    fn from(value: QuantRecipe) -> Self {
        Self {
            family: value.family,
            backend_target: value.backend,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveManifest {
    pub config: serde_json::Value,
    pub version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    #[serde(default)]
    pub tensor_quant_types: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant_recipe: Option<QuantRecipeStamp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationProvenance>,
}

impl ArchiveManifest {
    #[must_use]
    pub fn new(config: serde_json::Value) -> Self {
        Self {
            config,
            version: super::VERSION,
            model_type: None,
            tensor_quant_types: HashMap::new(),
            quant_recipe: None,
            calibration: None,
        }
    }

    pub fn to_metadata_bytes(&self) -> crate::Result<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| crate::Error::InvalidFormat(format!("archive manifest encode: {e}")))
    }

    pub fn from_metadata_bytes(bytes: &[u8]) -> crate::Result<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| crate::Error::InvalidFormat(format!("archive manifest decode: {e}")))
    }
}
