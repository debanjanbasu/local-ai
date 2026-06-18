#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Safetensors error: {0}")]
    Safetensors(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Model error: {0}")]
    Model(String),
}
