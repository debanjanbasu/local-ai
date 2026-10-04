#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Metal(#[from] local_metal::Error),

    #[error("invalid model format: {0}")]
    InvalidFormat(String),

    #[error("missing tensor: {0}")]
    MissingTensor(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("context creation error: {0}")]
    Context(String),

    #[error("tokenization error: {0}")]
    Tokenizer(String),

    #[error("generation error: {0}")]
    Generation(String),

    #[error("generation queue is full")]
    QueueFull,

    #[error("sampling error: {0}")]
    Sampling(String),

    #[error("context overflow: {0}")]
    ContextOverflow(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}
