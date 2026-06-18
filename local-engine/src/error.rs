#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Core error: {0}")]
    Core(#[from] local_core::Error),

    #[error("Metal error: {0}")]
    Metal(#[from] local_metal::Error),

    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Generation error: {0}")]
    Generation(String),

    #[error("Context too long: {0}")]
    ContextTooLong(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Sampling error: {0}")]
    Sampling(String),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("I/O error: {0}")]
    Io(String),

    #[error("Unsupported quant type discriminant: {0}")]
    UnsupportedQuantType(u8),

    #[error(
        "Block alignment error: tensor {tensor} has {weights} weights, not divisible by {block_size}"
    )]
    BlockAlignmentError {
        tensor: String,
        weights: usize,
        block_size: usize,
    },

    #[error("Missing imatrix for tensor: {0}")]
    MissingImatrix(String),

    #[error("Archive version mismatch: expected {expected}, found {found}")]
    ArchiveVersionMismatch { expected: u32, found: u32 },

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Context overflow: {0}")]
    ContextOverflow(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
}
