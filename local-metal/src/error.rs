#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("No Metal device found")]
    NoMetalDevice,

    #[error("No default Metal device")]
    NoDevice,

    #[error("Failed to create command queue")]
    NoCommandQueue,

    #[error("Buffer allocation failed (size={size}): {reason}")]
    BufferAllocation { size: usize, reason: String },

    #[error("Failed to create Metal library: {0}")]
    LibraryCreation(String),

    #[error("Failed to create pipeline state: {0}")]
    PipelineCreation(String),

    #[error("Shader function not found: {0}")]
    ShaderNotFound(String),

    #[error("Command buffer error: {0}")]
    CommandBuffer(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
