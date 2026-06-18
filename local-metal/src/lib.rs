pub mod batch;
pub mod buffer;
pub mod context;
pub mod kernels;
pub mod memory;
pub mod shaders;

mod error;
pub use error::Error;

pub type Result<T> = std::result::Result<T, Error>;
