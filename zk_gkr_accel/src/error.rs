use thiserror::Error;

#[derive(Error, Debug)]
pub enum AccelError {
    #[error("Backend not available: {0}")]
    BackendNotAvailable(&'static str),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Memory allocation failed")]
    MemoryAllocationFailed,

    #[error("Invalid matrix dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Computation failed: {0}")]
    ComputationFailed(String),

    #[error("Device not found: {0}")]
    DeviceNotFound(u32),

    #[error("Feature not supported: {0}")]
    FeatureNotSupported(String),

    #[error("GKR protocol error: {0}")]
    GkrError(#[from] zk_gkr::GkrError),

    #[error("Anyhow error: {0}")]
    AnyhowError(#[from] anyhow::Error),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}