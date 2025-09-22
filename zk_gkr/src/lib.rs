pub mod merkle_poseidon;
pub mod mle;
pub mod proof;
pub mod sumcheck;
pub mod transcript;

pub use ark_bn254::Fr;
pub use ark_ff::Field;

/// Domain separator tags for Fiat-Shamir transcript
pub const DOMAIN_GKR_V1: &[u8] = b"GKRv1";
pub const DOMAIN_GKR_U: &[u8] = b"GKR.u";
pub const DOMAIN_GKR_ROUND: &[u8] = b"GKR.round";
pub const DOMAIN_GKR_FINAL: &[u8] = b"GKR.final";

/// Error type for GKR operations
#[derive(Debug, thiserror::Error)]
pub enum GkrError {
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
    #[error("Sum-check verification failed: {0}")]
    SumCheckFailed(String),
    #[error("Merkle verification failed: {0}")]
    MerkleVerificationFailed(String),
    #[error("Transcript error: {0}")]
    TranscriptError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

pub type Result<T> = std::result::Result<T, GkrError>;
