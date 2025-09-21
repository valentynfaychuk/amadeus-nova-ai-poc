use ark_bn254::Fr;
// Removed unused imports Field and PrimeField (used in function bodies with full paths)
// use ark_ff::{Field, PrimeField};
// use std::io::Read;
use thiserror::Error;

pub mod commitment;
pub mod gemv;
pub mod freivalds;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Freivalds check failed at round {round}: lhs={lhs}, rhs={rhs}")]
    FreivaldsCheckFailed { round: usize, lhs: Fr, rhs: Fr },
    #[error("Invalid matrix dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: String, actual: String },
    #[error("Random number generation error")]
    RngError,
}

pub type EngineResult<T> = Result<T, EngineError>;

/// Configuration for the engine
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Total width K of the large layer W1 (16xK)
    pub k: usize,
    /// Tile size for streaming (e.g., 1024, 4096)
    pub tile_k: usize,
    /// Number of Freivalds rounds for probabilistic verification
    pub freivalds_rounds: usize,
    /// Scale numerator for quantization
    pub scale_num: u64,
    /// Random seed for deterministic execution
    pub seed: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            k: 4096,
            tile_k: 1024,
            freivalds_rounds: 32,
            scale_num: 3,
            seed: 42,
        }
    }
}

impl EngineConfig {
    pub fn demo() -> Self {
        Self::default()
    }
}

/// Convert field element to u64 for interfacing with integer math
pub fn field_to_u64(f: Fr) -> u64 {
    use ark_ff::PrimeField;
    let bigint = f.into_bigint();
    bigint.as_ref()[0]
}

/// Convert u64 to field element
pub fn u64_to_field(x: u64) -> Fr {
    Fr::from(x)
}

/// Convert i64 to field element (handles negative values)
pub fn i64_to_field(x: i64) -> Fr {
    if x >= 0 {
        Fr::from(x as u64)
    } else {
        -Fr::from((-x) as u64)
    }
}

/// Convert field element to i64 (handles negative values)
pub fn field_to_i64(f: Fr) -> i64 {
    use ark_ff::PrimeField;
    let bigint = f.into_bigint();
    let val = bigint.as_ref()[0];

    // Check if this is a negative value (greater than half the field)
    let half_field = Fr::MODULUS.as_ref()[0] / 2;
    if val > half_field {
        // This represents a negative number
        let neg_val = Fr::MODULUS.as_ref()[0] - val;
        -(neg_val as i64)
    } else {
        val as i64
    }
}