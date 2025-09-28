//! Expander SDK Matrix Multiplication POC
//!
//! High-performance matrix multiplication proofs using Polyhedra's Expander GKR system.
//! Proves y = W·x where W is private (witness) and x,y are public.

pub mod config;
pub mod circuit;
pub mod prover;
pub mod verifier;
pub mod types;
pub mod security_tests;

pub use types::*;
// Use real Expander implementations (falls back to CLI if API not available)
pub use prover::ExpanderMatrixProver;
pub use verifier::ExpanderMatrixVerifier;

/// High-level API for matrix multiplication proofs
#[derive(Debug)]
pub struct MatrixProofSystem {
    prover: ExpanderMatrixProver,
    verifier: ExpanderMatrixVerifier,
}

impl MatrixProofSystem {
    /// Create new proof system for given matrix dimensions
    pub fn new(m: usize, k: usize) -> anyhow::Result<Self> {
        let prover = ExpanderMatrixProver::new(m, k)?;
        let verifier = ExpanderMatrixVerifier::new(m, k)?;

        Ok(Self { prover, verifier })
    }

    /// Generate proof that y = W·x
    pub fn prove(
        &mut self,
        weights: &Matrix,    // m×k private matrix
        input: &Vector,      // k-dimensional public input
        output: &Vector,     // m-dimensional public output
    ) -> anyhow::Result<MatrixProof> {
        self.prover.prove(weights, input, output)
    }

    /// Verify proof of y = W·x (without knowing W)
    pub fn verify(
        &self,
        proof: &MatrixProof,
        input: &Vector,      // k-dimensional public input
        output: &Vector,     // m-dimensional public output
    ) -> anyhow::Result<bool> {
        self.verifier.verify(proof, input, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_small_matrix_proof() {
        let mut rng = rand::thread_rng();

        // Create 4×8 matrix and 8-dimensional input
        let m = 4;
        let k = 8;

        let weights = Matrix::random(m, k, &mut rng);
        let input = Vector::random(k, &mut rng);
        let output = weights.multiply(&input);

        let mut system = MatrixProofSystem::new(m, k).unwrap();

        // Generate proof
        let proof = system.prove(&weights, &input, &output).unwrap();

        // Verify proof
        let verified = system.verify(&proof, &input, &output).unwrap();
        assert!(verified);

        // Test with wrong output (should fail)
        let wrong_output = Vector::random(m, &mut rng);
        let wrong_verified = system.verify(&proof, &input, &wrong_output).unwrap();
        assert!(!wrong_verified);
    }
}