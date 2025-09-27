//! High-performance matrix multiplication verifier using Expander GKR

use crate::config::{MatrixMultConfig, MPIConfig, expander_utils};
use crate::circuit::MatrixMultCircuit;
use crate::types::{Vector, MatrixProof};
use anyhow::{Result, Context};
use std::time::Instant;

/// Expander-based matrix multiplication verifier
///
/// This wraps Expander's high-performance GKR verifier to verify proofs
/// for matrix multiplication y = W·x without knowing the private matrix W.
#[derive(Debug)]
pub struct ExpanderMatrixVerifier {
    circuit: MatrixMultCircuit,
    config: MatrixMultConfig,
    mpi_config: MPIConfig,
}

impl ExpanderMatrixVerifier {
    /// Create new verifier for given matrix dimensions
    pub fn new(m: usize, k: usize) -> Result<Self> {
        let circuit = MatrixMultCircuit::new(m, k)?;
        let config = MatrixMultConfig; // TODO: Use actual config
        let mpi_config = expander_utils::get_mpi_config();

        Ok(Self {
            circuit,
            config,
            mpi_config,
        })
    }

    /// Verify proof of y = W·x (without knowing W)
    pub fn verify(
        &self,
        proof: &MatrixProof,
        input: &Vector,
        output: &Vector,
    ) -> Result<bool> {
        let start_time = Instant::now();

        // Validate inputs
        self.validate_inputs(proof, input, output)?;

        // Verify using Expander
        let verification_result = self.verify_with_expander(proof, input, output)?;

        let verification_time = start_time.elapsed();
        println!("Verification completed in {:.2}ms", verification_time.as_secs_f64() * 1000.0);

        Ok(verification_result)
    }

    /// Validate input dimensions and consistency
    fn validate_inputs(&self, proof: &MatrixProof, input: &Vector, output: &Vector) -> Result<()> {
        if proof.m != self.circuit.m {
            return Err(anyhow::anyhow!(
                "Proof matrix row count mismatch: expected {}, got {}",
                self.circuit.m, proof.m
            ));
        }

        if proof.k != self.circuit.k {
            return Err(anyhow::anyhow!(
                "Proof matrix column count mismatch: expected {}, got {}",
                self.circuit.k, proof.k
            ));
        }

        if input.len() != self.circuit.k {
            return Err(anyhow::anyhow!(
                "Input vector length mismatch: expected {}, got {}",
                self.circuit.k, input.len()
            ));
        }

        if output.len() != self.circuit.m {
            return Err(anyhow::anyhow!(
                "Output vector length mismatch: expected {}, got {}",
                self.circuit.m, output.len()
            ));
        }

        // Check that claimed output matches provided output
        for i in 0..output.len() {
            if output.get(i) != proof.claimed_output.get(i) {
                return Err(anyhow::anyhow!(
                    "Output mismatch at index {}: proof claims {}, provided {}",
                    i, proof.claimed_output.get(i), output.get(i)
                ));
            }
        }

        Ok(())
    }

    /// Verify proof using Expander's high-performance verifier
    fn verify_with_expander(&self, proof: &MatrixProof, input: &Vector, output: &Vector) -> Result<bool> {
        // Option 1: Use Expander's Rust API (preferred)
        if let Ok(result) = self.verify_via_api(proof, input, output) {
            return Ok(result);
        }

        // Option 2: Fallback to CLI interface
        self.verify_via_cli(proof, input, output)
    }

    /// Verify proof using Expander's native Rust API
    fn verify_via_api(&self, _proof: &MatrixProof, _input: &Vector, _output: &Vector) -> Result<bool> {
        // TODO: Implement using actual Expander API when available
        //
        // The code would look something like:
        //
        // use gkr::{Circuit, Verifier};
        //
        // // Load circuit for verification
        // let (mut circuit, window) = Circuit::<MatrixMultConfig>::verifier_load_circuit(
        //     &circuit_path,
        //     &self.mpi_config
        // )?;
        //
        // // Prepare public inputs
        // let mut public_input = Vec::new();
        // public_input.extend(input.data.iter());
        // public_input.extend(output.data.iter());
        //
        // // Deserialize proof
        // let proof_obj = bincode::deserialize(&proof.proof_data)?;
        //
        // // Create verifier
        // let verifier = Verifier::<MatrixMultConfig>::new(self.mpi_config.clone());
        //
        // // Verify proof
        // let verification_result = verifier.verify(
        //     &mut circuit,
        //     &public_input,
        //     &proof.claimed_output,
        //     &pcs_params,
        //     &pcs_verification_key,
        //     &proof_obj
        // )?;
        //
        // Ok(verification_result)

        Err(anyhow::anyhow!("Expander API not yet integrated"))
    }

    /// Verify proof using Expander CLI (fallback)
    fn verify_via_cli(&self, proof: &MatrixProof, input: &Vector, output: &Vector) -> Result<bool> {
        use std::process::Command;

        // Generate temporary files for verification
        let temp_dir = tempfile::tempdir()?;
        let circuit_path = temp_dir.path().join("circuit.txt");
        let public_path = temp_dir.path().join("public.txt");
        let proof_path = temp_dir.path().join("proof.bin");

        // Create circuit and public input files
        self.circuit.generate_circuit_file(circuit_path.to_str().unwrap())?;
        self.circuit.generate_public_input_file(input, output, public_path.to_str().unwrap())?;

        // Write proof file
        std::fs::write(&proof_path, &proof.proof_data)
            .context("Failed to write proof file")?;

        // Build expander-exec verify command
        let mut cmd = Command::new("expander-exec");
        cmd.arg("verify")
            .arg("-c").arg(&circuit_path)
            .arg("-w").arg(&public_path)  // Public inputs act as "witness" for verification
            .arg("-i").arg(&proof_path)
            .arg("-f").arg("fr")  // BN254 field
            .arg("-p").arg("Raw") // Raw polynomial commitment
            .env("RUSTFLAGS", "-C target-cpu=native");

        // Execute command
        let output = cmd.output()
            .context("Failed to execute expander-exec verify")?;

        // Check verification result
        let success = output.status.success();

        if !success {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("Verification failed: {}", stderr);
        }

        Ok(success)
    }
}

/// Mock verifier for development without Expander dependency
pub struct MockExpanderVerifier {
    circuit: MatrixMultCircuit,
}

impl MockExpanderVerifier {
    pub fn new(m: usize, k: usize) -> Result<Self> {
        let circuit = MatrixMultCircuit::new(m, k)?;
        Ok(Self { circuit })
    }

    /// Verify mock proof (always succeeds for valid inputs)
    pub fn verify(
        &self,
        proof: &MatrixProof,
        input: &Vector,
        output: &Vector,
    ) -> Result<bool> {
        // Validate dimensions
        if proof.m != self.circuit.m || proof.k != self.circuit.k {
            return Ok(false);
        }

        if input.len() != self.circuit.k || output.len() != self.circuit.m {
            return Ok(false);
        }

        // Check that claimed output matches provided output
        for i in 0..output.len() {
            if output.get(i) != proof.claimed_output.get(i) {
                return Ok(false);
            }
        }

        // Mock verification: check that proof data looks reasonable
        let proof_str = String::from_utf8_lossy(&proof.proof_data);
        if proof_str.starts_with("MOCK_PROOF_") {
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::MockExpanderProver;
    use rand::thread_rng;

    #[test]
    fn test_mock_verifier_success() {
        let mut rng = thread_rng();

        let weights = crate::types::Matrix::random(4, 8, &mut rng);
        let input = Vector::random(8, &mut rng);
        let output = weights.multiply(&input);

        // Generate proof
        let mut prover = MockExpanderProver::new(4, 8).unwrap();
        let proof = prover.prove(&weights, &input, &output).unwrap();

        // Verify proof
        let verifier = MockExpanderVerifier::new(4, 8).unwrap();
        let verified = verifier.verify(&proof, &input, &output).unwrap();

        assert!(verified);
    }

    #[test]
    fn test_mock_verifier_failure() {
        let mut rng = thread_rng();

        let weights = crate::types::Matrix::random(4, 8, &mut rng);
        let input = Vector::random(8, &mut rng);
        let output = weights.multiply(&input);

        // Generate proof
        let mut prover = MockExpanderProver::new(4, 8).unwrap();
        let proof = prover.prove(&weights, &input, &output).unwrap();

        // Try to verify with wrong output
        let wrong_output = Vector::random(4, &mut rng);
        let verifier = MockExpanderVerifier::new(4, 8).unwrap();
        let verified = verifier.verify(&proof, &input, &wrong_output).unwrap();

        assert!(!verified);
    }

    #[test]
    fn test_dimension_validation() {
        let mut rng = thread_rng();

        let weights = crate::types::Matrix::random(4, 8, &mut rng);
        let input = Vector::random(8, &mut rng);
        let output = weights.multiply(&input);

        // Generate proof with 4×8 matrix
        let mut prover = MockExpanderProver::new(4, 8).unwrap();
        let proof = prover.prove(&weights, &input, &output).unwrap();

        // Try to verify with verifier configured for different dimensions
        let wrong_verifier = MockExpanderVerifier::new(6, 10).unwrap();
        let verified = wrong_verifier.verify(&proof, &input, &output).unwrap();

        assert!(!verified);
    }
}