//! High-performance matrix multiplication prover using Expander GKR

use crate::config::{MatrixMultConfig, MPIConfig, expander_utils};
use crate::circuit::MatrixMultCircuit;
use crate::types::{Matrix, Vector, MatrixProof};
use anyhow::{Result, Context};
use std::time::Instant;

/// Expander-based matrix multiplication prover
///
/// This wraps Expander's high-performance GKR prover to generate proofs
/// for matrix multiplication y = W·x where W is private.
#[derive(Debug)]
pub struct ExpanderMatrixProver {
    circuit: MatrixMultCircuit,
    config: MatrixMultConfig,
    mpi_config: MPIConfig,
}

impl ExpanderMatrixProver {
    /// Create new prover for given matrix dimensions
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

    /// Generate proof that y = W·x
    pub fn prove(
        &mut self,
        weights: &Matrix,
        input: &Vector,
        output: &Vector,
    ) -> Result<MatrixProof> {
        let start_time = Instant::now();

        // Validate inputs
        self.validate_inputs(weights, input, output)?;

        // Generate temporary files for Expander
        let temp_dir = tempfile::tempdir()?;
        let circuit_path = temp_dir.path().join("circuit.txt");
        let witness_path = temp_dir.path().join("witness.txt");
        let public_path = temp_dir.path().join("public.txt");
        let proof_path = temp_dir.path().join("proof.bin");

        // Create circuit and witness files
        self.circuit.generate_circuit_file(circuit_path.to_str().unwrap())?;
        self.circuit.generate_witness_file(weights, witness_path.to_str().unwrap())?;
        self.circuit.generate_public_input_file(input, output, public_path.to_str().unwrap())?;

        // Generate proof using Expander
        let proof_data = self.generate_expander_proof(
            &circuit_path,
            &witness_path,
            &proof_path,
        )?;

        let generation_time = start_time.elapsed().as_millis();
        let proof_size = proof_data.len();

        Ok(MatrixProof {
            proof_data,
            m: weights.rows,
            k: weights.cols,
            claimed_output: output.clone(),
            proof_size_bytes: proof_size,
            generation_time_ms: generation_time,
        })
    }

    /// Validate input dimensions and consistency
    fn validate_inputs(&self, weights: &Matrix, input: &Vector, output: &Vector) -> Result<()> {
        if weights.rows != self.circuit.m {
            return Err(anyhow::anyhow!(
                "Matrix row count mismatch: expected {}, got {}",
                self.circuit.m, weights.rows
            ));
        }

        if weights.cols != self.circuit.k {
            return Err(anyhow::anyhow!(
                "Matrix column count mismatch: expected {}, got {}",
                self.circuit.k, weights.cols
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

        // Verify that output = weights * input
        let computed_output = weights.multiply(input);
        for i in 0..output.len() {
            if output.get(i) != computed_output.get(i) {
                return Err(anyhow::anyhow!(
                    "Output verification failed at index {}: expected {}, got {}",
                    i, computed_output.get(i), output.get(i)
                ));
            }
        }

        Ok(())
    }

    /// Generate proof using Expander CLI via expander-exec binary
    fn generate_expander_proof(
        &self,
        circuit_path: &std::path::Path,
        witness_path: &std::path::Path,
        proof_path: &std::path::Path,
    ) -> Result<Vec<u8>> {
        use std::process::Command;

        // Use expander-exec CLI (built from Expander SDK dependencies)
        let mut cmd = Command::new("expander-exec");
        cmd.arg("prove")
            .arg("--circuit-file").arg(circuit_path)
            .arg("--witness-file").arg(witness_path)
            .arg("--output-proof-file").arg(proof_path);

        eprintln!("🔧 Running expander-exec prove...");
        eprintln!("   Circuit: {:?}", circuit_path);
        eprintln!("   Witness: {:?}", witness_path);
        eprintln!("   Output:  {:?}", proof_path);

        // Execute command
        let output = cmd.output()
            .context("Failed to execute expander-exec. Make sure Expander dependencies are enabled in Cargo.toml")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            eprintln!("❌ expander-exec failed!");
            eprintln!("   stdout: {}", stdout);
            eprintln!("   stderr: {}", stderr);
            return Err(anyhow::anyhow!(
                "expander-exec prove failed: {}",
                stderr
            ));
        }

        eprintln!("✅ Proof generated successfully");

        // Read generated proof file
        let proof_data = std::fs::read(proof_path)
            .context("Failed to read generated proof file")?;

        eprintln!("📦 Proof size: {} bytes", proof_data.len());

        Ok(proof_data)
    }
}

/// Mock prover for development without Expander dependency
#[derive(Debug)]
pub struct MockExpanderProver {
    circuit: MatrixMultCircuit,
}

impl MockExpanderProver {
    pub fn new(m: usize, k: usize) -> Result<Self> {
        let circuit = MatrixMultCircuit::new(m, k)?;
        Ok(Self { circuit })
    }

    /// Generate mock proof for testing
    pub fn prove(
        &mut self,
        weights: &Matrix,
        input: &Vector,
        output: &Vector,
    ) -> Result<MatrixProof> {
        let start_time = Instant::now();

        // Validate inputs (same as real prover)
        if weights.rows != self.circuit.m || weights.cols != self.circuit.k {
            return Err(anyhow::anyhow!("Matrix dimension mismatch"));
        }

        if input.len() != self.circuit.k || output.len() != self.circuit.m {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }

        // Verify computation
        let computed_output = weights.multiply(input);
        for i in 0..output.len() {
            if output.get(i) != computed_output.get(i) {
                return Err(anyhow::anyhow!("Output verification failed"));
            }
        }

        // Generate mock proof data
        let mock_proof_data = format!(
            "MOCK_PROOF_{}x{}_{}",
            weights.rows,
            weights.cols,
            chrono::Utc::now().timestamp()
        ).into_bytes();

        let generation_time = start_time.elapsed().as_millis().max(1); // Ensure at least 1ms for mock
        let proof_size = mock_proof_data.len();

        Ok(MatrixProof {
            proof_data: mock_proof_data,
            m: weights.rows,
            k: weights.cols,
            claimed_output: output.clone(),
            proof_size_bytes: proof_size,
            generation_time_ms: generation_time,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_mock_prover() {
        let mut rng = thread_rng();

        let weights = Matrix::random(4, 8, &mut rng);
        let input = Vector::random(8, &mut rng);
        let output = weights.multiply(&input);

        let mut prover = MockExpanderProver::new(4, 8).unwrap();
        let proof = prover.prove(&weights, &input, &output).unwrap();

        assert_eq!(proof.m, 4);
        assert_eq!(proof.k, 8);
        assert!(proof.proof_size_bytes > 0);
        assert!(proof.generation_time_ms > 0);
    }

    #[test]
    fn test_validation_errors() {
        let mut rng = thread_rng();
        let mut prover = MockExpanderProver::new(4, 8).unwrap();

        // Wrong matrix dimensions
        let wrong_weights = Matrix::random(3, 8, &mut rng);
        let input = Vector::random(8, &mut rng);
        let output = Vector::random(4, &mut rng);

        assert!(prover.prove(&wrong_weights, &input, &output).is_err());

        // Wrong computation
        let weights = Matrix::random(4, 8, &mut rng);
        let wrong_output = Vector::random(4, &mut rng);

        assert!(prover.prove(&weights, &input, &wrong_output).is_err());
    }
}