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

    /// Generate proof using Expander's high-performance prover
    fn generate_expander_proof(
        &self,
        circuit_path: &std::path::Path,
        witness_path: &std::path::Path,
        proof_path: &std::path::Path,
    ) -> Result<Vec<u8>> {
        // Option 1: Use Expander's Rust API (preferred)
        if let Ok(proof_data) = self.generate_proof_via_api(circuit_path, witness_path) {
            return Ok(proof_data);
        }

        // Option 2: Fallback to CLI interface
        self.generate_proof_via_cli(circuit_path, witness_path, proof_path)
    }

    /// Generate proof using Expander's native Rust API
    fn generate_proof_via_api(
        &self,
        _circuit_path: &std::path::Path,
        _witness_path: &std::path::Path,
    ) -> Result<Vec<u8>> {
        // TODO: Implement using actual Expander API when available
        //
        // The code would look something like:
        //
        // use gkr::{Circuit, Prover};
        //
        // let (mut circuit, window) = Circuit::<MatrixMultConfig>::prover_load_circuit(
        //     circuit_path.to_str().unwrap(),
        //     &self.mpi_config
        // )?;
        //
        // circuit.load_witness_allow_padding_testing_only(
        //     witness_path.to_str().unwrap(),
        //     &self.mpi_config
        // )?;
        //
        // circuit.evaluate();
        //
        // let mut prover = Prover::<MatrixMultConfig>::new(self.mpi_config.clone());
        // prover.prepare_mem(&circuit);
        //
        // let (claimed_v, proof) = prover.prove(
        //     &mut circuit,
        //     &pcs_params,
        //     &pcs_proving_key,
        //     &mut pcs_scratch
        // )?;
        //
        // // Serialize proof to bytes
        // let proof_bytes = bincode::serialize(&proof)?;
        // Ok(proof_bytes)

        Err(anyhow::anyhow!("Expander API not yet integrated"))
    }

    /// Generate proof using Expander CLI (fallback)
    fn generate_proof_via_cli(
        &self,
        circuit_path: &std::path::Path,
        witness_path: &std::path::Path,
        proof_path: &std::path::Path,
    ) -> Result<Vec<u8>> {
        use std::process::Command;

        // Build expander-exec command
        let mut cmd = Command::new("expander-exec");
        cmd.arg("prove")
            .arg("-c").arg(circuit_path)
            .arg("-w").arg(witness_path)
            .arg("-o").arg(proof_path)
            .arg("-f").arg("fr")  // BN254 field
            .arg("-p").arg("Raw") // Raw polynomial commitment
            .env("RUSTFLAGS", "-C target-cpu=native");

        // Execute command
        let output = cmd.output()
            .context("Failed to execute expander-exec")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!(
                "expander-exec prove failed: {}",
                stderr
            ));
        }

        // Read generated proof file
        let proof_data = std::fs::read(proof_path)
            .context("Failed to read generated proof file")?;

        Ok(proof_data)
    }
}

/// Mock prover for development without Expander dependency
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

        let generation_time = start_time.elapsed().as_millis();
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