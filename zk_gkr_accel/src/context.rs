use crate::{backend::Backend, error::AccelError};
use ark_bn254::Fr;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct ProofResult {
    pub proof: Vec<u8>, // Serialized GKR proof
    pub public_inputs: Vec<Fr>,
    pub proof_time_ms: u128,
    pub memory_usage_mb: Option<f64>,
}

pub trait AccelContext: Debug + Send + Sync {
    fn backend(&self) -> &dyn Backend;

    fn compute_and_prove(
        &mut self,
        weights: &[Vec<Fr>], // Matrix W (m x k)
        input: &[Fr],        // Input vector x (k elements)
        salt: &str,          // Random salt for Fiat-Shamir
    ) -> Result<ProofResult, AccelError>;

    fn verify_proof(
        &self,
        proof_data: &[u8],
        public_inputs: &[Fr],
        salt: &str,
    ) -> Result<bool, AccelError>;

    fn benchmark_computation(
        &mut self,
        matrix_sizes: &[(usize, usize)], // (m, k) pairs
        num_runs: usize,
    ) -> Result<Vec<BenchmarkResult>, AccelError>;
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub matrix_size: (usize, usize),
    pub proof_time_ms: u128,
    pub verify_time_ms: u128,
    pub memory_usage_mb: Option<f64>,
    pub throughput_ops_per_sec: f64,
}