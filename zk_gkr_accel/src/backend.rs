use crate::error::AccelError;
use ark_bn254::Fr;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub num_threads: Option<usize>,
    pub device_id: Option<u32>,
    pub memory_limit: Option<usize>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            num_threads: Some(rayon::current_num_threads()),
            device_id: Some(0),
            memory_limit: None,
        }
    }
}

pub trait Backend: Debug + Send + Sync {
    fn name(&self) -> &'static str;
    fn is_available(&self) -> bool;

    fn matrix_vector_multiply(
        &self,
        matrix: &[Vec<Fr>],
        vector: &[Fr],
        result: &mut [Fr],
    ) -> Result<(), AccelError>;

    fn sumcheck_round(
        &self,
        evaluations: &[Fr],
        challenges: &[Fr],
        result: &mut [Fr],
    ) -> Result<(), AccelError>;

    fn mle_evaluate(
        &self,
        coefficients: &[Fr],
        point: &[Fr],
    ) -> Result<Fr, AccelError>;

    fn mle_fold(
        &self,
        poly: &[Fr],
        challenge: Fr,
        result: &mut [Fr],
    ) -> Result<(), AccelError>;

    fn batch_scalar_multiply(
        &self,
        scalars: &[Fr],
        bases: &[Fr],
        result: &mut [Fr],
    ) -> Result<(), AccelError>;

    fn parallel_hash(
        &self,
        inputs: &[Fr],
        chunk_size: usize,
        result: &mut [Fr],
    ) -> Result<(), AccelError>;
}