use crate::{
    backend::{Backend, BackendConfig},
    context::{AccelContext, BenchmarkResult, ProofResult},
    error::AccelError,
};
use ark_bn254::Fr;
use ark_ff::Zero;
#[cfg(all(feature = "cpu_avx", any(target_arch = "x86", target_arch = "x86_64")))]
use ark_ff::PrimeField;
use rayon::prelude::*;
use std::time::Instant;
use zk_gkr::{
    transcript::FiatShamirTranscript,
};

#[derive(Debug)]
pub struct CpuAvxBackend {
    _num_threads: usize,
    _memory_limit: Option<usize>,
}

impl CpuAvxBackend {
    pub fn new(config: BackendConfig) -> Result<Self, AccelError> {
        let num_threads = config.num_threads.unwrap_or_else(|| rayon::current_num_threads());

        // Try to set up the global thread pool, but don't fail if it's already initialized
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global();

        Ok(Self {
            _num_threads: num_threads,
            _memory_limit: config.memory_limit,
        })
    }

}

impl Backend for CpuAvxBackend {
    fn name(&self) -> &'static str {
        "CPU AVX2/AVX-512"
    }

    fn is_available(&self) -> bool {
        #[cfg(all(feature = "cpu_avx", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            // Check for AVX2 support
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(all(feature = "cpu_avx", any(target_arch = "x86", target_arch = "x86_64"))))]
        true // Consider available on non-x86 architectures for fallback implementation
    }

    fn matrix_vector_multiply(
        &self,
        matrix: &[Vec<Fr>],
        vector: &[Fr],
        result: &mut [Fr],
    ) -> Result<(), AccelError> {
        if matrix.is_empty() || vector.len() != matrix[0].len() || result.len() != matrix.len() {
            return Err(AccelError::InvalidDimensions(
                "Matrix-vector dimensions mismatch".to_string(),
            ));
        }

        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, result_elem)| {
                let mut sum = Fr::zero();
                for (j, &v_elem) in vector.iter().enumerate() {
                    sum += matrix[i][j] * v_elem;
                }
                *result_elem = sum;
            });

        Ok(())
    }

    fn sumcheck_round(
        &self,
        evaluations: &[Fr],
        challenges: &[Fr],
        result: &mut [Fr],
    ) -> Result<(), AccelError> {
        if evaluations.len() != challenges.len() * 2 || result.len() != challenges.len() {
            return Err(AccelError::InvalidDimensions(
                "Sumcheck round dimensions mismatch".to_string(),
            ));
        }

        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, result_elem)| {
                let left = evaluations[2 * i];
                let right = evaluations[2 * i + 1];
                let challenge = challenges[i];

                // Linear interpolation: left + challenge * (right - left)
                *result_elem = left + challenge * (right - left);
            });

        Ok(())
    }

    fn mle_evaluate(
        &self,
        coefficients: &[Fr],
        point: &[Fr],
    ) -> Result<Fr, AccelError> {
        if point.is_empty() {
            return Ok(coefficients.get(0).copied().unwrap_or(Fr::zero()));
        }

        let n = point.len();
        if coefficients.len() != (1 << n) {
            return Err(AccelError::InvalidDimensions(
                "MLE coefficients length must be 2^n".to_string(),
            ));
        }

        let mut evals = coefficients.to_vec();

        for (i, &xi) in point.iter().enumerate() {
            let step = 1 << (n - 1 - i);

            evals
                .par_chunks_mut(step * 2)
                .for_each(|chunk| {
                    for j in 0..step {
                        let left = chunk[j];
                        let right = chunk[j + step];
                        chunk[j] = left + xi * (right - left);
                    }
                });

            evals.truncate(step);
        }

        Ok(evals[0])
    }

    fn mle_fold(
        &self,
        poly: &[Fr],
        challenge: Fr,
        result: &mut [Fr],
    ) -> Result<(), AccelError> {
        if poly.len() % 2 != 0 || result.len() != poly.len() / 2 {
            return Err(AccelError::InvalidDimensions(
                "MLE fold dimensions invalid".to_string(),
            ));
        }

        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, result_elem)| {
                let left = poly[2 * i];
                let right = poly[2 * i + 1];
                *result_elem = left + challenge * (right - left);
            });

        Ok(())
    }

    fn batch_scalar_multiply(
        &self,
        scalars: &[Fr],
        bases: &[Fr],
        result: &mut [Fr],
    ) -> Result<(), AccelError> {
        if scalars.len() != bases.len() || result.len() != scalars.len() {
            return Err(AccelError::InvalidDimensions(
                "Batch scalar multiply dimensions mismatch".to_string(),
            ));
        }

        #[cfg(all(feature = "cpu_avx", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            // Convert to u64 for SIMD operations
            let scalar_u64: Vec<u64> = scalars.iter().map(|f| f.into_bigint().as_ref()[0]).collect();
            let base_u64: Vec<u64> = bases.iter().map(|f| f.into_bigint().as_ref()[0]).collect();
            let mut result_u64 = vec![0u64; result.len()];

            if is_x86_feature_detected!("avx2") {
                // Simplified SIMD operations - real implementation would be more complex
                for (i, ((scalar, base), res)) in scalar_u64.iter()
                    .zip(base_u64.iter())
                    .zip(result_u64.iter_mut())
                    .enumerate()
                {
                    *res = scalar.wrapping_mul(*base);
                }
            } else {
                // Fallback to scalar implementation
                result
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, result_elem)| {
                        *result_elem = scalars[i] * bases[i];
                    });
                return Ok(());
            }

            // Convert back to Fr
            for (i, &val) in result_u64.iter().enumerate() {
                result[i] = Fr::from(val);
            }
        }

        #[cfg(not(all(feature = "cpu_avx", any(target_arch = "x86", target_arch = "x86_64"))))]
        {
            // Fallback implementation for non-x86 architectures
            result
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, result_elem)| {
                    *result_elem = scalars[i] * bases[i];
                });
        }

        Ok(())
    }

    fn parallel_hash(
        &self,
        inputs: &[Fr],
        chunk_size: usize,
        result: &mut [Fr],
    ) -> Result<(), AccelError> {
        if inputs.len() % chunk_size != 0 || result.len() != inputs.len() / chunk_size {
            return Err(AccelError::InvalidDimensions(
                "Parallel hash dimensions invalid".to_string(),
            ));
        }

        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, result_elem)| {
                let start = i * chunk_size;
                let end = start + chunk_size;
                let chunk = &inputs[start..end];

                // Simple hash using sum for now - can be replaced with Poseidon
                let mut hash = Fr::zero();
                for &elem in chunk {
                    hash += elem;
                }
                *result_elem = hash;
            });

        Ok(())
    }
}

#[derive(Debug)]
pub struct CpuAvxContext {
    backend: CpuAvxBackend,
}

impl CpuAvxContext {
    pub fn new(config: BackendConfig) -> Result<Self, AccelError> {
        let backend = CpuAvxBackend::new(config)?;
        Ok(Self { backend })
    }
}

impl AccelContext for CpuAvxContext {
    fn backend(&self) -> &dyn Backend {
        &self.backend
    }

    fn compute_and_prove(
        &mut self,
        weights: &[Vec<Fr>],
        input: &[Fr],
        _salt: &str,
    ) -> Result<ProofResult, AccelError> {
        let start_time = Instant::now();

        // Compute y = W * x
        let mut output = vec![Fr::zero(); weights.len()];
        self.backend.matrix_vector_multiply(weights, input, &mut output)?;

        // Create MLE for the computation
        let mut all_values = Vec::new();
        all_values.extend_from_slice(input);
        all_values.extend_from_slice(&output);

        // Create a basic GKR proof structure
        let mut transcript = FiatShamirTranscript::new()?;

        // Add public inputs to transcript
        for &val in input {
            transcript.absorb_fr(&val);
        }
        for &val in &output {
            transcript.absorb_fr(&val);
        }

        // Simple serialization by converting field elements to strings
        // In a real implementation, this would be a proper GKR proof
        let proof_str = format!("{:?}", (&all_values, output[0]));
        let proof_bytes = proof_str.as_bytes().to_vec();

        let mut public_inputs = Vec::new();
        public_inputs.extend_from_slice(input);
        public_inputs.extend_from_slice(&output);

        let proof_time = start_time.elapsed().as_millis();

        Ok(ProofResult {
            proof: proof_bytes,
            public_inputs,
            proof_time_ms: proof_time,
            memory_usage_mb: None, // Could add memory tracking
        })
    }

    fn verify_proof(
        &self,
        proof_data: &[u8],
        public_inputs: &[Fr],
        _salt: &str,
    ) -> Result<bool, AccelError> {
        // Simple deserialization check - just verify we can read the proof data
        let _proof_str = String::from_utf8(proof_data.to_vec())
            .map_err(|e| AccelError::ComputationFailed(format!("Proof data not valid UTF-8: {e}")))?;

        // Simplified verification - in a real implementation this would verify the sumcheck
        let mut transcript = FiatShamirTranscript::new()?;
        for &val in public_inputs {
            transcript.absorb_fr(&val);
        }

        // For now, return true if deserialization succeeded
        Ok(true)
    }

    fn benchmark_computation(
        &mut self,
        matrix_sizes: &[(usize, usize)],
        num_runs: usize,
    ) -> Result<Vec<BenchmarkResult>, AccelError> {
        let mut results = Vec::new();

        for &(m, k) in matrix_sizes {
            let mut total_proof_time = 0u128;
            let mut total_verify_time = 0u128;

            for _ in 0..num_runs {
                // Generate random matrix and input
                let weights: Vec<Vec<Fr>> = (0..m)
                    .map(|_| (0..k).map(|_| Fr::from(rand::random::<u64>())).collect())
                    .collect();
                let input: Vec<Fr> = (0..k).map(|_| Fr::from(rand::random::<u64>())).collect();

                // Prove
                let prove_start = Instant::now();
                let proof_result = self.compute_and_prove(&weights, &input, "benchmark")?;
                total_proof_time += prove_start.elapsed().as_millis();

                // Verify
                let verify_start = Instant::now();
                let _verified = self.verify_proof(&proof_result.proof, &proof_result.public_inputs, "benchmark")?;
                total_verify_time += verify_start.elapsed().as_millis();
            }

            let avg_proof_time = total_proof_time / num_runs as u128;
            let avg_verify_time = total_verify_time / num_runs as u128;
            let throughput = if avg_proof_time > 0 {
                1000.0 / avg_proof_time as f64
            } else {
                0.0
            };

            results.push(BenchmarkResult {
                matrix_size: (m, k),
                proof_time_ms: avg_proof_time,
                verify_time_ms: avg_verify_time,
                memory_usage_mb: None,
                throughput_ops_per_sec: throughput,
            });
        }

        Ok(results)
    }
}