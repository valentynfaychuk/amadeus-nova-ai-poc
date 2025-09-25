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
    proof::GkrProof,
    sumcheck::{SumCheckProver, SumCheckVerifier},
    transcript::FiatShamirTranscript,
    merkle_poseidon::PoseidonMerkleTree,
    mle::MleUtils,
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
        salt: &str,
    ) -> Result<ProofResult, AccelError> {
        let start_time = Instant::now();

        let m = weights.len();
        let k = input.len();

        // Compute y = W * x using accelerated backend
        let mut output = vec![Fr::zero(); m];
        self.backend.matrix_vector_multiply(weights, input, &mut output)?;

        // Convert to hypercube order and build Merkle trees
        let w_data = MleUtils::matrix_to_hypercube_order(weights, m, k)
            .map_err(|e| AccelError::ComputationFailed(format!("Failed to convert W to hypercube: {e}")))?;
        let x_data = MleUtils::vector_to_hypercube_order(input, k)
            .map_err(|e| AccelError::ComputationFailed(format!("Failed to convert X to hypercube: {e}")))?;

        let w_tree = PoseidonMerkleTree::build_tree(&w_data)
            .map_err(|e| AccelError::ComputationFailed(format!("Failed to build W tree: {e}")))?;
        let x_tree = PoseidonMerkleTree::build_tree(&x_data)
            .map_err(|e| AccelError::ComputationFailed(format!("Failed to build X tree: {e}")))?;

        let h_w = w_tree.root();
        let h_x = x_tree.root();

        // Create Fiat-Shamir transcript
        let mut transcript = FiatShamirTranscript::new_seeded(
            &h_w,
            &h_x,
            m,
            k,
            None, // model_id
            None, // vk_hash
            salt,
        ).map_err(|e| AccelError::ComputationFailed(format!("Failed to create transcript: {e}")))?;

        // Calculate padded dimensions
        let a = (m as f64).log2().ceil() as usize;
        let b = (k as f64).log2().ceil() as usize;

        // Derive challenge vector u
        let u = transcript.derive_u_vector(1 << a)
            .map_err(|e| AccelError::ComputationFailed(format!("Failed to derive u vector: {e}")))?;

        // Compute claimed value c = u^T * (W * x) = u^T * y
        let mut c = Fr::zero();
        for (i, &u_val) in u.iter().enumerate() {
            if i < output.len() {
                c += u_val * output[i];
            }
        }

        // Create SumCheck prover with correct parameter order
        let prover = SumCheckProver::new(
            u,
            w_data,
            x_data,
            a,
            b,
            w_tree,
            x_tree,
        ).map_err(|e| AccelError::ComputationFailed(format!("Failed to create prover: {e}")))?;

        // Note: u derivation already done above, don't repeat it

        // Generate sum-check proof
        let sumcheck_proof = prover.prove(&mut transcript)
            .map_err(|e| AccelError::ComputationFailed(format!("Failed to generate sum-check proof: {e}")))?;

        // Create complete GKR proof
        let gkr_proof = GkrProof::new(
            m, k, h_w, h_x, c, salt.to_string(), sumcheck_proof
        );

        // Serialize proof to bytes
        let proof_bytes = gkr_proof.to_bytes()
            .map_err(|e| AccelError::ComputationFailed(format!("Failed to serialize proof: {e}")))?;

        // Create public inputs with correct format
        let mut public_inputs = Vec::new();
        public_inputs.extend_from_slice(input);
        public_inputs.extend_from_slice(&output);
        public_inputs.push(c); // Add claimed value

        let proof_time = start_time.elapsed().as_millis();

        Ok(ProofResult {
            proof: proof_bytes,
            public_inputs,
            proof_time_ms: proof_time,
            memory_usage_mb: None,
        })
    }

    fn verify_proof(
        &self,
        proof_data: &[u8],
        public_inputs: &[Fr],
        salt: &str,
    ) -> Result<bool, AccelError> {
        // Deserialize GKR proof
        let gkr_proof = GkrProof::from_bytes(proof_data)
            .map_err(|e| AccelError::ComputationFailed(format!("Failed to deserialize proof: {e}")))?;

        // Extract dimensions and values from public inputs
        // Format: [input..., output..., claimed_value]
        let total_len = public_inputs.len();
        if total_len < 3 { // At least 1 input, 1 output, 1 claimed value
            return Err(AccelError::ComputationFailed("Invalid public inputs format".to_string()));
        }

        let m = gkr_proof.m;
        let k = gkr_proof.k;

        if total_len != k + m + 1 {
            return Err(AccelError::ComputationFailed(
                format!("Public inputs length {} doesn't match expected {}", total_len, k + m + 1)
            ));
        }

        let claimed_value = public_inputs[k+m];

        // Verify claimed value matches proof
        if gkr_proof.c != claimed_value {
            return Ok(false);
        }

        // Create verification transcript
        let mut transcript = FiatShamirTranscript::new_seeded(
            &gkr_proof.h_w,
            &gkr_proof.h_x,
            m,
            k,
            None, // model_id
            None, // vk_hash
            salt,
        ).map_err(|e| AccelError::ComputationFailed(format!("Failed to create verification transcript: {e}")))?;

        let a = (m as f64).log2().ceil() as usize;
        let u = transcript.derive_u_vector(1 << a)
            .map_err(|e| AccelError::ComputationFailed(format!("Failed to derive u vector: {e}")))?;

        // Create verification transcript (matching prover state)
        let mut verify_transcript = FiatShamirTranscript::new_seeded(
            &gkr_proof.h_w,
            &gkr_proof.h_x,
            m,
            k,
            None,
            None,
            salt,
        ).map_err(|e| AccelError::ComputationFailed(format!("Failed to create verify transcript: {e}")))?;

        // Skip u derivation to match prover state
        verify_transcript.derive_u_vector(1 << a)
            .map_err(|e| AccelError::ComputationFailed(format!("Failed to skip u derivation: {e}")))?;

        // Verify the sum-check proof using real cryptographic verification
        let verification_result = SumCheckVerifier::verify(
            &gkr_proof.sumcheck_proof,
            &u,
            &gkr_proof.h_w,
            &gkr_proof.h_x,
            gkr_proof.a,
            gkr_proof.b,
            &mut verify_transcript,
        ).map_err(|e| AccelError::ComputationFailed(format!("Sum-check verification failed: {e}")))?;

        Ok(verification_result)
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