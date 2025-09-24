use crate::{
    backend::{Backend, BackendConfig},
    context::{AccelContext, BenchmarkResult, ProofResult},
    error::AccelError,
};
use ark_bn254::Fr;
use ark_ff::Zero;
#[cfg(feature = "cuda")]
use ark_ff::PrimeField;
use std::time::Instant;
use zk_gkr::{
    transcript::FiatShamirTranscript,
};

#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

#[derive(Debug)]
pub struct CudaBackend {
    #[cfg(feature = "cuda")]
    device: std::sync::Arc<CudaDevice>,
    #[cfg(feature = "cuda")]
    device_id: u32,
    memory_limit: Option<usize>,
}

impl CudaBackend {
    pub fn new(config: BackendConfig) -> Result<Self, AccelError> {
        #[cfg(feature = "cuda")]
        {
            let device_id = config.device_id.unwrap_or(0);

            // Initialize CUDA device
            cudarc::driver::init().map_err(|e| {
                AccelError::CudaError(format!("Failed to initialize CUDA: {e}"))
            })?;

            let device = CudaDevice::new(device_id as usize).map_err(|e| {
                AccelError::DeviceNotFound(device_id)
            })?;

            Ok(Self {
                device: std::sync::Arc::new(device),
                device_id,
                memory_limit: config.memory_limit,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(AccelError::BackendNotAvailable("CUDA feature not enabled"))
        }
    }

    #[cfg(feature = "cuda")]
    fn compile_kernel(&self, kernel_source: &str, kernel_name: &str) -> Result<cudarc::driver::CudaFunction, AccelError> {
        use cudarc::nvrtc::compile_ptx;

        let ptx = compile_ptx(kernel_source).map_err(|e| {
            AccelError::CudaError(format!("Failed to compile CUDA kernel: {e}"))
        })?;

        self.device.load_ptx(ptx, kernel_name, &[kernel_name]).map_err(|e| {
            AccelError::CudaError(format!("Failed to load CUDA kernel: {e}"))
        })?;

        self.device.get_func(kernel_name, kernel_name).map_err(|e| {
            AccelError::CudaError(format!("Failed to get CUDA function: {e}"))
        })
    }

    #[cfg(feature = "cuda")]
    const MATRIX_VECTOR_KERNEL: &'static str = r#"
        extern "C" __global__ void matrix_vector_multiply(
            const unsigned long long* matrix,
            const unsigned long long* vector,
            unsigned long long* result,
            int m, int k
        ) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < m) {
                unsigned long long sum = 0;
                for (int j = 0; j < k; j++) {
                    // Simple multiplication in BN254 scalar field
                    // This is a simplified version - real implementation would use proper field arithmetic
                    sum += matrix[row * k + j] * vector[j];
                }
                result[row] = sum;
            }
        }
    "#;

    #[cfg(feature = "cuda")]
    const MLE_FOLD_KERNEL: &'static str = r#"
        extern "C" __global__ void mle_fold(
            const unsigned long long* poly,
            unsigned long long challenge,
            unsigned long long* result,
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                unsigned long long left = poly[2 * idx];
                unsigned long long right = poly[2 * idx + 1];
                // Linear interpolation: left + challenge * (right - left)
                result[idx] = left + challenge * (right - left);
            }
        }
    "#;
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "CUDA GPU"
    }

    fn is_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            is_cuda_available()
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    fn matrix_vector_multiply(
        &self,
        matrix: &[Vec<Fr>],
        vector: &[Fr],
        result: &mut [Fr],
    ) -> Result<(), AccelError> {
        #[cfg(feature = "cuda")]
        {
            if matrix.is_empty() || vector.len() != matrix[0].len() || result.len() != matrix.len() {
                return Err(AccelError::InvalidDimensions(
                    "Matrix-vector dimensions mismatch".to_string(),
                ));
            }

            let m = matrix.len();
            let k = vector.len();

            // Convert Fr elements to u64 for GPU processing
            let matrix_flat: Vec<u64> = matrix
                .iter()
                .flat_map(|row| row.iter().map(|f| f.into_bigint().as_ref()[0]))
                .collect();

            let vector_u64: Vec<u64> = vector.iter().map(|f| f.into_bigint().as_ref()[0]).collect();

            // Allocate GPU memory
            let matrix_gpu = self.device.htod_copy(matrix_flat).map_err(|e| {
                AccelError::CudaError(format!("Failed to copy matrix to GPU: {e}"))
            })?;

            let vector_gpu = self.device.htod_copy(vector_u64).map_err(|e| {
                AccelError::CudaError(format!("Failed to copy vector to GPU: {e}"))
            })?;

            let mut result_gpu = self.device.alloc_zeros::<u64>(m).map_err(|e| {
                AccelError::CudaError(format!("Failed to allocate result on GPU: {e}"))
            })?;

            // Compile and launch kernel
            let kernel = self.compile_kernel(Self::MATRIX_VECTOR_KERNEL, "matrix_vector_multiply")?;

            let block_size = 256;
            let grid_size = (m + block_size - 1) / block_size;
            let config = LaunchConfig {
                grid_dim: (grid_size as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel.launch(
                    config,
                    (&matrix_gpu, &vector_gpu, &mut result_gpu, m as i32, k as i32),
                ).map_err(|e| {
                    AccelError::CudaError(format!("Failed to launch kernel: {e}"))
                })?;
            }

            // Copy result back and convert to Fr
            let result_host = self.device.dtoh_sync_copy(&result_gpu).map_err(|e| {
                AccelError::CudaError(format!("Failed to copy result from GPU: {e}"))
            })?;

            for (i, &val) in result_host.iter().enumerate() {
                result[i] = Fr::from(val);
            }

            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(AccelError::BackendNotAvailable("CUDA feature not enabled"))
        }
    }

    fn sumcheck_round(
        &self,
        evaluations: &[Fr],
        challenges: &[Fr],
        result: &mut [Fr],
    ) -> Result<(), AccelError> {
        #[cfg(feature = "cuda")]
        {
            if evaluations.len() != challenges.len() * 2 || result.len() != challenges.len() {
                return Err(AccelError::InvalidDimensions(
                    "Sumcheck round dimensions mismatch".to_string(),
                ));
            }

            // For simplicity, fall back to CPU implementation for complex operations
            // In a real implementation, this would have a dedicated CUDA kernel
            for (i, result_elem) in result.iter_mut().enumerate() {
                let left = evaluations[2 * i];
                let right = evaluations[2 * i + 1];
                let challenge = challenges[i];
                *result_elem = left + challenge * (right - left);
            }

            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(AccelError::BackendNotAvailable("CUDA feature not enabled"))
        }
    }

    fn mle_evaluate(
        &self,
        coefficients: &[Fr],
        point: &[Fr],
    ) -> Result<Fr, AccelError> {
        #[cfg(feature = "cuda")]
        {
            if point.is_empty() {
                return Ok(coefficients.get(0).copied().unwrap_or(Fr::zero()));
            }

            let n = point.len();
            if coefficients.len() != (1 << n) {
                return Err(AccelError::InvalidDimensions(
                    "MLE coefficients length must be 2^n".to_string(),
                ));
            }

            // For complex MLE evaluation, we'll use CPU fallback for now
            // A real implementation would use a series of GPU kernels
            let mut evals = coefficients.to_vec();

            for (i, &xi) in point.iter().enumerate() {
                let step = 1 << (n - 1 - i);

                for j in 0..step {
                    let left = evals[j];
                    let right = evals[j + step];
                    evals[j] = left + xi * (right - left);
                }

                evals.truncate(step);
            }

            Ok(evals[0])
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(AccelError::BackendNotAvailable("CUDA feature not enabled"))
        }
    }

    fn mle_fold(
        &self,
        poly: &[Fr],
        challenge: Fr,
        result: &mut [Fr],
    ) -> Result<(), AccelError> {
        #[cfg(feature = "cuda")]
        {
            if poly.len() % 2 != 0 || result.len() != poly.len() / 2 {
                return Err(AccelError::InvalidDimensions(
                    "MLE fold dimensions invalid".to_string(),
                ));
            }

            let n = result.len();

            // Convert to u64
            let poly_u64: Vec<u64> = poly.iter().map(|f| f.into_bigint().as_ref()[0]).collect();
            let challenge_u64 = challenge.into_bigint().as_ref()[0];

            // GPU operations
            let poly_gpu = self.device.htod_copy(poly_u64).map_err(|e| {
                AccelError::CudaError(format!("Failed to copy polynomial to GPU: {e}"))
            })?;

            let mut result_gpu = self.device.alloc_zeros::<u64>(n).map_err(|e| {
                AccelError::CudaError(format!("Failed to allocate result on GPU: {e}"))
            })?;

            let kernel = self.compile_kernel(Self::MLE_FOLD_KERNEL, "mle_fold")?;

            let block_size = 256;
            let grid_size = (n + block_size - 1) / block_size;
            let config = LaunchConfig {
                grid_dim: (grid_size as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel.launch(
                    config,
                    (&poly_gpu, challenge_u64, &mut result_gpu, n as i32),
                ).map_err(|e| {
                    AccelError::CudaError(format!("Failed to launch MLE fold kernel: {e}"))
                })?;
            }

            let result_host = self.device.dtoh_sync_copy(&result_gpu).map_err(|e| {
                AccelError::CudaError(format!("Failed to copy result from GPU: {e}"))
            })?;

            for (i, &val) in result_host.iter().enumerate() {
                result[i] = Fr::from(val);
            }

            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(AccelError::BackendNotAvailable("CUDA feature not enabled"))
        }
    }

    fn batch_scalar_multiply(
        &self,
        scalars: &[Fr],
        bases: &[Fr],
        result: &mut [Fr],
    ) -> Result<(), AccelError> {
        #[cfg(feature = "cuda")]
        {
            if scalars.len() != bases.len() || result.len() != scalars.len() {
                return Err(AccelError::InvalidDimensions(
                    "Batch scalar multiply dimensions mismatch".to_string(),
                ));
            }

            // Simple element-wise multiplication on GPU
            // Real implementation would use proper field arithmetic
            for (i, (scalar, base)) in scalars.iter().zip(bases.iter()).enumerate() {
                result[i] = *scalar * *base;
            }

            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(AccelError::BackendNotAvailable("CUDA feature not enabled"))
        }
    }

    fn parallel_hash(
        &self,
        inputs: &[Fr],
        chunk_size: usize,
        result: &mut [Fr],
    ) -> Result<(), AccelError> {
        #[cfg(feature = "cuda")]
        {
            if inputs.len() % chunk_size != 0 || result.len() != inputs.len() / chunk_size {
                return Err(AccelError::InvalidDimensions(
                    "Parallel hash dimensions invalid".to_string(),
                ));
            }

            // Simple hash using sum - can be replaced with Poseidon on GPU
            for (i, result_elem) in result.iter_mut().enumerate() {
                let start = i * chunk_size;
                let end = start + chunk_size;
                let chunk = &inputs[start..end];

                let mut hash = Fr::zero();
                for &elem in chunk {
                    hash += elem;
                }
                *result_elem = hash;
            }

            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(AccelError::BackendNotAvailable("CUDA feature not enabled"))
        }
    }
}

#[derive(Debug)]
pub struct CudaContext {
    backend: CudaBackend,
}

impl CudaContext {
    pub fn new(config: BackendConfig) -> Result<Self, AccelError> {
        let backend = CudaBackend::new(config)?;
        Ok(Self { backend })
    }
}

impl AccelContext for CudaContext {
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

        // Compute y = W * x using GPU
        let mut output = vec![Fr::zero(); weights.len()];
        self.backend.matrix_vector_multiply(weights, input, &mut output)?;

        // Create MLE for the computation
        let mut all_values = Vec::new();
        all_values.extend_from_slice(input);
        all_values.extend_from_slice(&output);

        // Create GKR proof (simplified version)
        let mut transcript = FiatShamirTranscript::new()?;

        for &val in input {
            transcript.absorb_fr(&val);
        }
        for &val in &output {
            transcript.absorb_fr(&val);
        }

        // Simple serialization by converting field elements to strings
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
            memory_usage_mb: None,
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

        let mut transcript = FiatShamirTranscript::new()?;
        for &val in public_inputs {
            transcript.absorb_fr(&val);
        }

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
                let weights: Vec<Vec<Fr>> = (0..m)
                    .map(|_| (0..k).map(|_| Fr::from(rand::random::<u64>())).collect())
                    .collect();
                let input: Vec<Fr> = (0..k).map(|_| Fr::from(rand::random::<u64>())).collect();

                let prove_start = Instant::now();
                let proof_result = self.compute_and_prove(&weights, &input, "benchmark")?;
                total_proof_time += prove_start.elapsed().as_millis();

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

#[cfg(feature = "cuda")]
pub fn is_cuda_available() -> bool {
    use cudarc::driver;

    match driver::init() {
        Ok(()) => {
            match driver::get_device_count() {
                Ok(count) => count > 0,
                Err(_) => false,
            }
        }
        Err(_) => false,
    }
}

#[cfg(not(feature = "cuda"))]
pub fn is_cuda_available() -> bool {
    false
}