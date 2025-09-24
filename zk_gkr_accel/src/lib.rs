use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod backend;
pub mod context;
pub mod error;

#[cfg(feature = "cpu_avx")]
pub mod cpu_avx;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use backend::{Backend, BackendConfig};
pub use context::{AccelContext, ProofResult};
pub use error::AccelError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    CpuAvx,
    Cuda,
}

#[derive(Debug, Clone)]
pub struct ComputeConfig {
    pub backend: BackendType,
    pub device_id: Option<u32>,
    pub num_threads: Option<usize>,
    pub memory_limit: Option<usize>,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            backend: BackendType::CpuAvx,
            device_id: None,
            num_threads: None,
            memory_limit: None,
        }
    }
}

pub fn create_context(config: ComputeConfig) -> Result<Box<dyn AccelContext>, AccelError> {
    match config.backend {
        #[cfg(feature = "cpu_avx")]
        BackendType::CpuAvx => {
            let backend_config = BackendConfig {
                num_threads: config.num_threads,
                device_id: None,
                memory_limit: config.memory_limit,
            };
            Ok(Box::new(cpu_avx::CpuAvxContext::new(backend_config)?))
        }

        #[cfg(not(feature = "cpu_avx"))]
        BackendType::CpuAvx => Err(AccelError::BackendNotAvailable("cpu_avx feature not enabled")),

        #[cfg(feature = "cuda")]
        BackendType::Cuda => {
            let backend_config = BackendConfig {
                num_threads: None,
                device_id: config.device_id,
                memory_limit: config.memory_limit,
            };
            Ok(Box::new(cuda::CudaContext::new(backend_config)?))
        }

        #[cfg(not(feature = "cuda"))]
        BackendType::Cuda => Err(AccelError::BackendNotAvailable("cuda feature not enabled")),
    }
}

pub fn available_backends() -> Vec<BackendType> {
    let mut backends = Vec::new();

    #[cfg(feature = "cpu_avx")]
    backends.push(BackendType::CpuAvx);

    #[cfg(feature = "cuda")]
    {
        if cuda::is_cuda_available() {
            backends.push(BackendType::Cuda);
        }
    }

    backends
}