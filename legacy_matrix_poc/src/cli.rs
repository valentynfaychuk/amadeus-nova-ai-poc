use clap::Args;
use std::path::PathBuf;

#[derive(Args)]
pub struct ProveGkrArgs {
    /// Path to weights1 binary file (16×K, row-major i16)
    #[arg(long)]
    pub weights1_path: PathBuf,

    /// Path to input vector x0
    #[arg(long)]
    pub x0_path: PathBuf,

    /// Matrix dimensions (m rows, k columns)
    #[arg(long)]
    pub m: usize,

    #[arg(long)]
    pub k: usize,

    /// Random salt for Fiat-Shamir (hex string)
    #[arg(long, default_value = "deadbeef")]
    pub salt: String,

    /// Output directory for GKR proof files
    #[arg(long, default_value = "gkr_out")]
    pub out_dir: PathBuf,

    /// Optional model identifier
    #[arg(long)]
    pub model_id: Option<String>,

    /// Optional verification key hash
    #[arg(long)]
    pub vk_hash: Option<String>,

    /// Enable accelerated backend (requires --features accel)
    #[arg(long)]
    pub accel: bool,

    /// Acceleration backend type (cpu_avx, cuda)
    #[arg(long, default_value = "cpu_avx")]
    pub accel_backend: String,

    /// GPU device ID for CUDA backend
    #[arg(long, default_value = "0")]
    pub accel_device_id: u32,

    /// Number of threads for CPU backend
    #[arg(long)]
    pub accel_threads: Option<usize>,
}

#[derive(Args)]
pub struct VerifyGkrArgs {
    /// GKR proof file
    #[arg(long)]
    pub proof_path: PathBuf,

    /// Public inputs JSON file
    #[arg(long)]
    pub public_path: PathBuf,

    /// Also verify with tiny Groth16 tail
    #[arg(long)]
    pub with_tail: bool,
}
