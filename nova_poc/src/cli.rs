use clap::Args;
use std::path::PathBuf;

#[derive(Args)]
pub struct InferArgs {
    /// Width K of the large layer W1 (16xK)
    #[arg(long, default_value = "4096")]
    pub k: usize,

    /// Tile size for streaming (e.g., 1024, 4096)
    #[arg(long, default_value = "1024")]
    pub tile_k: usize,

    /// Scale numerator for quantization
    #[arg(long, default_value = "3")]
    pub scale_num: u64,

    /// Random seed for deterministic execution
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Number of Freivalds rounds for probabilistic verification
    #[arg(long, default_value = "32")]
    pub freivalds_rounds: usize,

    /// Path to weights1 binary file (16×K, row-major i16)
    #[arg(long)]
    pub weights1_path: Option<PathBuf>,

    /// Path to weights2 file (16×16)
    #[arg(long)]
    pub weights2_path: Option<PathBuf>,

    /// Path to input vector x0
    #[arg(long)]
    pub x0_path: Option<PathBuf>,

    /// Output JSON file path
    #[arg(long, default_value = "run.json")]
    pub out: PathBuf,

    /// Skip Freivalds verification for speed comparison
    #[arg(long)]
    pub skip_freivalds: bool,

    /// Use demo preset (generates random data)
    #[arg(long)]
    pub preset: Option<String>,
}

#[derive(Args)]
pub struct ProveArgs {
    /// Input run.json file from infer command
    pub run_json: PathBuf,

    /// Path to proving key (auto-load cached params if not specified)
    #[arg(long)]
    pub pk_path: Option<PathBuf>,

    /// Output directory for proof files
    #[arg(long, default_value = "out")]
    pub out_dir: PathBuf,
}

#[derive(Args)]
pub struct VerifyArgs {
    /// Input run.json file
    pub run_json: PathBuf,

    /// Verification key path
    #[arg(long)]
    pub vk_path: Option<PathBuf>,

    /// Proof file path
    #[arg(long)]
    pub proof_path: Option<PathBuf>,

    /// Public inputs JSON file
    #[arg(long)]
    pub public_inputs_path: Option<PathBuf>,

    /// Weights1 path to recompute Freivalds
    #[arg(long)]
    pub weights1_path: Option<PathBuf>,

    /// Skip Freivalds verification
    #[arg(long)]
    pub skip_freivalds: bool,

    /// Use prover's seed instead of transcript-bound seed (for tests)
    #[arg(long)]
    pub no_bind_randomness: bool,

    /// Allow k < 16 without reconstruction
    #[arg(long)]
    pub allow_low_k: bool,

    /// Optional extra entropy in the transcript (hex string)
    #[arg(long)]
    pub block_entropy: Option<String>,

    /// Tile size for recompute function (defaults to value from RunData)
    #[arg(long)]
    pub tile_k: Option<usize>,
}

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

#[derive(Args)]
pub struct SetupArgs {
    /// Output directory for keys
    #[arg(long, default_value = "keys")]
    pub out_dir: PathBuf,

    /// Force regeneration even if keys exist
    #[arg(long)]
    pub force: bool,
}