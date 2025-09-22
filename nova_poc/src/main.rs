use anyhow::Result;
use clap::{Parser, Subcommand};

mod cli;
use cli::*;

mod gkr;

#[derive(Parser)]
#[command(name = "nova_poc")]
#[command(about = "Nova POC: GKR Zero-Knowledge Proofs for Matrix-Vector Multiplication")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate GKR proof for matrix-vector multiplication (default mode)
    Prove(ProveGkrArgs),
    /// Verify GKR proof
    Verify(VerifyGkrArgs),
    /// Run complete demo: prove → verify (fast preset)
    Demo {
        /// Random seed for demo
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Matrix dimensions
        #[arg(long, default_value = "16")]
        m: usize,
        #[arg(long, default_value = "4096")]
        k: usize,
    },
    /// Setup and run benchmarks for different model sizes
    Benchmark {
        /// Matrix sizes to benchmark (comma-separated K values)
        #[arg(long, default_value = "4096,8192,16384")]
        sizes: String,
        /// Number of repeats per configuration
        #[arg(long, default_value = "3")]
        repeats: usize,
        /// Output CSV file
        #[arg(long, default_value = "benchmark_results.csv")]
        output: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Prove(args) => gkr::run_prove_gkr(args),
        Commands::Verify(args) => gkr::run_verify_gkr(args),
        Commands::Demo { seed, m, k } => gkr::run_demo(seed, m, k),
        Commands::Benchmark {
            sizes,
            repeats,
            output,
        } => gkr::run_benchmark(sizes, repeats, output),
    }
}
