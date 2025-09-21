use clap::{Parser, Subcommand};
use anyhow::Result;

mod cli;
mod infer;
mod prove;
mod verify;
mod demo;
mod formats;

use cli::*;

#[derive(Parser)]
#[command(name = "nova_poc")]
#[command(about = "Nova POC: Freivalds + GEMV + Tiny Groth16")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference with tiled GEMV and Freivalds auditing
    Infer(InferArgs),
    /// Generate tiny Groth16 proof for 16x16 tail layer
    Prove(ProveArgs),
    /// Verify proof and replay Freivalds check
    Verify(VerifyArgs),
    /// Setup proving/verification keys
    Setup(SetupArgs),
    /// Run complete demo: infer → prove → verify (fast preset)
    Demo {
        /// Random seed for demo
        #[arg(long, default_value = "42")]
        seed: u64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Infer(args) => infer::run_infer(args),
        Commands::Prove(args) => prove::run_prove(args),
        Commands::Verify(args) => verify::run_verify(args),
        Commands::Setup(args) => prove::run_setup(args),
        Commands::Demo { seed } => demo::run_demo(seed),
    }
}