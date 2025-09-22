use crate::cli::InferArgs;
use crate::formats::*;
use crate::demo;
use engine::EngineConfig;
use engine::gemv::{gemv_tiled, compute_layer_16x16};
use engine::commitment::{commit_alpha_sum_vec, commit_alpha_sum_w2};
use engine::freivalds::freivalds_check;
use anyhow::Result;
use std::fs::File;
use std::io::{Seek, SeekFrom};
use tempfile;

pub fn run_infer(args: InferArgs) -> Result<()> {
    println!("🚀 Starting inference with K={}, tile_k={}", args.k, args.tile_k);

    // Handle preset mode
    if let Some(preset) = &args.preset {
        match preset.as_str() {
            "demo" => {
                println!("🎲 Using demo preset with generated data");
                return run_demo_infer(args);
            }
            _ => anyhow::bail!("Unknown preset: {}", preset),
        }
    }

    // Validate required paths
    let weights1_path = args.weights1_path.clone().ok_or_else(|| anyhow::anyhow!("--weights1-path required when not using preset"))?;
    let weights2_path = args.weights2_path.clone().ok_or_else(|| anyhow::anyhow!("--weights2-path required when not using preset"))?;
    let x0_path = args.x0_path.clone().ok_or_else(|| anyhow::anyhow!("--x0-path required when not using preset"))?;

    // Load data
    println!("📂 Loading input data...");
    let w1_reader = load_weights1(&weights1_path, args.k)?;
    let w2 = load_weights2(&weights2_path)?;
    let x0 = load_vector(&x0_path)?;

    if x0.len() != args.k {
        anyhow::bail!("Input vector length {} doesn't match K={}", x0.len(), args.k);
    }

    println!("✅ Loaded W1 (16×{}), W2 (16×16), x0 ({})", args.k, x0.len());

    // Run the inference pipeline
    run_inference_pipeline(args, w1_reader, w2, x0)
}

fn run_demo_infer(args: InferArgs) -> Result<()> {
    println!("🎲 Generating random demo data...");
    let (w1_data, w2, x0) = demo::generate_demo_data(args.k, args.seed)?;
    // Write to temp file for demo since we need Seek trait
    let temp_file = tempfile::NamedTempFile::new()?;
    std::fs::write(temp_file.path(), &w1_data)?;
    let w1_reader = File::open(temp_file.path())?;

    println!("✅ Generated W1 (16×{}), W2 (16×16), x0 ({})", args.k, x0.len());

    run_inference_pipeline(args, w1_reader, w2, x0)
}

fn run_inference_pipeline(
    args: InferArgs,
    mut w1_reader: File,
    w2: [[i64; 16]; 16],
    x0: Vec<i64>,
) -> Result<()> {
    let _config = EngineConfig {
        k: args.k,
        tile_k: args.tile_k,
        freivalds_rounds: args.freivalds_rounds,
        scale_num: args.scale_num,
        seed: args.seed,
    };

    // Step 1: Compute commitments for inputs
    println!("🔒 Computing input commitments...");
    let h_x = commit_alpha_sum_vec(&x0);
    let h_w2 = commit_alpha_sum_w2(&w2);

    // Step 2: Tiled GEMV computation
    println!("⚡ Running tiled GEMV for large layer...");
    let start_time = std::time::Instant::now();
    let (y1, h_w1) = gemv_tiled(&mut w1_reader, &x0, args.k, args.tile_k)?;
    let gemv_time = start_time.elapsed();
    println!("✅ GEMV completed in {:.3}s", gemv_time.as_secs_f64());

    // Step 3: Freivalds verification (optional)
    let freivalds_result = if args.skip_freivalds {
        println!("⏭️  Skipping Freivalds verification");
        None
    } else {
        println!("🔍 Running Freivalds verification ({} rounds)...", args.freivalds_rounds);
        w1_reader.seek(SeekFrom::Start(0))?; // Reset for Freivalds
        let start_time = std::time::Instant::now();

        match freivalds_check(
            &mut w1_reader,
            &x0,
            &y1,
            args.k,
            args.tile_k,
            args.freivalds_rounds,
            args.seed,
        ) {
            Ok(()) => {
                let freivalds_time = start_time.elapsed();
                println!("✅ Freivalds passed in {:.3}s", freivalds_time.as_secs_f64());
                Some(FreivaldsResult {
                    seed: args.seed,
                    rounds: args.freivalds_rounds,
                    passed: true,
                    failed_round: None,
                    soundness_bits: args.freivalds_rounds as f64,
                })
            }
            Err(engine::EngineError::FreivaldsCheckFailed { round, lhs, rhs }) => {
                println!("❌ Freivalds failed at round {}", round);
                println!("   LHS: {:?}, RHS: {:?}", lhs, rhs);
                Some(FreivaldsResult {
                    seed: args.seed,
                    rounds: args.freivalds_rounds,
                    passed: false,
                    failed_round: Some(round),
                    soundness_bits: 0.0,
                })
            }
            Err(e) => return Err(e.into()),
        }
    };

    // Step 4: Compute 16×16 tail layer
    println!("🧮 Computing 16×16 tail layer...");
    let y2 = compute_layer_16x16(&w2, &y1, args.scale_num);

    // Step 5: Compute output commitments
    println!("🔒 Computing output commitments...");
    let h_y1 = commit_alpha_sum_vec(&y1.to_vec());
    let h_y = commit_alpha_sum_vec(&y2.to_vec());

    // Step 6: Save results
    println!("💾 Saving results to {}...", args.out.display());
    let w2_vec = w2.iter().map(|row| row.to_vec()).collect();
    let run_data = RunData {
        config: RunConfig {
            k: args.k,
            tile_k: args.tile_k,
            scale_num: args.scale_num,
            seed: args.seed,
            freivalds_rounds: args.freivalds_rounds,
            skip_freivalds: args.skip_freivalds,
            model_id: None, // Will be set during proving if needed
            vk_hash: None,  // Will be set during proving
        },
        x0,
        y1: y1.to_vec(),
        y2: y2.to_vec(),
        w2: w2_vec,
        commitments: Commitments {
            h_w1: field_to_string(h_w1),
            h_w2: field_to_string(h_w2),
            h_x: field_to_string(h_x),
            h_y1: field_to_string(h_y1),
            h_y: field_to_string(h_y),
        },
        freivalds_result: freivalds_result.clone(),
    };

    save_run_data(&args.out, &run_data)?;

    // Summary
    println!();
    println!("📊 Inference Summary:");
    println!("  • Large layer (16×{}): y1 = W1 · x0", args.k);
    println!("  • Tail layer (16×16): y2 = floor((W2 · y1) * {} / 2)", args.scale_num);
    if let Some(ref freivalds) = freivalds_result {
        if freivalds.passed {
            println!("  • Freivalds: ✅ PASSED ({} rounds, ~2^{:.0} soundness)",
                     freivalds.rounds, -freivalds.soundness_bits);
        } else {
            println!("  • Freivalds: ❌ FAILED at round {}", freivalds.failed_round.unwrap());
        }
    }
    println!("  • Commitments computed and saved");
    println!("  • Ready for proving with: nova_poc prove {}", args.out.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_demo_infer() {
        let temp_file = NamedTempFile::new().unwrap();
        let args = InferArgs {
            k: 256, // Small for testing
            tile_k: 128,
            scale_num: 3,
            seed: 42,
            freivalds_rounds: 16, // Minimum supported rounds
            weights1_path: None,
            weights2_path: None,
            x0_path: None,
            out: temp_file.path().to_path_buf(),
            skip_freivalds: true, // Skip for test speed
            preset: Some("demo".to_string()),
        };

        run_infer(args).unwrap();

        // Check that output file exists and is valid
        let run_data = load_run_data(temp_file.path()).unwrap();
        assert_eq!(run_data.config.k, 256);
        assert_eq!(run_data.x0.len(), 256);
        assert_eq!(run_data.y1.len(), 16);
        assert_eq!(run_data.y2.len(), 16);
        assert!(run_data.freivalds_result.is_some());
    }
}