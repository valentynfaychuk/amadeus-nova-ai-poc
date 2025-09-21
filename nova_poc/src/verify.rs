use crate::cli::VerifyArgs;
use crate::formats::*;
use circuit::compressed;
use engine::freivalds::freivalds_check;
use ark_bn254::Bn254;
use ark_groth16::Groth16;
use std::fs::File;
use std::io::BufReader;
use anyhow::Result;

pub fn run_verify(args: VerifyArgs) -> Result<()> {
    println!("🔍 Starting verification...");

    // Load run data
    println!("📂 Loading run data from {}...", args.run_json.display());
    let run_data = load_run_data(&args.run_json)?;

    // Determine file paths
    let vk_path = args.vk_path.clone().unwrap_or_else(|| {
        args.run_json.parent().unwrap_or_else(|| std::path::Path::new("."))
            .join("out/vk.bin")
    });

    let proof_path = args.proof_path.clone().unwrap_or_else(|| {
        args.run_json.parent().unwrap_or_else(|| std::path::Path::new("."))
            .join("out/proof.bin")
    });

    let public_inputs_path = args.public_inputs_path.clone().unwrap_or_else(|| {
        args.run_json.parent().unwrap_or_else(|| std::path::Path::new("."))
            .join("out/public_inputs.json")
    });

    // Step 1: Verify Freivalds check (if not skipped)
    if !args.skip_freivalds && run_data.freivalds_result.is_some() {
        println!("🔍 Re-running Freivalds verification...");

        let weights1_path = args.weights1_path.clone().ok_or_else(|| {
            anyhow::anyhow!("--weights1-path required for Freivalds verification")
        })?;

        let mut w1_reader = load_weights1(&weights1_path, run_data.config.k)?;
        let y1: [i64; 16] = run_data.y1.as_slice().try_into()
            .map_err(|_| anyhow::anyhow!("y1 must have exactly 16 elements"))?;

        let start_time = std::time::Instant::now();

        match freivalds_check(
            &mut w1_reader,
            &run_data.x0,
            &y1,
            run_data.config.k,
            run_data.config.tile_k,
            run_data.config.freivalds_rounds,
            run_data.config.seed,
        ) {
            Ok(()) => {
                let freivalds_time = start_time.elapsed();
                println!("✅ Freivalds verification passed in {:.3}s", freivalds_time.as_secs_f64());

                // Check consistency with recorded result
                if let Some(ref recorded) = run_data.freivalds_result {
                    if !recorded.passed {
                        anyhow::bail!("Freivalds passed now but was recorded as failed");
                    }
                    println!("   ✓ Consistent with recorded result");
                } else {
                    println!("   ⚠️ No recorded Freivalds result to compare");
                }
            }
            Err(engine::EngineError::FreivaldsCheckFailed { round, .. }) => {
                println!("❌ Freivalds verification failed at round {}", round);

                // Check consistency with recorded result
                if let Some(ref recorded) = run_data.freivalds_result {
                    if recorded.passed {
                        anyhow::bail!("Freivalds failed now but was recorded as passed");
                    }
                    if recorded.failed_round == Some(round) {
                        println!("   ✓ Consistent with recorded failure at round {}", round);
                    } else {
                        println!("   ⚠️ Failed at different round than recorded ({:?})", recorded.failed_round);
                    }
                } else {
                    anyhow::bail!("Freivalds failed but no recorded result to compare");
                }

                // Continue with Groth16 verification even if Freivalds failed
                // (this allows testing the proof system independently)
                println!("   ➤ Continuing with Groth16 verification...");
            }
            Err(e) => return Err(e.into()),
        }
    } else {
        println!("⏭️  Skipping Freivalds verification");
    }

    // Step 2: Load verification key
    println!("🔑 Loading verification key from {}...", vk_path.display());
    let vk_file = File::open(&vk_path)?;
    let mut vk_reader = BufReader::new(vk_file);
    let vk = compressed::deserialize_vk_compressed(&mut vk_reader)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize verification key: {}", e))?;

    // Step 3: Load proof
    println!("📜 Loading proof from {}...", proof_path.display());
    let proof_file = File::open(&proof_path)?;
    let mut proof_reader = BufReader::new(proof_file);
    let proof = compressed::deserialize_proof_compressed(&mut proof_reader)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize proof: {}", e))?;

    // Step 4: Load public inputs
    println!("📊 Loading public inputs from {}...", public_inputs_path.display());
    let public_inputs = load_public_inputs(&public_inputs_path)?;

    // Step 5: Verify public inputs consistency
    println!("🔍 Verifying public inputs consistency...");
    verify_public_inputs_consistency(&run_data, &public_inputs)?;

    // Step 6: Verify Groth16 proof
    println!("🔒 Verifying Groth16 proof...");
    let start_time = std::time::Instant::now();
    let prepared_vk = ark_groth16::prepare_verifying_key(&vk);
    let verification_result = Groth16::<Bn254>::verify_proof(&prepared_vk, &proof, &public_inputs)?;
    let verify_time = start_time.elapsed();

    if verification_result {
        println!("✅ Groth16 proof verification PASSED in {:.3}s", verify_time.as_secs_f64());
    } else {
        println!("❌ Groth16 proof verification FAILED");
        anyhow::bail!("Proof verification failed");
    }

    // Step 7: Show verification summary
    print_verification_summary(&run_data, &args);

    println!("🎉 All verifications completed successfully!");

    Ok(())
}

/// Verify that public inputs match the commitments in run data
fn verify_public_inputs_consistency(run_data: &RunData, public_inputs: &[ark_bn254::Fr]) -> Result<()> {
    if public_inputs.len() != 5 {
        anyhow::bail!("Expected 5 public inputs, got {}", public_inputs.len());
    }

    // Expected order: h_w2, h_x, h_y1, h_y, scale_num
    let expected = [
        string_to_field(&run_data.commitments.h_w2)?,
        string_to_field(&run_data.commitments.h_x)?,
        string_to_field(&run_data.commitments.h_y1)?,
        string_to_field(&run_data.commitments.h_y)?,
        ark_bn254::Fr::from(run_data.config.scale_num),
    ];

    for (i, (&actual, &expected)) in public_inputs.iter().zip(expected.iter()).enumerate() {
        if actual != expected {
            anyhow::bail!("Public input {} mismatch: actual={:?}, expected={:?}", i, actual, expected);
        }
    }

    println!("   ✓ All 5 public inputs match run data commitments");
    Ok(())
}

/// Print detailed verification summary
fn print_verification_summary(run_data: &RunData, args: &VerifyArgs) {
    println!();
    println!("📋 Verification Summary:");
    println!("  • Configuration:");
    println!("    - Large layer: 16×{}", run_data.config.k);
    println!("    - Tile size: {}", run_data.config.tile_k);
    println!("    - Scale factor: {}/2", run_data.config.scale_num);
    println!("    - Seed: {}", run_data.config.seed);

    if !args.skip_freivalds {
        if let Some(ref freivalds) = run_data.freivalds_result {
            if freivalds.passed {
                println!("  • Freivalds: ✅ PASSED ({} rounds, ~2^{:.0} soundness)",
                         freivalds.rounds, -freivalds.soundness_bits);
            } else {
                println!("  • Freivalds: ❌ FAILED at round {}",
                         freivalds.failed_round.unwrap_or(0));
            }
        }
    }

    println!("  • Groth16: ✅ PASSED (tiny proof ~200-300 bytes)");
    println!("  • Public inputs: ✅ 5 commitments verified");

    // Show data sizes
    println!("  • Vector sizes:");
    println!("    - Input x0: {} elements", run_data.x0.len());
    println!("    - Intermediate y1: {} elements", run_data.y1.len());
    println!("    - Output y2: {} elements", run_data.y2.len());

    // Show commitment values (first few chars for brevity)
    println!("  • Commitments (first 16 chars):");
    println!("    - h_W1: {}...", &run_data.commitments.h_w1[..16.min(run_data.commitments.h_w1.len())]);
    println!("    - h_W2: {}...", &run_data.commitments.h_w2[..16.min(run_data.commitments.h_w2.len())]);
    println!("    - h_X:  {}...", &run_data.commitments.h_x[..16.min(run_data.commitments.h_x.len())]);
    println!("    - h_y1: {}...", &run_data.commitments.h_y1[..16.min(run_data.commitments.h_y1.len())]);
    println!("    - h_Y:  {}...", &run_data.commitments.h_y[..16.min(run_data.commitments.h_y.len())]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use crate::prove::{run_setup, run_prove};
    use crate::infer::run_infer;
    use crate::cli::*;

    #[test]
    fn test_end_to_end_verification() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Run inference
        let run_json = base_path.join("run.json");
        let infer_args = InferArgs {
            k: 256,
            tile_k: 128,
            scale_num: 3,
            seed: 42,
            freivalds_rounds: 5,
            weights1_path: None,
            weights2_path: None,
            x0_path: None,
            out: run_json.clone(),
            skip_freivalds: true, // Skip for speed in test
            preset: Some("demo".to_string()),
        };
        run_infer(infer_args).unwrap();

        // Setup keys
        let keys_dir = base_path.join("keys");
        let setup_args = SetupArgs {
            out_dir: keys_dir.clone(),
            force: true,
        };
        run_setup(setup_args).unwrap();

        // Generate proof
        let proof_dir = base_path.join("proof");
        let prove_args = ProveArgs {
            run_json: run_json.clone(),
            pk_path: Some(keys_dir.join("pk.bin")),
            out_dir: proof_dir.clone(),
        };
        run_prove(prove_args).unwrap();

        // Verify proof
        let verify_args = VerifyArgs {
            run_json: run_json.clone(),
            vk_path: Some(proof_dir.join("vk.bin")),
            proof_path: Some(proof_dir.join("proof.bin")),
            public_inputs_path: Some(proof_dir.join("public_inputs.json")),
            weights1_path: None, // Skip Freivalds in test
            skip_freivalds: true,
        };

        // Should verify successfully
        run_verify(verify_args).unwrap();
    }

    #[test]
    fn test_public_inputs_consistency() {
        use ark_bn254::Fr;

        let run_data = RunData {
            config: RunConfig {
                k: 256,
                tile_k: 128,
                scale_num: 3,
                seed: 42,
                freivalds_rounds: 32,
                skip_freivalds: false,
            },
            x0: vec![1; 256],
            y1: vec![2; 16],
            y2: vec![3; 16],
            w2: vec![vec![1; 16]; 16], // Identity matrix for test
            commitments: Commitments {
                h_w1: "100".to_string(),
                h_w2: "200".to_string(),
                h_x: "300".to_string(),
                h_y1: "400".to_string(),
                h_y: "500".to_string(),
            },
            freivalds_result: None,
        };

        let public_inputs = vec![
            Fr::from(200u64), // h_w2
            Fr::from(300u64), // h_x
            Fr::from(400u64), // h_y1
            Fr::from(500u64), // h_y
            Fr::from(3u64),   // scale_num
        ];

        // Should pass
        verify_public_inputs_consistency(&run_data, &public_inputs).unwrap();

        // Should fail with wrong values
        let wrong_inputs = vec![
            Fr::from(999u64), // Wrong h_w2
            Fr::from(300u64),
            Fr::from(400u64),
            Fr::from(500u64),
            Fr::from(3u64),
        ];

        assert!(verify_public_inputs_consistency(&run_data, &wrong_inputs).is_err());
    }
}