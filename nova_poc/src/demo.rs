use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
// use std::io::Write;
use anyhow::Result;

/// Generate demo data for testing: random W1 (16×K), W2 (16×16), and x0 (K)
pub fn generate_demo_data(k: usize, seed: u64) -> Result<(Vec<u8>, [[i64; 16]; 16], Vec<i64>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate random input vector x0 (small integers)
    let x0: Vec<i64> = (0..k).map(|_| rng.gen_range(-10..=10)).collect();

    // Generate random W1 weights (16×K, small integers)
    let mut w1_data = Vec::new();
    for _tile_start in (0..k).step_by(1024) {
        for _i in 0..16 {
            for _j in 0..std::cmp::min(1024, k) {
                let weight = rng.gen_range(-127..=127) as i16;
                w1_data.extend_from_slice(&weight.to_le_bytes());
            }
        }
    }

    // Generate random W2 weights (16×16, small integers)
    let mut w2 = [[0i64; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            w2[i][j] = rng.gen_range(-10..=10);
        }
    }

    Ok((w1_data, w2, x0))
}

/// Run demo end-to-end pipeline: infer → prove → verify
pub fn run_demo(seed: u64) -> Result<()> {
    use crate::infer;
    use crate::prove;
    use crate::verify;
    use crate::cli::*;
    // use std::path::PathBuf;
    use tempfile::TempDir;

    println!("🚀 Running Nova POC demo with seed={}", seed);

    // Create temporary directory
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    // Generate demo data
    let (w1_data, w2, x0) = generate_demo_data(4096, seed)?;

    // Save demo data to temp files
    let w1_path = temp_path.join("w1.bin");
    let w2_path = temp_path.join("w2.json");
    let x0_path = temp_path.join("x0.json");
    let run_path = temp_path.join("run.json");

    std::fs::write(&w1_path, w1_data)?;
    std::fs::write(&w2_path, serde_json::to_string_pretty(&w2)?)?;
    std::fs::write(&x0_path, serde_json::to_string_pretty(&x0)?)?;

    let start_time = std::time::Instant::now();

    // Step 1: Infer
    println!("📊 Step 1: Running inference with tiled GEMV...");
    let infer_args = InferArgs {
        k: 4096,
        tile_k: 1024,
        scale_num: 3,
        seed,
        freivalds_rounds: 32,
        weights1_path: Some(w1_path.clone()),
        weights2_path: Some(w2_path),
        x0_path: Some(x0_path),
        out: run_path.clone(),
        skip_freivalds: false,
        preset: None,
    };

    infer::run_infer(infer_args)?;
    let infer_time = start_time.elapsed();
    println!("✅ Inference completed in {:.2}s", infer_time.as_secs_f64());

    // Step 2: Setup keys
    let keys_dir = temp_path.join("keys");
    std::fs::create_dir_all(&keys_dir)?;

    println!("🔑 Step 2: Setting up proving keys...");
    let setup_args = SetupArgs {
        out_dir: keys_dir.clone(),
        force: true,
    };

    prove::run_setup(setup_args)?;
    let setup_time = start_time.elapsed() - infer_time;
    println!("✅ Key setup completed in {:.2}s", setup_time.as_secs_f64());

    // Step 3: Prove
    println!("🔒 Step 3: Generating tiny Groth16 proof...");
    let prove_args = ProveArgs {
        run_json: run_path.clone(),
        pk_path: Some(keys_dir.join("pk.bin")),
        out_dir: temp_path.join("proof_out"),
    };

    prove::run_prove(prove_args)?;
    let prove_time = start_time.elapsed() - infer_time - setup_time;
    println!("✅ Proof generated in {:.2}s", prove_time.as_secs_f64());

    // Step 4: Verify
    println!("🔍 Step 4: Verifying proof and Freivalds check...");
    let verify_args = VerifyArgs {
        run_json: run_path,
        vk_path: Some(keys_dir.join("vk.bin")),
        proof_path: Some(temp_path.join("proof_out/proof.bin")),
        public_inputs_path: Some(temp_path.join("proof_out/public_inputs.json")),
        weights1_path: Some(w1_path),
        skip_freivalds: false,
        no_bind_randomness: false,
        allow_low_k: true, // Demo might use small k
        block_entropy: None,
        tile_k: None,
    };

    verify::run_verify(verify_args)?;
    let verify_time = start_time.elapsed() - infer_time - setup_time - prove_time;
    println!("✅ Verification completed in {:.2}s", verify_time.as_secs_f64());

    let total_time = start_time.elapsed();
    println!();
    println!("🎉 Demo completed successfully!");
    println!("📈 Timing breakdown:");
    println!("  • Inference: {:.2}s", infer_time.as_secs_f64());
    println!("  • Key setup: {:.2}s", setup_time.as_secs_f64());
    println!("  • Proving:   {:.2}s", prove_time.as_secs_f64());
    println!("  • Verify:    {:.2}s", verify_time.as_secs_f64());
    println!("  • Total:     {:.2}s", total_time.as_secs_f64());

    // Check proof size
    let proof_path = temp_path.join("proof_out/proof.bin");
    let public_inputs_path = temp_path.join("proof_out/public_inputs.json");

    if let Ok(proof_size) = std::fs::metadata(&proof_path).map(|m| m.len()) {
        if let Ok(inputs_size) = std::fs::metadata(&public_inputs_path).map(|m| m.len()) {
            let total_tx_size = proof_size + inputs_size;
            println!();
            println!("📦 Transaction size:");
            println!("  • Proof:        {} bytes", proof_size);
            println!("  • Public inputs: {} bytes", inputs_size);
            println!("  • Total TX:      {} bytes", total_tx_size);

            if total_tx_size < 1024 {
                println!("✅ Total transaction size < 1 KB target!");
            } else {
                println!("⚠️  Total transaction size exceeds 1 KB target");
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_demo_data() {
        let (w1_data, w2, x0) = generate_demo_data(1024, 42).unwrap();

        // Check sizes
        assert_eq!(x0.len(), 1024);
        assert_eq!(w1_data.len(), 16 * 1024 * 2); // 16 rows * 1024 cols * 2 bytes per i16

        // Check W2 dimensions
        assert_eq!(w2.len(), 16);
        for row in &w2 {
            assert_eq!(row.len(), 16);
        }

        // Check reproducibility
        let (w1_data2, _w2_2, x0_2) = generate_demo_data(1024, 42).unwrap();
        assert_eq!(w1_data, w1_data2);
        assert_eq!(x0, x0_2);
    }
}