use crate::cli::{ProveArgs, SetupArgs};
use crate::formats::*;
use circuit::{TinyTailCircuit, compressed};
use ark_bn254::{Bn254, Fr};
// use ark_ff::{Field, PrimeField};
use ark_groth16::{Groth16, ProvingKey};
use ark_snark::SNARK;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use rand::rngs::OsRng;
use std::fs::File;
use std::io::{BufWriter, BufReader, Write};
use anyhow::Result;

pub fn run_setup(args: SetupArgs) -> Result<()> {
    let pk_path = args.out_dir.join("pk.bin");
    let vk_path = args.out_dir.join("vk.bin");

    // Check if keys already exist
    if !args.force && pk_path.exists() && vk_path.exists() {
        println!("🔑 Keys already exist in {}", args.out_dir.display());
        println!("   Use --force to regenerate");
        return Ok(());
    }

    println!("🔧 Setting up proving and verification keys...");
    std::fs::create_dir_all(&args.out_dir)?;

    // Create a blank circuit for setup
    let blank_circuit = create_blank_circuit();

    println!("⚙️  Running circuit-specific setup...");
    let mut rng = OsRng;
    let start_time = std::time::Instant::now();
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(blank_circuit, &mut rng)?;
    let setup_time = start_time.elapsed();

    println!("💾 Saving keys to disk...");

    // Save proving key (uncompressed for performance)
    let pk_file = File::create(&pk_path)?;
    let mut pk_writer = BufWriter::new(pk_file);
    pk.serialize_uncompressed(&mut pk_writer)?;
    pk_writer.flush()?;

    // Save verification key (compressed)
    let vk_file = File::create(&vk_path)?;
    let mut vk_writer = BufWriter::new(vk_file);
    compressed::serialize_vk_compressed(&vk, &mut vk_writer)
        .map_err(|e| anyhow::anyhow!("Failed to serialize verification key: {}", e))?;
    vk_writer.flush()?;

    println!("✅ Key setup completed in {:.2}s", setup_time.as_secs_f64());
    println!("📁 Keys saved to:");
    println!("   • Proving key:      {}", pk_path.display());
    println!("   • Verification key: {}", vk_path.display());

    // Show key sizes
    if let Ok(pk_size) = std::fs::metadata(&pk_path).map(|m| m.len()) {
        if let Ok(vk_size) = std::fs::metadata(&vk_path).map(|m| m.len()) {
            println!("📊 Key sizes:");
            println!("   • Proving key:   {:.1} KB", pk_size as f64 / 1024.0);
            println!("   • Verification key: {} bytes", vk_size);
        }
    }

    Ok(())
}

pub fn run_prove(args: ProveArgs) -> Result<()> {
    println!("🔒 Starting proof generation...");

    // Load run data
    println!("📂 Loading run data from {}...", args.run_json.display());
    let run_data = load_run_data(&args.run_json)?;

    // Determine proving key path
    let pk_path = if let Some(path) = args.pk_path {
        path
    } else {
        // Auto-load from default location
        let default_path = std::path::PathBuf::from("keys").join("pk.bin");
        if !default_path.exists() {
            anyhow::bail!("No proving key found. Run 'nova_poc setup' first or specify --pk-path");
        }
        default_path
    };

    // Create output directory
    std::fs::create_dir_all(&args.out_dir)?;

    // Load proving key
    println!("🔑 Loading proving key from {}...", pk_path.display());
    let pk_file = File::open(&pk_path)?;
    let mut pk_reader = BufReader::new(pk_file);
    let pk = ProvingKey::<Bn254>::deserialize_uncompressed(&mut pk_reader)?;

    // Create circuit with witness data
    println!("🔧 Building circuit with witness data...");
    let circuit = build_circuit_from_run_data(&run_data)?;

    // Generate proof
    println!("⚡ Generating Groth16 proof...");
    let mut rng = OsRng;
    let start_time = std::time::Instant::now();
    let proof = Groth16::<Bn254>::prove(&pk, circuit, &mut rng)?;
    let prove_time = start_time.elapsed();

    // Prepare public inputs in the correct order
    let public_inputs = build_public_inputs(&run_data)?;

    // Save proof (compressed)
    let proof_path = args.out_dir.join("proof.bin");
    let proof_file = File::create(&proof_path)?;
    let mut proof_writer = BufWriter::new(proof_file);
    compressed::serialize_proof_compressed(&proof, &mut proof_writer)
        .map_err(|e| anyhow::anyhow!("Failed to serialize proof: {}", e))?;
    proof_writer.flush()?;

    // Save public inputs
    let inputs_path = args.out_dir.join("public_inputs.json");
    save_public_inputs(&inputs_path, &public_inputs)?;

    // Copy verification key to output directory for convenience
    let vk_src = pk_path.parent().unwrap_or_else(|| std::path::Path::new(".")).join("vk.bin");
    let vk_dst = args.out_dir.join("vk.bin");
    if vk_src.exists() {
        std::fs::copy(&vk_src, &vk_dst)?;
    }

    println!("✅ Proof generation completed in {:.2}s", prove_time.as_secs_f64());
    println!("📁 Proof files saved to:");
    println!("   • Proof:        {}", proof_path.display());
    println!("   • Public inputs: {}", inputs_path.display());
    println!("   • Verification key: {}", vk_dst.display());

    // Show file sizes
    if let Ok(proof_size) = std::fs::metadata(&proof_path).map(|m| m.len()) {
        if let Ok(inputs_size) = std::fs::metadata(&inputs_path).map(|m| m.len()) {
            let total_tx_size = proof_size + inputs_size;
            println!("📊 Transaction size:");
            println!("   • Proof:        {} bytes", proof_size);
            println!("   • Public inputs: {} bytes", inputs_size);
            println!("   • Total TX:      {} bytes", total_tx_size);

            if total_tx_size < 1024 {
                println!("✅ Total transaction size < 1 KB target!");
            } else {
                println!("⚠️  Total transaction size exceeds 1 KB target");
            }
        }
    }

    println!("🚀 Ready for verification with: nova_poc verify {}", args.run_json.display());

    Ok(())
}

/// Create a blank circuit for key generation
fn create_blank_circuit() -> TinyTailCircuit {
    let zero = Fr::from(0u64);
    TinyTailCircuit {
        w2: [[zero; 16]; 16],
        y1: [zero; 16],
        y2: [zero; 16],
        scale_num: Fr::from(2u64), // Use a valid scale_num for setup
        h_w2: zero,
        h_x: zero,
        h_y1: zero,
        h_y: zero,
        div_quotients: [zero; 16],
        div_remainders: [zero; 16],
    }
}

/// Build circuit with actual witness data from run
fn build_circuit_from_run_data(run_data: &RunData) -> Result<TinyTailCircuit> {
    // Use the actual W2 matrix from the run data
    let mut w2 = [[Fr::from(0u64); 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let val = run_data.w2[i][j];
            w2[i][j] = if val >= 0 {
                Fr::from(val as u64)
            } else {
                -Fr::from((-val) as u64)
            };
        }
    }

    // Convert vectors to field elements
    let y1: Vec<Fr> = run_data.y1.iter().map(|&x| {
        if x >= 0 {
            Fr::from(x as u64)
        } else {
            -Fr::from((-x) as u64)
        }
    }).collect();

    let y2: Vec<Fr> = run_data.y2.iter().map(|&x| {
        if x >= 0 {
            Fr::from(x as u64)
        } else {
            -Fr::from((-x) as u64)
        }
    }).collect();

    // Compute division witnesses for floor operation
    let scale_num = Fr::from(run_data.config.scale_num);
    let mut div_quotients = [Fr::from(0u64); 16];
    let mut div_remainders = [Fr::from(0u64); 16];

    for i in 0..16 {
        // For each output y2[i], we need: (W2·y1)[i] * scale_num = 2 * y2[i] + remainder
        // Compute (W2·y1)[i] = Σ W2[i][j] * y1[j]
        let mut w2_y1_i = Fr::from(0u64);
        for j in 0..16 {
            w2_y1_i += w2[i][j] * y1[j];
        }

        // lhs = (W2·y1)[i] * scale_num
        let lhs = w2_y1_i * scale_num;
        let two_times_quotient = Fr::from(2u64) * y2[i];

        // remainder = lhs - 2 * quotient
        let remainder = lhs - two_times_quotient;

        div_quotients[i] = y2[i];
        div_remainders[i] = remainder;
    }

    // Convert commitments from strings to field elements
    let h_w2 = string_to_field(&run_data.commitments.h_w2)?;
    let h_x = string_to_field(&run_data.commitments.h_x)?;
    let h_y1 = string_to_field(&run_data.commitments.h_y1)?;
    let h_y = string_to_field(&run_data.commitments.h_y)?;

    let circuit = TinyTailCircuit {
        w2,
        y1: y1.try_into().map_err(|_| anyhow::anyhow!("y1 length != 16"))?,
        y2: y2.try_into().map_err(|_| anyhow::anyhow!("y2 length != 16"))?,
        scale_num,
        h_w2,
        h_x,
        h_y1,
        h_y,
        div_quotients,
        div_remainders,
    };

    Ok(circuit)
}

/// Build public inputs in the exact order expected by the circuit
fn build_public_inputs(run_data: &RunData) -> Result<Vec<Fr>> {
    // Order: h_w2, h_x, h_y1, h_y, scale_num
    let mut inputs = Vec::new();

    inputs.push(string_to_field(&run_data.commitments.h_w2)?);
    inputs.push(string_to_field(&run_data.commitments.h_x)?);
    inputs.push(string_to_field(&run_data.commitments.h_y1)?);
    inputs.push(string_to_field(&run_data.commitments.h_y)?);
    inputs.push(Fr::from(run_data.config.scale_num));

    Ok(inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_blank_circuit_setup() {
        let circuit = create_blank_circuit();
        let mut rng = OsRng;

        // Should be able to generate keys without error
        let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(circuit, &mut rng).unwrap();

        // Keys should be non-trivial
        assert!(!pk.vk.gamma_g2.is_zero());
        assert!(!vk.gamma_g2.is_zero());
    }

    #[test]
    fn test_key_serialization() {
        let temp_dir = TempDir::new().unwrap();
        let args = SetupArgs {
            out_dir: temp_dir.path().to_path_buf(),
            force: true,
        };

        run_setup(args).unwrap();

        // Check that files exist
        assert!(temp_dir.path().join("pk.bin").exists());
        assert!(temp_dir.path().join("vk.bin").exists());

        // Check that we can load them back
        let pk_file = File::open(temp_dir.path().join("pk.bin")).unwrap();
        let mut pk_reader = BufReader::new(pk_file);
        let _pk = ProvingKey::<Bn254>::deserialize_uncompressed(&mut pk_reader).unwrap();

        let vk_file = File::open(temp_dir.path().join("vk.bin")).unwrap();
        let mut vk_reader = BufReader::new(vk_file);
        let _vk = compressed::deserialize_vk_compressed(&mut vk_reader)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize verification key: {}", e)).unwrap();
    }
}