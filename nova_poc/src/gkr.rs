use crate::cli::{ProveGkrArgs, VerifyGkrArgs};
use crate::formats::*;
use ark_serialize::CanonicalSerialize;
use anyhow::Result;
use std::fs;
use std::path::Path;
use zk_gkr::{Fr, Field};
use zk_gkr::transcript::FiatShamirTranscript;
use zk_gkr::merkle_poseidon::PoseidonMerkleTree;
use zk_gkr::mle::MleUtils;
use zk_gkr::sumcheck::{SumCheckProver, SumCheckVerifier};
use zk_gkr::proof::{GkrProof, GkrPublicInputs};

/// Run GKR proof generation
pub fn run_prove_gkr(args: ProveGkrArgs) -> Result<()> {
    println!("🔐 Starting GKR proof generation...");
    println!("   Matrix dimensions: {}×{}", args.m, args.k);

    // Create output directory
    fs::create_dir_all(&args.out_dir)?;

    // Load and convert data
    println!("📂 Loading data...");
    let w_matrix = load_weights_matrix(&args.weights1_path, args.m, args.k)?;
    let x_vector = load_x_vector(&args.x0_path, args.k)?;

    // Convert to field elements
    let w_fr = matrix_i16_to_fr(&w_matrix);
    let x_fr = vector_i16_to_fr(&x_vector);

    // Pad to powers of 2 and convert to hypercube order
    let a = (args.m as f64).log2().ceil() as usize;
    let b = (args.k as f64).log2().ceil() as usize;

    println!("   Padded dimensions: 2^{}×2^{} = {}×{}", a, b, 1 << a, 1 << b);

    let w_data = MleUtils::matrix_to_hypercube_order(&w_fr, args.m, args.k)?;
    let x_data = MleUtils::vector_to_hypercube_order(&x_fr, args.k)?;

    // Build Merkle trees
    println!("🌳 Building Merkle commitments...");
    let w_tree = PoseidonMerkleTree::build_tree(&w_data)?;
    let x_tree = PoseidonMerkleTree::build_tree(&x_data)?;

    let h_w = w_tree.root();
    let h_x = x_tree.root();

    println!("   h_W = 0x{}", field_to_hex(&h_w));
    println!("   h_X = 0x{}", field_to_hex(&h_x));

    // Create Fiat-Shamir transcript and derive challenge vector u
    println!("🎲 Deriving challenge vector...");
    let mut transcript = FiatShamirTranscript::new_seeded(
        &h_w,
        &h_x,
        args.m,
        args.k,
        args.model_id.as_deref(),
        args.vk_hash.as_deref(),
        &args.salt,
    )?;

    let u = transcript.derive_u_vector(1 << a)?;

    // Compute claimed value c = u^T * (W * x)
    println!("⚡ Computing claimed value...");
    let c = compute_claimed_value(&u, &w_fr, &x_fr, args.m, args.k)?;
    println!("   Claimed value c = 0x{}", field_to_hex(&c));

    // Create sum-check prover and generate proof
    println!("🔍 Generating sum-check proof...");
    let prover = SumCheckProver::new(u.clone(), w_data, x_data, a, b, w_tree, x_tree)?;

    // Reset transcript for proof generation
    let mut proof_transcript = FiatShamirTranscript::new_seeded(
        &h_w,
        &h_x,
        args.m,
        args.k,
        args.model_id.as_deref(),
        args.vk_hash.as_deref(),
        &args.salt,
    )?;

    // Skip u derivation since it's already done
    proof_transcript.derive_u_vector(1 << a)?;

    let sumcheck_proof = prover.prove(&mut proof_transcript)?;

    // Create complete GKR proof
    let gkr_proof = GkrProof::new(
        args.m,
        args.k,
        h_w,
        h_x,
        c,
        args.salt.clone(),
        sumcheck_proof,
    );

    // Save proof and public inputs
    let proof_path = args.out_dir.join("gkr_proof.bin");
    let public_path = args.out_dir.join("public.json");

    println!("💾 Saving files...");
    gkr_proof.save_to_file(&proof_path.to_string_lossy())?;

    let public_inputs = gkr_proof.public_inputs(args.model_id, args.vk_hash);
    public_inputs.save_to_file(&public_path.to_string_lossy())?;

    let proof_size = gkr_proof.size_bytes()?;

    println!("✅ GKR proof generation complete!");
    println!("   Proof file: {}", proof_path.display());
    println!("   Public inputs: {}", public_path.display());
    println!("   Proof size: {} bytes ({:.1} KB)", proof_size, proof_size as f64 / 1024.0);

    Ok(())
}

/// Run GKR proof verification
pub fn run_verify_gkr(args: VerifyGkrArgs) -> Result<()> {
    println!("🔍 Starting GKR proof verification...");

    let start_time = std::time::Instant::now();

    // Load proof and public inputs
    println!("📂 Loading proof files...");
    let proof = GkrProof::load_from_file(&args.proof_path.to_string_lossy())?;
    let public_inputs = GkrPublicInputs::load_from_file(&args.public_path.to_string_lossy())?;

    println!("   Matrix dimensions: {}×{}", public_inputs.m, public_inputs.k);
    println!("   Claimed value: 0x{}", field_to_hex(&public_inputs.c));

    // Verify consistency between proof and public inputs
    if proof.m != public_inputs.m || proof.k != public_inputs.k {
        anyhow::bail!("Dimension mismatch between proof and public inputs");
    }

    if proof.h_w != public_inputs.h_w || proof.h_x != public_inputs.h_x {
        anyhow::bail!("Merkle root mismatch between proof and public inputs");
    }

    if proof.c != public_inputs.c {
        anyhow::bail!("Claimed value mismatch between proof and public inputs");
    }

    // Create verification transcript
    println!("🎲 Recreating challenge vector...");
    let mut transcript = FiatShamirTranscript::new_seeded(
        &public_inputs.h_w,
        &public_inputs.h_x,
        public_inputs.m,
        public_inputs.k,
        public_inputs.model_id.as_deref(),
        public_inputs.vk_hash.as_deref(),
        &public_inputs.salt,
    )?;

    let a = (public_inputs.m as f64).log2().ceil() as usize;
    let u = transcript.derive_u_vector(1 << a)?;

    // Verify sum-check proof
    println!("🔍 Verifying sum-check proof...");
    let mut verify_transcript = FiatShamirTranscript::new_seeded(
        &public_inputs.h_w,
        &public_inputs.h_x,
        public_inputs.m,
        public_inputs.k,
        public_inputs.model_id.as_deref(),
        public_inputs.vk_hash.as_deref(),
        &public_inputs.salt,
    )?;

    // Skip u derivation to match prover state
    verify_transcript.derive_u_vector(1 << a)?;

    let verification_result = SumCheckVerifier::verify(
        &proof.sumcheck_proof,
        &u,
        &public_inputs.h_w,
        &public_inputs.h_x,
        proof.a,
        proof.b,
        &mut verify_transcript,
    )?;

    let verification_time = start_time.elapsed();

    if verification_result {
        println!("✅ GKR proof verification PASSED!");
        println!("   Verification time: {:.2} ms", verification_time.as_secs_f64() * 1000.0);

        if args.with_tail {
            println!("⚠️  Note: Groth16 tail verification not yet implemented");
        }
    } else {
        println!("❌ GKR proof verification FAILED!");
        anyhow::bail!("Verification failed");
    }

    Ok(())
}

/// Load weights matrix from binary file
fn load_weights_matrix(path: &Path, m: usize, k: usize) -> Result<Vec<Vec<i16>>> {
    let data = fs::read(path)?;
    let expected_size = m * k * 2; // 2 bytes per i16

    if data.len() != expected_size {
        anyhow::bail!(
            "Weight file size {} doesn't match expected size {} for {}×{} matrix",
            data.len(),
            expected_size,
            m,
            k
        );
    }

    let mut matrix = Vec::with_capacity(m);
    for i in 0..m {
        let mut row = Vec::with_capacity(k);
        for j in 0..k {
            let idx = (i * k + j) * 2;
            let value = i16::from_le_bytes([data[idx], data[idx + 1]]);
            row.push(value);
        }
        matrix.push(row);
    }

    Ok(matrix)
}

/// Load x vector from JSON file
fn load_x_vector(path: &Path, k: usize) -> Result<Vec<i16>> {
    let json_str = fs::read_to_string(path)?;
    let vector: Vec<i16> = serde_json::from_str(&json_str)?;

    if vector.len() != k {
        anyhow::bail!(
            "Vector length {} doesn't match expected length {}",
            vector.len(),
            k
        );
    }

    Ok(vector)
}

/// Convert i16 matrix to field elements
fn matrix_i16_to_fr(matrix: &[Vec<i16>]) -> Vec<Vec<Fr>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|&x| i16_to_fr(x)).collect())
        .collect()
}

/// Convert i16 vector to field elements
fn vector_i16_to_fr(vector: &[i16]) -> Vec<Fr> {
    vector.iter().map(|&x| i16_to_fr(x)).collect()
}

/// Convert i16 to field element (handling negative values)
fn i16_to_fr(x: i16) -> Fr {
    if x >= 0 {
        Fr::from(x as u64)
    } else {
        // For negative values, use field arithmetic
        Fr::zero() - Fr::from((-x) as u64)
    }
}

/// Compute claimed value c = u^T * (W * x)
fn compute_claimed_value(
    u: &[Fr],
    w_matrix: &[Vec<Fr>],
    x_vector: &[Fr],
    m: usize,
    k: usize,
) -> Result<Fr> {
    // Compute y = W * x
    let mut y = Vec::with_capacity(m);
    for i in 0..m {
        let mut sum = Fr::zero();
        for j in 0..k {
            if i < w_matrix.len() && j < w_matrix[i].len() && j < x_vector.len() {
                sum += w_matrix[i][j] * x_vector[j];
            }
        }
        y.push(sum);
    }

    // Compute c = u^T * y
    let mut c = Fr::zero();
    for (i, &y_i) in y.iter().enumerate() {
        if i < u.len() {
            c += u[i] * y_i;
        }
    }

    Ok(c)
}

/// Convert field element to hex string
fn field_to_hex(field: &Fr) -> String {
    let mut bytes = Vec::new();
    field.serialize_compressed(&mut bytes).unwrap();
    hex::encode(bytes)
}