//! zkCUDA Matrix Multiplication Demo
//!
//! Demonstrates GPU-accelerated zero-knowledge proofs for matrix multiplication
//! using Polyhedra's zkCUDA framework.

use cuda_poc::*;

fn main() -> anyhow::Result<()> {
    println!("🚀 zkCUDA Matrix Multiplication Proof Demo");
    println!("==========================================\n");

    // Create proof system
    println!("⚙️  Initializing zkCUDA proof system...");
    let system = MatrixProofSystem::new()?;

    // Create test matrices
    println!("📐 Creating 64×32 and 32×64 test matrices...");
    let mut mat_a: Vec<Vec<M31>> = vec![];
    for i in 0..64 {
        mat_a.push(vec![]);
        for j in 0..32 {
            mat_a[i].push(M31::from((i * 233 + j + 1) as u32));
        }
    }

    let mut mat_b: Vec<Vec<M31>> = vec![];
    for i in 0..32 {
        mat_b.push(vec![]);
        for j in 0..64 {
            mat_b[i].push(M31::from((i * 2333 + j + 1) as u32));
        }
    }

    // Calculate expected result
    let mut expected_result = M31::zero();
    for i in 0..64 {
        for j in 0..64 {
            for k in 0..32 {
                expected_result += mat_a[i][k] * mat_b[k][j];
            }
        }
    }
    println!("  Expected result: {:?}", expected_result);

    // Generate proof
    println!("\n🔐 Generating zero-knowledge proof on GPU...");
    let start = std::time::Instant::now();
    let result = system.prove(&mat_a, &mat_b)?;
    let prove_time = start.elapsed();

    println!("  ✅ Proof generated in {:.2?}", prove_time);
    println!("  🔢 Computed result: {:?}", result);

    // Verify proof
    println!("\n🔍 Verifying proof...");
    let start = std::time::Instant::now();
    let verified = system.verify(&mat_a, &mat_b, expected_result)?;
    let verify_time = start.elapsed();

    if verified {
        println!("  ✅ Proof verified successfully in {:.2?}", verify_time);
    } else {
        println!("  ❌ Proof verification failed!");
        return Err(anyhow::anyhow!("Verification failed"));
    }

    // Summary
    println!("\n📊 Performance Summary:");
    println!("  Matrix size: 64×32 × 32×64");
    println!("  Total operations: {} multiplications", 64 * 64 * 32);
    println!("  Proving time: {:.2?}", prove_time);
    println!("  Verification time: {:.2?}", verify_time);
    println!("  Speedup: {:.1}× faster verification", prove_time.as_secs_f64() / verify_time.as_secs_f64());

    println!("\n✨ Demo completed successfully!");

    Ok(())
}
