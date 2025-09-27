//! Matrix Multiplication Proof Demo using Expander SDK
//!
//! This demonstrates high-performance matrix multiplication proofs:
//! - Generate proof that y = W·x where W is private
//! - Verify proof without knowing W
//! - Compare performance against standard implementation

use expander_matrix_poc::*;
use rand::{thread_rng, Rng};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("🚀 Expander SDK Matrix Multiplication Proof Demo");
    println!("================================================\n");

    // Test different matrix sizes to show performance scaling
    let test_sizes = vec![
        (4, 8),      // Small test
        (16, 64),    // Medium test
        (32, 256),   // Larger test
        (64, 512),   // Performance test
    ];

    for (m, k) in test_sizes {
        println!("📐 Testing {}×{} matrix multiplication proof", m, k);
        run_matrix_proof_test(m, k)?;
        println!();
    }

    println!("✅ All tests completed successfully!");
    Ok(())
}

fn run_matrix_proof_test(m: usize, k: usize) -> anyhow::Result<()> {
    let mut rng = thread_rng();

    // Generate test data
    println!("  🎲 Generating random test data...");
    let weights = Matrix::random(m, k, &mut rng);
    let input = Vector::random(k, &mut rng);
    let output = weights.multiply(&input);

    // Create proof system
    println!("  ⚙️  Initializing Expander proof system...");
    let mut system = MatrixProofSystem::new(m, k)?;

    // Generate proof
    println!("  🔐 Generating matrix multiplication proof...");
    let prove_start = Instant::now();
    let proof = system.prove(&weights, &input, &output)?;
    let prove_time = prove_start.elapsed();

    println!("    ✅ Proof generated in {:.2}ms", prove_time.as_secs_f64() * 1000.0);
    println!("    📦 Proof size: {} bytes ({:.2} KB)", proof.proof_size_bytes, proof.proof_size_bytes as f64 / 1024.0);

    // Verify proof
    println!("  🔍 Verifying proof...");
    let verify_start = Instant::now();
    let verified = system.verify(&proof, &input, &output)?;
    let verify_time = verify_start.elapsed();

    if verified {
        println!("    ✅ Proof verified successfully in {:.2}ms", verify_time.as_secs_f64() * 1000.0);
    } else {
        println!("    ❌ Proof verification failed!");
        return Err(anyhow::anyhow!("Verification failed"));
    }

    // Test security: try to verify with wrong output
    println!("  🛡️  Testing security with wrong output...");
    let wrong_output = Vector::random(m, &mut rng);
    let security_verified = system.verify(&proof, &input, &wrong_output)?;

    if !security_verified {
        println!("    ✅ Security test passed: wrong output correctly rejected");
    } else {
        println!("    ❌ Security test failed: wrong output accepted!");
        return Err(anyhow::anyhow!("Security test failed"));
    }

    // Performance summary
    let matrix_ops = m * k; // Number of multiplications in W·x
    let prove_throughput = matrix_ops as f64 / prove_time.as_secs_f64();
    let verify_throughput = matrix_ops as f64 / verify_time.as_secs_f64();

    println!("  📊 Performance Summary:");
    println!("    Matrix operations: {} ({}×{})", matrix_ops, m, k);
    println!("    Proving throughput: {:.0} ops/sec", prove_throughput);
    println!("    Verification throughput: {:.0} ops/sec", verify_throughput);
    println!("    Proof overhead: {:.1}× vs direct computation",
        prove_time.as_secs_f64() / compute_baseline_time(m, k).as_secs_f64());

    Ok(())
}

/// Compute baseline matrix multiplication time for comparison
fn compute_baseline_time(m: usize, k: usize) -> std::time::Duration {
    let mut rng = thread_rng();
    let weights = Matrix::random(m, k, &mut rng);
    let input = Vector::random(k, &mut rng);

    let start = Instant::now();
    let _output = weights.multiply(&input);
    start.elapsed()
}

#[cfg(test)]
mod demo_tests {
    use super::*;

    #[test]
    fn test_demo_functionality() {
        // Run a small test to ensure demo works
        run_matrix_proof_test(4, 8).expect("Demo test should succeed");
    }

    #[test]
    fn test_performance_scaling() {
        let sizes = vec![(2, 4), (4, 8), (8, 16)];
        let mut prev_time = std::time::Duration::from_nanos(0);

        for (m, k) in sizes {
            let baseline_time = compute_baseline_time(m, k);

            // Ensure time scales reasonably with problem size
            if prev_time > std::time::Duration::from_nanos(0) {
                let growth_factor = baseline_time.as_nanos() as f64 / prev_time.as_nanos() as f64;
                assert!(growth_factor > 1.0, "Time should increase with problem size");
                assert!(growth_factor < 100.0, "Time growth should be reasonable");
            }

            prev_time = baseline_time;
        }
    }
}