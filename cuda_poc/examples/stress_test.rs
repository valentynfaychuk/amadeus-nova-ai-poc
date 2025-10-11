//! Stress Test - Multiple iterations to test performance and GPU utilization
//!
//! Runs many iterations of the 64×32 × 32×64 matrix multiplication
//! to stress test the system and observe GPU behavior.

use cuda_poc::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("🚀 zkCUDA Matrix Multiplication Stress Test");
    println!("===========================================\n");

    let iterations = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    println!("Configuration:");
    println!("  - Matrix size: 64×32 × 32×64");
    println!("  - Iterations: {}", iterations);
    println!("  - Total operations per iter: 131,072 multiplications\n");

    // Create proof system
    println!("⚙️  Initializing zkCUDA proof system...");
    let system = MatrixProofSystem::new()?;
    println!("✅ System initialized\n");

    // Create test matrices once
    println!("📐 Creating test matrices...");
    let mut mat_a: Vec<Vec<M31>> = vec![];
    for i in 0..64 {
        mat_a.push(vec![]);
        for j in 0..32 {
            mat_a[i].push(M31::from(((i * 233 + j + 1) % 2147483647) as u32));
        }
    }

    let mut mat_b: Vec<Vec<M31>> = vec![];
    for i in 0..32 {
        mat_b.push(vec![]);
        for j in 0..64 {
            mat_b[i].push(M31::from(((i * 2333 + j + 1) % 2147483647) as u32));
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
    println!("✅ Test data prepared\n");

    // Warmup
    println!("🔥 Warming up (5 iterations)...");
    for _ in 0..5 {
        let _result = system.prove(&mat_a, &mat_b)?;
    }
    println!("✅ Warmup complete\n");

    // Run stress test
    println!("🔐 Running stress test...");
    let mut prove_times = Vec::new();
    let mut verify_times = Vec::new();

    let total_start = Instant::now();

    for i in 0..iterations {
        let prove_start = Instant::now();
        let result = system.prove(&mat_a, &mat_b)?;
        let prove_time = prove_start.elapsed();
        prove_times.push(prove_time);

        let verify_start = Instant::now();
        let verified = system.verify(&mat_a, &mat_b, expected_result)?;
        let verify_time = verify_start.elapsed();
        verify_times.push(verify_time);

        if !verified || result != expected_result {
            println!("❌ Iteration {} failed verification!", i);
            return Err(anyhow::anyhow!("Verification failed at iteration {}", i));
        }

        if (i + 1) % 10 == 0 {
            println!("  Progress: {}/{} iterations", i + 1, iterations);
        }
    }

    let total_time = total_start.elapsed();

    // Calculate statistics
    let avg_prove = prove_times.iter().sum::<std::time::Duration>() / prove_times.len() as u32;
    let avg_verify = verify_times.iter().sum::<std::time::Duration>() / verify_times.len() as u32;
    let min_prove = prove_times.iter().min().unwrap();
    let max_prove = prove_times.iter().max().unwrap();
    let min_verify = verify_times.iter().min().unwrap();
    let max_verify = verify_times.iter().max().unwrap();

    let total_ops = iterations as u64 * 131_072;
    let throughput = total_ops as f64 / total_time.as_secs_f64();

    println!("\n✅ Stress test completed successfully!\n");
    println!("📊 Performance Statistics:");
    println!("  Total time: {:.2?}", total_time);
    println!("  Total operations: {} million", total_ops / 1_000_000);
    println!("  Overall throughput: {:.0} ops/sec", throughput);
    println!();
    println!("  Proving times:");
    println!("    - Average: {:.2?}", avg_prove);
    println!("    - Min: {:.2?}", min_prove);
    println!("    - Max: {:.2?}", max_prove);
    println!();
    println!("  Verification times:");
    println!("    - Average: {:.2?}", avg_verify);
    println!("    - Min: {:.2?}", min_verify);
    println!("    - Max: {:.2?}", max_verify);
    println!();
    println!("  Iterations/second: {:.1}", iterations as f64 / total_time.as_secs_f64());

    Ok(())
}
