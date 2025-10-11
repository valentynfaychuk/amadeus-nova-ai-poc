//! Large-scale zkCUDA Matrix Multiplication Benchmark
//!
//! Tests GPU-accelerated zero-knowledge proofs with realistic matrix sizes.

use cuda_poc::*;

fn main() -> anyhow::Result<()> {
    println!("🚀 zkCUDA Large Matrix Multiplication Benchmark");
    println!("================================================\n");

    // Test multiple sizes
    let sizes = vec![
        (64, 32, "Tiny (demo size)"),
        (256, 32, "Small"),
        (512, 32, "Medium"),
        (1024, 32, "Large"),
        (2048, 32, "Very Large"),
    ];

    // Create proof system once
    println!("⚙️  Initializing zkCUDA proof system...");
    let system = MatrixProofSystem::new()?;
    println!("✅ System initialized\n");

    for (rows, cols, label) in sizes {
        println!("📐 Testing {} - Matrix: {}×{} × {}×64", label, rows, cols, cols);

        // Create test matrices
        let mut mat_a: Vec<Vec<M31>> = vec![];
        for i in 0..rows {
            mat_a.push(vec![]);
            for j in 0..cols {
                mat_a[i].push(M31::from(((i * 233 + j + 1) % 2147483647) as u32));
            }
        }

        let mut mat_b: Vec<Vec<M31>> = vec![];
        for i in 0..cols {
            mat_b.push(vec![]);
            for j in 0..64 {
                mat_b[i].push(M31::from(((i * 2333 + j + 1) % 2147483647) as u32));
            }
        }

        // Calculate expected result on CPU
        println!("  Computing expected result on CPU...");
        let cpu_start = std::time::Instant::now();
        let mut expected_result = M31::zero();
        for i in 0..rows {
            for j in 0..64 {
                for k in 0..cols {
                    expected_result += mat_a[i][k] * mat_b[k][j];
                }
            }
        }
        let cpu_time = cpu_start.elapsed();
        println!("  CPU computation: {:.2?}", cpu_time);

        // Generate proof
        println!("  🔐 Generating zero-knowledge proof...");
        let prove_start = std::time::Instant::now();
        let result = system.prove(&mat_a, &mat_b)?;
        let prove_time = prove_start.elapsed();

        // Verify proof
        let verify_start = std::time::Instant::now();
        let verified = system.verify(&mat_a, &mat_b, expected_result)?;
        let verify_time = verify_start.elapsed();

        // Calculate total operations
        let total_ops = rows * 64 * cols;

        if verified && result == expected_result {
            println!("  ✅ Proof verified successfully!");
            println!("  📊 Stats:");
            println!("     - Total operations: {} multiplications", total_ops);
            println!("     - CPU time: {:.2?}", cpu_time);
            println!("     - Proving time: {:.2?}", prove_time);
            println!("     - Verification time: {:.2?}", verify_time);
            println!("     - Speedup vs CPU: {:.2}×", cpu_time.as_secs_f64() / prove_time.as_secs_f64());
            println!("     - Throughput: {:.0} ops/sec", total_ops as f64 / prove_time.as_secs_f64());
        } else {
            println!("  ❌ Verification failed!");
            return Err(anyhow::anyhow!("Verification failed for size {}", label));
        }

        println!();
    }

    println!("✨ All benchmarks completed successfully!");
    Ok(())
}
