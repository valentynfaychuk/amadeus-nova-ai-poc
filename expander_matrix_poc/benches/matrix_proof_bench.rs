use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use expander_matrix_poc::{MatrixProofSystem, Matrix, Vector};
use rand::thread_rng;

fn benchmark_proof_generation(c: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = c.benchmark_group("proof_generation");

    // Test different matrix sizes
    for (m, k) in [(4, 8), (8, 16), (16, 32), (32, 64)].iter() {
        let m = *m;
        let k = *k;

        // Calculate total operations (matrix multiplication)
        let ops = m * k;
        group.throughput(Throughput::Elements(ops as u64));

        // Prepare test data
        let weights = Matrix::random(m, k, &mut rng);
        let input = Vector::random(k, &mut rng);
        let output = weights.multiply(&input);

        let mut system = MatrixProofSystem::new(m, k).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", m, k)),
            &(weights.clone(), input.clone(), output.clone()),
            |b, (w, x, y)| {
                b.iter(|| {
                    system.prove(black_box(w), black_box(x), black_box(y))
                });
            },
        );
    }

    group.finish();
}

fn benchmark_proof_verification(c: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = c.benchmark_group("proof_verification");

    // Test different matrix sizes
    for (m, k) in [(4, 8), (8, 16), (16, 32), (32, 64)].iter() {
        let m = *m;
        let k = *k;

        // Prepare test data and generate proof
        let weights = Matrix::random(m, k, &mut rng);
        let input = Vector::random(k, &mut rng);
        let output = weights.multiply(&input);

        let mut system = MatrixProofSystem::new(m, k).unwrap();
        let proof = system.prove(&weights, &input, &output).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", m, k)),
            &(proof, input.clone(), output.clone()),
            |b, (p, x, y)| {
                b.iter(|| {
                    system.verify(black_box(p), black_box(x), black_box(y))
                });
            },
        );
    }

    group.finish();
}

fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = c.benchmark_group("matrix_multiplication");

    // Benchmark raw matrix multiplication for comparison
    for (m, k) in [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128)].iter() {
        let m = *m;
        let k = *k;

        let ops = m * k;
        group.throughput(Throughput::Elements(ops as u64));

        let weights = Matrix::random(m, k, &mut rng);
        let input = Vector::random(k, &mut rng);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", m, k)),
            &(weights.clone(), input.clone()),
            |b, (w, x)| {
                b.iter(|| {
                    w.multiply(black_box(x))
                });
            },
        );
    }

    group.finish();
}

fn benchmark_end_to_end(c: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = c.benchmark_group("end_to_end");

    // Benchmark complete prove + verify cycle
    for (m, k) in [(4, 8), (16, 32), (32, 64)].iter() {
        let m = *m;
        let k = *k;

        let ops = m * k;
        group.throughput(Throughput::Elements(ops as u64));

        let weights = Matrix::random(m, k, &mut rng);
        let input = Vector::random(k, &mut rng);
        let output = weights.multiply(&input);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", m, k)),
            &(weights.clone(), input.clone(), output.clone()),
            |b, (w, x, y)| {
                b.iter(|| {
                    let mut system = MatrixProofSystem::new(m, k).unwrap();
                    let proof = system.prove(black_box(w), black_box(x), black_box(y)).unwrap();
                    let verified = system.verify(&proof, x, y).unwrap();
                    assert!(verified);
                    verified
                });
            },
        );
    }

    group.finish();
}

fn benchmark_scaling(c: &mut Criterion) {
    let mut rng = thread_rng();

    // Benchmark to understand scaling behavior
    let mut group = c.benchmark_group("scaling_analysis");

    // Keep one dimension fixed, vary the other
    let fixed_m = 16;
    for k in [8, 16, 32, 64, 128].iter() {
        let k = *k;

        let ops = fixed_m * k;
        group.throughput(Throughput::Elements(ops as u64));

        let weights = Matrix::random(fixed_m, k, &mut rng);
        let input = Vector::random(k, &mut rng);
        let output = weights.multiply(&input);

        let mut system = MatrixProofSystem::new(fixed_m, k).unwrap();

        group.bench_with_input(
            BenchmarkId::new("varying_k", format!("16x{}", k)),
            &(weights.clone(), input.clone(), output.clone()),
            |b, (w, x, y)| {
                b.iter(|| {
                    system.prove(black_box(w), black_box(x), black_box(y))
                });
            },
        );
    }

    // Now fix k and vary m
    let fixed_k = 32;
    for m in [4, 8, 16, 32, 64].iter() {
        let m = *m;

        let ops = m * fixed_k;
        group.throughput(Throughput::Elements(ops as u64));

        let weights = Matrix::random(m, fixed_k, &mut rng);
        let input = Vector::random(fixed_k, &mut rng);
        let output = weights.multiply(&input);

        let mut system = MatrixProofSystem::new(m, fixed_k).unwrap();

        group.bench_with_input(
            BenchmarkId::new("varying_m", format!("{}x32", m)),
            &(weights.clone(), input.clone(), output.clone()),
            |b, (w, x, y)| {
                b.iter(|| {
                    system.prove(black_box(w), black_box(x), black_box(y))
                });
            },
        );
    }

    group.finish();
}

// Mock vs Real Expander comparison (for future use)
#[cfg(feature = "real_expander")]
fn benchmark_mock_vs_real(c: &mut Criterion) {
    use expander_matrix_poc::{ExpanderMatrixProver, ExpanderMatrixVerifier};
    let mut rng = thread_rng();

    let mut group = c.benchmark_group("implementation_comparison");

    let m = 32;
    let k = 64;

    let weights = Matrix::random(m, k, &mut rng);
    let input = Vector::random(k, &mut rng);
    let output = weights.multiply(&input);

    // Benchmark Mock implementation
    let mut mock_system = MatrixProofSystem::new(m, k).unwrap();
    group.bench_function("mock_32x64", |b| {
        b.iter(|| {
            mock_system.prove(black_box(&weights), black_box(&input), black_box(&output))
        });
    });

    // Benchmark Real Expander implementation
    let mut real_prover = ExpanderMatrixProver::new(m, k).unwrap();
    group.bench_function("expander_32x64", |b| {
        b.iter(|| {
            real_prover.prove(black_box(&weights), black_box(&input), black_box(&output))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_proof_generation,
    benchmark_proof_verification,
    benchmark_matrix_multiplication,
    benchmark_end_to_end,
    benchmark_scaling
);

criterion_main!(benches);