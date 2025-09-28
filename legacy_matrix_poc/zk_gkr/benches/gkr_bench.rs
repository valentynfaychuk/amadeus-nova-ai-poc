use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use zk_gkr::{
    mle::MleUtils,
    sumcheck::{SumCheckProver, SumCheckVerifier},
    transcript::FiatShamirTranscript,
    merkle_poseidon::PoseidonMerkleTree,
    Fr,
};
use ark_ff::UniformRand;
use rand::thread_rng;

/// Helper to generate random matrix and vectors for testing
fn generate_test_data(m: usize, k: usize) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>) {
    let mut rng = thread_rng();

    // Generate random m×k matrix (flattened)
    let weights: Vec<Fr> = (0..m * k)
        .map(|_| Fr::rand(&mut rng))
        .collect();

    // Generate random k-dimensional input
    let input: Vec<Fr> = (0..k)
        .map(|_| Fr::rand(&mut rng))
        .collect();

    // Compute output y = W·x
    let mut output = vec![Fr::from(0u64); m];
    for i in 0..m {
        for j in 0..k {
            output[i] += weights[i * k + j] * input[j];
        }
    }

    (weights, input, output)
}

fn benchmark_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("legacy_proof_generation");

    // Test different matrix sizes (same as Expander POC)
    for (m, k) in [(4, 8), (8, 16), (16, 32), (32, 64)].iter() {
        let m = *m;
        let k = *k;

        // Calculate total operations
        let ops = m * k;
        group.throughput(Throughput::Elements(ops as u64));

        // Prepare test data
        let (weights, input, output) = generate_test_data(m, k);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", m, k)),
            &(weights.clone(), input.clone(), output.clone()),
            |b, (w, x, y)| {
                b.iter(|| {
                    // Create transcript
                    let mut transcript = FiatShamirTranscript::new().unwrap();

                    // Derive u vector (for output dimension)
                    let u = transcript.derive_u_vector(1 << ((m as f64).log2().ceil() as usize)).unwrap();

                    // Create Merkle trees
                    let w_tree = PoseidonMerkleTree::build_tree(black_box(w)).unwrap();
                    let x_tree = PoseidonMerkleTree::build_tree(black_box(x)).unwrap();

                    // Create prover and generate proof
                    // a = log2(m) = number of output bits, b = log2(k) = number of input bits
                    let a = (m as f64).log2().ceil() as usize;
                    let b = (k as f64).log2().ceil() as usize;

                    let prover = SumCheckProver::new(
                        u.clone(),
                        black_box(w).clone(),
                        black_box(x).clone(),
                        a,
                        b,
                        w_tree,
                        x_tree,
                    ).unwrap();

                    let proof = prover.prove(&mut transcript).unwrap();
                    proof
                });
            },
        );
    }

    group.finish();
}

fn benchmark_proof_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("legacy_proof_verification");

    // Test different matrix sizes
    for (m, k) in [(4, 8), (8, 16), (16, 32), (32, 64)].iter() {
        let m = *m;
        let k = *k;

        // Prepare test data and generate proof
        let (weights, input, output) = generate_test_data(m, k);

        // Generate proof once for verification benchmark
        let mut transcript = FiatShamirTranscript::new().unwrap();
        let u = transcript.derive_u_vector(1 << ((m as f64).log2().ceil() as usize)).unwrap();

        let w_tree = PoseidonMerkleTree::build_tree(&weights).unwrap();
        let x_tree = PoseidonMerkleTree::build_tree(&input).unwrap();

        let a = (m as f64).log2().ceil() as usize;
        let b = (k as f64).log2().ceil() as usize;

        let prover = SumCheckProver::new(
            u.clone(),
            weights.clone(),
            input.clone(),
            a,
            b,
            w_tree.clone(),
            x_tree.clone(),
        ).unwrap();

        let proof = prover.prove(&mut transcript).unwrap();

        let a = (m as f64).log2().ceil() as usize;
        let b = (k as f64).log2().ceil() as usize;

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", m, k)),
            &(proof, u, w_tree.root(), x_tree.root()),
            |bench, (p, u_vec, w_root, x_root)| {
                bench.iter(|| {
                    let mut verifier_transcript = FiatShamirTranscript::new().unwrap();
                    // Skip u derivation to match prover state
                    verifier_transcript.derive_u_vector(1 << a).unwrap();

                    let result = SumCheckVerifier::verify(
                        black_box(p),
                        black_box(u_vec),
                        black_box(w_root),
                        black_box(x_root),
                        a,
                        b,
                        &mut verifier_transcript,
                    ).unwrap();
                    result
                });
            },
        );
    }

    group.finish();
}

fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("legacy_matrix_multiplication");

    // Benchmark raw matrix multiplication for comparison
    for (m, k) in [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128)].iter() {
        let m = *m;
        let k = *k;

        let ops = m * k;
        group.throughput(Throughput::Elements(ops as u64));

        let (weights, input, _) = generate_test_data(m, k);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", m, k)),
            &(weights.clone(), input.clone()),
            |b, (w, x)| {
                b.iter(|| {
                    let mut output = vec![Fr::from(0u64); m];
                    for i in 0..m {
                        for j in 0..k {
                            output[i] += black_box(&w[i * k + j]) * black_box(&x[j]);
                        }
                    }
                    output
                });
            },
        );
    }

    group.finish();
}

fn benchmark_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("legacy_end_to_end");

    // Benchmark complete prove + verify cycle
    for (m, k) in [(4, 8), (16, 32), (32, 64)].iter() {
        let m = *m;
        let k = *k;

        let ops = m * k;
        group.throughput(Throughput::Elements(ops as u64));

        let (weights, input, output) = generate_test_data(m, k);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", m, k)),
            &(weights.clone(), input.clone(), output.clone()),
            |b, (w, x, y)| {
                b.iter(|| {
                    // Prove
                    let mut transcript = FiatShamirTranscript::new().unwrap();
                    let u = transcript.derive_u_vector(1 << ((m as f64).log2().ceil() as usize)).unwrap();

                    let w_tree = PoseidonMerkleTree::build_tree(black_box(w)).unwrap();
                    let x_tree = PoseidonMerkleTree::build_tree(black_box(x)).unwrap();

                    let a = (m as f64).log2().ceil() as usize;
                    let b = (k as f64).log2().ceil() as usize;

                    let prover = SumCheckProver::new(
                        u.clone(),
                        black_box(w).clone(),
                        black_box(x).clone(),
                        a,
                        b,
                        w_tree.clone(),
                        x_tree.clone(),
                    ).unwrap();

                    let proof = prover.prove(&mut transcript).unwrap();

                    // Verify
                    let mut verifier_transcript = FiatShamirTranscript::new().unwrap();
                    // Skip u derivation to match prover state
                    verifier_transcript.derive_u_vector(1 << a).unwrap();

                    let verified = SumCheckVerifier::verify(
                        &proof,
                        &u,
                        &w_tree.root(),
                        &x_tree.root(),
                        a,
                        b,
                        &mut verifier_transcript,
                    );
                    assert!(verified.is_ok());
                    verified.unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_scaling(c: &mut Criterion) {
    // Benchmark to understand scaling behavior
    let mut group = c.benchmark_group("legacy_scaling_analysis");

    // Keep one dimension fixed, vary the other
    let fixed_m = 16;
    for k in [8, 16, 32, 64, 128].iter() {
        let k = *k;

        let ops = fixed_m * k;
        group.throughput(Throughput::Elements(ops as u64));

        let (weights, input, output) = generate_test_data(fixed_m, k);

        group.bench_with_input(
            BenchmarkId::new("varying_k", format!("16x{}", k)),
            &(weights.clone(), input.clone(), output.clone()),
            |b, (w, x, y)| {
                b.iter(|| {
                    let mut transcript = FiatShamirTranscript::new().unwrap();
                    let u = transcript.derive_u_vector(1 << ((fixed_m as f64).log2().ceil() as usize)).unwrap();

                    let w_tree = PoseidonMerkleTree::build_tree(black_box(w)).unwrap();
                    let x_tree = PoseidonMerkleTree::build_tree(black_box(x)).unwrap();

                    let a = (fixed_m as f64).log2().ceil() as usize;
                    let b = (k as f64).log2().ceil() as usize;

                    let prover = SumCheckProver::new(
                        u.clone(),
                        black_box(w).clone(),
                        black_box(x).clone(),
                        a,
                        b,
                        w_tree,
                        x_tree,
                    ).unwrap();

                    prover.prove(&mut transcript).unwrap()
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

        let (weights, input, output) = generate_test_data(m, fixed_k);

        group.bench_with_input(
            BenchmarkId::new("varying_m", format!("{}x32", m)),
            &(weights.clone(), input.clone(), output.clone()),
            |b, (w, x, y)| {
                b.iter(|| {
                    let mut transcript = FiatShamirTranscript::new().unwrap();
                    let u = transcript.derive_u_vector(1 << ((m as f64).log2().ceil() as usize)).unwrap();

                    let w_tree = PoseidonMerkleTree::build_tree(black_box(w)).unwrap();
                    let x_tree = PoseidonMerkleTree::build_tree(black_box(x)).unwrap();

                    let a = (m as f64).log2().ceil() as usize;
                    let b_dim = (fixed_k as f64).log2().ceil() as usize;

                    let prover = SumCheckProver::new(
                        u.clone(),
                        black_box(w).clone(),
                        black_box(x).clone(),
                        a,
                        b_dim,
                        w_tree,
                        x_tree,
                    ).unwrap();

                    prover.prove(&mut transcript).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("legacy_components");

    let m = 32;
    let k = 64;
    let (weights, input, _) = generate_test_data(m, k);

    // Benchmark MLE evaluation
    group.bench_function("mle_evaluation_32x64", |b| {
        let mut rng = thread_rng();
        let point: Vec<Fr> = (0..((m * k) as f64).log2().ceil() as usize)
            .map(|_| Fr::rand(&mut rng))
            .collect();

        b.iter(|| {
            MleUtils::evaluate_mle_direct(&black_box(&weights), &black_box(&point)).unwrap()
        });
    });

    // Benchmark Merkle tree creation
    group.bench_function("merkle_tree_32x64", |b| {
        b.iter(|| {
            PoseidonMerkleTree::build_tree(black_box(&weights)).unwrap()
        });
    });

    // Benchmark transcript operations
    group.bench_function("transcript_derive_u", |b| {
        let num_vars = ((m * k) as f64).log2().ceil() as usize;

        b.iter(|| {
            let mut transcript = FiatShamirTranscript::new().unwrap();
            transcript.derive_u_vector(black_box(num_vars)).unwrap()
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
    benchmark_scaling,
    benchmark_components
);

criterion_main!(benches);