use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ark_bn254::Fr;
use ark_ff::PrimeField;
use zk_gkr_accel::{create_context, BackendType, ComputeConfig};

fn generate_test_matrix(m: usize, k: usize) -> (Vec<Vec<Fr>>, Vec<Fr>) {
    let weights: Vec<Vec<Fr>> = (0..m)
        .map(|i| {
            (0..k)
                .map(|j| Fr::from((i * k + j + 1) as u64))
                .collect()
        })
        .collect();

    let input: Vec<Fr> = (0..k).map(|i| Fr::from((i + 1) as u64)).collect();

    (weights, input)
}

fn bench_matrix_vector_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_vector_multiply");

    let sizes = vec![
        (16, 256),
        (32, 512),
        (64, 1024),
        (128, 2048),
    ];

    for &(m, k) in &sizes {
        let (weights, input) = generate_test_matrix(m, k);
        let ops_per_iteration = m * k;

        group.throughput(Throughput::Elements(ops_per_iteration as u64));

        // Benchmark CPU AVX backend
        #[cfg(feature = "cpu_avx")]
        {
            let config = ComputeConfig {
                backend: BackendType::CpuAvx,
                ..Default::default()
            };

            if let Ok(mut context) = create_context(config) {
                group.bench_with_input(
                    BenchmarkId::new("CPU_AVX", format!("{}x{}", m, k)),
                    &(m, k),
                    |b, _| {
                        b.iter(|| {
                            let result = context.compute_and_prove(
                                black_box(&weights),
                                black_box(&input),
                                black_box("benchmark"),
                            );
                            black_box(result)
                        })
                    },
                );
            }
        }

        // Benchmark CUDA backend if available
        #[cfg(feature = "cuda")]
        {
            if zk_gkr_accel::cuda::is_cuda_available() {
                let config = ComputeConfig {
                    backend: BackendType::Cuda,
                    device_id: Some(0),
                    ..Default::default()
                };

                if let Ok(mut context) = create_context(config) {
                    group.bench_with_input(
                        BenchmarkId::new("CUDA", format!("{}x{}", m, k)),
                        &(m, k),
                        |b, _| {
                            b.iter(|| {
                                let result = context.compute_and_prove(
                                    black_box(&weights),
                                    black_box(&input),
                                    black_box("benchmark"),
                                );
                                black_box(result)
                            })
                        },
                    );
                }
            }
        }
    }

    group.finish();
}

fn bench_mle_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mle_operations");

    let sizes = vec![8, 10, 12, 14]; // 2^n coefficients

    for &n in &sizes {
        let num_coeffs = 1 << n;
        let coefficients: Vec<Fr> = (0..num_coeffs)
            .map(|i| Fr::from((i + 1) as u64))
            .collect();
        let point: Vec<Fr> = (0..n).map(|i| Fr::from((i + 1) as u64)).collect();

        group.throughput(Throughput::Elements(num_coeffs as u64));

        // Benchmark CPU AVX backend
        #[cfg(feature = "cpu_avx")]
        {
            let config = ComputeConfig {
                backend: BackendType::CpuAvx,
                ..Default::default()
            };

            if let Ok(context) = create_context(config) {
                group.bench_with_input(
                    BenchmarkId::new("CPU_AVX_MLE", format!("2^{}", n)),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            let result = context.backend().mle_evaluate(
                                black_box(&coefficients),
                                black_box(&point),
                            );
                            black_box(result)
                        })
                    },
                );
            }
        }

        // Benchmark CUDA backend if available
        #[cfg(feature = "cuda")]
        {
            if zk_gkr_accel::cuda::is_cuda_available() {
                let config = ComputeConfig {
                    backend: BackendType::Cuda,
                    device_id: Some(0),
                    ..Default::default()
                };

                if let Ok(context) = create_context(config) {
                    group.bench_with_input(
                        BenchmarkId::new("CUDA_MLE", format!("2^{}", n)),
                        &n,
                        |b, _| {
                            b.iter(|| {
                                let result = context.backend().mle_evaluate(
                                    black_box(&coefficients),
                                    black_box(&point),
                                );
                                black_box(result)
                            })
                        },
                    );
                }
            }
        }
    }

    group.finish();
}

fn bench_sumcheck_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_round");

    let sizes = vec![128, 256, 512, 1024];

    for &size in &sizes {
        let evaluations: Vec<Fr> = (0..size * 2)
            .map(|i| Fr::from((i + 1) as u64))
            .collect();
        let challenges: Vec<Fr> = (0..size)
            .map(|i| Fr::from((i + 1) as u64))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark CPU AVX backend
        #[cfg(feature = "cpu_avx")]
        {
            let config = ComputeConfig {
                backend: BackendType::CpuAvx,
                ..Default::default()
            };

            if let Ok(context) = create_context(config) {
                group.bench_with_input(
                    BenchmarkId::new("CPU_AVX_SUMCHECK", size),
                    &size,
                    |b, _| {
                        let mut result = vec![Fr::zero(); size];
                        b.iter(|| {
                            let res = context.backend().sumcheck_round(
                                black_box(&evaluations),
                                black_box(&challenges),
                                black_box(&mut result),
                            );
                            black_box(res)
                        })
                    },
                );
            }
        }

        // Benchmark CUDA backend if available
        #[cfg(feature = "cuda")]
        {
            if zk_gkr_accel::cuda::is_cuda_available() {
                let config = ComputeConfig {
                    backend: BackendType::Cuda,
                    device_id: Some(0),
                    ..Default::default()
                };

                if let Ok(context) = create_context(config) {
                    group.bench_with_input(
                        BenchmarkId::new("CUDA_SUMCHECK", size),
                        &size,
                        |b, _| {
                            let mut result = vec![Fr::zero(); size];
                            b.iter(|| {
                                let res = context.backend().sumcheck_round(
                                    black_box(&evaluations),
                                    black_box(&challenges),
                                    black_box(&mut result),
                                );
                                black_box(res)
                            })
                        },
                    );
                }
            }
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_vector_multiply,
    bench_mle_operations,
    bench_sumcheck_round
);
criterion_main!(benches);