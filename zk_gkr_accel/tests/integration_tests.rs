use ark_bn254::Fr;
use ark_ff::Zero;
#[allow(unused_imports)]
use ark_ff::PrimeField;  // May be needed on some architectures
use zk_gkr_accel::{available_backends, create_context, BackendType, ComputeConfig};

fn generate_test_data(m: usize, k: usize) -> (Vec<Vec<Fr>>, Vec<Fr>, Vec<Fr>) {
    let weights: Vec<Vec<Fr>> = (0..m)
        .map(|i| {
            (0..k)
                .map(|j| Fr::from((i * k + j + 1) as u64))
                .collect()
        })
        .collect();

    let input: Vec<Fr> = (0..k).map(|i| Fr::from((i + 1) as u64)).collect();

    // Compute expected output: y = W * x
    let mut expected_output = vec![Fr::zero(); m];
    for i in 0..m {
        let mut sum = Fr::zero();
        for j in 0..k {
            sum += weights[i][j] * input[j];
        }
        expected_output[i] = sum;
    }

    (weights, input, expected_output)
}

#[test]
fn test_available_backends() {
    let backends = available_backends();

    // On systems without AVX or CUDA support, no backends may be available
    // This is expected behavior
    println!("Available backends: {:?}", backends);

    // CPU AVX should be available only on x86/x86_64 with AVX2 support
    #[cfg(all(feature = "cpu_avx", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if is_x86_feature_detected!("avx2") {
            assert!(backends.contains(&BackendType::CpuAvx));
        }
    }

    // CUDA depends on hardware availability
    #[cfg(feature = "cuda")]
    {
        use zk_gkr_accel::cuda::is_cuda_available;
        if is_cuda_available() {
            assert!(backends.contains(&BackendType::Cuda));
        }
    }

    // This test should pass even if no backends are available
    // as that's valid behavior on unsupported hardware
}

#[test]
#[cfg(feature = "cpu_avx")]
fn test_cpu_avx_context_creation() {
    let config = ComputeConfig {
        backend: BackendType::CpuAvx,
        num_threads: Some(2),
        memory_limit: Some(1024 * 1024 * 100), // 100MB
        ..Default::default()
    };

    let context = create_context(config);
    assert!(context.is_ok(), "CPU AVX context creation should succeed");

    let context = context.unwrap();
    assert_eq!(context.backend().name(), "CPU AVX2/AVX-512");
    assert!(context.backend().is_available());
}

#[test]
#[cfg(all(feature = "cuda", not(test)))] // Skip in most test runs where CUDA may not be available
fn test_cuda_context_creation() {
    use zk_gkr_accel::cuda::is_cuda_available;

    if !is_cuda_available() {
        println!("CUDA not available, skipping test");
        return;
    }

    let config = ComputeConfig {
        backend: BackendType::Cuda,
        device_id: Some(0),
        memory_limit: Some(1024 * 1024 * 512), // 512MB
        ..Default::default()
    };

    let context = create_context(config);
    assert!(context.is_ok(), "CUDA context creation should succeed");

    let context = context.unwrap();
    assert_eq!(context.backend().name(), "CUDA GPU");
    assert!(context.backend().is_available());
}

#[test]
#[cfg(feature = "cpu_avx")]
fn test_cpu_avx_matrix_vector_multiply() {
    let config = ComputeConfig {
        backend: BackendType::CpuAvx,
        ..Default::default()
    };

    let context = create_context(config).unwrap();
    let (weights, input, expected) = generate_test_data(4, 8);

    let mut result = vec![Fr::zero(); 4];
    let res = context.backend().matrix_vector_multiply(&weights, &input, &mut result);

    assert!(res.is_ok(), "Matrix-vector multiplication should succeed");
    assert_eq!(result, expected, "Results should match expected values");
}

#[test]
#[cfg(feature = "cpu_avx")]
fn test_cpu_avx_mle_operations() {
    let config = ComputeConfig {
        backend: BackendType::CpuAvx,
        ..Default::default()
    };

    let context = create_context(config).unwrap();

    // Test MLE evaluation
    let coefficients = vec![
        Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64),
    ];
    let point = vec![Fr::from(1u64), Fr::from(2u64)]; // 2-variable evaluation

    let result = context.backend().mle_evaluate(&coefficients, &point);
    assert!(result.is_ok(), "MLE evaluation should succeed");

    // Test MLE folding
    let poly = vec![
        Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64),
    ];
    let challenge = Fr::from(5u64);
    let mut fold_result = vec![Fr::zero(); 2];

    let res = context.backend().mle_fold(&poly, challenge, &mut fold_result);
    assert!(res.is_ok(), "MLE folding should succeed");

    // Verify folding: left + challenge * (right - left)
    assert_eq!(fold_result[0], Fr::from(1u64) + Fr::from(5u64) * (Fr::from(2u64) - Fr::from(1u64)));
    assert_eq!(fold_result[1], Fr::from(3u64) + Fr::from(5u64) * (Fr::from(4u64) - Fr::from(3u64)));
}

#[test]
#[cfg(feature = "cpu_avx")]
fn test_cpu_avx_sumcheck_round() {
    let config = ComputeConfig {
        backend: BackendType::CpuAvx,
        ..Default::default()
    };

    let context = create_context(config).unwrap();

    let evaluations = vec![
        Fr::from(1u64), Fr::from(2u64), // Pair 1
        Fr::from(3u64), Fr::from(4u64), // Pair 2
    ];
    let challenges = vec![Fr::from(5u64), Fr::from(6u64)];
    let mut result = vec![Fr::zero(); 2];

    let res = context.backend().sumcheck_round(&evaluations, &challenges, &mut result);
    assert!(res.is_ok(), "Sumcheck round should succeed");

    // Verify interpolation: left + challenge * (right - left)
    let expected_0 = Fr::from(1u64) + Fr::from(5u64) * (Fr::from(2u64) - Fr::from(1u64));
    let expected_1 = Fr::from(3u64) + Fr::from(6u64) * (Fr::from(4u64) - Fr::from(3u64));

    assert_eq!(result[0], expected_0);
    assert_eq!(result[1], expected_1);
}

#[test]
#[cfg(feature = "cpu_avx")]
fn test_cpu_avx_end_to_end_proof() {
    let config = ComputeConfig {
        backend: BackendType::CpuAvx,
        ..Default::default()
    };

    let mut context = create_context(config).unwrap();
    let (weights, input, _expected) = generate_test_data(8, 16);

    // Generate proof
    let proof_result = context.compute_and_prove(&weights, &input, "test_salt");
    assert!(proof_result.is_ok(), "Proof generation should succeed");

    let proof_result = proof_result.unwrap();
    assert!(!proof_result.proof.is_empty(), "Proof should not be empty");
    assert!(!proof_result.public_inputs.is_empty(), "Public inputs should not be empty");
    // Proof time can be 0 for very fast operations
    assert!(proof_result.proof_time_ms < u128::MAX, "Proof time should be valid");

    // Verify proof
    let verification = context.verify_proof(
        &proof_result.proof,
        &proof_result.public_inputs,
        "test_salt",
    );
    assert!(verification.is_ok(), "Proof verification should succeed");
    assert!(verification.unwrap(), "Proof should be valid");
}

#[test]
#[cfg(all(feature = "cuda", not(test)))]
fn test_cuda_end_to_end_proof() {
    use zk_gkr_accel::cuda::is_cuda_available;

    if !is_cuda_available() {
        println!("CUDA not available, skipping test");
        return;
    }

    let config = ComputeConfig {
        backend: BackendType::Cuda,
        device_id: Some(0),
        ..Default::default()
    };

    let mut context = create_context(config).unwrap();
    let (weights, input, _expected) = generate_test_data(8, 16);

    let proof_result = context.compute_and_prove(&weights, &input, "test_salt");
    assert!(proof_result.is_ok(), "CUDA proof generation should succeed");

    let proof_result = proof_result.unwrap();
    let verification = context.verify_proof(
        &proof_result.proof,
        &proof_result.public_inputs,
        "test_salt",
    );
    assert!(verification.is_ok(), "CUDA proof verification should succeed");
    assert!(verification.unwrap(), "CUDA proof should be valid");
}

#[test]
#[cfg(feature = "cpu_avx")]
fn test_benchmark_computation() {
    let config = ComputeConfig {
        backend: BackendType::CpuAvx,
        ..Default::default()
    };

    let mut context = create_context(config).unwrap();

    let matrix_sizes = vec![(4, 8), (8, 16)];
    let num_runs = 2;

    let results = context.benchmark_computation(&matrix_sizes, num_runs);
    assert!(results.is_ok(), "Benchmarking should succeed");

    let results = results.unwrap();
    assert_eq!(results.len(), 2, "Should have results for both matrix sizes");

    for (i, result) in results.iter().enumerate() {
        assert_eq!(result.matrix_size, matrix_sizes[i]);
        assert!(result.proof_time_ms < u128::MAX, "Proof time should be valid");
        assert!(result.verify_time_ms < u128::MAX, "Verify time should be valid");
        assert!(result.throughput_ops_per_sec >= 0.0, "Throughput should be non-negative");
    }
}

#[test]
fn test_invalid_dimensions() {
    let config = ComputeConfig {
        backend: BackendType::CpuAvx,
        ..Default::default()
    };

    #[cfg(feature = "cpu_avx")]
    {
        let context = create_context(config).unwrap();

        // Test mismatched matrix-vector dimensions
        let weights = vec![vec![Fr::from(1u64); 4]; 3]; // 3x4 matrix
        let input = vec![Fr::from(1u64); 5]; // Wrong size vector (should be 4)
        let mut result = vec![Fr::zero(); 3];

        let res = context.backend().matrix_vector_multiply(&weights, &input, &mut result);
        assert!(res.is_err(), "Should fail with dimension mismatch");

        match res.unwrap_err() {
            zk_gkr_accel::AccelError::InvalidDimensions(_) => {},
            _ => panic!("Should be InvalidDimensions error"),
        }
    }
}