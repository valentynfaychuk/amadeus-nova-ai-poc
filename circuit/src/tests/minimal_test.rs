use crate::*;
use ark_bn254::Fr;
use ark_r1cs_std::fields::fp::FpVar;
use ark_relations::r1cs::ConstraintSystem;

#[test]
fn test_minimal_division_logic() {
    // Test just the division constraint logic with minimal circuit
    let cs = ConstraintSystem::<Fr>::new_ref();

    // Test case: 1 = 2 * 0 + 1
    let numed = FpVar::<Fr>::new_witness(cs.clone(), || Ok(Fr::from(1u64))).unwrap();
    let quotient = FpVar::<Fr>::new_witness(cs.clone(), || Ok(Fr::from(0u64))).unwrap();
    let remainder_val = Fr::from(1u64);

    // Test Boolean conversion
    let remainder_bit = Boolean::new_witness(cs.clone(), || Ok(!remainder_val.is_zero())).unwrap();
    let remainder = FpVar::<Fr>::from(remainder_bit);

    // Test constraint: numed = 2 * quotient + remainder
    let two = FpVar::<Fr>::constant(Fr::from(2u64));
    let result = (&two * &quotient + &remainder).enforce_equal(&numed);

    match result {
        Ok(_) => {
            let satisfied = cs.is_satisfied().unwrap();
            println!(
                "Division constraint test: {}",
                if satisfied { "✅ PASS" } else { "❌ FAIL" }
            );
            assert!(satisfied, "Division constraint should be satisfied");
        }
        Err(e) => {
            panic!("Failed to create division constraint: {:?}", e);
        }
    }
}

#[test]
fn test_commitment_computation() {
    // Test just the commitment computation logic
    let cs = ConstraintSystem::<Fr>::new_ref();

    // Simple 2x2 matrix for testing
    let matrix = [
        [Fr::from(1u64), Fr::from(2u64)],
        [Fr::from(3u64), Fr::from(4u64)],
    ];

    // Compute commitment: h = Σ matrix[i][j] * α^(i*2+j)
    let alpha = Fr::from(5u64);
    let mut expected = Fr::from(0u64);
    let mut alpha_pow = Fr::from(1u64);

    for i in 0..2 {
        for j in 0..2 {
            expected += matrix[i][j] * alpha_pow;
            alpha_pow *= alpha;
        }
    }

    // Now test in circuit
    let mut matrix_vars = Vec::new();
    for i in 0..2 {
        let mut row = Vec::new();
        for j in 0..2 {
            let var = FpVar::<Fr>::new_witness(cs.clone(), || Ok(matrix[i][j])).unwrap();
            row.push(var);
        }
        matrix_vars.push(row);
    }

    // Compute commitment in circuit
    let alpha_var = FpVar::<Fr>::constant(alpha);
    let mut alpha_pow_var = FpVar::<Fr>::constant(Fr::from(1u64));
    let mut commitment = FpVar::<Fr>::constant(Fr::from(0u64));

    for i in 0..2 {
        for j in 0..2 {
            commitment += &matrix_vars[i][j] * &alpha_pow_var;
            alpha_pow_var *= &alpha_var;
        }
    }

    // Check that computed commitment matches expected
    let expected_var = FpVar::<Fr>::new_witness(cs.clone(), || Ok(expected)).unwrap();
    let result = commitment.enforce_equal(&expected_var);

    match result {
        Ok(_) => {
            let satisfied = cs.is_satisfied().unwrap();
            println!(
                "Commitment test: {}",
                if satisfied { "✅ PASS" } else { "❌ FAIL" }
            );
            println!("Expected: {}", expected);
            assert!(satisfied, "Commitment constraint should be satisfied");
        }
        Err(e) => {
            panic!("Failed to create commitment constraint: {:?}", e);
        }
    }
}
