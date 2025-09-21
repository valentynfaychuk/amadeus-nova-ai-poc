use super::*;
use ark_relations::r1cs::ConstraintSystem;

#[test]
fn test_circuit_constraint_debug() {
    // Create a simple test with properly computed commitments
    let w2 = [[Fr::from(1u64); 16]; 16]; // Identity matrix
    let y1 = [Fr::from(2u64); 16]; // Input vector [2, 2, ...]
    let _scale_num = Fr::from(3u64);

    // Expected computation: y2[i] = floor((W2·y1)[i] * scale_num / 2)
    // With identity: (W2·y1)[i] = y1[i] = 2
    // So: y2[i] = floor(2 * 3 / 2) = floor(3) = 3
    let y2 = [Fr::from(3u64); 16];

    // Division witnesses: 2 * 3 = 2 * 3 + 0, so quotient=3, remainder=0
    let _div_quotients = [Fr::from(3u64); 16];
    let _div_remainders = [Fr::from(0u64); 16];

    // Compute commitments properly using the same logic as the circuit
    let alpha = Fr::from(5u64);
    let beta = Fr::from(7u64);

    // h_w2 = commit_matrix(w2)
    let mut h_w2 = Fr::from(0u64);
    let mut alpha_pow = Fr::from(1u64);
    for i in 0..16 {
        for j in 0..16 {
            h_w2 += w2[i][j] * alpha_pow;
            alpha_pow *= alpha;
        }
    }

    // h_y1 = commit_vector(y1)
    let mut h_y1 = Fr::from(0u64);
    let mut beta_pow = Fr::from(1u64);
    for i in 0..16 {
        h_y1 += y1[i] * beta_pow;
        beta_pow *= beta;
    }

    // h_y = commit_vector(y2)
    let mut h_y = Fr::from(0u64);
    beta_pow = Fr::from(1u64);
    for i in 0..16 {
        h_y += y2[i] * beta_pow;
        beta_pow *= beta;
    }

    // Use much smaller values to avoid range check issues
    let w2_small = [[Fr::from(1u64); 16]; 16]; // Keep identity
    let y1_small = [Fr::from(1u64); 16]; // Smaller input vector [1, 1, ...]
    let scale_num_small = Fr::from(1u64); // Use scale_num = 1 to simplify

    // With identity and y1=[1], scale_num=1:
    // t_i = W2·y1 = sum of 16 ones = 16
    // numed = t_i * scale_num = 16 * 1 = 16
    // y2[i] = floor(numed / 2) = floor(16 / 2) = 8
    let y2_small = [Fr::from(8u64); 16];

    // Division witnesses: 16 = 2 * 8 + 0, so quotient=8, remainder=0
    let div_quotients_small = [Fr::from(8u64); 16];
    let div_remainders_small = [Fr::from(0u64); 16];

    // Recompute commitments for smaller values
    let mut h_w2_small = Fr::from(0u64);
    let mut alpha_pow = Fr::from(1u64);
    for i in 0..16 {
        for j in 0..16 {
            h_w2_small += w2_small[i][j] * alpha_pow;
            alpha_pow *= alpha;
        }
    }

    let mut h_y1_small = Fr::from(0u64);
    let mut beta_pow = Fr::from(1u64);
    for i in 0..16 {
        h_y1_small += y1_small[i] * beta_pow;
        beta_pow *= beta;
    }

    let mut h_y_small = Fr::from(0u64);
    beta_pow = Fr::from(1u64);
    for i in 0..16 {
        h_y_small += y2_small[i] * beta_pow;
        beta_pow *= beta;
    }

    let circuit = TinyTailCircuit {
        w2: w2_small,
        y1: y1_small,
        y2: y2_small,
        scale_num: scale_num_small,
        h_w2: h_w2_small,
        h_x: Fr::from(42u64), // Dummy value for h_x (not checked in circuit)
        h_y1: h_y1_small,
        h_y: h_y_small,
        div_quotients: div_quotients_small,
        div_remainders: div_remainders_small,
    };

    println!("🔍 Testing circuit constraint satisfaction...");
    println!("h_w2_small: {}", h_w2_small);
    println!("h_y1_small: {}", h_y1_small);
    println!("h_y_small: {}", h_y_small);

    // Debug the computation step by step
    println!("🔍 Debug computation:");
    println!("W2 (identity): {:?}", w2_small[0]); // First row
    println!("y1 (all 1s): {:?}", &y1_small[0..4]); // First few elements
    println!("scale_num: {}", scale_num_small);

    // Manual computation: t_i = W2·y1 = 1*1 + 1*1 + ... = 16 (sum of 16 ones)
    // numed = t_i * scale_num = 16 * 1 = 16
    // floor(numed / 2) = floor(16/2) = 8
    // So quotient should be 8, remainder should be 0
    println!("Expected: t_i = 16, numed = 16, quotient = 8, remainder = 0");
    println!("But we used: quotient = 0, remainder = 1");

    // Test constraint generation
    let cs = ConstraintSystem::<Fr>::new_ref();
    let result = circuit.generate_constraints(cs.clone());

    match result {
        Ok(_) => {
            let satisfied = cs.is_satisfied().unwrap();
            println!("Num variables: {}", cs.num_witness_variables() + cs.num_instance_variables());
            println!("Num constraints: {}", cs.num_constraints());

            if satisfied {
                println!("✅ Constraints satisfied!");
            } else {
                println!("❌ Constraints not satisfied!");

                // Try to get more debugging info
                if let Ok(unsatisfied) = cs.which_is_unsatisfied() {
                    if let Some(constraint_name) = unsatisfied {
                        println!("Unsatisfied constraint: {}", constraint_name);
                    }
                }
            }

            assert!(satisfied, "Circuit constraints should be satisfied");
        }
        Err(e) => {
            panic!("Failed to generate constraints: {:?}", e);
        }
    }
}