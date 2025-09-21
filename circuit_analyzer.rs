use ark_bn254::Fr;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};
use circuit::{LinChainCircuit, Quant, N, L};

fn main() {
    // Create a test circuit with dummy values
    let z = Fr::from(0u64);
    let circuit = LinChainCircuit {
        w: [[[z; N]; N]; L],
        x0: [z; N],
        y_out: [z; N],
        h_w: z,
        q: Quant {
            scale_num: Fr::from(1u64),
            scale_den: Fr::from(2u64)
        },
        div_quotients: [[z; N]; L],
        div_remainders: [[z; N]; L],
    };

    // Create constraint system and generate constraints
    let cs = ConstraintSystem::<Fr>::new_ref();
    circuit.generate_constraints(cs.clone()).expect("Failed to generate constraints");

    // Print constraint statistics
    println!("=== Circuit Complexity Analysis ===");
    println!("Matrix dimensions: N={}, L={} layers", N, L);
    println!("Total constraints: {}", cs.num_constraints());
    println!("Number of variables: {}", cs.num_instance_variables() + cs.num_witness_variables());
    println!("Public inputs: {}", cs.num_instance_variables() - 1); // -1 for the constant ONE variable
    println!("Private witnesses: {}", cs.num_witness_variables());

    // Calculate theoretical complexity
    let weight_vars = L * N * N; // Weight matrix elements
    let intermediate_vars = L * N; // Intermediate computation results per layer
    let division_vars = L * N * 2; // Quotients and remainders
    let range_check_bits = 16; // bits per range check

    println!("\n=== Theoretical Breakdown ===");
    println!("Weight variables: {} ({}x{}x{})", weight_vars, L, N, N);
    println!("Intermediate variables: {} ({}x{})", intermediate_vars, L, N);
    println!("Division variables: {} ({}x{}x2)", division_vars, L, N);
    println!("Range check constraints: ~{} per variable", range_check_bits);

    // Estimate constraint breakdown
    let matrix_mult_constraints = L * N * N; // One constraint per multiplication
    let division_constraints = L * N; // Division verification constraints
    let range_check_constraints = (weight_vars * 8) + ((N + N) * 16) + (L * N * 16); // 8-bit weights, 16-bit inputs/outputs/quotients
    let commitment_constraints = weight_vars; // Weight commitment computation

    println!("\n=== Estimated Constraint Breakdown ===");
    println!("Matrix multiplication: {}", matrix_mult_constraints);
    println!("Division verification: {}", division_constraints);
    println!("Range checking: {}", range_check_constraints);
    println!("Weight commitment: {}", commitment_constraints);
    println!("Total estimated: {}", matrix_mult_constraints + division_constraints + range_check_constraints + commitment_constraints);
}