use ark_bn254::Fr;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};
use circuit::{TinyTailCircuit, N};

fn main() {
    // Create a test circuit with dummy values (new tiny tail circuit)
    let z = Fr::from(0u64);
    let circuit = TinyTailCircuit {
        w2: [[z; N]; N],
        y1: [z; N],
        y2: [z; N],
        scale_num: Fr::from(3u64),
        h_w2: z,
        h_x: z,
        h_y1: z,
        h_y: z,
        div_quotients: [z; N],
        div_remainders: [z; N],
    };

    // Create constraint system and generate constraints
    let cs = ConstraintSystem::<Fr>::new_ref();
    circuit.generate_constraints(cs.clone()).expect("Failed to generate constraints");

    // Print constraint statistics
    println!("=== Tiny Tail Circuit Complexity Analysis ===");
    println!("Matrix dimensions: 16x16 tail layer only");
    println!("Total constraints: {}", cs.num_constraints());
    println!("Number of variables: {}", cs.num_instance_variables() + cs.num_witness_variables());
    println!("Public inputs: {}", cs.num_instance_variables() - 1); // -1 for the constant ONE variable
    println!("Private witnesses: {}", cs.num_witness_variables());

    // Calculate theoretical complexity for tiny circuit
    let weight_vars = N * N; // 16×16 weight matrix W2
    let vector_vars = N * 3; // y1, y2, and computed y2
    let division_vars = N * 2; // Quotients and remainders

    println!("\n=== Theoretical Breakdown ===");
    println!("Weight variables (W2): {} ({}x{})", weight_vars, N, N);
    println!("Vector variables: {} (3 x {})", vector_vars, N);
    println!("Division variables: {} ({}x2)", division_vars, N);

    // Estimate constraint breakdown for tiny circuit
    let matrix_mult_constraints = N * N; // Matrix-vector multiplication
    let division_constraints = N; // Division verification constraints
    let commitment_constraints = N * 3; // Three vector commitments + one matrix commitment
    let range_check_constraints = (weight_vars * 16) + (vector_vars * 16) + 8; // 16-bit range checks + 8-bit scale_num

    println!("\n=== Estimated Constraint Breakdown ===");
    println!("Matrix multiplication (16x16): {}", matrix_mult_constraints);
    println!("Division verification: {}", division_constraints);
    println!("Commitment verification: {}", commitment_constraints);
    println!("Range checking: {}", range_check_constraints);
    println!("Total estimated: {}", matrix_mult_constraints + division_constraints + commitment_constraints + range_check_constraints);

    println!("\n=== Tiny Groth16 Benefits ===");
    println!("• Proof size: ~200-300 bytes (compressed)");
    println!("• Public inputs: 5 field elements (~160 bytes)");
    println!("• Total transaction: < 1 KB");
    println!("• Large computation verified off-chain with Freivalds");
}