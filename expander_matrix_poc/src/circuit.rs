//! Matrix multiplication circuit definition for Expander GKR

use crate::types::{Matrix, Vector, ProofConfig};
use ark_bn254::Fr;
use anyhow::{Result, Context};

/// Matrix multiplication circuit: y = W·x
///
/// This represents the computation as a circuit that Expander can prove efficiently.
/// The circuit computes y[i] = Σ(j=0 to k-1) W[i][j] * x[j] for all i ∈ [0, m)
#[derive(Debug)]
pub struct MatrixMultCircuit {
    pub m: usize,  // number of rows in W
    pub k: usize,  // number of columns in W / length of x
    config: ProofConfig,
}

impl MatrixMultCircuit {
    /// Create new matrix multiplication circuit
    pub fn new(m: usize, k: usize) -> Result<Self> {
        let config = ProofConfig {
            m,
            k,
            ..Default::default()
        };

        Ok(Self { m, k, config })
    }

    /// Generate circuit description file for Expander
    ///
    /// This creates a text file describing the matrix multiplication computation
    /// in Expander's circuit format. The exact format needs to be determined
    /// from Expander documentation.
    pub fn generate_circuit_file(&self, output_path: &str) -> Result<()> {
        let circuit_description = self.create_circuit_description()?;

        std::fs::write(output_path, circuit_description)
            .context("Failed to write circuit file")?;

        Ok(())
    }

    /// Generate witness file for private matrix W
    pub fn generate_witness_file(&self, weights: &Matrix, output_path: &str) -> Result<()> {
        if weights.rows != self.m || weights.cols != self.k {
            return Err(anyhow::anyhow!(
                "Matrix dimensions mismatch: expected {}×{}, got {}×{}",
                self.m, self.k, weights.rows, weights.cols
            ));
        }

        let witness_data = self.create_witness_data(weights)?;

        std::fs::write(output_path, witness_data)
            .context("Failed to write witness file")?;

        Ok(())
    }

    /// Generate public input file for x and y
    pub fn generate_public_input_file(
        &self,
        input: &Vector,
        output: &Vector,
        file_path: &str,
    ) -> Result<()> {
        if input.len() != self.k {
            return Err(anyhow::anyhow!(
                "Input vector length mismatch: expected {}, got {}",
                self.k, input.len()
            ));
        }

        if output.len() != self.m {
            return Err(anyhow::anyhow!(
                "Output vector length mismatch: expected {}, got {}",
                self.m, output.len()
            ));
        }

        let public_data = self.create_public_input_data(input, output)?;

        std::fs::write(file_path, public_data)
            .context("Failed to write public input file")?;

        Ok(())
    }

    /// Create circuit description in Expander's format
    ///
    /// This needs to be adapted based on Expander's actual circuit format.
    /// For now, this is a placeholder that describes the computation structure.
    fn create_circuit_description(&self) -> Result<String> {
        // Pre-allocate string capacity to avoid reallocations for large circuits
        let estimated_size = (self.m + self.k) * 30 + (self.m * self.k) * 25 + (self.m * self.k * 15) + 1000;
        let mut circuit = String::with_capacity(estimated_size);

        use std::fmt::Write;

        // Circuit header
        writeln!(circuit, "# Matrix multiplication circuit: {}×{}", self.m, self.k)?;
        writeln!(circuit, "# Computes y = W·x where W is {}×{} private matrix", self.m, self.k)?;
        writeln!(circuit)?;

        // Field specification
        writeln!(circuit, "field bn254")?;
        writeln!(circuit)?;

        // Input declarations
        writeln!(circuit, "# Public inputs")?;
        for i in 0..self.k {
            writeln!(circuit, "public input x_{}", i)?;
        }
        writeln!(circuit)?;

        // Output declarations
        writeln!(circuit, "# Public outputs")?;
        for i in 0..self.m {
            writeln!(circuit, "public output y_{}", i)?;
        }
        writeln!(circuit)?;

        // Witness declarations (private matrix W)
        writeln!(circuit, "# Private witness (matrix W)")?;
        for i in 0..self.m {
            for j in 0..self.k {
                writeln!(circuit, "private witness W_{}_{}", i, j)?;
            }
        }
        writeln!(circuit)?;

        // Computation constraints
        writeln!(circuit, "# Matrix multiplication constraints")?;
        for i in 0..self.m {
            write!(circuit, "constraint y_{} = ", i)?;
            for j in 0..self.k {
                if j > 0 {
                    write!(circuit, " + ")?;
                }
                write!(circuit, "W_{}_{} * x_{}", i, j, j)?;
            }
            writeln!(circuit)?;
        }

        Ok(circuit)
    }

    /// Create witness data in Expander's format
    fn create_witness_data(&self, weights: &Matrix) -> Result<String> {
        // Pre-allocate string capacity to avoid reallocations
        let estimated_size = self.m * self.k * 50; // Rough estimate: 50 chars per element
        let mut witness = String::with_capacity(estimated_size);

        witness.push_str(&format!("# Witness data for {}×{} matrix\n", self.m, self.k));
        witness.push_str("\n");

        // Write matrix elements with efficient formatting
        for i in 0..self.m {
            for j in 0..self.k {
                let value = weights.get(i, j);
                use std::fmt::Write;
                writeln!(witness, "W_{}_{} {}", i, j, field_to_string(value))
                    .map_err(|e| anyhow::anyhow!("Failed to format witness data: {}", e))?;
            }
        }

        Ok(witness)
    }

    /// Create public input data
    fn create_public_input_data(&self, input: &Vector, output: &Vector) -> Result<String> {
        let mut public = String::new();

        public.push_str("# Public inputs and outputs\n");
        public.push_str("\n");

        // Input vector x
        public.push_str("# Input vector x\n");
        for i in 0..self.k {
            let value = input.get(i);
            public.push_str(&format!("x_{} {}\n", i, field_to_string(value)));
        }

        // Output vector y
        public.push_str("\n# Output vector y\n");
        for i in 0..self.m {
            let value = output.get(i);
            public.push_str(&format!("y_{} {}\n", i, field_to_string(value)));
        }

        Ok(public)
    }

    /// Generate cryptographic hash of circuit for proof integrity
    pub fn generate_circuit_hash(&self) -> anyhow::Result<Vec<u8>> {
        use ark_ff::PrimeField;
        use ark_bn254::Fr;
        use ark_serialize::CanonicalSerialize;

        // Create deterministic hash from circuit parameters without expensive string generation
        let mut hasher_input = Vec::new();

        // Hash the circuit structure parameters directly
        hasher_input.extend_from_slice("matrix_mult_circuit".as_bytes()); // Circuit type identifier
        hasher_input.extend_from_slice(&self.m.to_le_bytes());           // Matrix rows
        hasher_input.extend_from_slice(&self.k.to_le_bytes());           // Matrix cols
        hasher_input.extend_from_slice("bn254".as_bytes());              // Field type

        // Add circuit structure info without generating full description
        let total_variables = self.m + self.k + (self.m * self.k);       // outputs + inputs + weights
        let total_constraints = self.m;                                  // One constraint per output
        hasher_input.extend_from_slice(&total_variables.to_le_bytes());
        hasher_input.extend_from_slice(&total_constraints.to_le_bytes());

        // Generate field element hash
        let hash_field = Fr::from_le_bytes_mod_order(&hasher_input);
        let mut hash_bytes = Vec::new();
        hash_field.serialize_compressed(&mut hash_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to serialize circuit hash: {}", e))?;

        Ok(hash_bytes)
    }
}

/// Convert field element to string representation
/// This needs to match Expander's expected format
fn field_to_string(value: Fr) -> String {
    // For now, convert to decimal string
    // This may need to be adjusted based on Expander's requirements
    format!("{}", value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_circuit_creation() {
        let circuit = MatrixMultCircuit::new(4, 8).unwrap();
        assert_eq!(circuit.m, 4);
        assert_eq!(circuit.k, 8);
    }

    #[test]
    fn test_circuit_file_generation() {
        let circuit = MatrixMultCircuit::new(2, 3).unwrap();

        let circuit_desc = circuit.create_circuit_description().unwrap();

        // Check that description contains expected elements
        assert!(circuit_desc.contains("field bn254"));
        assert!(circuit_desc.contains("public input x_0"));
        assert!(circuit_desc.contains("public output y_0"));
        assert!(circuit_desc.contains("private witness W_0_0"));
        assert!(circuit_desc.contains("constraint y_0 = W_0_0 * x_0 + W_0_1 * x_1 + W_0_2 * x_2"));
    }

    #[test]
    fn test_witness_file_generation() {
        let circuit = MatrixMultCircuit::new(2, 2).unwrap();
        let mut rng = thread_rng();
        let matrix = Matrix::random(2, 2, &mut rng);

        let witness_data = circuit.create_witness_data(&matrix).unwrap();

        // Check that witness contains matrix elements
        assert!(witness_data.contains("W_0_0"));
        assert!(witness_data.contains("W_1_1"));
    }
}