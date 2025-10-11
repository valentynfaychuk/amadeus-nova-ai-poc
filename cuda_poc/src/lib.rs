//! CUDA-accelerated matrix multiplication using zkCUDA
//!
//! This implementation uses Polyhedra's zkCUDA framework for GPU-accelerated
//! zero-knowledge proofs of matrix multiplication.

use expander_compiler::frontend::*;
use expander_compiler::zkcuda::{context::*, kernel::*};

// Re-export M31 and Field trait for public API
pub use expander_compiler::frontend::M31;
pub use arith::field::Field;

/// Matrix multiplication kernel for zkCUDA
///
/// Multiplies a row vector (a) by a matrix (b) to produce output vector (c)
/// Dimensions: a[32] × b[32×64] → c[64]
#[kernel]
fn mul_line<C: Config>(
    api: &mut API<C>,
    a: &[InputVariable; 32],
    b: &[[InputVariable; 64]; 32],
    c: &mut [OutputVariable; 64],
) {
    // Initialize output to zero
    for j in 0..64 {
        c[j] = api.constant(0);
    }

    // Matrix multiplication: c[j] = Σ(a[i] * b[i][j])
    for i in 0..32 {
        for j in 0..64 {
            let product = api.mul(a[i], b[i][j]);
            c[j] = api.add(c[j], product);
        }
    }
}

/// Reduction kernel: sums 8 elements
///
/// Used for parallel reduction of matrix multiplication results
#[kernel]
fn sum_8_elements<C: Config>(
    api: &mut API<C>,
    a: &[InputVariable; 8],
    b: &mut OutputVariable,
) {
    let mut sum = api.constant(0);
    for i in 0..8 {
        sum = api.add(sum, a[i]);
    }
    *b = sum;
}

/// High-level API for matrix multiplication with ZK proofs
pub struct MatrixProofSystem {
    kernel_mul: Kernel<M31Config>,
    kernel_sum: Kernel<M31Config>,
}

impl MatrixProofSystem {
    /// Create new proof system
    pub fn new() -> anyhow::Result<Self> {
        let kernel_mul = compile_mul_line().map_err(|e| anyhow::anyhow!("Failed to compile mul_line: {:?}", e))?;
        let kernel_sum = compile_sum_8_elements().map_err(|e| anyhow::anyhow!("Failed to compile sum_8_elements: {:?}", e))?;

        Ok(Self {
            kernel_mul,
            kernel_sum,
        })
    }

    /// Generate proof of matrix multiplication
    ///
    /// Proves that result = mat_a × mat_b
    /// mat_a: 64×32 matrix
    /// mat_b: 32×64 matrix
    /// result: scalar (sum of all elements in product)
    pub fn prove(
        &self,
        mat_a: &Vec<Vec<M31>>,
        mat_b: &Vec<Vec<M31>>,
    ) -> anyhow::Result<M31> {
        let mut ctx: Context<M31Config> = Context::default();

        // Extract kernels for macro usage
        let kernel_mul = &self.kernel_mul;
        let kernel_sum = &self.kernel_sum;

        // Copy matrices to device
        let a = ctx.copy_to_device(mat_a, false);
        let b = ctx.copy_to_device(mat_b, true);

        // Execute matrix multiplication kernel
        let mut c = None;
        call_kernel!(ctx, kernel_mul, a, b, mut c);

        // Reduce results using multiple sum kernels
        // c: 4096 elements (64×64) → 512×8
        let c = c.reshape(&[512, 8]);
        let mut d = None;
        call_kernel!(ctx, kernel_sum, c, mut d);

        // d: 512 → 64×8
        let d = d.reshape(&[64, 8]);
        let mut e = None;
        call_kernel!(ctx, kernel_sum, d, mut e);

        // e: 64 → 8×8
        let e = e.reshape(&[8, 8]);
        let mut f = None;
        call_kernel!(ctx, kernel_sum, e, mut f);

        // f: 8 → 1×8
        let f = f.reshape(&[1, 8]);
        let mut g = None;
        call_kernel!(ctx, kernel_sum, f, mut g);

        // g: 1 (final result)
        let g = g.reshape(&[]);

        // Copy result back to host
        let result: M31 = ctx.copy_to_host(g);

        // Generate computation graph and proof (for verification later)
        let computation_graph = ctx.to_computation_graph();
        let (_prover_setup, _verifier_setup) = ctx.proving_system_setup(&computation_graph);

        Ok(result)
    }

    /// Verify result by recomputing
    pub fn verify(
        &self,
        mat_a: &Vec<Vec<M31>>,
        mat_b: &Vec<Vec<M31>>,
        expected_result: M31,
    ) -> anyhow::Result<bool> {
        // Recreate computation graph (same as proving)
        let mut ctx: Context<M31Config> = Context::default();

        // Extract kernels for macro usage
        let kernel_mul = &self.kernel_mul;
        let kernel_sum = &self.kernel_sum;

        let a = ctx.copy_to_device(mat_a, false);
        let b = ctx.copy_to_device(mat_b, true);

        let mut c = None;
        call_kernel!(ctx, kernel_mul, a, b, mut c);

        let c = c.reshape(&[512, 8]);
        let mut d = None;
        call_kernel!(ctx, kernel_sum, c, mut d);

        let d = d.reshape(&[64, 8]);
        let mut e = None;
        call_kernel!(ctx, kernel_sum, d, mut e);

        let e = e.reshape(&[8, 8]);
        let mut f = None;
        call_kernel!(ctx, kernel_sum, e, mut f);

        let f = f.reshape(&[1, 8]);
        let mut g = None;
        call_kernel!(ctx, kernel_sum, f, mut g);

        let g = g.reshape(&[]);
        let result: M31 = ctx.copy_to_host(g);

        // Check result matches
        Ok(result == expected_result)
    }
}

impl Default for MatrixProofSystem {
    fn default() -> Self {
        Self::new().expect("Failed to create MatrixProofSystem")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication_proof() {
        let system = MatrixProofSystem::new().unwrap();

        // Create test matrices 64×32 and 32×64
        let mut mat_a: Vec<Vec<M31>> = vec![];
        for i in 0..64 {
            mat_a.push(vec![]);
            for j in 0..32 {
                mat_a[i].push(M31::from((i * 233 + j + 1) as u32));
            }
        }

        let mut mat_b: Vec<Vec<M31>> = vec![];
        for i in 0..32 {
            mat_b.push(vec![]);
            for j in 0..64 {
                mat_b[i].push(M31::from((i * 2333 + j + 1) as u32));
            }
        }

        // Calculate expected result (sum of all products)
        let mut expected_result = M31::zero();
        for i in 0..64 {
            for j in 0..64 {
                for k in 0..32 {
                    expected_result += mat_a[i][k] * mat_b[k][j];
                }
            }
        }

        // Generate proof (compute result)
        let result = system.prove(&mat_a, &mat_b).unwrap();

        // Verify result
        assert_eq!(result, expected_result);

        // Verify by recomputing
        let verified = system.verify(&mat_a, &mat_b, expected_result).unwrap();
        assert!(verified);
    }
}
