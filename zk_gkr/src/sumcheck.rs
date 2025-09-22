use crate::{Fr, Result, GkrError};
use crate::transcript::FiatShamirTranscript;
use crate::merkle_poseidon::PoseidonMerkleTree;
use crate::mle::{MleUtils, MleOpenProof};
use ark_ff::Field;
use serde::{Serialize, Deserialize};

/// Univariate polynomial representation (coefficients in ascending degree order)
#[derive(Debug, Clone)]
pub struct UnivariatePolynomial {
    pub coefficients: Vec<Fr>,
}

impl UnivariatePolynomial {
    /// Create a new polynomial with given coefficients
    pub fn new(coefficients: Vec<Fr>) -> Self {
        Self { coefficients }
    }

    /// Evaluate polynomial at a point
    pub fn evaluate(&self, x: &Fr) -> Fr {
        let mut result = Fr::zero();
        let mut power = Fr::one();

        for coeff in &self.coefficients {
            result += *coeff * power;
            power *= x;
        }

        result
    }

    /// Get the degree of the polynomial
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }
}

/// Sum-check proof structure
#[derive(Debug, Clone)]
pub struct SumCheckProof {
    /// Claimed sum
    pub claimed_sum: Fr,
    /// Univariate polynomials for each round
    pub round_polynomials: Vec<UnivariatePolynomial>,
    /// Challenges used in each round
    pub challenges: Vec<Fr>,
    /// MLE opening proof for W at final point
    pub w_opening: MleOpenProof,
    /// MLE opening proof for X at final point
    pub x_opening: MleOpenProof,
    /// Final evaluation point
    pub final_point: Vec<Fr>,
}

/// Sum-check prover for the GKR protocol
pub struct SumCheckProver {
    /// Vector u for row compression
    u: Vec<Fr>,
    /// Matrix W data (in hypercube order)
    w_data: Vec<Fr>,
    /// Vector x data (in hypercube order)
    x_data: Vec<Fr>,
    /// Dimensions
    a: usize, // log2(m)
    b: usize, // log2(k)
    /// Merkle trees
    w_tree: PoseidonMerkleTree,
    x_tree: PoseidonMerkleTree,
}

impl SumCheckProver {
    /// Create a new sum-check prover
    pub fn new(
        u: Vec<Fr>,
        w_data: Vec<Fr>,
        x_data: Vec<Fr>,
        a: usize,
        b: usize,
        w_tree: PoseidonMerkleTree,
        x_tree: PoseidonMerkleTree,
    ) -> Result<Self> {
        let expected_w_size = 1 << (a + b);
        let expected_x_size = 1 << b;
        let expected_u_size = 1 << a;

        if w_data.len() != expected_w_size {
            return Err(GkrError::InvalidDimensions(
                format!("W data size {} doesn't match expected 2^{}", w_data.len(), a + b)
            ));
        }

        if x_data.len() != expected_x_size {
            return Err(GkrError::InvalidDimensions(
                format!("X data size {} doesn't match expected 2^{}", x_data.len(), b)
            ));
        }

        if u.len() != expected_u_size {
            return Err(GkrError::InvalidDimensions(
                format!("U vector size {} doesn't match expected 2^{}", u.len(), a)
            ));
        }

        Ok(Self {
            u,
            w_data,
            x_data,
            a,
            b,
            w_tree,
            x_tree,
        })
    }

    /// Prove the sum-check relation: sum over hypercube of g(z) = c
    /// where g(z) = U(z[0..a]) * W(z) * X(z[a..a+b])
    pub fn prove(&self, transcript: &mut FiatShamirTranscript) -> Result<SumCheckProof> {
        let total_vars = self.a + self.b;

        // Compute the claimed sum
        let claimed_sum = self.compute_claimed_sum()?;

        let mut current_g_table = self.build_initial_g_table()?;
        let mut challenges = Vec::new();
        let mut round_polynomials = Vec::new();

        // Sum-check rounds
        for round in 0..total_vars {
            // Compute univariate polynomial G_t(X) for this round
            let g_poly = self.compute_round_polynomial(&current_g_table, round)?;

            // Add polynomial to transcript and get challenge
            transcript.absorb_fr_vec(&g_poly.coefficients);
            let challenge = transcript.get_round_challenge(round)?;

            challenges.push(challenge);
            round_polynomials.push(g_poly.clone());

            // Update g_table by fixing the current variable to the challenge
            current_g_table = self.fix_variable(&current_g_table, &challenge, round)?;
        }

        // At this point, we should have a single value
        if current_g_table.len() != 1 {
            return Err(GkrError::SumCheckFailed(
                "Final g table should have exactly one element".to_string()
            ));
        }

        let final_point = challenges.clone();

        // Generate MLE opening proofs for the final evaluation
        let w_root = self.w_tree.root();
        let x_root = self.x_tree.root();

        let w_opening = MleUtils::prove_mle_opening(
            &w_root,
            &self.w_data,
            &final_point,
            &self.w_tree,
        )?;

        let x_opening = MleUtils::prove_mle_opening(
            &x_root,
            &self.x_data,
            &final_point[self.a..],
            &self.x_tree,
        )?;

        Ok(SumCheckProof {
            claimed_sum,
            round_polynomials,
            challenges,
            w_opening,
            x_opening,
            final_point,
        })
    }

    /// Compute the claimed sum: sum over all z of g(z)
    fn compute_claimed_sum(&self) -> Result<Fr> {
        let total_size = 1 << (self.a + self.b);
        let mut sum = Fr::zero();

        for z_index in 0..total_size {
            let g_value = self.evaluate_g_at_index(z_index)?;
            sum += g_value;
        }

        Ok(sum)
    }

    /// Evaluate g at a specific hypercube index
    fn evaluate_g_at_index(&self, z_index: usize) -> Result<Fr> {
        // Extract i and j from z_index
        let i_mask = (1 << self.a) - 1;
        let i = z_index & i_mask;
        let j = z_index >> self.a;

        if i >= self.u.len() || j >= (1 << self.b) {
            return Ok(Fr::zero());
        }

        // g(z) = U(i) * W(z_index) * X(j)
        let u_val = self.u[i];
        let w_val = self.w_data[z_index];
        let x_val = self.x_data[j];

        Ok(u_val * w_val * x_val)
    }

    /// Build initial table of g values over the hypercube
    fn build_initial_g_table(&self) -> Result<Vec<Fr>> {
        let total_size = 1 << (self.a + self.b);
        let mut g_table = Vec::with_capacity(total_size);

        for z_index in 0..total_size {
            g_table.push(self.evaluate_g_at_index(z_index)?);
        }

        Ok(g_table)
    }

    /// Compute the univariate polynomial for a given round
    fn compute_round_polynomial(&self, g_table: &[Fr], round: usize) -> Result<UnivariatePolynomial> {
        let current_size = g_table.len();

        // We're computing sum over the first variable, treating it as X
        // G(X) = sum_{b in {0,1}^{remaining_vars}} g(X, b)

        let mut coeffs = vec![Fr::zero(); 4]; // Degree at most 3

        // Evaluate at points 0, 1, 2, 3 and interpolate
        for eval_point in 0..4 {
            let x = Fr::from(eval_point as u64);
            let mut g_x_sum = Fr::zero();

            // Sum over all assignments to remaining variables
            for b_index in 0..(current_size / 2) {
                let g_0 = g_table[2 * b_index];
                let g_1 = g_table[2 * b_index + 1];

                // g(x, b) = (1-x) * g(0, b) + x * g(1, b)
                let g_x_b = (Fr::one() - x) * g_0 + x * g_1;
                g_x_sum += g_x_b;
            }

            // Use Lagrange interpolation to find coefficients
            for i in 0..4 {
                let mut basis = Fr::one();
                for j in 0..4 {
                    if i != j {
                        let num = Fr::from(eval_point as u64) - Fr::from(j as u64);
                        let den = Fr::from(i as u64) - Fr::from(j as u64);
                        basis *= num * den.inverse().unwrap();
                    }
                }
                coeffs[i] += basis * g_x_sum;
            }
        }

        Ok(UnivariatePolynomial::new(coeffs))
    }

    /// Fix a variable in the g table to a specific value
    fn fix_variable(&self, g_table: &[Fr], challenge: &Fr, round: usize) -> Result<Vec<Fr>> {
        let current_size = g_table.len();
        let mut next_g_table = Vec::with_capacity(current_size / 2);

        for i in 0..(current_size / 2) {
            let g_0 = g_table[2 * i];
            let g_1 = g_table[2 * i + 1];

            // Linear interpolation: g(challenge) = (1 - challenge) * g_0 + challenge * g_1
            let g_challenge = (Fr::one() - challenge) * g_0 + challenge * g_1;
            next_g_table.push(g_challenge);
        }

        Ok(next_g_table)
    }
}

/// Sum-check verifier for the GKR protocol
pub struct SumCheckVerifier;

impl SumCheckVerifier {
    /// Verify a sum-check proof
    pub fn verify(
        proof: &SumCheckProof,
        u: &[Fr],
        h_w: &Fr,
        h_x: &Fr,
        a: usize,
        b: usize,
        transcript: &mut FiatShamirTranscript,
    ) -> Result<bool> {
        let total_vars = a + b;

        if proof.round_polynomials.len() != total_vars {
            return Ok(false);
        }

        if proof.challenges.len() != total_vars {
            return Ok(false);
        }

        let mut current_claim = proof.claimed_sum;

        // Verify each round
        for round in 0..total_vars {
            let poly = &proof.round_polynomials[round];

            // Check degree bound (should be at most 3)
            if poly.degree() > 3 {
                return Ok(false);
            }

            // Check consistency: G_t(0) + G_t(1) should equal current claim
            let g_0 = poly.evaluate(&Fr::zero());
            let g_1 = poly.evaluate(&Fr::one());

            if g_0 + g_1 != current_claim {
                return Ok(false);
            }

            // Absorb polynomial and get challenge
            transcript.absorb_fr_vec(&poly.coefficients);
            let challenge = transcript.get_round_challenge(round)?;

            if challenge != proof.challenges[round] {
                return Ok(false);
            }

            // Update claim for next round
            current_claim = poly.evaluate(&challenge);
        }

        // Final check: verify MLE openings and recompute g(r)
        let final_point = &proof.final_point;

        // Verify W opening
        let w_valid = MleUtils::verify_mle_opening(h_w, final_point, &proof.w_opening)?;
        if !w_valid {
            return Ok(false);
        }

        // Verify X opening
        let x_valid = MleUtils::verify_mle_opening(h_x, &final_point[a..], &proof.x_opening)?;
        if !x_valid {
            return Ok(false);
        }

        // Compute U(r[0..a]) directly (since u is public)
        let u_at_r = MleUtils::evaluate_mle_direct(u, &final_point[0..a])?;

        // Recompute g(r) = U(r[0..a]) * W(r) * X(r[a..])
        let expected_final_value = u_at_r * proof.w_opening.value * proof.x_opening.value;

        if expected_final_value != current_claim {
            return Ok(false);
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mle::MleUtils;
    use ark_std::test_rng;

    #[test]
    fn test_univariate_polynomial() {
        let poly = UnivariatePolynomial::new(vec![
            Fr::from(1u64), // constant
            Fr::from(2u64), // x
            Fr::from(3u64), // x^2
        ]);

        // p(x) = 1 + 2x + 3x^2
        // p(2) = 1 + 4 + 12 = 17
        let result = poly.evaluate(&Fr::from(2u64));
        assert_eq!(result, Fr::from(17u64));

        assert_eq!(poly.degree(), 2);
    }

    #[test]
    fn test_sum_check_small_example() {
        // Small 2x2 matrix example
        let m = 2;
        let k = 2;
        let a = 1; // log2(2)
        let b = 1; // log2(2)

        // u = [1, 2]
        let u = vec![Fr::from(1u64), Fr::from(2u64)];

        // W matrix in hypercube order: [[1,2],[3,4]] -> [1,3,2,4]
        let w_matrix = vec![
            vec![Fr::from(1u64), Fr::from(2u64)],
            vec![Fr::from(3u64), Fr::from(4u64)],
        ];
        let w_data = MleUtils::matrix_to_hypercube_order(&w_matrix, m, k).unwrap();

        // x = [5, 6]
        let x_vector = vec![Fr::from(5u64), Fr::from(6u64)];
        let x_data = MleUtils::vector_to_hypercube_order(&x_vector, k).unwrap();

        // Build merkle trees
        let w_tree = PoseidonMerkleTree::build_tree(&w_data).unwrap();
        let x_tree = PoseidonMerkleTree::build_tree(&x_data).unwrap();

        // Create prover
        let prover = SumCheckProver::new(
            u.clone(),
            w_data.clone(),
            x_data.clone(),
            a,
            b,
            w_tree,
            x_tree,
        ).unwrap();

        // Expected sum: u^T * (W * x)
        // W * x = [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
        // u^T * [17, 39] = 1*17 + 2*39 = 17 + 78 = 95
        let expected_sum = Fr::from(95u64);

        // Test claimed sum computation
        let computed_sum = prover.compute_claimed_sum().unwrap();
        assert_eq!(computed_sum, expected_sum);
    }
}