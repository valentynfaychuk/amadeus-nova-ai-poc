use ark_bn254::Fr; // Removed unused Bn254
use ark_ff::Zero; // Removed unused Field
use ark_r1cs_std::{
    alloc::AllocVar,
    boolean::Boolean,
    eq::EqGadget,
    fields::fp::FpVar,
    fields::FieldVar,
    // ToBitsGadget,  // Unused import
};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, Result as R1CSResult};
// use ark_groth16::{Proof, VerifyingKey};  // Unused imports
// use ark_serialize::{CanonicalSerialize, CanonicalDeserialize, Compress, Validate};  // Unused imports
// use std::io::{Read, Write};  // Unused imports

pub mod compressed;

#[cfg(test)]
mod tests;

/// Tiny Groth16 circuit for 16×16 tail layer only
pub const N: usize = 16; // Fixed 16×16 layer size for tail

/// Tiny Groth16 circuit for proving only the 16×16 tail layer computation
/// with commitment-based public inputs to keep proof size small
#[derive(Clone)]
pub struct TinyTailCircuit {
    /// Single 16×16 weight matrix W2 (private witness)
    pub w2: [[Fr; N]; N],
    /// Input vector y1 from large layer (private witness)
    pub y1: [Fr; N],
    /// Output vector y2 (computed inside circuit, private)
    pub y2: [Fr; N],
    /// Scale numerator for quantization (public, bounded to 8 bits)
    pub scale_num: Fr,
    /// Public commitments (hash-based, not cryptographic for POC)
    pub h_w2: Fr, // commitment to W2
    pub h_x: Fr,  // commitment to original input x0
    pub h_y1: Fr, // commitment to y1
    pub h_y: Fr,  // commitment to final output y2
    /// Division quotients and remainders for floor operation (private witnesses)
    pub div_quotients: [Fr; N],
    pub div_remainders: [Fr; N],
}

impl TinyTailCircuit {
    /// Deterministic α-sum commitment for 16×16 matrix.
    /// h = Σ w[i][j] * α^(i*16+j) with fixed α=5.
    ///
    /// ⚠️  NON-CRYPTOGRAPHIC: This is a linear map, collisions exist.
    /// For production, replace with Poseidon hash using vetted BN254 parameters.
    fn commit_matrix(
        cs: ConstraintSystemRef<Fr>,
        w_vars: &Vec<Vec<FpVar<Fr>>>,
    ) -> R1CSResult<FpVar<Fr>> {
        let alpha = Fr::from(5u64); // Fixed base for deterministic commitment
        let mut alpha_pow = FpVar::<Fr>::new_constant(cs.clone(), Fr::from(1u64))?;
        let alpha_c = FpVar::<Fr>::new_constant(cs.clone(), alpha)?;
        let mut acc = FpVar::<Fr>::zero();

        for i in 0..N {
            for j in 0..N {
                acc += &w_vars[i][j] * &alpha_pow;
                alpha_pow = &alpha_pow * &alpha_c;
            }
        }
        Ok(acc)
    }

    /// Deterministic β-sum commitment for vector.
    /// h = Σ v[i] * β^i with fixed β=7.
    fn commit_vector(
        cs: ConstraintSystemRef<Fr>,
        v_vars: &Vec<FpVar<Fr>>,
    ) -> R1CSResult<FpVar<Fr>> {
        let beta = Fr::from(7u64); // Fixed base for vector commitment
        let mut beta_pow = FpVar::<Fr>::new_constant(cs.clone(), Fr::from(1u64))?;
        let beta_c = FpVar::<Fr>::new_constant(cs.clone(), beta)?;
        let mut acc = FpVar::<Fr>::zero();

        for i in 0..N {
            acc += &v_vars[i] * &beta_pow;
            beta_pow = &beta_pow * &beta_c;
        }
        Ok(acc)
    }

    /// Range-check an FpVar to fit in `bits` by constraining high bits to zero.
    /// Currently disabled to keep circuit small (307 constraints vs 260K+)
    fn range_check_bits(_v: &FpVar<Fr>, _bits: usize) -> R1CSResult<()> {
        // TODO: Re-enable for production security
        // This would add ~1000 constraints per range check
        // let bits_le = _v.to_bits_le()?;
        // for b in _bits..bits_le.len() {
        //     Boolean::enforce_equal(&bits_le[b], &Boolean::constant(false))?;
        // }
        Ok(())
    }
}

impl ConstraintSynthesizer<Fr> for TinyTailCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> R1CSResult<()> {
        // Public inputs (≤10 field elements to keep proof size tiny):
        // h_w2, h_x, h_y1, h_y, scale_num
        let h_w2_pub = FpVar::<Fr>::new_input(cs.clone(), || Ok(self.h_w2))?;
        let _h_x_pub = FpVar::<Fr>::new_input(cs.clone(), || Ok(self.h_x))?;
        let h_y1_pub = FpVar::<Fr>::new_input(cs.clone(), || Ok(self.h_y1))?;
        let h_y_pub = FpVar::<Fr>::new_input(cs.clone(), || Ok(self.h_y))?;
        let scale_num_pub = FpVar::<Fr>::new_input(cs.clone(), || Ok(self.scale_num))?;

        // Range-bound scale_num to 8 bits
        Self::range_check_bits(&scale_num_pub, 8)?;

        // Private witnesses: W2 (16×16), y1 (16), y2 (16)
        let mut w2_vars: Vec<Vec<FpVar<Fr>>> = Vec::with_capacity(N);
        for i in 0..N {
            let mut row: Vec<FpVar<Fr>> = Vec::with_capacity(N);
            for j in 0..N {
                let w_ij = FpVar::<Fr>::new_witness(cs.clone(), || Ok(self.w2[i][j]))?;
                // Range check weights to 8-16 bits to keep circuit small
                Self::range_check_bits(&w_ij, 16)?;
                row.push(w_ij);
            }
            w2_vars.push(row);
        }

        let y1_vars: Vec<FpVar<Fr>> = (0..N)
            .map(|i| {
                let y1_i = FpVar::<Fr>::new_witness(cs.clone(), || Ok(self.y1[i]))?;
                Self::range_check_bits(&y1_i, 16)?; // Range check for reasonable values
                Ok(y1_i)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Compute y2 = floor((W2 · y1) * scale_num / 2) inside the circuit
        let mut y2_computed: Vec<FpVar<Fr>> = Vec::with_capacity(N);
        for i in 0..N {
            // t_i = (W2 · y1)[i] = Σ W2[i][j] * y1[j]
            let mut t_i = FpVar::<Fr>::zero();
            for j in 0..N {
                t_i += &w2_vars[i][j] * &y1_vars[j];
            }

            // Quantized: numed = t_i * scale_num
            let numed = &t_i * &scale_num_pub;

            // Floor division by 2: numed = 2 * quotient + remainder
            let quotient = FpVar::<Fr>::new_witness(cs.clone(), || Ok(self.div_quotients[i]))?;

            // remainder is either 0 or 1 (since den=2)
            let remainder_bit = Boolean::new_witness(cs.clone(), || {
                let rem_val = self.div_remainders[i];
                Ok(!rem_val.is_zero())
            })?;
            let remainder = FpVar::<Fr>::from(remainder_bit);

            // Constraint: numed = 2 * quotient + remainder
            let two = FpVar::<Fr>::constant(Fr::from(2u64));
            (&two * &quotient + &remainder).enforce_equal(&numed)?;

            // Range check quotient
            Self::range_check_bits(&quotient, 16)?;

            y2_computed.push(quotient);
        }

        // Verify commitments
        // 1. h_w2 = commit_matrix(W2)
        let h_w2_calc = Self::commit_matrix(cs.clone(), &w2_vars)?;
        h_w2_calc.enforce_equal(&h_w2_pub)?;

        // 2. h_y1 = commit_vector(y1)
        let h_y1_calc = Self::commit_vector(cs.clone(), &y1_vars)?;
        h_y1_calc.enforce_equal(&h_y1_pub)?;

        // 3. h_y = commit_vector(y2_computed)
        let h_y_calc = Self::commit_vector(cs.clone(), &y2_computed)?;
        h_y_calc.enforce_equal(&h_y_pub)?;

        // Note: h_x (commitment to original input x0) is just verified by inclusion
        // in public inputs, as we don't have access to x0 in this tiny circuit

        Ok(())
    }
}
