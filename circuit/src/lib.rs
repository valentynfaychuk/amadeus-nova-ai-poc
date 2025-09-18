use ark_bn254::Fr;
use ark_ff::{Field, Zero};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, Result as R1CSResult};
use ark_r1cs_std::{
    alloc::AllocVar,
    boolean::Boolean,
    eq::EqGadget,
    fields::fp::FpVar,
    fields::FieldVar,
    ToBitsGadget,
};

/// Small fixed dimensions for POC.
pub const N: usize = 3;     // vector/matrix width
pub const L: usize = 2;     // number of layers in the forward

/// Public quantization config (simple per-tensor scale = NUM/DEN).
#[derive(Clone, Copy)]
pub struct Quant {
    pub scale_num: Fr, // integer encoded in field
    pub scale_den: Fr, // integer encoded in field (>0), fixed to 2 for POC
}

#[derive(Clone)]
pub struct LinChainCircuit {
    /// L layers of weights (private)
    pub w: [[[Fr; N]; N]; L],
    /// Input and expected output (public)
    pub x0: [Fr; N],
    pub y_out: [Fr; N],
    /// Public "commitment" to all weights (single field element).
    pub h_w: Fr,
    /// Public quantization config
    pub q: Quant,
    /// Division quotients and remainders for each layer (private witnesses)
    pub div_quotients: [[Fr; N]; L],
    pub div_remainders: [[Fr; N]; L],
}

impl LinChainCircuit {
    /// Deterministic α-sum commitment over weights.
    /// h = Σ w_i * α^i with fixed α=5.
    ///
    /// ⚠️  NON-CRYPTOGRAPHIC: This is a linear map, collisions exist.
    /// For production, replace with Poseidon hash using vetted BN254 parameters.
    fn commit_weights(cs: ConstraintSystemRef<Fr>, w_vars: &Vec<Vec<Vec<FpVar<Fr>>>>) -> R1CSResult<FpVar<Fr>> {
        let alpha = Fr::from(5u64); // Fixed base for deterministic commitment
        let mut alpha_pow = FpVar::<Fr>::new_constant(cs.clone(), Fr::from(1u64))?;
        let alpha_c = FpVar::<Fr>::new_constant(cs.clone(), alpha)?;
        let mut acc = FpVar::<Fr>::zero();

        for l in 0..L {
            for i in 0..N {
                for j in 0..N {
                    acc += &w_vars[l][i][j] * &alpha_pow;
                    alpha_pow = &alpha_pow * &alpha_c;
                }
            }
        }
        Ok(acc)
    }

    /// Range-check an FpVar to fit in `bits` by constraining high bits to zero.
    fn range_check_bits(v: &FpVar<Fr>, bits: usize) -> R1CSResult<()> {
        let bits_le = v.to_bits_le()?;
        // Force all bits above our limit to be zero
        for b in bits..bits_le.len() {
            Boolean::enforce_equal(&bits_le[b], &Boolean::constant(false))?;
        }
        Ok(())
    }
}

impl ConstraintSynthesizer<Fr> for LinChainCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> R1CSResult<()> {
        // Public inputs: x0, y_out, h_w, q (scale_num, scale_den, qmin, qmax)
        let mut x_vars: Vec<FpVar<Fr>> = (0..N)
            .map(|j| FpVar::<Fr>::new_input(cs.clone(), || Ok(self.x0[j])))
            .collect::<Result<_,_>>()?;
        let y_pub: Vec<FpVar<Fr>> = (0..N)
            .map(|i| FpVar::<Fr>::new_input(cs.clone(), || Ok(self.y_out[i])))
            .collect::<Result<_,_>>()?;
        let h_w_pub = FpVar::<Fr>::new_input(cs.clone(), || Ok(self.h_w))?;
        let scale_num = FpVar::<Fr>::new_input(cs.clone(), || Ok(self.q.scale_num))?;
        let scale_den = FpVar::<Fr>::new_input(cs.clone(), || Ok(self.q.scale_den))?;

        // Range-bound public inputs to prevent warping
        for xi in &x_vars { Self::range_check_bits(xi, 16)?; }
        for yi in &y_pub  { Self::range_check_bits(yi, 16)?; }

        // POC: fix den == 2 and enforce it's nonzero
        scale_den.enforce_equal(&FpVar::constant(Fr::from(2u64)))?;
        let inv2 = FpVar::constant(Fr::from(2u64).inverse().unwrap());
        (&scale_den * &inv2).enforce_equal(&FpVar::constant(Fr::from(1u64)))?;

        // Private weights
        let mut w_vars: Vec<Vec<Vec<FpVar<Fr>>>> = Vec::with_capacity(L);
        for l in 0..L {
            let mut mat: Vec<Vec<FpVar<Fr>>> = Vec::with_capacity(N);
            for i in 0..N {
                let mut row: Vec<FpVar<Fr>> = Vec::with_capacity(N);
                for j in 0..N {
                    row.push(FpVar::<Fr>::new_witness(cs.clone(), || Ok(self.w[l][i][j]))?);
                    // Range check like INT8 (|v| < 2^8); we keep witnesses small in the POC anyway
                    Self::range_check_bits(row.last().unwrap(), 8)?;
                }
                mat.push(row);
            }
            w_vars.push(mat);
        }

        // Bind weights to public "commitment"
        let h_w_calc = Self::commit_weights(cs.clone(), &w_vars)?;
        h_w_calc.enforce_equal(&h_w_pub)?;

        // Forward pass: for l in 0..L { x_{l+1} = clip( floor( (W_l * x_l) * num / den ), qmin, qmax ) }
        // Note: we model floor via: (sum * num) = den * y + r, with 0 <= r < den. We **assume** chosen inputs avoid clipping.
        for l in 0..L {
            // t = W_l * x_l
            let mut t: Vec<FpVar<Fr>> = Vec::with_capacity(N);
            for i in 0..N {
                let mut acc = FpVar::<Fr>::zero();
                for j in 0..N {
                    acc += &w_vars[l][i][j] * &x_vars[j];
                }
                t.push(acc);
            }
            // y_int via floor((t * num) / den)
            let mut next_x: Vec<FpVar<Fr>> = Vec::with_capacity(N);
            for i in 0..N {
                let numed = &t[i] * &scale_num;             // t * num
                // Use the pre-computed witness values
                let y_i = FpVar::<Fr>::new_witness(cs.clone(), || Ok(self.div_quotients[l][i]))?;

                // Use Boolean directly for remainder (den=2 → r ∈ {0,1})
                let r_bit = Boolean::new_witness(cs.clone(), || {
                    let rem_val = self.div_remainders[l][i];
                    Ok(!rem_val.is_zero())
                })?;
                let r_i = FpVar::<Fr>::from(r_bit.clone());
                (&scale_den * &y_i + &r_i).enforce_equal(&numed)?;

                // Range check y_i to be reasonable (skip signed range for POC)
                Self::range_check_bits(&y_i, 16)?; // INT16 range for headroom
                // Carry
                next_x.push(y_i);
            }
            x_vars = next_x;
        }

        // Final output must match y_out (public)
        for i in 0..N {
            x_vars[i].enforce_equal(&y_pub[i])?;
        }
        Ok(())
    }
}