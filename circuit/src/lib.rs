
use ark_bn254::Fr;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, Result as R1CSResult};
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::fields::FieldVar;
use ark_r1cs_std::eq::EqGadget;

pub const N: usize = 3;

#[derive(Clone)]
pub struct LinLayerCircuit {
    pub w: [[Fr; N]; N],
    pub x: [Fr; N],
    pub y: [Fr; N],
}

impl ConstraintSynthesizer<Fr> for LinLayerCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> R1CSResult<()> {
        let mut x_vars: Vec<FpVar<Fr>> = Vec::with_capacity(N);
        for j in 0..N {
            x_vars.push(FpVar::<Fr>::new_input(cs.clone(), || Ok(self.x[j]))?);
        }
        let mut y_vars: Vec<FpVar<Fr>> = Vec::with_capacity(N);
        for i in 0..N {
            y_vars.push(FpVar::<Fr>::new_input(cs.clone(), || Ok(self.y[i]))?);
        }
        let mut w_vars: Vec<Vec<FpVar<Fr>>> = Vec::with_capacity(N);
        for i in 0..N {
            let mut row: Vec<FpVar<Fr>> = Vec::with_capacity(N);
            for j in 0..N {
                row.push(FpVar::<Fr>::new_witness(cs.clone(), || Ok(self.w[i][j]))?);
            }
            w_vars.push(row);
        }
        for i in 0..N {
            let mut acc = FpVar::<Fr>::zero();
            for j in 0..N {
                acc += &w_vars[i][j] * &x_vars[j];
            }
            acc.enforce_equal(&y_vars[i])?;
        }
        Ok(())
    }
}
