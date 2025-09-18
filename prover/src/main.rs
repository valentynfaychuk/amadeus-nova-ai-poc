use ark_bn254::{Bn254, Fr};
use ark_ff::PrimeField;
use ark_groth16::Groth16;
use ark_snark::SNARK;
use ark_serialize::CanonicalSerialize;
use ark_crypto_primitives::sponge::{
    poseidon::{PoseidonConfig, PoseidonSponge},
    CryptographicSponge,
};
use rand::rngs::OsRng;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use circuit::{LinChainCircuit, Quant, N, L};

#[derive(Deserialize)]
struct InputJson {
    // L layers of NxN weights
    w: [[[u64; N]; N]; L],
    x: [u64; N],
    y: [u64; N],
    // quantization config
    scale_num: u64,
    scale_den: u64,
    qmin: i64,
    qmax: i64,
}

fn to_fr(x: u64) -> Fr { Fr::from(x) }
fn to_fr_i(x: i64) -> Fr { if x>=0 { Fr::from(x as u64) } else { -Fr::from((-x) as u64) } }

fn field_to_u64(f: Fr) -> u64 {
    // Convert field element to u64 by extracting the underlying integer
    // This assumes the field element represents a small integer
    use ark_ff::PrimeField;
    let bigint = f.into_bigint();
    bigint.as_ref()[0] // Get the least significant 64-bit word
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: prover <input.json> [out_dir]");
        std::process::exit(1);
    }
    let input_path = &args[1];
    let out_dir = if args.len() >= 3 { &args[2] } else { "out" };
    std::fs::create_dir_all(out_dir)?;

    let file = File::open(input_path)?;
    let rdr = BufReader::new(file);
    let input: InputJson = serde_json::from_reader(rdr)?;

    // Convert inputs to Fr
    let w_fr = input.w.map(|mat| mat.map(|row| row.map(to_fr)));
    let x_fr = input.x.map(to_fr);
    let y_fr = input.y.map(to_fr);
    let q = Quant {
        scale_num: to_fr(input.scale_num),
        scale_den: to_fr(input.scale_den),
        qmin: to_fr_i(input.qmin),
        qmax: to_fr_i(input.qmax),
    };

    // Poseidon commitment to weights (matches circuit)
    // Using the same parameters as in the circuit
    let ark = vec![vec![Fr::from(0u64); 3]; 39]; // 8 + 31 rounds, rate + capacity
    let mds = vec![vec![Fr::from(1u64); 3]; 3]; // Simple MDS matrix for rate=2, capacity=1
    let poseidon_config = PoseidonConfig::<Fr>::new(8, 31, 5, mds, ark, 2, 1);
    let mut sponge = PoseidonSponge::<Fr>::new(&poseidon_config);

    // Absorb all weights in the same order as the circuit: w[l][i][j]
    for l in 0..L {
        for i in 0..N {
            for j in 0..N {
                sponge.absorb(&w_fr[l][i][j]);
            }
        }
    }

    // Squeeze one field element as the commitment
    let h_w = sponge.squeeze_field_elements(1)[0];

    // Compute the forward pass with witnesses to get intermediate values
    let mut layer_outputs = [[Fr::from(0u64); N]; L];
    let mut div_quotients = [[Fr::from(0u64); N]; L];
    let mut div_remainders = [[Fr::from(0u64); N]; L];

    let mut current_input = x_fr;
    for l in 0..L {
        // Matrix multiplication: t = W_l * current_input
        let mut t = [Fr::from(0u64); N];
        for i in 0..N {
            let mut acc = Fr::from(0u64);
            for j in 0..N {
                acc += w_fr[l][i][j] * current_input[j];
            }
            t[i] = acc;
        }

        // Quantization: y = floor((t * scale_num) / scale_den)
        for i in 0..N {
            // Convert to integers for proper floor division
            let t_int = field_to_u64(t[i]);
            let scale_num_int = field_to_u64(q.scale_num);
            let scale_den_int = field_to_u64(q.scale_den);

            let numed_int = t_int * scale_num_int;
            let quot_int = numed_int / scale_den_int; // Integer floor division
            let rem_int = numed_int % scale_den_int;   // Integer remainder

            div_quotients[l][i] = Fr::from(quot_int);
            div_remainders[l][i] = Fr::from(rem_int);
            layer_outputs[l][i] = Fr::from(quot_int); // The output of this layer
        }

        // Update input for next layer
        current_input = layer_outputs[l];
    }

    let circuit_with_witness = LinChainCircuit {
        w: w_fr,
        x0: x_fr,
        y_out: y_fr,
        h_w,
        q,
        layer_outputs,
        div_quotients,
        div_remainders,
    };

    // Blank circuit of same shape (zeros)
    let z = Fr::from(0u64);
    let circuit_blank = LinChainCircuit {
        w: [[[z; N]; N]; L],
        x0: [z; N],
        y_out: [z; N],
        h_w: z,
        q: Quant { scale_num: z, scale_den: Fr::from(1u64), qmin: z, qmax: z },
        layer_outputs: [[z; N]; L],
        div_quotients: [[z; N]; L],
        div_remainders: [[z; N]; L],
    };

    let mut rng = OsRng;
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(circuit_blank, &mut rng)?;

    let proof = Groth16::<Bn254>::prove(&pk, circuit_with_witness, &mut rng)?;

    let mut f_vk = BufWriter::new(File::create(format!("{}/vk.bin", out_dir))?);
    vk.serialize_uncompressed(&mut f_vk)?;

    let mut f_pf = BufWriter::new(File::create(format!("{}/proof.bin", out_dir))?);
    proof.serialize_uncompressed(&mut f_pf)?;

    // Public inputs (order must match verifier & circuit): x0 || y_out || h_w || q
    let mut pubs: Vec<Fr> = Vec::new();
    pubs.extend(x_fr.iter());
    pubs.extend(y_fr.iter());
    pubs.push(h_w);
    pubs.push(q.scale_num);
    pubs.push(q.scale_den);
    pubs.push(q.qmin);
    pubs.push(q.qmax);
    let pubs_u64: Vec<String> = pubs.iter().map(|v| v.into_bigint().to_string()).collect();
    std::fs::write(format!("{}/public_inputs.json", out_dir), serde_json::to_string_pretty(&pubs_u64)?)?;

    let ok = Groth16::<Bn254>::verify(&vk, &pubs, &proof)?;
    println!("Local verify: {}", ok);

    println!("Wrote outputs to '{}'", out_dir);
    Ok(())
}