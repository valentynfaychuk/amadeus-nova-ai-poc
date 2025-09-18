
use ark_bn254::{Bn254, Fr};
use ark_groth16::Groth16;
use ark_snark::SNARK;
use ark_serialize::CanonicalSerialize;
use rand::rngs::OsRng;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use circuit::{LinLayerCircuit, N};

#[derive(Deserialize)]
struct InputJson {
    w: [[u64; N]; N],
    x: [u64; N],
    y: [u64; N],
}

fn to_fr(x: u64) -> Fr { Fr::from(x) }

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

    let w_fr = input.w.map(|row| row.map(to_fr));
    let x_fr = input.x.map(to_fr);
    let y_fr = input.y.map(to_fr);

    let circuit_with_witness = LinLayerCircuit { w: w_fr, x: x_fr, y: y_fr };

    let zero = Fr::from(0u64);
    let circuit_blank = LinLayerCircuit { w: [[zero; N]; N], x: [zero; N], y: [zero; N] };

    let mut rng = OsRng;
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(circuit_blank, &mut rng)?;

    let proof = Groth16::<Bn254>::prove(&pk, circuit_with_witness, &mut rng)?;

    let mut f_vk = BufWriter::new(File::create(format!("{}/vk.bin", out_dir))?);
    vk.serialize_uncompressed(&mut f_vk)?;

    let mut f_pf = BufWriter::new(File::create(format!("{}/proof.bin", out_dir))?);
    proof.serialize_uncompressed(&mut f_pf)?;

    let pubs: Vec<u64> = input.x.iter().chain(input.y.iter()).cloned().collect();
    std::fs::write(format!("{}/public_inputs.json", out_dir), serde_json::to_string_pretty(&pubs)?)?;

    let pub_fr: Vec<Fr> = pubs.iter().map(|v| Fr::from(*v)).collect();
    let ok = Groth16::<Bn254>::verify(&vk, &pub_fr, &proof)?;
    println!("Local verify: {}", ok);

    println!("Wrote outputs to '{}'", out_dir);
    Ok(())
}
