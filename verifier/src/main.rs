
use ark_bn254::{Bn254, Fr};
use ark_groth16::{Groth16, Proof, VerifyingKey};
use ark_snark::SNARK;
use ark_serialize::CanonicalDeserialize;
use std::fs::File;
use std::io::BufReader;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: verifier <vk.bin> <proof.bin> <public_inputs.json>");
        std::process::exit(1);
    }
    let vk_path = &args[1];
    let pf_path = &args[2];
    let pub_path = &args[3];

    let vk: VerifyingKey<Bn254> = {
        let f = File::open(vk_path)?;
        let rdr = BufReader::new(f);
        VerifyingKey::deserialize_uncompressed(rdr)?
    };

    let proof: Proof<Bn254> = {
        let f = File::open(pf_path)?;
        let rdr = BufReader::new(f);
        Proof::deserialize_uncompressed(rdr)?
    };

    let pubs_u64: Vec<u64> = {
        let f = File::open(pub_path)?;
        let rdr = BufReader::new(f);
        serde_json::from_reader(rdr)?
    };
    let pubs_fr: Vec<Fr> = pubs_u64.iter().map(|v| Fr::from(*v)).collect();

    let ok = Groth16::<Bn254>::verify(&vk, &pubs_fr, &proof)?;
    println!("Verify result: {}", ok);
    Ok(())
}
