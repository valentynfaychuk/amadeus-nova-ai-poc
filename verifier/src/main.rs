use ark_bn254::{Bn254, Fr};
use ark_ff::PrimeField;
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

    // Read public inputs as decimal strings (Fr elements)
    let pubs_str: Vec<String> = {
        let f = File::open(pub_path)?;
        let rdr = BufReader::new(f);
        serde_json::from_reader(rdr)?
    };
    let mut pubs_fr: Vec<Fr> = Vec::with_capacity(pubs_str.len());
    for s in &pubs_str {
        // Parse string to u64 array then to BigInt
        let val = s.parse::<num_bigint::BigUint>()
            .map_err(|e| anyhow::anyhow!("Failed to parse bigint: {}", e))?;
        let bytes = val.to_bytes_le();
        let mut limbs = [0u64; 4];
        for (i, chunk) in bytes.chunks(8).enumerate() {
            if i >= 4 { break; }
            let mut arr = [0u8; 8];
            for (j, &byte) in chunk.iter().enumerate() {
                arr[j] = byte;
            }
            limbs[i] = u64::from_le_bytes(arr);
        }
        let bi = <Fr as PrimeField>::BigInt::new(limbs);
        pubs_fr.push(Fr::from_bigint(bi).expect("not in field"));
    }

    let ok = Groth16::<Bn254>::verify(&vk, &pubs_fr, &proof)?;
    println!("Verify result: {}", ok);
    Ok(())
}