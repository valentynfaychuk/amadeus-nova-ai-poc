use ark_bn254::{Bn254, Fr};
use ark_ff::PrimeField;
use ark_groth16::{Groth16, Proof, VerifyingKey};
use ark_snark::SNARK;
use ark_serialize::CanonicalDeserialize;
use std::fs::File;
use std::io::BufReader;

/// Batch verify many proofs (same vk) from a manifest JSON:
/// [
///   {"vk":"out/vk.bin","proof":"out/proof.bin","pubs":"out/public_inputs.json"},
///   ...
/// ]
#[derive(serde::Deserialize)]
struct Item { vk: String, proof: String, pubs: String }

fn read_vk(path: &str) -> anyhow::Result<VerifyingKey<Bn254>> {
    let f = File::open(path)?;
    let rdr = BufReader::new(f);
    Ok(VerifyingKey::deserialize_uncompressed(rdr)?)
}
fn read_proof(path: &str) -> anyhow::Result<Proof<Bn254>> {
    let f = File::open(path)?;
    let rdr = BufReader::new(f);
    Ok(Proof::deserialize_uncompressed(rdr)?)
}
fn read_pubs(path: &str) -> anyhow::Result<Vec<Fr>> {
    let f = File::open(path)?;
    let rdr = BufReader::new(f);
    let ss: Vec<String> = serde_json::from_reader(rdr)?;
    let mut out = Vec::with_capacity(ss.len());
    for s in ss {
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
        out.push(Fr::from_bigint(bi).expect("not in field"));
    }
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: aggregator <manifest.json>");
        std::process::exit(1);
    }
    let manifest_path = &args[1];
    let items: Vec<Item> = {
        let f = File::open(manifest_path)?;
        let rdr = BufReader::new(f);
        serde_json::from_reader(rdr)?
    };
    for (idx, it) in items.iter().enumerate() {
        let vk = read_vk(&it.vk)?;
        let pf = read_proof(&it.proof)?;
        let pubs = read_pubs(&it.pubs)?;
        let ok = Groth16::<Bn254>::verify(&vk, &pubs, &pf)?;
        println!("#{idx}: {}", ok);
    }
    Ok(())
}
