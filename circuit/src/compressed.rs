use ark_bn254::Bn254;
use ark_groth16::{Proof, VerifyingKey};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use std::io::{Read, Write};

/// Serialize Groth16 proof using compression to minimize size (~200-300 bytes)
pub fn serialize_proof_compressed<W: Write>(
    proof: &Proof<Bn254>,
    writer: &mut W,
) -> Result<(), Box<dyn std::error::Error>> {
    proof.serialize_with_mode(writer, Compress::Yes)?;
    Ok(())
}

/// Deserialize compressed Groth16 proof
pub fn deserialize_proof_compressed<R: Read>(
    reader: &mut R,
) -> Result<Proof<Bn254>, Box<dyn std::error::Error>> {
    let proof = Proof::<Bn254>::deserialize_with_mode(reader, Compress::Yes, Validate::Yes)?;
    Ok(proof)
}

/// Serialize verification key using compression
pub fn serialize_vk_compressed<W: Write>(
    vk: &VerifyingKey<Bn254>,
    writer: &mut W,
) -> Result<(), Box<dyn std::error::Error>> {
    vk.serialize_with_mode(writer, Compress::Yes)?;
    Ok(())
}

/// Deserialize compressed verification key
pub fn deserialize_vk_compressed<R: Read>(
    reader: &mut R,
) -> Result<VerifyingKey<Bn254>, Box<dyn std::error::Error>> {
    let vk = VerifyingKey::<Bn254>::deserialize_with_mode(reader, Compress::Yes, Validate::Yes)?;
    Ok(vk)
}

/// Estimate compressed proof size in bytes (typically 192-256 bytes for BN254)
pub fn estimate_compressed_proof_size() -> usize {
    // BN254 G1 point compressed: ~32 bytes
    // BN254 G2 point compressed: ~64 bytes
    // Groth16 proof has: 2 G1 points + 1 G2 point = ~128 bytes + overhead
    // With serialization overhead, expect ~200-300 bytes
    256
}

/// Estimate verification key size in bytes
pub fn estimate_compressed_vk_size() -> usize {
    // VK contains multiple G1/G2 points depending on circuit size
    // For small circuits like 16x16, expect ~1-2KB compressed
    2048
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TinyTailCircuit;
    use ark_bn254::Fr;
    use ark_groth16::Groth16;
    use ark_snark::SNARK;
    use rand::rngs::OsRng;

    #[test]
    fn test_proof_compression_roundtrip() {
        // Create a valid circuit for testing with proper witness computation
        // Create actual identity matrix (not all 1s)
        let mut w2 = [[Fr::from(0u64); 16]; 16];
        for i in 0..16 {
            w2[i][i] = Fr::from(1u64); // Set diagonal to 1
        }
        let y1 = [Fr::from(2u64); 16]; // Input vector
        let scale_num = Fr::from(3u64);

        // Compute y2 = floor((W2 · y1) * scale_num / 2)
        // With identity matrix and y1=[2], we get: (W2 · y1)[i] = 2
        // numed = 2 * 3 = 6
        // y2[i] = floor(6 / 2) = floor(3) = 3
        let y2 = [Fr::from(3u64); 16];

        // Compute division witnesses: 6 = 2 * 3 + 0, so quotient=3, remainder=0
        let div_quotients = [Fr::from(3u64); 16];
        let div_remainders = [Fr::from(0u64); 16];

        // Compute commitments using the same functions as the circuit
        use crate::TinyTailCircuit;
        let alpha = Fr::from(5u64);
        let beta = Fr::from(7u64);

        // h_w2 = commit_matrix(w2)
        let mut h_w2 = Fr::from(0u64);
        let mut alpha_pow = Fr::from(1u64);
        for i in 0..16 {
            for j in 0..16 {
                h_w2 += w2[i][j] * alpha_pow;
                alpha_pow *= alpha;
            }
        }

        // h_y1 = commit_vector(y1)
        let mut h_y1 = Fr::from(0u64);
        let mut beta_pow = Fr::from(1u64);
        for i in 0..16 {
            h_y1 += y1[i] * beta_pow;
            beta_pow *= beta;
        }

        // h_y = commit_vector(y2)
        let mut h_y = Fr::from(0u64);
        beta_pow = Fr::from(1u64);
        for i in 0..16 {
            h_y += y2[i] * beta_pow;
            beta_pow *= beta;
        }

        let circuit = TinyTailCircuit {
            w2,
            y1,
            y2,
            scale_num,
            h_w2,
            h_x: Fr::from(42u64), // Dummy value for h_x
            h_y1,
            h_y,
            div_quotients,
            div_remainders,
        };

        let mut rng = OsRng;
        let (pk, _vk) =
            Groth16::<Bn254>::circuit_specific_setup(circuit.clone(), &mut rng).unwrap();
        let proof = Groth16::<Bn254>::prove(&pk, circuit, &mut rng).unwrap();

        // Test proof compression
        let mut proof_bytes = Vec::new();
        serialize_proof_compressed(&proof, &mut proof_bytes).unwrap();

        let proof_size = proof_bytes.len();
        println!("Compressed proof size: {} bytes", proof_size);

        // Should be significantly smaller than uncompressed
        assert!(proof_size <= estimate_compressed_proof_size());
        assert!(proof_size >= 100); // Sanity check - shouldn't be too small

        // Test roundtrip
        let mut cursor = std::io::Cursor::new(proof_bytes);
        let proof_decoded = deserialize_proof_compressed(&mut cursor).unwrap();

        // Proofs should be equal (this tests serialization correctness)
        let mut original_bytes = Vec::new();
        serialize_proof_compressed(&proof, &mut original_bytes).unwrap();

        let mut decoded_bytes = Vec::new();
        serialize_proof_compressed(&proof_decoded, &mut decoded_bytes).unwrap();

        assert_eq!(original_bytes, decoded_bytes);
    }

    #[test]
    fn test_vk_compression_roundtrip() {
        // Use a simple blank circuit for VK testing (constraints don't need to be satisfied for setup)
        let zero = Fr::from(0u64);
        let circuit = TinyTailCircuit {
            w2: [[zero; 16]; 16],
            y1: [zero; 16],
            y2: [zero; 16],
            scale_num: Fr::from(3u64),
            h_w2: zero,
            h_x: zero,
            h_y1: zero,
            h_y: zero,
            div_quotients: [zero; 16],
            div_remainders: [zero; 16],
        };

        let mut rng = OsRng;
        let (_pk, vk) = Groth16::<Bn254>::circuit_specific_setup(circuit, &mut rng).unwrap();

        // Test VK compression
        let mut vk_bytes = Vec::new();
        serialize_vk_compressed(&vk, &mut vk_bytes).unwrap();

        let vk_size = vk_bytes.len();
        println!("Compressed VK size: {} bytes", vk_size);

        assert!(vk_size <= estimate_compressed_vk_size());

        // Test roundtrip
        let mut cursor = std::io::Cursor::new(vk_bytes);
        let vk_decoded = deserialize_vk_compressed(&mut cursor).unwrap();

        // VKs should be equal
        let mut original_bytes = Vec::new();
        serialize_vk_compressed(&vk, &mut original_bytes).unwrap();

        let mut decoded_bytes = Vec::new();
        serialize_vk_compressed(&vk_decoded, &mut decoded_bytes).unwrap();

        assert_eq!(original_bytes, decoded_bytes);
    }
}
