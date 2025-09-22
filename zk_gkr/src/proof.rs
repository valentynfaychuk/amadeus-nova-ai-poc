use crate::{Fr, Result, GkrError};
use crate::sumcheck::SumCheckProof;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use serde::{Serialize, Deserialize};
use std::io::{Read, Write};

/// Complete GKR proof structure
#[derive(Debug, Clone)]
pub struct GkrProof {
    /// Matrix dimensions (original, before padding)
    pub m: usize,
    pub k: usize,
    /// Padded dimensions (powers of 2)
    pub a: usize, // log2(padded_m)
    pub b: usize, // log2(padded_k)
    /// Merkle roots
    pub h_w: Fr,
    pub h_x: Fr,
    /// Claimed scalar value c = u^T * (W * x)
    pub c: Fr,
    /// Salt used in Fiat-Shamir
    pub salt: String,
    /// Sum-check proof
    pub sumcheck_proof: SumCheckProof,
}

/// Public inputs for GKR verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GkrPublicInputs {
    /// Matrix dimensions
    pub m: usize,
    pub k: usize,
    /// Merkle roots
    pub h_w: Fr,
    pub h_x: Fr,
    /// Claimed value
    pub c: Fr,
    /// Optional model identifier
    pub model_id: Option<String>,
    /// Optional verification key hash
    pub vk_hash: Option<String>,
    /// Random salt
    pub salt: String,
}

impl GkrProof {
    /// Create a new GKR proof
    pub fn new(
        m: usize,
        k: usize,
        h_w: Fr,
        h_x: Fr,
        c: Fr,
        salt: String,
        sumcheck_proof: SumCheckProof,
    ) -> Self {
        let a = (m as f64).log2().ceil() as usize;
        let b = (k as f64).log2().ceil() as usize;

        Self {
            m,
            k,
            a,
            b,
            h_w,
            h_x,
            c,
            salt,
            sumcheck_proof,
        }
    }

    /// Serialize proof to binary format
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();

        // Serialize dimensions
        bytes.extend_from_slice(&self.m.to_le_bytes());
        bytes.extend_from_slice(&self.k.to_le_bytes());
        bytes.extend_from_slice(&self.a.to_le_bytes());
        bytes.extend_from_slice(&self.b.to_le_bytes());

        // Serialize field elements
        self.h_w.serialize_compressed(&mut bytes)
            .map_err(|e| GkrError::SerializationError(format!("h_w serialization failed: {:?}", e)))?;

        self.h_x.serialize_compressed(&mut bytes)
            .map_err(|e| GkrError::SerializationError(format!("h_x serialization failed: {:?}", e)))?;

        self.c.serialize_compressed(&mut bytes)
            .map_err(|e| GkrError::SerializationError(format!("c serialization failed: {:?}", e)))?;

        // Serialize salt
        let salt_bytes = self.salt.as_bytes();
        bytes.extend_from_slice(&(salt_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(salt_bytes);

        // Serialize sum-check proof using binary format
        let sumcheck_bytes = self.serialize_sumcheck_proof()?;
        bytes.extend_from_slice(&(sumcheck_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&sumcheck_bytes);

        Ok(bytes)
    }

    /// Deserialize proof from binary format
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = 0;

        // Read dimensions
        if data.len() < cursor + 16 {
            return Err(GkrError::SerializationError("Insufficient data for dimensions".to_string()));
        }

        let m = u32::from_le_bytes([data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]]) as usize;
        cursor += 4;
        let k = u32::from_le_bytes([data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]]) as usize;
        cursor += 4;
        let a = u32::from_le_bytes([data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]]) as usize;
        cursor += 4;
        let b = u32::from_le_bytes([data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]]) as usize;
        cursor += 4;

        // Read field elements
        let h_w = Fr::deserialize_compressed(&data[cursor..])
            .map_err(|e| GkrError::SerializationError(format!("h_w deserialization failed: {:?}", e)))?;
        cursor += h_w.serialized_size(ark_serialize::Compress::Yes);

        let h_x = Fr::deserialize_compressed(&data[cursor..])
            .map_err(|e| GkrError::SerializationError(format!("h_x deserialization failed: {:?}", e)))?;
        cursor += h_x.serialized_size(ark_serialize::Compress::Yes);

        let c = Fr::deserialize_compressed(&data[cursor..])
            .map_err(|e| GkrError::SerializationError(format!("c deserialization failed: {:?}", e)))?;
        cursor += c.serialized_size(ark_serialize::Compress::Yes);

        // Read salt
        if data.len() < cursor + 4 {
            return Err(GkrError::SerializationError("Insufficient data for salt length".to_string()));
        }

        let salt_len = u32::from_le_bytes([data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]]) as usize;
        cursor += 4;

        if data.len() < cursor + salt_len {
            return Err(GkrError::SerializationError("Insufficient data for salt".to_string()));
        }

        let salt = String::from_utf8(data[cursor..cursor + salt_len].to_vec())
            .map_err(|e| GkrError::SerializationError(format!("Salt UTF-8 decoding failed: {}", e)))?;
        cursor += salt_len;

        // Read sum-check proof
        if data.len() < cursor + 4 {
            return Err(GkrError::SerializationError("Insufficient data for sumcheck length".to_string()));
        }

        let sumcheck_len = u32::from_le_bytes([data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]]) as usize;
        cursor += 4;

        if data.len() < cursor + sumcheck_len {
            return Err(GkrError::SerializationError("Insufficient data for sumcheck proof".to_string()));
        }

        let sumcheck_bytes = &data[cursor..cursor + sumcheck_len];
        let sumcheck_proof = Self::deserialize_sumcheck_proof(sumcheck_bytes)?;

        Ok(Self {
            m,
            k,
            a,
            b,
            h_w,
            h_x,
            c,
            salt,
            sumcheck_proof,
        })
    }

    /// Save proof to file
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes)
            .map_err(|e| GkrError::SerializationError(format!("Failed to write proof file: {}", e)))?;
        Ok(())
    }

    /// Load proof from file
    pub fn load_from_file(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| GkrError::SerializationError(format!("Failed to read proof file: {}", e)))?;
        Self::from_bytes(&bytes)
    }

    /// Get proof size in bytes
    pub fn size_bytes(&self) -> Result<usize> {
        Ok(self.to_bytes()?.len())
    }

    /// Extract public inputs from the proof
    pub fn public_inputs(&self, model_id: Option<String>, vk_hash: Option<String>) -> GkrPublicInputs {
        GkrPublicInputs {
            m: self.m,
            k: self.k,
            h_w: self.h_w,
            h_x: self.h_x,
            c: self.c,
            model_id,
            vk_hash,
            salt: self.salt.clone(),
        }
    }

    /// Serialize sum-check proof to binary format
    fn serialize_sumcheck_proof(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();

        // Serialize claimed sum
        self.sumcheck_proof.claimed_sum.serialize_compressed(&mut bytes)
            .map_err(|e| GkrError::SerializationError(format!("Claimed sum serialization failed: {:?}", e)))?;

        // Serialize round polynomials
        bytes.extend_from_slice(&(self.sumcheck_proof.round_polynomials.len() as u32).to_le_bytes());
        for poly in &self.sumcheck_proof.round_polynomials {
            bytes.extend_from_slice(&(poly.coefficients.len() as u32).to_le_bytes());
            for coeff in &poly.coefficients {
                coeff.serialize_compressed(&mut bytes)
                    .map_err(|e| GkrError::SerializationError(format!("Coefficient serialization failed: {:?}", e)))?;
            }
        }

        // Serialize challenges
        bytes.extend_from_slice(&(self.sumcheck_proof.challenges.len() as u32).to_le_bytes());
        for challenge in &self.sumcheck_proof.challenges {
            challenge.serialize_compressed(&mut bytes)
                .map_err(|e| GkrError::SerializationError(format!("Challenge serialization failed: {:?}", e)))?;
        }

        // For MLE openings, we'll use a simplified serialization
        // This is a placeholder - in practice you'd want more robust serialization
        bytes.extend_from_slice(b"MLE_PLACEHOLDER");

        Ok(bytes)
    }

    /// Deserialize sum-check proof from binary format
    fn deserialize_sumcheck_proof(data: &[u8]) -> Result<SumCheckProof> {
        use crate::sumcheck::{SumCheckProof, UnivariatePolynomial};
        use crate::mle::MleOpenProof;

        let mut cursor = 0;

        // Deserialize claimed sum
        let claimed_sum = Fr::deserialize_compressed(&data[cursor..])
            .map_err(|e| GkrError::SerializationError(format!("Claimed sum deserialization failed: {:?}", e)))?;
        cursor += claimed_sum.serialized_size(ark_serialize::Compress::Yes);

        // For now, create a dummy proof structure
        // In a full implementation, you'd deserialize all fields properly
        let round_polynomials = vec![UnivariatePolynomial::new(vec![Fr::zero()])];
        let challenges = vec![Fr::zero()];
        let w_opening = MleOpenProof {
            value: Fr::zero(),
            fold_values: vec![],
            merkle_paths: vec![],
        };
        let x_opening = MleOpenProof {
            value: Fr::zero(),
            fold_values: vec![],
            merkle_paths: vec![],
        };
        let final_point = vec![Fr::zero()];

        Ok(SumCheckProof {
            claimed_sum,
            round_polynomials,
            challenges,
            w_opening,
            x_opening,
            final_point,
        })
    }
}

impl GkrPublicInputs {
    /// Save public inputs to JSON file
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| GkrError::SerializationError(format!("JSON serialization failed: {}", e)))?;

        std::fs::write(path, json)
            .map_err(|e| GkrError::SerializationError(format!("Failed to write public inputs file: {}", e)))?;

        Ok(())
    }

    /// Load public inputs from JSON file
    pub fn load_from_file(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| GkrError::SerializationError(format!("Failed to read public inputs file: {}", e)))?;

        serde_json::from_str(&json)
            .map_err(|e| GkrError::SerializationError(format!("JSON deserialization failed: {}", e)))
    }

    /// Convert field elements to hex strings for JSON compatibility
    pub fn to_hex_format(&self) -> serde_json::Value {
        serde_json::json!({
            "m": self.m,
            "k": self.k,
            "h_w": field_to_hex(&self.h_w),
            "h_x": field_to_hex(&self.h_x),
            "c": field_to_hex(&self.c),
            "model_id": self.model_id,
            "vk_hash": self.vk_hash,
            "salt": self.salt
        })
    }

    /// Create from hex format JSON
    pub fn from_hex_format(value: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            m: value["m"].as_u64().ok_or_else(|| GkrError::SerializationError("Missing m".to_string()))? as usize,
            k: value["k"].as_u64().ok_or_else(|| GkrError::SerializationError("Missing k".to_string()))? as usize,
            h_w: field_from_hex(value["h_w"].as_str().ok_or_else(|| GkrError::SerializationError("Missing h_w".to_string()))?)?,
            h_x: field_from_hex(value["h_x"].as_str().ok_or_else(|| GkrError::SerializationError("Missing h_x".to_string()))?)?,
            c: field_from_hex(value["c"].as_str().ok_or_else(|| GkrError::SerializationError("Missing c".to_string()))?)?,
            model_id: value["model_id"].as_str().map(|s| s.to_string()),
            vk_hash: value["vk_hash"].as_str().map(|s| s.to_string()),
            salt: value["salt"].as_str().ok_or_else(|| GkrError::SerializationError("Missing salt".to_string()))?.to_string(),
        })
    }
}

/// Convert field element to hex string
fn field_to_hex(field: &Fr) -> String {
    let mut bytes = Vec::new();
    field.serialize_compressed(&mut bytes).unwrap();
    hex::encode(bytes)
}

/// Convert hex string to field element
fn field_from_hex(hex_str: &str) -> Result<Fr> {
    let bytes = hex::decode(hex_str)
        .map_err(|e| GkrError::SerializationError(format!("Invalid hex: {}", e)))?;

    Fr::deserialize_compressed(&*bytes)
        .map_err(|e| GkrError::SerializationError(format!("Field deserialization failed: {:?}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sumcheck::{SumCheckProof, UnivariatePolynomial};
    use crate::mle::MleOpenProof;
    use ark_std::test_rng;

    fn create_dummy_proof() -> GkrProof {
        let sumcheck_proof = SumCheckProof {
            claimed_sum: Fr::from(42u64),
            round_polynomials: vec![
                UnivariatePolynomial::new(vec![Fr::from(1u64), Fr::from(2u64)]),
            ],
            challenges: vec![Fr::from(3u64)],
            w_opening: MleOpenProof {
                value: Fr::from(4u64),
                fold_values: vec![vec![Fr::from(5u64)]],
                merkle_paths: vec![],
            },
            x_opening: MleOpenProof {
                value: Fr::from(6u64),
                fold_values: vec![vec![Fr::from(7u64)]],
                merkle_paths: vec![],
            },
            final_point: vec![Fr::from(8u64)],
        };

        GkrProof::new(
            16,
            4096,
            Fr::from(100u64),
            Fr::from(200u64),
            Fr::from(300u64),
            "deadbeef".to_string(),
            sumcheck_proof,
        )
    }

    #[test]
    fn test_proof_serialization() {
        let proof = create_dummy_proof();

        let bytes = proof.to_bytes().unwrap();
        let restored_proof = GkrProof::from_bytes(&bytes).unwrap();

        assert_eq!(proof.m, restored_proof.m);
        assert_eq!(proof.k, restored_proof.k);
        assert_eq!(proof.h_w, restored_proof.h_w);
        assert_eq!(proof.h_x, restored_proof.h_x);
        assert_eq!(proof.c, restored_proof.c);
        assert_eq!(proof.salt, restored_proof.salt);
    }

    #[test]
    fn test_public_inputs_hex_format() {
        let public_inputs = GkrPublicInputs {
            m: 16,
            k: 4096,
            h_w: Fr::from(123u64),
            h_x: Fr::from(456u64),
            c: Fr::from(789u64),
            model_id: Some("test_model".to_string()),
            vk_hash: Some("test_vk".to_string()),
            salt: "deadbeef".to_string(),
        };

        let hex_format = public_inputs.to_hex_format();
        let restored = GkrPublicInputs::from_hex_format(&hex_format).unwrap();

        assert_eq!(public_inputs.m, restored.m);
        assert_eq!(public_inputs.k, restored.k);
        assert_eq!(public_inputs.h_w, restored.h_w);
        assert_eq!(public_inputs.h_x, restored.h_x);
        assert_eq!(public_inputs.c, restored.c);
        assert_eq!(public_inputs.model_id, restored.model_id);
        assert_eq!(public_inputs.vk_hash, restored.vk_hash);
        assert_eq!(public_inputs.salt, restored.salt);
    }

    #[test]
    fn test_field_hex_conversion() {
        let field = Fr::from(12345u64);
        let hex_str = field_to_hex(&field);
        let restored = field_from_hex(&hex_str).unwrap();
        assert_eq!(field, restored);
    }
}