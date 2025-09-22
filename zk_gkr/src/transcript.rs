use crate::{Fr, Result, GkrError, DOMAIN_GKR_V1, DOMAIN_GKR_U, DOMAIN_GKR_ROUND, DOMAIN_GKR_FINAL};
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use sha2::{Sha256, Digest};

/// Fiat-Shamir transcript for GKR protocol
#[derive(Debug, Clone)]
pub struct FiatShamirTranscript {
    state: Vec<u8>,
}

impl FiatShamirTranscript {
    /// Create a new transcript with initial domain separation
    pub fn new() -> Result<Self> {
        let mut transcript = Self {
            state: Vec::new(),
        };

        // Domain separation for GKR protocol
        transcript.absorb_bytes(DOMAIN_GKR_V1);
        Ok(transcript)
    }

    /// Absorb bytes into the transcript
    pub fn absorb_bytes(&mut self, data: &[u8]) {
        self.state.extend_from_slice(data);
    }

    /// Absorb a field element
    pub fn absorb_fr(&mut self, element: &Fr) {
        let mut bytes = Vec::new();
        element.serialize_compressed(&mut bytes)
            .expect("Field element serialization should not fail");
        self.absorb_bytes(&bytes);
    }

    /// Absorb multiple field elements
    pub fn absorb_fr_vec(&mut self, elements: &[Fr]) {
        for element in elements {
            self.absorb_fr(element);
        }
    }

    /// Absorb dimension parameters
    pub fn absorb_dimensions(&mut self, m: usize, k: usize, a: usize, b: usize) {
        self.absorb_bytes(&m.to_le_bytes());
        self.absorb_bytes(&k.to_le_bytes());
        self.absorb_bytes(&a.to_le_bytes());
        self.absorb_bytes(&b.to_le_bytes());
    }

    /// Absorb string data (e.g., model_id, vk_hash)
    pub fn absorb_string(&mut self, s: &str) {
        self.absorb_bytes(s.as_bytes());
    }

    /// Absorb hex-encoded salt
    pub fn absorb_salt(&mut self, salt_hex: &str) -> Result<()> {
        let salt_bytes = hex::decode(salt_hex)
            .map_err(|e| GkrError::TranscriptError(format!("Invalid hex salt: {}", e)))?;
        self.absorb_bytes(&salt_bytes);
        Ok(())
    }

    /// Squeeze a field element from the transcript
    pub fn squeeze_fr(&mut self) -> Result<Fr> {
        // Hash current state using SHA256
        let mut hasher = Sha256::new();
        hasher.update(&self.state);
        let hash_bytes = hasher.finalize();

        // Convert hash to field element (take first 31 bytes to stay in field)
        let mut field_bytes = [0u8; 32];
        field_bytes[..31].copy_from_slice(&hash_bytes[..31]);
        // Clear the top bit to ensure we're in the field
        field_bytes[31] = 0;

        let fr_element = Fr::from_le_bytes_mod_order(&field_bytes);

        // Update state with the squeezed value to ensure different outputs on subsequent calls
        self.absorb_fr(&fr_element);

        Ok(fr_element)
    }

    /// Squeeze multiple field elements
    pub fn squeeze_fr_vec(&mut self, count: usize) -> Result<Vec<Fr>> {
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            result.push(self.squeeze_fr()?);
        }
        Ok(result)
    }

    /// Derive challenge vector u for row compression
    pub fn derive_u_vector(&mut self, m: usize) -> Result<Vec<Fr>> {
        self.absorb_bytes(DOMAIN_GKR_U);
        self.squeeze_fr_vec(m)
    }

    /// Get round challenge for sum-check
    pub fn get_round_challenge(&mut self, round: usize) -> Result<Fr> {
        self.absorb_bytes(DOMAIN_GKR_ROUND);
        self.absorb_bytes(&round.to_le_bytes());
        self.squeeze_fr()
    }

    /// Get final challenge for terminal check
    pub fn get_final_challenge(&mut self) -> Result<Fr> {
        self.absorb_bytes(DOMAIN_GKR_FINAL);
        self.squeeze_fr()
    }

    /// Create a transcript seeded with specific data for reproducible challenges
    pub fn new_seeded(
        h_w: &Fr,
        h_x: &Fr,
        m: usize,
        k: usize,
        model_id: Option<&str>,
        vk_hash: Option<&str>,
        salt: &str,
    ) -> Result<Self> {
        let mut transcript = Self::new()?;

        // Absorb all the inputs that determine the challenge vector
        transcript.absorb_fr(h_w);
        transcript.absorb_fr(h_x);

        let a = (m as f64).log2().ceil() as usize;
        let b = (k as f64).log2().ceil() as usize;
        transcript.absorb_dimensions(m, k, a, b);

        if let Some(model_id) = model_id {
            transcript.absorb_string(model_id);
        }

        if let Some(vk_hash) = vk_hash {
            transcript.absorb_string(vk_hash);
        }

        transcript.absorb_salt(salt)?;

        Ok(transcript)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_deterministic() {
        let mut transcript1 = FiatShamirTranscript::new().unwrap();
        let mut transcript2 = FiatShamirTranscript::new().unwrap();

        transcript1.absorb_bytes(b"test");
        transcript2.absorb_bytes(b"test");

        let challenge1 = transcript1.squeeze_fr().unwrap();
        let challenge2 = transcript2.squeeze_fr().unwrap();

        assert_eq!(challenge1, challenge2);
    }

    #[test]
    fn test_transcript_different_inputs() {
        let mut transcript1 = FiatShamirTranscript::new().unwrap();
        let mut transcript2 = FiatShamirTranscript::new().unwrap();

        transcript1.absorb_bytes(b"test1");
        transcript2.absorb_bytes(b"test2");

        let challenge1 = transcript1.squeeze_fr().unwrap();
        let challenge2 = transcript2.squeeze_fr().unwrap();

        assert_ne!(challenge1, challenge2);
    }

    #[test]
    fn test_seeded_transcript() {
        let h_w = Fr::from(123u64);
        let h_x = Fr::from(456u64);
        let salt = "deadbeef";

        let mut transcript1 = FiatShamirTranscript::new_seeded(
            &h_w, &h_x, 16, 4096, Some("model1"), Some("vk1"), salt
        ).unwrap();

        let mut transcript2 = FiatShamirTranscript::new_seeded(
            &h_w, &h_x, 16, 4096, Some("model1"), Some("vk1"), salt
        ).unwrap();

        let u1 = transcript1.derive_u_vector(16).unwrap();
        let u2 = transcript2.derive_u_vector(16).unwrap();

        assert_eq!(u1, u2);
    }
}