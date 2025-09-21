use ark_bn254::Fr;
use ark_groth16::VerifyingKey;
use ark_bn254::Bn254;
use ark_serialize::CanonicalSerialize;
use ark_ff::{Zero, Field};
use std::io::{Read, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt};
use sha2::{Sha256, Digest};
use crate::{EngineResult, EngineError, i64_to_field};

/// Compute VK hash for transcript binding
pub fn vk_hash(vk: &VerifyingKey<Bn254>) -> [u8; 32] {
    let mut hasher = Sha256::new();

    // Serialize VK in compressed format
    let mut vk_bytes = Vec::new();
    vk.serialize_compressed(&mut vk_bytes).unwrap();

    hasher.update(&vk_bytes);
    hasher.finalize().into()
}

/// Derive transcript-bound seed
pub fn derive_seed(
    vk_hash: &[u8; 32],
    model_id: Option<&str>,
    h_w1: &str,
    h_w2: &str,
    h_x: &str,
    h_y1: &str,
    h_y: &str,
    block_entropy: Option<&str>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();

    // Domain separator
    hasher.update(b"FREIVALDSv1");

    // VK hash
    hasher.update(vk_hash);

    // Model ID (optional)
    if let Some(mid) = model_id {
        hasher.update(mid.as_bytes());
    }

    // Commitments
    hasher.update(h_w1.as_bytes());
    hasher.update(h_w2.as_bytes());
    hasher.update(h_x.as_bytes());
    hasher.update(h_y1.as_bytes());
    hasher.update(h_y.as_bytes());

    // Block entropy (optional)
    if let Some(entropy) = block_entropy {
        hasher.update(entropy.as_bytes());
    }

    hasher.finalize().into()
}

/// Freivalds probabilistic verification of y1 = W1 · x0
/// Performs k rounds of checks with soundness error ≤ 2^(-k)
pub fn freivalds_check<R: Read + Seek>(
    mut w1_reader: R,
    x0: &[i64],
    y1: &[i64; 16],
    k: usize,
    tile_k: usize,
    num_rounds: usize,
    seed: u64,
) -> EngineResult<()> {
    if x0.len() != k {
        return Err(EngineError::InvalidDimensions {
            expected: format!("input vector length {}", k),
            actual: format!("got length {}", x0.len()),
        });
    }

    // Convert inputs to field elements
    let x0_fr: Vec<Fr> = x0.iter().map(|&x| i64_to_field(x)).collect();
    let y1_fr: Vec<Fr> = y1.iter().map(|&y| i64_to_field(y)).collect();

    for round in 0..num_rounds {
        // Generate random vector r ∈ F^16 deterministically
        let r = generate_random_vector_old(seed, round);

        // Compute lhs = dot(r, y1)
        let mut lhs = Fr::from(0u64);
        for i in 0..16 {
            lhs += r[i] * y1_fr[i];
        }

        // Compute rhs = dot(W1^T * r, x0) by streaming W1
        w1_reader.seek(SeekFrom::Start(0))?; // Reset to beginning
        let r_len = r.len();
        let r_array: [Fr; 16] = r.try_into().map_err(|_| EngineError::InvalidDimensions {
            expected: "r vector length 16".to_string(),
            actual: format!("got length {}", r_len),
        })?;
        let u = compute_w1_transpose_times_r(&mut w1_reader, &r_array, k, tile_k)?;

        let mut rhs = Fr::from(0u64);
        for j in 0..k {
            rhs += u[j] * x0_fr[j];
        }

        // Check if lhs == rhs
        if lhs != rhs {
            return Err(EngineError::FreivaldsCheckFailed { round, lhs, rhs });
        }
    }

    Ok(())
}

/// Freivalds verification with transcript-bound randomness
pub fn freivalds_check_bound<R: Read + Seek>(
    mut w1_reader: R,
    x0: &[i64],
    y1: &[i64; 16],
    k: usize,
    tile_k: usize,
    num_rounds: usize,
    transcript_seed: [u8; 32],
) -> EngineResult<(Vec<[Fr; 16]>, Vec<Fr>)> {
    if x0.len() != k {
        return Err(EngineError::InvalidDimensions {
            expected: format!("input vector length {}", k),
            actual: format!("got length {}", x0.len()),
        });
    }

    // Convert inputs to field elements
    let x0_fr: Vec<Fr> = x0.iter().map(|&x| i64_to_field(x)).collect();
    let y1_fr: Vec<Fr> = y1.iter().map(|&y| i64_to_field(y)).collect();

    let mut r_matrix = Vec::with_capacity(num_rounds);
    let mut s_vector = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        // Generate random vector r ∈ F^16 deterministically from transcript
        let r = generate_random_vector(transcript_seed, round as u32);

        // Store r for reconstruction
        r_matrix.push(r);

        // Compute lhs = dot(r, y1)
        let mut lhs = Fr::from(0u64);
        for i in 0..16 {
            lhs += r[i] * y1_fr[i];
        }

        // Compute rhs = dot(W1^T * r, x0) by streaming W1
        w1_reader.seek(SeekFrom::Start(0))?; // Reset to beginning
        let u = compute_w1_transpose_times_r(&mut w1_reader, &r, k, tile_k)?;

        let mut rhs = Fr::from(0u64);
        for j in 0..k {
            rhs += u[j] * x0_fr[j];
        }

        // Store scalar for reconstruction
        s_vector.push(rhs);

        // Check if lhs == rhs
        if lhs != rhs {
            return Err(EngineError::FreivaldsCheckFailed { round, lhs, rhs });
        }
    }

    Ok((r_matrix, s_vector))
}

/// Generate deterministic random vector r ∈ F^16 for Freivalds round
pub fn generate_random_vector(seed: [u8; 32], round: u32) -> [Fr; 16] {
    let mut out = [Fr::from(0u64); 16];
    let mut ctr: u32 = 0;

    for i in 0..16 {
        let mut hasher = Sha256::new();
        hasher.update(b"FREIVALDS_R");
        hasher.update(&seed);
        hasher.update(&round.to_be_bytes());
        hasher.update(&ctr.to_be_bytes());
        let digest = hasher.finalize();

        // Map to field; use first 8 bytes as u64 then convert to field element
        let mut bytes_array = [0u8; 8];
        bytes_array.copy_from_slice(&digest[..8]);
        let val = u64::from_le_bytes(bytes_array);
        let fr = Fr::from(val);
        out[i] = fr;
        ctr = ctr.wrapping_add(1);
    }

    out
}

/// Solve linear 16x16 system over Fr using Gaussian elimination
pub fn solve_linear_16(mut a_t: [[Fr; 16]; 16], mut b: [Fr; 16]) -> Option<[Fr; 16]> {
    for col in 0..16 {
        // find pivot
        let mut piv = None;
        for r in col..16 {
            if !a_t[r][col].is_zero() {
                piv = Some(r);
                break;
            }
        }
        let p = piv?;
        if p != col {
            a_t.swap(p, col);
            b.swap(p, col);
        }

        let inv = a_t[col][col].inverse()?;
        for j in col..16 {
            a_t[col][j] *= inv;
        }
        b[col] *= inv;

        for r in 0..16 {
            if r == col {
                continue;
            }
            let f = a_t[r][col];
            if f.is_zero() {
                continue;
            }
            for j in col..16 {
                a_t[r][j] -= f * a_t[col][j];
            }
            b[r] -= f * b[col];
        }
    }
    Some(b)
}

/// Compute matrix rank over Fr
pub fn matrix_rank(matrix: &[[Fr; 16]; 16]) -> usize {
    let mut a = *matrix;
    let mut rank = 0;

    for col in 0..16 {
        // find pivot
        let mut piv = None;
        for r in rank..16 {
            if !a[r][col].is_zero() {
                piv = Some(r);
                break;
            }
        }

        if let Some(p) = piv {
            if p != rank {
                a.swap(p, rank);
            }

            let inv = a[rank][col].inverse().unwrap();
            for j in col..16 {
                a[rank][j] *= inv;
            }

            for r in 0..16 {
                if r == rank {
                    continue;
                }
                let f = a[r][col];
                if f.is_zero() {
                    continue;
                }
                for j in col..16 {
                    a[r][j] -= f * a[rank][j];
                }
            }

            rank += 1;
        }
    }

    rank
}

/// Old interface for compatibility
fn generate_random_vector_old(seed: u64, round: usize) -> Vec<Fr> {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update(round.to_le_bytes());
    let hash = hasher.finalize();

    let mut r = Vec::with_capacity(16);
    for i in 0..16 {
        // Use 8 bytes from hash for each field element
        let start = (i * 8) % 32;
        let end = start + 8;
        let bytes = &hash[start..end];

        let mut val_bytes = [0u8; 8];
        val_bytes.copy_from_slice(bytes);
        let val = u64::from_le_bytes(val_bytes);

        // Reduce modulo field size to get uniform distribution
        r.push(Fr::from(val));
    }

    // If we need more than 4 elements, rehash
    if r.len() < 16 {
        let mut hasher2 = Sha256::new();
        hasher2.update(hash);
        let hash2 = hasher2.finalize();

        for i in 4..16 {
            let start = ((i - 4) * 8) % 32;
            let end = start + 8;
            let bytes = &hash2[start..end];

            let mut val_bytes = [0u8; 8];
            val_bytes.copy_from_slice(bytes);
            let val = u64::from_le_bytes(val_bytes);

            r.push(Fr::from(val));
        }
    }

    r.truncate(16);
    r
}

/// Compute u = W1^T * r by streaming W1 in tiles
/// Returns vector u ∈ F^k such that u[j] = Σ_i W1[i,j] * r[i]
fn compute_w1_transpose_times_r<R: Read>(
    w1_reader: &mut R,
    r: &[Fr; 16],
    k: usize,
    tile_k: usize,
) -> EngineResult<Vec<Fr>> {
    let mut u = vec![Fr::from(0u64); k];

    // Process in tiles
    for tile_start in (0..k).step_by(tile_k) {
        let tile_end = std::cmp::min(k, tile_start + tile_k);
        let tile_width = tile_end - tile_start;

        // Read current tile: 16 × tile_width
        for i in 0..16 {
            for j in 0..tile_width {
                let weight = w1_reader.read_i16::<LittleEndian>()? as i64;
                let weight_fr = i64_to_field(weight);

                // u[tile_start + j] += W1[i, tile_start + j] * r[i]
                u[tile_start + j] += weight_fr * r[i];
            }
        }
    }

    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use byteorder::WriteBytesExt;

    #[test]
    fn test_freivalds_correct_computation() {
        // Create a simple 16×4 matrix and vector
        let k = 4;
        let tile_k = 2;
        let mut w1_data = Vec::new();
        let mut w1_matrix = vec![vec![0i16; k]; 16];

        // Fill with simple test data
        for i in 0..16 {
            for j in 0..k {
                w1_matrix[i][j] = (i + j + 1) as i16;
            }
        }

        // Serialize in tile order
        for tile_start in (0..k).step_by(tile_k) {
            let tile_end = std::cmp::min(k, tile_start + tile_k);
            for i in 0..16 {
                for j in tile_start..tile_end {
                    w1_data.write_i16::<LittleEndian>(w1_matrix[i][j]).unwrap();
                }
            }
        }

        let x0 = vec![1i64, 2, 3, 4];

        // Compute correct y1 = W1 * x0
        let mut y1 = [0i64; 16];
        for i in 0..16 {
            for j in 0..k {
                y1[i] += (w1_matrix[i][j] as i64) * x0[j];
            }
        }

        // Freivalds check should pass
        let cursor = Cursor::new(w1_data);
        let result = freivalds_check(cursor, &x0, &y1, k, tile_k, 10, 42);
        assert!(result.is_ok());
    }

    #[test]
    fn test_freivalds_detects_error() {
        let k = 4;
        let tile_k = 2;
        let mut w1_data = Vec::new();

        // Create identity-like matrix
        for i in 0..16 {
            for _j in 0..k {
                w1_data.write_i16::<LittleEndian>(if i < k { 1 } else { 0 }).unwrap();
            }
        }

        let x0 = vec![1i64, 2, 3, 4];
        let mut y1_wrong = [0i64; 16];
        y1_wrong[0] = 999; // Deliberately wrong

        // Freivalds check should fail
        let cursor = Cursor::new(w1_data);
        let result = freivalds_check(cursor, &x0, &y1_wrong, k, tile_k, 5, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_vector_generation() {
        let r1 = generate_random_vector_old(42, 0);
        let r2 = generate_random_vector_old(42, 1);
        let r3 = generate_random_vector_old(43, 0);

        assert_eq!(r1.len(), 16);
        assert_eq!(r2.len(), 16);
        assert_eq!(r3.len(), 16);

        // Different rounds should give different vectors
        assert_ne!(r1, r2);
        // Different seeds should give different vectors
        assert_ne!(r1, r3);
        // Same seed and round should give same vector
        let r1_repeat = generate_random_vector_old(42, 0);
        assert_eq!(r1, r1_repeat);
    }
}