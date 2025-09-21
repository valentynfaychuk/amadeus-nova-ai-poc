use ark_bn254::Fr;
use ark_ff::Field;
use std::io::{Read, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt};
use sha2::{Sha256, Digest};
use crate::{EngineResult, EngineError, i64_to_field};

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
        let r = generate_random_vector(seed, round);

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

        // Check if lhs == rhs
        if lhs != rhs {
            return Err(EngineError::FreivaldsCheckFailed { round, lhs, rhs });
        }
    }

    Ok(())
}

/// Generate deterministic random vector r ∈ F^16 for Freivalds round
fn generate_random_vector(seed: u64, round: usize) -> Vec<Fr> {
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
    r: &[Fr],
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
        let r1 = generate_random_vector(42, 0);
        let r2 = generate_random_vector(42, 1);
        let r3 = generate_random_vector(43, 0);

        assert_eq!(r1.len(), 16);
        assert_eq!(r2.len(), 16);
        assert_eq!(r3.len(), 16);

        // Different rounds should give different vectors
        assert_ne!(r1, r2);
        // Different seeds should give different vectors
        assert_ne!(r1, r3);
        // Same seed and round should give same vector
        let r1_repeat = generate_random_vector(42, 0);
        assert_eq!(r1, r1_repeat);
    }
}