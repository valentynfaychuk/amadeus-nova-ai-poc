use ark_bn254::Fr;
// use ark_ff::Field;  // Unused import
use std::io::Read;
use byteorder::{LittleEndian, ReadBytesExt};
use crate::EngineResult;  // Removed unused field_to_u64, u64_to_field

/// Alpha value for deterministic alpha-sum commitment
/// This is non-cryptographic but sufficient for POC
const ALPHA: u64 = 5;

/// Beta value for vector commitments
const BETA: u64 = 7;

/// Compute alpha-sum commitment for vector
/// h_X = Σ x[j] * β^j with β = 7
pub fn commit_alpha_sum_vec(x: &[i64]) -> Fr {
    let beta = Fr::from(BETA);
    let mut beta_pow = Fr::from(1u64);
    let mut acc = Fr::from(0u64);

    for &val in x {
        let val_fr = if val >= 0 {
            Fr::from(val as u64)
        } else {
            -Fr::from((-val) as u64)
        };
        acc += val_fr * beta_pow;
        beta_pow *= beta;
    }
    acc
}

/// Compute alpha-sum commitment for W1 matrix while streaming
/// h_W1 = Σ W1[i,j] * α^(index(i,j)) with α = 5
/// Uses row-major, K-tiling order as documented
pub fn commit_alpha_sum_w1<R: Read>(
    reader: &mut R,
    rows: usize,
    k: usize,
    tile_k: usize
) -> EngineResult<Fr> {
    let alpha = Fr::from(ALPHA);
    let mut alpha_pow = Fr::from(1u64);
    let mut acc = Fr::from(0u64);

    // Stream in tiles, maintaining the same order as GEMV
    for tile_start in (0..k).step_by(tile_k) {
        let tile_end = std::cmp::min(k, tile_start + tile_k);
        let tile_width = tile_end - tile_start;

        // For each row in this tile
        for _i in 0..rows {
            // Read one row's worth of this tile
            for _j in 0..tile_width {
                let weight = reader.read_i16::<LittleEndian>()? as i64;
                let weight_fr = if weight >= 0 {
                    Fr::from(weight as u64)
                } else {
                    -Fr::from((-weight) as u64)
                };

                acc += weight_fr * alpha_pow;
                alpha_pow *= alpha;
            }
        }
    }
    Ok(acc)
}

/// Compute alpha-sum commitment for a 16x16 matrix W2
pub fn commit_alpha_sum_w2(w2: &[[i64; 16]; 16]) -> Fr {
    let alpha = Fr::from(ALPHA);
    let mut alpha_pow = Fr::from(1u64);
    let mut acc = Fr::from(0u64);

    for i in 0..16 {
        for j in 0..16 {
            let weight_fr = if w2[i][j] >= 0 {
                Fr::from(w2[i][j] as u64)
            } else {
                -Fr::from((-w2[i][j]) as u64)
            };
            acc += weight_fr * alpha_pow;
            alpha_pow *= alpha;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use byteorder::WriteBytesExt;

    #[test]
    fn test_commit_vec() {
        let x = vec![1, 2, 3];
        let h_x = commit_alpha_sum_vec(&x);

        // h_X = 1*7^0 + 2*7^1 + 3*7^2 = 1 + 14 + 147 = 162
        let expected = Fr::from(162u64);
        assert_eq!(h_x, expected);
    }

    #[test]
    fn test_commit_w1_streaming() {
        // Create a simple 2x4 matrix streamed as tiles
        let matrix = [
            [1i16, 2, 3, 4],  // row 0
            [5i16, 6, 7, 8],  // row 1
        ];

        // Serialize as binary stream (row-major, tile order)
        let mut data = Vec::new();
        let tile_k = 2;

        // Tile 0: columns 0-1
        for i in 0..2 {
            for j in 0..2 {
                data.write_i16::<LittleEndian>(matrix[i][j]).unwrap();
            }
        }

        // Tile 1: columns 2-3
        for i in 0..2 {
            for j in 2..4 {
                data.write_i16::<LittleEndian>(matrix[i][j]).unwrap();
            }
        }

        let mut cursor = Cursor::new(data);
        let h_w1 = commit_alpha_sum_w1(&mut cursor, 2, 4, tile_k).unwrap();

        // With tiled order:
        // Tile 0 (cols 0-1): α^0*1 + α^1*2 + α^2*5 + α^3*6 = 1 + 10 + 125 + 750 = 886
        // Tile 1 (cols 2-3): α^4*3 + α^5*4 + α^6*7 + α^7*8 = 1875 + 5000 + 109375 + 156250 = 272500
        // Total: 886 + 272500 = 273386
        // But the actual calculation should be progressive per the function
        let mut expected_calc = Fr::from(0u64);
        let alpha = Fr::from(5u64);
        let mut alpha_pow = Fr::from(1u64);

        // Simulate the same order as the function
        for i in 0..2 {
            for j in 0..2 {
                expected_calc += Fr::from(matrix[i][j] as u64) * alpha_pow;
                alpha_pow *= alpha;
            }
        }
        for i in 0..2 {
            for j in 2..4 {
                expected_calc += Fr::from(matrix[i][j] as u64) * alpha_pow;
                alpha_pow *= alpha;
            }
        }

        let expected = expected_calc;
        assert_eq!(h_w1, expected);
    }
}