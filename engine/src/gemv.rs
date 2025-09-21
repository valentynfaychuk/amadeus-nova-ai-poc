use ark_bn254::Fr;
use ark_ff::Field;
use std::io::Read;
use byteorder::{LittleEndian, ReadBytesExt};
use crate::{EngineResult, commitment::commit_alpha_sum_w1};

/// Tiled GEMV implementation for streaming large matrices
/// Computes y1 = W1 · x0 where W1 is 16×K and x0 is length K
/// Returns both the result y1 and the commitment h_W1
pub fn gemv_tiled<R: Read>(
    mut w1_reader: R,
    x0: &[i64],
    k: usize,
    tile_k: usize,
) -> EngineResult<([i64; 16], Fr)> {
    if x0.len() != k {
        return Err(crate::EngineError::InvalidDimensions {
            expected: format!("input vector length {}", k),
            actual: format!("got length {}", x0.len()),
        });
    }

    let mut y1 = [0i64; 16];
    let alpha = Fr::from(5u64); // Alpha for commitment
    let mut alpha_pow = Fr::from(1u64);
    let mut h_w1_acc = Fr::from(0u64);

    // Process in tiles to avoid loading entire matrix into memory
    for tile_start in (0..k).step_by(tile_k) {
        let tile_end = std::cmp::min(k, tile_start + tile_k);
        let tile_width = tile_end - tile_start;

        // Read current tile: 16 × tile_width
        let mut w1_tile = vec![vec![0i16; tile_width]; 16];

        for i in 0..16 {
            for j in 0..tile_width {
                w1_tile[i][j] = w1_reader.read_i16::<LittleEndian>()?;

                // Update commitment in the same order as reading
                let weight = w1_tile[i][j] as i64;
                let weight_fr = if weight >= 0 {
                    Fr::from(weight as u64)
                } else {
                    -Fr::from((-weight) as u64)
                };
                h_w1_acc += weight_fr * alpha_pow;
                alpha_pow *= alpha;
            }
        }

        // Compute y1 += W1_tile @ x0[tile_start..tile_end]
        for i in 0..16 {
            for j in 0..tile_width {
                y1[i] += (w1_tile[i][j] as i64) * x0[tile_start + j];
            }
        }
    }

    Ok((y1, h_w1_acc))
}

/// Compute small 16×16 layer: y2 = floor((W2 · y1) * scale_num / 2)
pub fn compute_layer_16x16(w2: &[[i64; 16]; 16], y1: &[i64; 16], scale_num: u64) -> [i64; 16] {
    let mut y2 = [0i64; 16];

    for i in 0..16 {
        let mut acc = 0i64;
        for j in 0..16 {
            acc += w2[i][j] * y1[j];
        }
        // Quantized floor: y = floor((acc * scale_num) / 2)
        // Use proper floor division for negative numbers
        let numerator = acc * (scale_num as i64);
        y2[i] = if numerator >= 0 {
            numerator / 2
        } else {
            // For negative numbers: floor(a/b) = (a - (b-1)) / b
            (numerator - 1) / 2
        };
    }

    y2
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use byteorder::WriteBytesExt;

    #[test]
    fn test_gemv_tiled_vs_naive() {
        // Create a test matrix 16×8 with tile_k=4
        let k = 8;
        let tile_k = 4;
        let mut w1_data = Vec::new();
        let mut w1_matrix = vec![vec![0i16; k]; 16];

        // Fill with test data
        for i in 0..16 {
            for j in 0..k {
                w1_matrix[i][j] = ((i + 1) * (j + 1)) as i16;
            }
        }

        // Serialize in tile order for streaming
        for tile_start in (0..k).step_by(tile_k) {
            let tile_end = std::cmp::min(k, tile_start + tile_k);
            for i in 0..16 {
                for j in tile_start..tile_end {
                    w1_data.write_i16::<LittleEndian>(w1_matrix[i][j]).unwrap();
                }
            }
        }

        let x0: Vec<i64> = (1..=k).map(|i| i as i64).collect();

        // Test tiled GEMV
        let cursor = Cursor::new(w1_data);
        let (y1_tiled, _h_w1) = gemv_tiled(cursor, &x0, k, tile_k).unwrap();

        // Compute naive GEMV for comparison
        let mut y1_naive = [0i64; 16];
        for i in 0..16 {
            for j in 0..k {
                y1_naive[i] += (w1_matrix[i][j] as i64) * x0[j];
            }
        }

        assert_eq!(y1_tiled, y1_naive);
    }

    #[test]
    fn test_compute_layer_16x16() {
        let mut w2 = [[0i64; 16]; 16];
        let mut y1 = [0i64; 16];

        // Simple test case
        for i in 0..16 {
            w2[i][i] = 2; // Identity * 2
            y1[i] = i as i64 + 1; // [1, 2, 3, ..., 16]
        }

        let y2 = compute_layer_16x16(&w2, &y1, 3);

        // Expected: y2[i] = floor((2 * (i+1) * 3) / 2) = floor(3 * (i+1)) = 3 * (i+1)
        for i in 0..16 {
            let expected = 3 * (i as i64 + 1);
            assert_eq!(y2[i], expected);
        }
    }

    #[test]
    fn test_floor_division_negative() {
        // Test negative floor division specifically
        let w2 = [[-1i64; 16]; 16]; // All -1s
        let y1 = [1i64; 16]; // All 1s
        let scale_num = 3;

        let y2 = compute_layer_16x16(&w2, &y1, scale_num);

        // acc = -1 * 1 * 16 = -16
        // numerator = -16 * 3 = -48
        // floor(-48 / 2) = floor(-24) = -24
        let expected = -24i64;
        assert_eq!(y2[0], expected);

        // Test borderline case: numerator = -3
        // floor(-3 / 2) = floor(-1.5) = -2
        let w2_border = [[0i64; 16]; 16];
        let mut w2_test = w2_border;
        w2_test[0][0] = -1;
        let y1_test = [3i64; 16];
        let y2_border = compute_layer_16x16(&w2_test, &y1_test, 1);
        // acc = -1 * 3 = -3, numerator = -3 * 1 = -3
        // floor(-3 / 2) = -2
        assert_eq!(y2_border[0], -2);
    }
}