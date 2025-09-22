use anyhow::Result;
use ark_bn254::Fr;
use ark_ff::PrimeField;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
// use flate2::read::GzDecoder;

/// Run data structure saved to run.json
#[derive(Serialize, Deserialize, Clone)]
pub struct RunData {
    /// Configuration used for this run
    pub config: RunConfig,
    /// Input vector x0 (length K)
    pub x0: Vec<i64>,
    /// Output of large layer y1 (length 16)
    pub y1: Vec<i64>,
    /// Final output y2 (length 16)
    pub y2: Vec<i64>,
    /// 16×16 tail layer weights (for circuit witness)
    pub w2: Vec<Vec<i64>>,
    /// Commitments (as decimal strings for JSON compatibility)
    pub commitments: Commitments,
    /// Freivalds audit result
    pub freivalds_result: Option<FreivaldsResult>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RunConfig {
    pub k: usize,
    pub tile_k: usize,
    pub scale_num: u64,
    pub seed: u64,
    pub freivalds_rounds: usize,
    pub skip_freivalds: bool,
    /// Optional model identifier for transcript binding
    pub model_id: Option<String>,
    /// VK hash for transcript binding (hex string)
    pub vk_hash: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Commitments {
    /// Commitment to W1 (large layer weights)
    pub h_w1: String,
    /// Commitment to W2 (16x16 tail weights)
    pub h_w2: String,
    /// Commitment to input x0
    pub h_x: String,
    /// Commitment to intermediate y1
    pub h_y1: String,
    /// Commitment to final output y2
    pub h_y: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct FreivaldsResult {
    pub seed: u64,
    pub rounds: usize,
    pub passed: bool,
    pub failed_round: Option<usize>,
    pub soundness_bits: f64, // log2(soundness error)
}

/// Load 16×K weights from binary file (i16, little endian, row-major)
/// Returns a File that implements both Read and Seek
pub fn load_weights1<P: AsRef<Path>>(path: P, _k: usize) -> Result<File> {
    let file = File::open(&path)?;
    // For now, we don't support .gz files in places that need Seek
    // TODO: Add support by decompressing to memory or temp file
    Ok(file)
}

/// Load 16×16 weights from JSON file
pub fn load_weights2<P: AsRef<Path>>(path: P) -> Result<[[i64; 16]; 16]> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let weights: Vec<Vec<i64>> = serde_json::from_reader(reader)?;

    if weights.len() != 16 {
        anyhow::bail!("Expected 16 rows, got {}", weights.len());
    }

    let mut result = [[0i64; 16]; 16];
    for (i, row) in weights.iter().enumerate() {
        if row.len() != 16 {
            anyhow::bail!("Row {} has {} columns, expected 16", i, row.len());
        }
        for (j, &val) in row.iter().enumerate() {
            result[i][j] = val;
        }
    }

    Ok(result)
}

/// Load input vector from JSON file
pub fn load_vector<P: AsRef<Path>>(path: P) -> Result<Vec<i64>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let vector: Vec<i64> = serde_json::from_reader(reader)?;
    Ok(vector)
}

/// Save run data to JSON file
pub fn save_run_data<P: AsRef<Path>>(path: P, data: &RunData) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, data)?;
    Ok(())
}

/// Load run data from JSON file
pub fn load_run_data<P: AsRef<Path>>(path: P) -> Result<RunData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let data: RunData = serde_json::from_reader(reader)?;
    Ok(data)
}

/// Convert field element to decimal string for JSON
pub fn field_to_string(f: Fr) -> String {
    f.into_bigint().to_string()
}

/// Convert decimal string to field element
pub fn string_to_field(s: &str) -> Result<Fr> {
    use num_bigint::BigUint;
    let bigint: BigUint = s.parse()?;
    Ok(Fr::from(bigint))
}

/// Save public inputs as JSON array of decimal strings
pub fn save_public_inputs<P: AsRef<Path>>(path: P, inputs: &[Fr]) -> Result<()> {
    let strings: Vec<String> = inputs.iter().map(|f| field_to_string(*f)).collect();
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &strings)?;
    Ok(())
}

/// Load public inputs from JSON array of decimal strings
pub fn load_public_inputs<P: AsRef<Path>>(path: P) -> Result<Vec<Fr>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let strings: Vec<String> = serde_json::from_reader(reader)?;

    let mut inputs = Vec::new();
    for s in strings {
        inputs.push(string_to_field(&s)?);
    }
    Ok(inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_field_string_roundtrip() {
        let original = Fr::from(12345u64);
        let string = field_to_string(original);
        let recovered = string_to_field(&string).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_run_data_serialization() {
        let data = RunData {
            config: RunConfig {
                k: 4096,
                tile_k: 1024,
                scale_num: 3,
                seed: 42,
                freivalds_rounds: 32,
                skip_freivalds: false,
                model_id: Some("test-model".to_string()),
                vk_hash: Some("deadbeef".to_string()),
            },
            x0: vec![1, 2, 3, 4],
            y1: vec![5; 16],
            y2: vec![6; 16],
            w2: vec![vec![1; 16]; 16], // Identity matrix for test
            commitments: Commitments {
                h_w1: "123".to_string(),
                h_w2: "456".to_string(),
                h_x: "789".to_string(),
                h_y1: "101112".to_string(),
                h_y: "131415".to_string(),
            },
            freivalds_result: Some(FreivaldsResult {
                seed: 42,
                rounds: 32,
                passed: true,
                failed_round: None,
                soundness_bits: 32.0,
            }),
        };

        let temp_file = NamedTempFile::new().unwrap();
        save_run_data(temp_file.path(), &data).unwrap();
        let loaded = load_run_data(temp_file.path()).unwrap();

        assert_eq!(data.config.k, loaded.config.k);
        assert_eq!(data.x0, loaded.x0);
        assert_eq!(data.commitments.h_w1, loaded.commitments.h_w1);
    }
}
