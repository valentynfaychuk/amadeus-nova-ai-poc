//! Type definitions for matrix multiplication proofs

use ark_bn254::Fr;
use serde::{Deserialize, Serialize};

/// Matrix represented as row-major Vec<Vec<Fr>>
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<Fr>>,
}

impl Matrix {
    /// Create new matrix with given dimensions
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![vec![Fr::from(0u64); cols]; rows];
        Self { rows, cols, data }
    }

    /// Create random matrix for testing
    pub fn random(rows: usize, cols: usize, rng: &mut impl rand::Rng) -> Self {
        let data = (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| Fr::from(rng.gen::<u64>() % 1000))
                    .collect()
            })
            .collect();

        Self { rows, cols, data }
    }

    /// Multiply matrix by vector: y = M·x
    pub fn multiply(&self, vector: &Vector) -> Vector {
        assert_eq!(self.cols, vector.len());

        let result_data: Vec<Fr> = (0..self.rows)
            .map(|i| {
                let mut sum = Fr::from(0u64);
                for j in 0..self.cols {
                    sum += self.data[i][j] * vector.data[j];
                }
                sum
            })
            .collect();

        Vector::from_data(result_data)
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Fr {
        self.data[row][col]
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: Fr) {
        self.data[row][col] = value;
    }
}

/// Vector represented as Vec<Fr>
#[derive(Debug, Clone)]
pub struct Vector {
    pub data: Vec<Fr>,
}

impl Vector {
    /// Create new zero vector
    pub fn new(len: usize) -> Self {
        Self {
            data: vec![Fr::from(0u64); len],
        }
    }

    /// Create vector from data
    pub fn from_data(data: Vec<Fr>) -> Self {
        Self { data }
    }

    /// Create random vector for testing
    pub fn random(len: usize, rng: &mut impl rand::Rng) -> Self {
        let data = (0..len)
            .map(|_| Fr::from(rng.gen::<u64>() % 1000))
            .collect();

        Self { data }
    }

    /// Get vector length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if vector is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Fr {
        self.data[index]
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: Fr) {
        self.data[index] = value;
    }
}

/// Proof of matrix multiplication y = W·x
#[derive(Debug, Clone)]
pub struct MatrixProof {
    /// Serialized GKR proof from Expander
    pub proof_data: Vec<u8>,

    /// Matrix dimensions
    pub m: usize,  // rows
    pub k: usize,  // cols

    /// Proof metadata
    pub claimed_output: Vector,
    pub proof_size_bytes: usize,
    pub generation_time_ms: u128,
}

/// Configuration for matrix proof system
#[derive(Debug, Clone)]
pub struct ProofConfig {
    /// Matrix dimensions
    pub m: usize,
    pub k: usize,

    /// Field type (BN254 for now)
    pub field_type: FieldType,

    /// Hash function for Fiat-Shamir
    pub hash_type: HashType,

    /// Polynomial commitment scheme
    pub pcs_type: PcsType,
}

#[derive(Debug, Clone)]
pub enum FieldType {
    BN254,
    M31Ext3,
    Goldilocks,
}

#[derive(Debug, Clone)]
pub enum HashType {
    Keccak256,
    SHA256,
    Poseidon,
}

#[derive(Debug, Clone)]
pub enum PcsType {
    Raw,
    Orion,
    Hyrax,
    KZG,
}

impl Default for ProofConfig {
    fn default() -> Self {
        Self {
            m: 16,
            k: 1024,
            field_type: FieldType::BN254,
            hash_type: HashType::Keccak256,
            pcs_type: PcsType::Raw,
        }
    }
}