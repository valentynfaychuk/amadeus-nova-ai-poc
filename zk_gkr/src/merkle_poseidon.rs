use crate::{Fr, Result, GkrError};
use ark_ff::{Zero, PrimeField};
use ark_serialize::CanonicalSerialize;
use sha2::{Sha256, Digest};

/// SHA256-based binary Merkle tree
#[derive(Debug, Clone)]
pub struct PoseidonMerkleTree {
    /// Tree nodes stored level by level (bottom up)
    nodes: Vec<Vec<Fr>>,
    /// Number of leaves
    leaf_count: usize,
    /// Tree height (log2 of next power of 2 >= leaf_count)
    height: usize,
}

/// Merkle path for opening a specific leaf
#[derive(Debug, Clone)]
pub struct MerklePath {
    /// Leaf index
    pub leaf_index: usize,
    /// Sibling hashes along the path to root
    pub siblings: Vec<Fr>,
    /// Path bits (0 = left, 1 = right)
    pub path_bits: Vec<bool>,
}

impl PoseidonMerkleTree {
    /// Build a SHA256-Merkle tree from leaves in hypercube order
    pub fn build_tree(leaves: &[Fr]) -> Result<Self> {
        if leaves.is_empty() {
            return Err(GkrError::InvalidDimensions("Empty leaves".to_string()));
        }

        let leaf_count = leaves.len();
        let height = (leaf_count as f64).log2().ceil() as usize;
        let padded_size = 1 << height;

        let mut nodes = Vec::new();

        // Bottom level: pad leaves to next power of 2
        let mut current_level = leaves.to_vec();
        current_level.resize(padded_size, Fr::zero());
        nodes.push(current_level.clone());

        // Build tree bottom-up
        for _level in 0..height {
            let mut next_level = Vec::new();
            let current_size = current_level.len();

            for i in (0..current_size).step_by(2) {
                let left = current_level[i];
                let right = if i + 1 < current_size {
                    current_level[i + 1]
                } else {
                    Fr::zero()
                };

                // Hash left and right children using SHA256
                let parent = Self::sha256_hash2(&left, &right)?;
                next_level.push(parent);
            }

            current_level = next_level;
            nodes.push(current_level.clone());
        }

        Ok(Self {
            nodes,
            leaf_count,
            height,
        })
    }

    /// Get the root of the tree
    pub fn root(&self) -> Fr {
        self.nodes[self.height][0]
    }

    /// Get a leaf value by index
    pub fn get_leaf(&self, index: usize) -> Result<Fr> {
        if index >= self.leaf_count {
            return Err(GkrError::InvalidDimensions(format!("Leaf index {} out of bounds", index)));
        }
        Ok(self.nodes[0][index])
    }

    /// Generate a Merkle path for the given leaf index
    pub fn open(&self, leaf_index: usize) -> Result<MerklePath> {
        if leaf_index >= self.leaf_count {
            return Err(GkrError::InvalidDimensions(format!("Leaf index {} out of bounds", leaf_index)));
        }

        let mut siblings = Vec::new();
        let mut path_bits = Vec::new();
        let mut current_index = leaf_index;

        // Traverse from leaf to root, collecting siblings
        for level in 0..self.height {
            let is_right = current_index % 2 == 1;
            let sibling_index = if is_right {
                current_index - 1
            } else {
                current_index + 1
            };

            // Get sibling (may be zero if out of bounds)
            let sibling = if sibling_index < self.nodes[level].len() {
                self.nodes[level][sibling_index]
            } else {
                Fr::zero()
            };

            siblings.push(sibling);
            path_bits.push(is_right);

            current_index /= 2;
        }

        Ok(MerklePath {
            leaf_index,
            siblings,
            path_bits,
        })
    }

    /// Verify a Merkle path against a root
    pub fn verify(
        root: &Fr,
        leaf_index: usize,
        leaf_value: &Fr,
        path: &MerklePath,
    ) -> Result<bool> {
        if path.leaf_index != leaf_index {
            return Ok(false);
        }

        if path.siblings.len() != path.path_bits.len() {
            return Ok(false);
        }

        let mut current_hash = *leaf_value;

        // Traverse path from leaf to root
        for (sibling, is_right) in path.siblings.iter().zip(path.path_bits.iter()) {
            let (left, right) = if *is_right {
                (*sibling, current_hash)
            } else {
                (current_hash, *sibling)
            };

            current_hash = Self::sha256_hash2(&left, &right)?;
        }

        Ok(current_hash == *root)
    }

    /// Hash two field elements using SHA256
    fn sha256_hash2(left: &Fr, right: &Fr) -> Result<Fr> {
        let mut hasher = Sha256::new();

        // Serialize both field elements
        let mut left_bytes = Vec::new();
        left.serialize_compressed(&mut left_bytes)
            .map_err(|e| GkrError::SerializationError(format!("Left serialization failed: {:?}", e)))?;

        let mut right_bytes = Vec::new();
        right.serialize_compressed(&mut right_bytes)
            .map_err(|e| GkrError::SerializationError(format!("Right serialization failed: {:?}", e)))?;

        // Hash the concatenated bytes
        hasher.update(&left_bytes);
        hasher.update(&right_bytes);
        let hash_bytes = hasher.finalize();

        // Convert hash back to field element (take first 31 bytes to stay in field)
        let mut field_bytes = [0u8; 32];
        field_bytes[..31].copy_from_slice(&hash_bytes[..31]);
        // Clear the top bit to ensure we're in the field
        field_bytes[31] = 0;

        // Convert to Fr using from_bytes_mod_order
        let hash_fr = Fr::from_le_bytes_mod_order(&field_bytes);
        Ok(hash_fr)
    }

    /// Convert matrix indices (i, j) to flat hypercube index for MLE compatibility
    pub fn matrix_to_hypercube_index(i: usize, j: usize, a: usize, b: usize) -> usize {
        // Bit-reverse the indices for hypercube ordering
        let i_bits = Self::int_to_bits(i, a);
        let j_bits = Self::int_to_bits(j, b);

        // Concatenate i_bits and j_bits
        let mut combined_bits = i_bits;
        combined_bits.extend(j_bits);

        Self::bits_to_int(&combined_bits)
    }

    /// Convert vector index j to hypercube index
    pub fn vector_to_hypercube_index(j: usize, b: usize) -> usize {
        let j_bits = Self::int_to_bits(j, b);
        Self::bits_to_int(&j_bits)
    }

    /// Convert integer to bit representation (little-endian)
    fn int_to_bits(mut n: usize, bit_count: usize) -> Vec<bool> {
        let mut bits = Vec::with_capacity(bit_count);
        for _ in 0..bit_count {
            bits.push(n & 1 == 1);
            n >>= 1;
        }
        bits
    }

    /// Convert bit representation to integer (little-endian)
    fn bits_to_int(bits: &[bool]) -> usize {
        let mut result = 0;
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                result |= 1 << i;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_build_and_verify() {
        let leaves = vec![
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
        ];

        let tree = PoseidonMerkleTree::build_tree(&leaves).unwrap();
        let root = tree.root();

        // Test opening each leaf
        for i in 0..leaves.len() {
            let path = tree.open(i).unwrap();
            let is_valid = PoseidonMerkleTree::verify(&root, i, &leaves[i], &path).unwrap();
            assert!(is_valid);
        }
    }

    #[test]
    fn test_merkle_path_invalid_leaf() {
        let leaves = vec![Fr::from(1u64), Fr::from(2u64)];
        let tree = PoseidonMerkleTree::build_tree(&leaves).unwrap();
        let root = tree.root();

        let path = tree.open(0).unwrap();
        // Try to verify with wrong leaf value
        let is_valid = PoseidonMerkleTree::verify(&root, 0, &Fr::from(999u64), &path).unwrap();
        assert!(!is_valid);
    }

    #[test]
    fn test_hypercube_indexing() {
        // Test matrix indexing
        let index = PoseidonMerkleTree::matrix_to_hypercube_index(1, 2, 2, 2);
        // i=1 (bits: [1,0]), j=2 (bits: [0,1]) -> combined: [1,0,0,1] -> 9
        assert_eq!(index, 9);

        // Test vector indexing
        let index = PoseidonMerkleTree::vector_to_hypercube_index(3, 3);
        // j=3 (bits: [1,1,0]) -> 3
        assert_eq!(index, 3);
    }

    #[test]
    fn test_single_leaf_tree() {
        let leaves = vec![Fr::from(42u64)];
        let tree = PoseidonMerkleTree::build_tree(&leaves).unwrap();
        let root = tree.root();

        let path = tree.open(0).unwrap();
        let is_valid = PoseidonMerkleTree::verify(&root, 0, &leaves[0], &path).unwrap();
        assert!(is_valid);
    }
}