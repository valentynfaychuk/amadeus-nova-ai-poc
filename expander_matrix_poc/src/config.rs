//! Expander GKR configuration for matrix multiplication

use crate::types::{FieldType, HashType, PcsType};

// Import Expander configuration macros and types
// Note: These imports will need to be adjusted based on actual Expander API
// use config_macros::declare_gkr_config;
// use gkr::gkr_scheme::GKRScheme;

/// GKR Configuration for BN254 field with Keccak hash
///
/// This follows Expander's configuration pattern:
/// declare_gkr_config!(
///     ConfigName,
///     FieldType::BN254,
///     HashType::Keccak256,
///     PcsType::Raw,
///     GKRScheme::Vanilla
/// );
#[derive(Debug)]
pub struct MatrixMultConfig;

// TODO: Uncomment when Expander is available
// declare_gkr_config!(
//     MatrixMultConfig,
//     FieldType::BN254,
//     FiatShamirHashType::Keccak256,
//     PolynomialCommitmentType::Raw,
//     GKRScheme::Vanilla
// );

/// Configuration builder for different setups
pub struct ConfigBuilder {
    field_type: FieldType,
    hash_type: HashType,
    pcs_type: PcsType,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            field_type: FieldType::BN254,
            hash_type: HashType::Keccak256,
            pcs_type: PcsType::Raw,
        }
    }

    pub fn field_type(mut self, field_type: FieldType) -> Self {
        self.field_type = field_type;
        self
    }

    pub fn hash_type(mut self, hash_type: HashType) -> Self {
        self.hash_type = hash_type;
        self
    }

    pub fn pcs_type(mut self, pcs_type: PcsType) -> Self {
        self.pcs_type = pcs_type;
        self
    }

    /// Build configuration (placeholder for now)
    pub fn build(self) -> MatrixMultConfig {
        MatrixMultConfig
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper functions for Expander integration
pub mod expander_utils {
    use super::*;

    /// Convert our field type to Expander's field type
    pub fn to_expander_field(field_type: &FieldType) -> &'static str {
        match field_type {
            FieldType::BN254 => "fr",
            FieldType::M31Ext3 => "m31ext3",
            FieldType::Goldilocks => "goldilocks",
        }
    }

    /// Convert our hash type to Expander's hash type
    pub fn to_expander_hash(hash_type: &HashType) -> &'static str {
        match hash_type {
            HashType::Keccak256 => "keccak",
            HashType::SHA256 => "sha256",
            HashType::Poseidon => "poseidon",
        }
    }

    /// Get MPI configuration for single-node usage
    pub fn get_mpi_config() -> MPIConfig {
        MPIConfig::single_node()
    }
}

/// MPI Configuration for Expander
/// This will need to be adapted based on actual Expander MPI requirements
#[derive(Debug, Clone)]
pub struct MPIConfig {
    pub rank: usize,
    pub size: usize,
}

impl MPIConfig {
    /// Single-node configuration (no MPI parallelism)
    pub fn single_node() -> Self {
        Self { rank: 0, size: 1 }
    }

    /// Multi-node configuration
    pub fn multi_node(rank: usize, size: usize) -> Self {
        Self { rank, size }
    }
}