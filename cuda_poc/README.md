# CUDA POC - zkCUDA Matrix Multiplication

GPU-accelerated zero-knowledge proofs for matrix multiplication using Polyhedra's zkCUDA framework.

## Overview

This implementation uses the **ExpanderCompilerCollection** zkCUDA programming model to create GPU-accelerated GKR proofs. It provides:

- ✅ **GPU Acceleration**: Automatic CUDA kernel generation for proof computation
- ✅ **Zero-Knowledge**: Matrix multiplication proofs without revealing the matrices
- ✅ **High Performance**: Parallel execution on GPU using zkCUDA framework
- ✅ **Clean API**: Simple Rust interface for proving and verification

## zkCUDA Architecture

```
┌─────────────────────────┐
│   Rust zkCUDA Kernels   │  ← #[kernel] macros
│  (mul_line, sum_8)      │
└──────────┬──────────────┘
           │ compile_*()
           ↓
┌─────────────────────────┐
│   GKR Circuit Graph     │  ← Automatic circuit generation
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│   GPU Execution         │  ← zkSMs (zk Streaming Multiprocessors)
│   (CUDA Backend)        │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│   Zero-Knowledge Proof  │  ← Compact proof output
└─────────────────────────┘
```

## Usage

### Basic Example

```rust
use cuda_poc::*;

// Create proof system
let system = MatrixProofSystem::new()?;

// Create matrices (64×32 and 32×64)
let mat_a = create_matrix_64x32();
let mat_b = create_matrix_32x64();

// Generate proof on GPU
let (result, proof) = system.prove(&mat_a, &mat_b)?;

// Verify proof
let verified = system.verify(&mat_a, &mat_b, result, &proof)?;
assert!(verified);
```

### Run Demo

```bash
# Build and run
cargo run -p cuda_poc --example demo --release

# Expected output:
# 🚀 zkCUDA Matrix Multiplication Proof Demo
# ✅ Proof generated in ~XXms
# ✅ Proof verified successfully in ~XXms
```

## Implementation Details

### Kernels

1. **`mul_line`**: Matrix multiplication kernel
   - Input: Row vector `a[32]`, Matrix `b[32×64]`
   - Output: Result vector `c[64]`
   - Operation: `c[j] = Σ(a[i] * b[i][j])`

2. **`sum_8_elements`**: Reduction kernel
   - Input: Array `a[8]`
   - Output: Sum of all elements
   - Used for parallel reduction of results

### Proof Flow

1. **Matrix Multiplication**: 64×32 × 32×64 → 64×64 result matrix (4096 elements)
2. **Parallel Reduction**: Multiple reduction stages using `sum_8_elements`
   - 4096 → 512 → 64 → 8 → 1
3. **Proof Generation**: Convert computation graph to ZK proof
4. **Verification**: Verify proof using public computation graph

## Dependencies

- **ExpanderCompilerCollection** (zkcuda branch): zkCUDA compiler and runtime
- M31 field arithmetic for circuit operations

## Performance

| Matrix Size | Proving Time | Verification | Proof Size |
|-------------|--------------|--------------|------------|
| 64×32×32×64 | ~XXms (GPU)  | ~XXms        | ~XX KB     |

*Benchmarks TBD on actual CUDA hardware*

## Comparison with Legacy

| Feature | Legacy (CPU) | zkCUDA (GPU) |
|---------|--------------|--------------|
| Backend | Custom GKR   | Expander     |
| Acceleration | AVX/SIMD | CUDA         |
| Proving | Sequential   | Parallel     |
| Field | BN254        | M31          |

## Building

### Prerequisites

- Rust nightly (required by ExpanderCompilerCollection)
- CUDA toolkit (for GPU acceleration)

### Build Commands

```bash
# Build (will auto-detect CUDA)
cargo build -p cuda_poc --release

# Run tests
cargo test -p cuda_poc

# Run example
cargo run -p cuda_poc --example demo --release
```

## References

- [Polyhedra zkCUDA Docs](https://docs.polyhedra.network/expander/cuda/)
- [ExpanderCompilerCollection](https://github.com/PolyhedraZK/ExpanderCompilerCollection)
- [Expander GKR](https://github.com/PolyhedraZK/Expander)
