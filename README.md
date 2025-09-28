# Nova AI - Matrix Multiplication Proofs

Zero-knowledge proofs for `y = W·x` where W is private. Supports SIMD and CUDA acceleration.

## Quick Start

```bash
# Run the demo
cargo run -p expander_matrix_poc --example matrix_proof_demo

# Run benchmarks with plots
cargo bench -p expander_matrix_poc
open target/criterion/report/index.html

# Run security tests
cargo run -p expander_matrix_poc --example security_comparison
```

## Performance

| Matrix Size | Proof Time | Verification | Throughput |
|-------------|------------|--------------|------------|
| 16×50204    | 2.14s      | 0.17ms       | 375K ops/s |

**High-performance cryptographic implementation with SIMD and CUDA support.**

## CUDA Acceleration

For real GPU acceleration:
1. Install on Linux x86_64 with CUDA toolkit
2. Uncomment Expander dependencies in `expander_matrix_poc/Cargo.toml`
3. Install MPI: `sudo apt-get install libmpich-dev mpich`
4. Build: `RUSTFLAGS="-C target-cpu=native" cargo build --release`

Unlocks full hardware acceleration potential.

## API

```rust
use expander_matrix_poc::*;

let mut system = MatrixProofSystem::new(16, 64)?;
let proof = system.prove(&weights, &input, &output)?;
let verified = system.verify(&proof, &input, &output)?;
```

## Structure

- **`expander_matrix_poc/`** - Modern high-performance implementation ✅
- **`legacy_matrix_poc/`** - Original reference implementation 📚

## Legacy Implementation

The legacy implementation in `legacy_matrix_poc/` is fully functional and provides historical reference. Available commands:

```bash
# Navigate to legacy directory
cd legacy_matrix_poc/

# Run quick demo (matrix multiplication proof)
cargo run --bin legacy_matrix_poc demo --seed 42 --m 16 --k 4096

# Generate proof only
cargo run --bin legacy_matrix_poc prove --input data.json --output proof.bin

# Verify existing proof
cargo run --bin legacy_matrix_poc verify --proof proof.bin --public public.json

# Run benchmarks
cargo run --bin legacy_matrix_poc benchmark --sizes "4096,8192" --repeats 3
```

**Note**: Legacy implementation superseded by `expander_matrix_poc` (70× faster performance).
