# Nova AI: Freivalds + GEMV + Tiny Groth16 ZK System

A zero-knowledge proof system combining **tiled GEMV inference**, **Freivalds probabilistic auditing**, and **tiny Groth16 proofs** for efficient verification of large linear algebraic computations while maintaining proof sizes under 1 KB.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Large Layer   │    │   Freivalds      │    │  Tiny Groth16   │
│   W1 (16×K)     │ -> │   Auditor        │ -> │  Tail Proof     │
│   Tiled GEMV    │    │   k rounds       │    │  (16×16 only)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        |                       |                       |
   Streams tiles           Probabilistic            Commitment-based
   No huge RAM              ~2^(-k) error           < 1 KB total TX
```

### Key Components

1. **Engine Crate** (`engine/`): GEMV tiling, commitments, Freivalds checks
2. **Circuit Crate** (`circuit/`): Tiny 16×16 Groth16 circuit with compressed proofs
3. **Nova POC Binary** (`nova_poc/`): CLI tool with `infer`/`prove`/`verify` commands

## Quick Start

### Demo Mode (Fastest)

```bash
# Run complete end-to-end demo in ~10 seconds
cargo run --release -p nova_poc -- demo --seed 42
```

This generates random data and runs the full pipeline: inference → key setup → proof → verification.

### Production-Scale Example (K=50240)

```bash
# 1. Generate test data for large-scale inference (16×50240 matrix)
python3 scripts/gen.py --k 50240

# 2. Run the generated script (or use manual commands below)
./run_inference.sh

# Manual commands:
cargo run --release -p nova_poc -- infer \
    --k 50240 \
    --tile-k 4096 \
    --weights1-path data/w1_16x50240.bin \
    --weights2-path data/w2_16x16.json \
    --x0-path data/x0_50240.json \
    --out data/run_k50240.json

# 3. Generate proof (first time will setup keys)
cargo run --release -p nova_poc -- setup --out-dir keys  # One-time
cargo run --release -p nova_poc -- prove data/run_k50240.json --pk-path keys/pk.bin --out-dir data/out

# 4. Verify proof
cargo run --release -p nova_poc -- verify data/run_k50240.json --weights1-path data/w1_16x50240.bin
```

### Manual Pipeline

```bash
# 1. Generate/prepare your data files
cargo run --release -p nova_poc -- infer \
    --preset demo \
    --k 4096 \
    --tile-k 1024 \
    --freivalds-rounds 32 \
    --out run.json

# 2. Setup proving keys (one-time)
cargo run --release -p nova_poc -- setup --out-dir keys

# 3. Generate proof
cargo run --release -p nova_poc -- prove run.json --pk-path keys/pk.bin --out-dir proof_out

# 4. Verify proof
cargo run --release -p nova_poc -- verify run.json \
    --vk-path proof_out/vk.bin \
    --proof-path proof_out/proof.bin \
    --public-inputs-path proof_out/public_inputs.json \
    --weights1-path path/to/weights1.bin
```

## Technical Specifications

### Tiled GEMV Engine

- **Purpose**: Compute `y1 = W1 · x0` for large matrices (16×K) without loading entire matrix into RAM
- **Streaming**: Processes matrix in tiles (default 1024×16), supports K=4096, 8192, etc.
- **Commitment**: Computes α-sum commitment `h_W1 = Σ W[i,j] * α^(index)` during streaming

```rust
// Example: 16×4096 matrix processed in 1024-width tiles
let (y1, h_w1) = gemv_tiled(&mut w1_reader, &x0, 4096, 1024)?;
```

### Freivalds Probabilistic Auditor

- **Purpose**: Verify `y1 = W1 · x0` without full recomputation
- **Method**: k rounds of randomized checking with soundness error ≤ 2^(-k)
- **Streaming**: Re-reads W1 matrix for each round, computes `u = W1^T * r`

```rust
// 32 rounds gives ~2^(-32) soundness error
freivalds_check(&mut w1_reader, &x0, &y1, k, tile_k, 32, seed)?;
```

### Tiny Groth16 Circuit

- **Scope**: Only proves the small 16×16 tail layer computation
- **Formula**: `y2 = floor((W2 · y1) * scale_num / 2)` with exact floor semantics
- **Public Inputs**: 5 commitments only (~160 bytes)
  - `h_w2`: Commitment to 16×16 weight matrix W2
  - `h_x`: Commitment to original input x0
  - `h_y1`: Commitment to intermediate result y1
  - `h_y`: Commitment to final output y2
  - `scale_num`: Quantization parameter (≤8 bits)

### Transaction Size Breakdown

| Component | Size | Details |
|-----------|------|---------|
| Compressed Proof | ~200-300 bytes | BN254 Groth16 with compression |
| Public Inputs | ~160 bytes | 5 field elements as JSON |
| **Total Transaction** | **< 1 KB** | ✅ **Meets target** |
## File Formats

### Input Files

- **W1 weights**: Binary i16, little-endian, row-major (16×K)
- **W2 weights**: JSON 16×16 matrix
- **Input vector**: JSON array of i64 values

### Output Files

- **run.json**: Complete execution trace with commitments
- **proof.bin**: Compressed Groth16 proof
- **public_inputs.json**: Field elements as decimal strings
- **vk.bin**: Compressed verification key

## CLI Reference

### `nova_poc infer`

Runs tiled GEMV inference with Freivalds auditing:

```bash
nova_poc infer [OPTIONS]

Options:
  --k <K>                     Width K of large layer (default: 4096)
  --tile-k <TILE_K>          Tile size for streaming (default: 1024)
  --scale-num <SCALE_NUM>    Quantization numerator (default: 3)
  --seed <SEED>              Random seed (default: 42)
  --freivalds-rounds <N>     Verification rounds (default: 32)
  --weights1-path <PATH>     W1 binary file (16×K)
  --weights2-path <PATH>     W2 JSON file (16×16)
  --x0-path <PATH>           Input vector JSON
  --out <PATH>               Output run.json (default: run.json)
  --skip-freivalds           Skip probabilistic verification
  --preset <PRESET>          Use demo preset (generates random data)
```

### `nova_poc prove`

Generates tiny Groth16 proof for tail layer:

```bash
nova_poc prove <RUN_JSON> [OPTIONS]

Options:
  --pk-path <PATH>       Proving key path
  --out-dir <DIR>        Output directory (default: out)
```

### `nova_poc verify`

Verifies proof and optionally re-runs Freivalds:

```bash
nova_poc verify <RUN_JSON> [OPTIONS]

Options:
  --vk-path <PATH>             Verification key
  --proof-path <PATH>          Proof file
  --public-inputs-path <PATH>  Public inputs JSON
  --weights1-path <PATH>       W1 for Freivalds re-verification (required unless --skip-freivalds)
  --skip-freivalds             Skip Freivalds re-verification
```

### `nova_poc setup`

One-time proving/verification key generation:

```bash
nova_poc setup [OPTIONS]

Options:
  --out-dir <DIR>    Key output directory (default: keys)
  --force            Regenerate even if keys exist
```

### `nova_poc demo`

End-to-end demonstration with random data:

```bash
nova_poc demo [OPTIONS]

Options:
  --seed <SEED>    Random seed (default: 42)
```

## Performance Characteristics

### Typical Timings (MacBook Pro M1)

| Operation | K=4096 | K=8192 | Notes |
|-----------|--------|--------|-------|
| Tiled GEMV | ~0.06s | ~0.12s | Linear in K |
| Freivalds (32 rounds) | ~1.1s | ~2.2s | Linear in K×rounds |
| Key Setup | ~1.2s | ~1.2s | Independent of K |
| Proof Generation | ~5-10s | ~5-10s | Independent of K |
| Verification | ~0.02s | ~0.02s | Independent of K |

### Memory Usage

- **GEMV**: O(tile_k) = ~1024 weights in memory at once
- **Freivalds**: O(K) for input vectors, streams matrix
- **Proof**: O(1) circuit size (16×16 only)

## Security Model

### Cryptographic Assumptions

1. **Groth16 Security**: Relies on discrete log assumption in BN254
2. **Commitment Binding**: α-sum is **non-cryptographic** (POC only)
   - ⚠️ **Replace with Poseidon hash for production**
3. **Freivalds Soundness**: Error probability ≤ 2^(-k) for k rounds

### Threat Model

- **Proves**: Correct computation of 16×16 tail layer with exact quantization
- **Assumes**: Large layer computation verified probabilistically via Freivalds
- **Limitations**: Non-cryptographic commitments allow collisions

## Development

### Build Requirements

```bash
# Rust 1.70+ with default features
cargo --version

# All dependencies included in Cargo.toml
cargo build --release
```

### Testing

```bash
# Test individual components
cargo test -p engine
cargo test -p circuit
cargo test -p nova_poc

# Run circuit complexity analysis
cargo run -p circuit --bin analyze

# Integration test
./target/release/nova_poc demo --seed 123
```

### Architecture Extension

To add cryptographic commitments:

1. Replace α-sum in `engine/src/commitment.rs` with Poseidon hash
2. Update circuit constraints in `circuit/src/lib.rs`
3. Add Poseidon gadgets to dependencies

## Examples

### Data Generation Script

The `scripts/gen.py` script generates all required test data:

```bash
# Generate data for default size (K=50240)
python3 scripts/gen.py

# Generate data for custom size
python3 scripts/gen.py --k 8192 --seed 123

# What it creates:
# - data/w1_16x{K}.bin    # Large layer weights (binary, 16×K)
# - data/w2_16x16.json    # Tail layer weights (JSON, 16×16)
# - data/x0_{K}.json      # Input vector (JSON, K elements)
# - run_inference.sh      # Ready-to-run script
```

### Custom Weight Files

```bash
# Manual creation for specific dimensions
python3 -c "
import numpy as np
# Create 16×1024 random weights
weights = np.random.randint(-100, 100, (16, 1024), dtype=np.int16)
weights.tofile('w1.bin')
"

# Create W2 matrix
echo '[[1,0,0...],[0,1,0...]...]' > w2.json  # 16×16 matrix

# Create input vector
echo '[1, 2, 3, ..., 1024]' > x0.json

# Run inference
nova_poc infer --k 1024 \
    --weights1-path w1.bin \
    --weights2-path w2.json \
    --x0-path x0.json \
    --out my_run.json
```

### Batch Verification

```bash
# Generate multiple proofs
for seed in 42 123 456; do
    nova_poc infer --preset demo --seed $seed --out run_$seed.json
    nova_poc prove run_$seed.json --out-dir proof_$seed
done

# Verify all proofs (option 1: with Freivalds re-verification)
for seed in 42 123 456; do
    nova_poc verify run_$seed.json --proof-path proof_$seed/proof.bin --weights1-path w1.bin
done

# Or verify without Freivalds (option 2: faster)
for seed in 42 123 456; do
    nova_poc verify run_$seed.json --proof-path proof_$seed/proof.bin --skip-freivalds
done
```

## Roadmap

- [x] Tiled GEMV with streaming
- [x] Freivalds probabilistic auditor
- [x] Tiny Groth16 circuit (16×16)
- [x] Compressed proof serialization
- [x] CLI with full pipeline
- [x] Demo mode with <10s completion
- [ ] Cryptographic commitments (Poseidon)
- [ ] Batch proof aggregation
- [ ] GPU acceleration for large K
- [ ] WebAssembly compilation

## License

MIT License - see [LICENSE](LICENSE) for details.
