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
# 0. Cleanup old data (optional)
rm -rf keys data

# 1. Generate test data for large-scale inference (16×50240 matrix)
python3 scripts/gen.py --k 50240

# 2. Run the generated script (or use manual commands below)
cargo run --release -p nova_poc -- infer \
    --k 50240 \
    --tile-k 4096 \
    --weights1-path data/w1_16x50240.bin \
    --weights2-path data/w2_16x16.json \
    --x0-path data/x0_50240.json \
    --scale-num 3 \
    --freivalds-rounds 32 \
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

Verifies proof with enhanced security features:

```bash
nova_poc verify <RUN_JSON> [OPTIONS]

Options:
  --vk-path <PATH>             Verification key
  --proof-path <PATH>          Proof file
  --public-inputs-path <PATH>  Public inputs JSON
  --weights1-path <PATH>       W1 for Freivalds re-verification (required unless --skip-freivalds)
  --skip-freivalds             Skip Freivalds re-verification
  --no-bind-randomness         Use prover's seed instead of transcript-bound seed (for tests)
  --allow-low-k                Allow k < 16 without y1 reconstruction
  --block-entropy <HEX>        Optional extra entropy in the transcript (hex string)
  --tile-k <USIZE>             Tile size for recompute function (defaults to value from RunData)
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

## Security Analysis

### 🔐 Enhanced Security Features (v2.0)

**Version 2.0 introduces critical security enhancements that bind the Freivalds verifier to the SNARK proof, preventing sophisticated fraud attempts:**

#### **1. Transcript-Bound Randomness (Fiat-Shamir)**
- **Purpose**: Prevents prover from choosing "easy" Freivalds seeds
- **Method**: Derives seeds from immutable transcript containing VK hash and all commitments
- **Transcript Format**: `"FREIVALDSv1" || vk_hash || model_id || h_w1 || h_w2 || h_x || h_y1 || h_y || block_entropy`
- **Security**: Eliminates prover's ability to bias randomness generation

#### **2. Weight Integrity Verification**
- **Purpose**: Detects tampering with W1 weight files during verification
- **Method**: Recomputes `h_W1` commitment by streaming weights in exact gemv order
- **Validation**: Rejects verification if `h_W1_recomputed ≠ h_W1_stored`
- **Security**: Prevents weight file substitution attacks

#### **3. Hidden State Reconstruction (k ≥ 16)**
- **Purpose**: Binds Freivalds transcript to SNARK's hidden intermediate state y1
- **Method**: Solves `R^T * y1 = s` where R is matrix of Freivalds random vectors
- **Validation**: Reconstructed y1 must match commitment `h_y1` via β-sum
- **Security**: Prevents mismatched y1 attacks and strengthens binding between layers

#### **4. Enhanced CLI Security Controls**
```bash
# Production verification (all security features enabled)
nova_poc verify run.json --weights1-path w1.bin

# Test mode with relaxed security (⚠️ NOT for production)
nova_poc verify run.json --no-bind-randomness --allow-low-k

# Additional entropy for high-security environments
nova_poc verify run.json --block-entropy "$(head -c 16 /dev/urandom | hexdump -e '16/1 "%02x"')"
```

### Attack Resistance & Defense Mechanisms

Nova AI employs **multiple independent layers of security** to defend against various attack vectors:

#### **Primary Defense: Freivalds Algorithm**
- **Purpose**: Probabilistically verifies large layer computation (`W1 · x0 = y1`)
- **Detection Probability**: `1 - 2^(-k)` where k = number of rounds
- **Default**: 32 rounds → ~99.9999999% fraud detection
- **Method**: Re-streams entire W1 matrix during verification

#### **Secondary Defense: Groth16 Public Input Validation**
- **Purpose**: Cryptographically validates commitment consistency
- **Detection Probability**: 100% (deterministic)
- **Method**: Recomputes all commitments during verification

#### **Tertiary Defense: Circuit Constraints**
- **Purpose**: Enforces exact quantized division semantics
- **Detection Probability**: 100% (cryptographic proof)
- **Method**: 307 constraints validate 16×16 tail computation

### Attack Simulation Results

Run `scripts/run_attacks.sh` to reproduce these security tests:

| Attack Type | Prover Strategy | v1.0 Defense | v2.0 Enhanced Defense | Result |
|-------------|----------------|---------------|----------------------|--------|
| **Naive Fraud** | Fake y1/y2, wrong commitments | Freivalds detection | Same + transcript binding | ❌ **BLOCKED** |
| **Weight Substitution** | Replace W1 file with modified weights | Manual validation | Automatic h_W1 recomputation | ❌ **BLOCKED** |
| **Seed Manipulation** | Choose favorable Freivalds seed | Randomness check | Transcript-bound derivation | ❌ **BLOCKED** |
| **y1 Mismatch** | Valid W1*x0 but wrong y1 in proof | Limited detection | y1 reconstruction binding | ❌ **BLOCKED** |
| **Bypass Attempt** | Skip enhanced security checks | N/A | CLI flag validation | ❌ **BLOCKED** |

```bash
# Test all attack vectors (including new v2.0 attacks)
./scripts/run_attacks.sh

# Expected output:
# 🛡️  Testing Enhanced Security Features (v2.0)
# ✅ SECURITY SUCCESS: All attacks detected and blocked
# ⏱️  Detection time: ~15s (includes new security checks)
# 🔒 False positive rate: ~2^(-32) (astronomically low)
# 💪 Enhanced binding: Freivalds ↔ SNARK proven secure
```

#### **Economic Security Model**
- **Honest Verification Cost**: ~13 seconds for production scale (K=50240)
- **Fraud Detection Cost**: ~13 seconds (same as honest case)
- **Economic Incentive**: Honest behavior is no more expensive than fraud attempts

### Cryptographic Assumptions

1. **Groth16 Security**: Relies on discrete log assumption in BN254
2. **Commitment Binding**: α-sum is **non-cryptographic** (POC only)
   - ⚠️ **Replace with Poseidon hash for production**
3. **Freivalds Soundness**: Error probability ≤ 2^(-k) for k rounds

### Enhanced Verification Workflow (v2.0)

The v2.0 verification process now includes additional security checks:

```bash
# 1. Basic verification (all security features enabled by default)
nova_poc verify run.json --weights1-path w1.bin

# 2. High-security verification with additional entropy
ENTROPY=$(head -c 16 /dev/urandom | hexdump -e '16/1 "%02x"')
nova_poc verify run.json --weights1-path w1.bin --block-entropy $ENTROPY

# 3. Verification steps performed internally:
#    a) Load VK and compute vk_hash for transcript binding
#    b) Recompute h_W1 by streaming weights file
#    c) Derive transcript-bound seed from VK + commitments
#    d) Run Freivalds with transcript-bound randomness
#    e) Reconstruct y1 from Freivalds matrix (if k ≥ 16)
#    f) Validate y1 against h_y1 commitment
#    g) Verify Groth16 proof with public inputs
```

#### **Security Verification Checklist**

✅ **VK Hash Computation**: Ensures transcript is bound to specific circuit
✅ **Weight Integrity**: `h_W1_recomputed == h_W1_stored`
✅ **Transcript Binding**: Seed derived from immutable transcript
✅ **Freivalds Execution**: 32 rounds with transcript-bound randomness
✅ **Matrix Rank Check**: `rank(R) ≥ 16` for unique y1 solution
✅ **y1 Reconstruction**: Solve `R^T * y1 = s` and validate commitment
✅ **Groth16 Verification**: Cryptographic proof validation

### Production Security Guidelines

#### **🔒 Mandatory Requirements (v2.0)**
1. **NEVER use `--skip-freivalds` in production**
   - This flag is for testing/debugging only
   - Skipping Freivalds significantly reduces security

2. **NEVER use `--no-bind-randomness` in production**
   - This disables transcript binding (v2.0 security feature)
   - Only use for compatibility testing with v1.0 behavior

3. **Use sufficient Freivalds rounds (≥32 recommended)**
   - Each round halves the fraud probability
   - 32 rounds → 2^(-32) fraud probability (~1 in 4 billion)

4. **Ensure k ≥ 16 for full y1 reconstruction security**
   - y1 reconstruction requires at least 16 Freivalds rounds
   - Use `--allow-low-k` only for testing smaller dimensions

5. **Validate all public inputs independently**
   - Don't trust commitment values in run.json files
   - v2.0 automatically recomputes h_W1 commitment during verification

6. **Protect weight files from tampering**
   - v2.0 detects weight file modifications via h_W1 verification
   - Store weight files with integrity checksums in production

#### **⚠️ Security Limitations**
- **Non-cryptographic commitments**: α-sum allows collisions (replace with Poseidon for production)
- **Probabilistic security**: Freivalds has ~2^(-32) false negative rate
- **Implementation attacks**: Side-channel attacks not considered in this analysis

### Threat Model

- **✅ Proves**: Correct computation of 16×16 tail layer with exact quantization
- **✅ Detects**: Invalid large layer computations with high probability
- **✅ Prevents**: Commitment inconsistencies and constraint violations
- **⚠️ Assumes**: Implementation is side-channel resistant
- **⚠️ Limitations**: Non-cryptographic commitments in POC version

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

# Security analysis & attack simulation
./scripts/run_attacks.sh
```

### Architecture Extension

To add cryptographic commitments:

1. Replace α-sum in `engine/src/commitment.rs` with Poseidon hash
2. Update circuit constraints in `circuit/src/lib.rs`
3. Add Poseidon gadgets to dependencies

## Benchmarking

### Quick Start (Python-only workflow)

```bash
# 1) (optional) create venv and install bench deps
python3 -m venv .venv && source .venv/bin/activate
pip install -U psutil pandas matplotlib pyyaml

# 2) Demo (fast) — single config
python3 scripts/bench.py \
  --grid K=4096 \
  --tile-k 1024 \
  --rounds 16 \
  --threads auto \
  --modes one_pass \
  --repeats 1 \
  --out results_demo.csv

# 3) Full grid (longer)
python3 scripts/bench.py \
  --grid K=4096,12288,16384,24576 \
  --tile-k 1024,4096,8192 \
  --rounds 16,32,64 \
  --threads 1,auto \
  --modes one_pass,k_pass_legacy \
  --repeats 3 \
  --out results.csv

# 4) Plot
python3 scripts/plot_bench.py --in results.csv --outdir plots
```

### Production-Scale Presets

```bash
# 7B-like model dimensions
python3 scripts/bench.py --grid K=12288 --tile-k 8192 --rounds 16 --threads auto --modes one_pass --repeats 3 --out 7b.csv

# Mid-size model
python3 scripts/bench.py --grid K=16384 --tile-k 8192 --rounds 16 --threads auto --modes one_pass --repeats 3 --out mid.csv

# Large model
python3 scripts/bench.py --grid K=24576 --tile-k 16384 --rounds 16 --threads auto --modes one_pass --repeats 3 --out large.csv

# GKR vs Freivalds comparison
python3 scripts/bench.py --grid K=4096,16384 --tile-k 4096 --rounds 16 --threads auto --modes one_pass,gkr --repeats 2 --out comparison.csv
```

### Benchmark Features

The benchmarking suite measures:
- **Wall time, CPU time (user/sys), and peak memory** for each stage
- **I/O metrics** (bytes read/written, best-effort on macOS)
- **Proof and transaction sizes** to verify < 1 KB constraint
- **Thread scaling** with configurable RAYON_NUM_THREADS
- **One-pass vs k-pass Freivalds vs GKR** performance comparison

Configuration grid:
- `K ∈ {4096, 12288, 16384, 24576}` - Matrix width (LLM layer dimensions)
- `tile_k ∈ {1024, 4096, 8192, 16384}` - Streaming tile size (Freivalds only)
- `rounds ∈ {16, 32, 64}` - Freivalds security parameter (Freivalds only)
- `threads ∈ {1, auto}` - Thread count (auto = all cores)
- `mode ∈ {one_pass, k_pass_legacy, gkr}` - Verification algorithm

### Data Generation

The benchmark uses **realistic LLM-like quantization**:
- **INT8 weights & activations** with per-row weight scales
- **INT32 accumulators** safe for K ≤ 24576
- **Xavier initialization** (σ = 1/√K) for weights
- **Deterministic generation** with configurable seeds

### Output Analysis

The plotting script generates:
- **Wall time vs K** grouped by rounds and mode
- **Stage breakdown** showing time distribution
- **Speedup analysis** comparing one-pass vs k-pass
- **Proof size tracking** to verify < 1 KB constraint
- **Thread scaling efficiency** plots
- **Summary report** with key metrics

### Notes

- The benchmark suite sets `RAYON_NUM_THREADS` automatically per run
- **One-pass Freivalds** should be ≫ faster than k-pass (≥10× for k=16–32)
- **GKR mode** provides cryptographic security without matrix access during verification
- Proof bytes: Freivalds (~200–300 B), GKR (~KB to tens of KB)
- I/O counters may be unavailable on macOS (shown as None)
- Each configuration is run multiple times; first run may be slower (cold cache)

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

# Traditional inference + Freivalds verification
nova_poc infer --k 1024 \
    --weights1-path w1.bin \
    --weights2-path w2.json \
    --x0-path x0.json \
    --out my_run.json

# Alternative: GKR proof (no verifier matrix access)
nova_poc prove-gkr \
    --weights1-path w1.bin \
    --x0-path x0.json \
    --m 16 --k 1024 \
    --out-dir gkr_proof
```

### Batch Verification

```bash
# Traditional Freivalds workflow
for seed in 42 123 456; do
    nova_poc infer --preset demo --seed $seed --out run_$seed.json
    nova_poc prove run_$seed.json --out-dir proof_$seed
    nova_poc verify run_$seed.json --proof-path proof_$seed/proof.bin --weights1-path w1.bin
done

# GKR workflow (no matrix access during verification)
for K in 4096 8192 16384; do
    nova_poc prove-gkr \
        --weights1-path data/w1_16x${K}.bin \
        --x0-path data/x0_${K}.json \
        --m 16 --k $K \
        --salt "batch_${K}" \
        --out-dir gkr_${K}

    nova_poc verify-gkr \
        --proof-path gkr_${K}/gkr_proof.bin \
        --public-path gkr_${K}/public.json
done
```

## GKR Mode (Zero-Knowledge Matrix-Vector Multiplication)

### Overview

The GKR (Goldwasser-Kalai-Rothblum) mode provides a **non-interactive zero-knowledge proof** for matrix-vector multiplication `y = W·x` without requiring the verifier to access the weight matrix `W`. This replaces the verifier-side Freivalds check with a cryptographic proof that has **polylogarithmic verification time**.

### Protocol Summary

The protocol proves the scalar claim:
```
c = Σᵢ Σⱼ uᵢ · Wᵢⱼ · xⱼ
```

Where:
- **W** is the private weight matrix (16×K)
- **x** is the public input vector (length K)
- **u** is a challenge vector derived via Fiat-Shamir
- **c** is the claimed scalar result

### Key Features

- **Commitments**: Poseidon-Merkle trees over matrix/vector elements
- **Sum-Check**: Multi-round interactive protocol made non-interactive via Fiat-Shamir
- **MLE Openings**: Binary folding proofs for multilinear extension evaluations
- **Proof Size**: O(log(m·k)) Merkle paths + O(m+k) field elements
- **Verification**: Milliseconds, independent of matrix size

### Usage

#### Generate GKR Proof
```bash
# Build required binary
cargo build --release -p nova_poc

# Generate proof for 16×4096 matrix
nova_poc prove-gkr \
  --weights1-path data/w1_16x4096.bin \
  --x0-path data/x0_4096.json \
  --m 16 \
  --k 4096 \
  --salt deadbeef1234 \
  --out-dir gkr_proof \
  --model-id "llm_layer_1" \
  --vk-hash "abc123"
```

#### Verify GKR Proof
```bash
# Verify proof (no access to weight matrix)
nova_poc verify-gkr \
  --proof-path gkr_proof/gkr_proof.bin \
  --public-path gkr_proof/public.json
```

#### Optional: Verify with Groth16 Tail
```bash
# Combine GKR + Groth16 for downstream binding
nova_poc verify-gkr \
  --proof-path gkr_proof/gkr_proof.bin \
  --public-path gkr_proof/public.json \
  --with-tail
```

### Technical Specifications

#### Dimensions and Padding
- Matrix dimensions padded to powers of 2: `m = 2^a`, `k = 2^b`
- Hypercube indexing for MLE compatibility
- Support for realistic LLM dimensions (K ∈ {12288, 16384, 24576})

#### Cryptographic Primitives
- **Field**: BN254 scalar field (Fr)
- **Hash**: Poseidon for BN254 (Merkle tree nodes)
- **Commitment**: Binary Merkle trees with Poseidon internal nodes
- **Randomness**: Fiat-Shamir transcript with domain separation

#### Proof Structure
```json
{
  "m": 16,
  "k": 4096,
  "h_w": "0x1234...",  // Merkle root of W
  "h_x": "0x5678...",  // Merkle root of x
  "c": "0x9abc...",    // Claimed scalar value
  "salt": "deadbeef",
  "sumcheck_proof": {
    "claimed_sum": "0x...",
    "round_polynomials": [...],
    "challenges": [...],
    "w_opening": {...},
    "x_opening": {...}
  }
}
```

#### Security Properties
- **Soundness**: Cheating prover caught with overwhelming probability
- **Zero-Knowledge**: Verifier learns nothing about W beyond public claim
- **Non-Interactive**: Single proof, no verifier interaction required
- **Binding**: Fiat-Shamir transcript binds to model_id, vk_hash, dimensions

#### Performance Characteristics
- **Proof Generation**: O(m·k) field operations + O(log(m·k)) Merkle proofs
- **Proof Size**: ~KB to tens of KB (vs. GB for naive approaches)
- **Verification Time**: ~milliseconds (independent of matrix size)
- **Memory**: O(log(m·k)) verifier memory (vs. O(m·k) for Freivalds)

### Integration with Existing System

The GKR mode integrates seamlessly with the existing Nova POC architecture:

1. **Inference Stage**: Unchanged - computes `y = W·x` as before
2. **Proof Stage**: Choice between Freivalds (`prove`) or GKR (`prove-gkr`)
3. **Verification**: Freivalds requires matrix access; GKR does not
4. **Groth16 Tail**: Optional downstream binding for both modes

### Acceptance Criteria

✅ **GKR proof generation** produces valid binary proof and JSON public inputs
✅ **Verification time** under 100ms for m=16, k∈[12k..24k]
✅ **No matrix access** during verification (only Merkle openings)
✅ **Tamper resistance** - any bit flip causes verification failure
✅ **Proof size** scales as O(log(mk)) not O(mk)
✅ **Optional Groth16** keeps total payload under 1KB when combined

## Roadmap

### ✅ Completed (v1.0)
- [x] Tiled GEMV with streaming
- [x] Freivalds probabilistic auditor
- [x] Tiny Groth16 circuit (16×16)
- [x] Compressed proof serialization
- [x] CLI with full pipeline
- [x] Demo mode with <10s completion

### ✅ Completed (v2.0 - Security Enhancements)
- [x] **Transcript-bound randomness (Fiat-Shamir)** for Freivalds
- [x] **Weight integrity verification** during verify
- [x] **y1 reconstruction binding** from Freivalds matrix
- [x] **Enhanced CLI security controls** with new flags
- [x] **Comprehensive attack simulation** testing

### 🚧 Planned (v3.0)
- [ ] Cryptographic commitments (Poseidon hash replacement)
- [ ] Batch proof aggregation for multiple inferences
- [ ] GPU acceleration for large K dimensions
- [ ] WebAssembly compilation for browser deployment
- [ ] Zero-knowledge model architecture hiding
- [ ] Recursive proof composition

## License

MIT License - see [LICENSE](LICENSE) for details.
