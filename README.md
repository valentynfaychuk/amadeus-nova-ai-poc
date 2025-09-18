# Nova AI - Zero-Knowledge Neural Network Inference

Proves knowledge of private 2-layer neural network weights that produce given outputs from public inputs, with quantization and Poseidon commitment.

## What it does

- **Private**: 2×3×3 weight matrices (18 values total)
- **Public**: Input vector, output vector, quantization params, weight commitment
- **Proves**: `y = quantize(W₁ × quantize(W₀ × x))` where W₀, W₁ are private weights
- **Commitment**: Poseidon hash of all weights prevents weight tampering
- **Quantization**: Integer division with remainder constraints: `(input × scale_num) = scale_den × output + remainder`

## Usage

```bash
# Generate test data
python3 scripts/infer.py > data/instance.json

# Build and run
cargo build --release
./target/release/prover data/instance.json out/
./target/release/verifier out/vk.bin out/proof.bin out/public_inputs.json

# Batch verification
echo '[{"vk":"out/vk.bin","proof":"out/proof.bin","pubs":"out/public_inputs.json"}]' > batch.json
./target/release/aggregator batch.json
```

## Architecture

- **circuit/**: R1CS constraint system (matrix ops + quantization + Poseidon)
- **prover/**: Groth16 proof generation with witness computation
- **verifier/**: Standalone proof verification
- **aggregator/**: Batch proof verification
- **scripts/**: Test data generation