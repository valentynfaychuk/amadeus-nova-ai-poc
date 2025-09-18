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
# INFERENCE
# In prod will be running on an H100 beast with the weights loaded into HBM
# The weights hash is added to this model's instance on-chain, presumably
# 120B @ INT16 OSS GPT (240GB of weights)
python3 scripts/infer.py > data/instance.json

# COMPUTOR (MINER)
# The most heavy-weight lifting, mostly because can't be done directly inside H100 -
# creates cryptographic proof with public inputs and outputs of the inference and
# seeded by the blockchain recent block and the model weights
cargo build --release
./target/release/prover data/instance.json out/

# VERIFIER (VALIDATOR)
# Verification is the fastest step, can do single and batch verification
./target/release/verifier out/vk.bin out/proof.bin out/public_inputs.json
echo '[{"vk":"out/vk.bin","proof":"out/proof.bin","pubs":"out/public_inputs.json"}]' > batch.json
./target/release/aggregator batch.json
```

## Architecture

- **circuit/**: R1CS constraint system (matrix ops + quantization + Poseidon)
- **prover/**: Groth16 proof generation with witness computation
- **verifier/**: Standalone proof verification
- **aggregator/**: Batch proof verification
- **scripts/**: Test data generation
