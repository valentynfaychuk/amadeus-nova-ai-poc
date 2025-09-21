#!/bin/bash
# Auto-generated script to run inference with K=4096

echo "🚀 Running Nova POC inference with K=4096"
echo "   This will compute: y2 = W2 · (W1 · x0) with quantization"
echo ""

# Build the project if needed
cargo build --release -p nova_poc

# Run inference with the generated data
./target/release/nova_poc infer \
    --k 4096 \
    --tile-k 4096 \
    --weights1-path data/w1_16x4096.bin \
    --weights2-path data/w2_16x16.json \
    --x0-path data/x0_4096.json \
    --scale-num 3 \
    --freivalds-rounds 32 \
    --out data/run_k4096.json

echo ""
echo "✅ Inference complete! Results saved to data/run_k4096.json"
echo ""
echo "Next steps:"
echo "  1. Setup keys (one-time): ./target/release/nova_poc setup --out-dir keys"
echo "  2. Generate proof: ./target/release/nova_poc prove data/run_k4096.json --pk-path keys/pk.bin --out-dir data/out"
echo "  3. Verify proof:  ./target/release/nova_poc verify data/run_k4096.json --weights1-path data/w1_16x4096.bin"
echo "  Or verify without Freivalds: ./target/release/nova_poc verify data/run_k4096.json --skip-freivalds"
