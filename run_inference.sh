#!/bin/bash
# Auto-generated script to run inference with K=50240

echo "🚀 Running Nova POC inference with K=50240"
echo "   This will compute: y2 = W2 · (W1 · x0) with quantization"
echo ""

# Build the project if needed
cargo build --release -p nova_poc

# Run inference with the generated data
./target/release/nova_poc infer \
    --k 50240 \
    --tile-k 4096 \
    --weights1-path data/w1_16x50240.bin \
    --weights2-path data/w2_16x16.json \
    --x0-path data/x0_50240.json \
    --scale-num 3 \
    --freivalds-rounds 32 \
    --out data/run_k50240.json

echo ""
echo "✅ Inference complete! Results saved to data/run_k50240.json"
echo ""
echo "Next steps:"
echo "  1. Setup keys (one-time): ./target/release/nova_poc setup --out-dir keys"
echo "  2. Generate proof: ./target/release/nova_poc prove data/run_k50240.json --pk-path keys/pk.bin --out-dir data/out"
echo "  3. Verify proof (v2.0 enhanced): ./target/release/nova_poc verify data/run_k50240.json --weights1-path data/w1_16x50240.bin"
echo "     Optional security flags:"
echo "       --block-entropy <hex>     # Extra entropy for high-security environments"
echo "       --no-bind-randomness      # Disable transcript binding (⚠️ testing only)"
echo "       --allow-low-k             # Allow k < 16 without y1 reconstruction"
echo "       --skip-freivalds          # Skip Freivalds (⚠️ reduced security)"
