#!/usr/bin/env python3
"""
Generate test data for Nova POC with configurable dimensions.
Usage: python3 scripts/gen.py [--k K] [--seed SEED]
"""

import numpy as np
import json
import argparse
import os

def generate_test_data(k=50240, seed=42):
    """Generate W1, W2, and x0 files for testing."""

    np.random.seed(seed)

    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    print(f"🎲 Generating test data with K={k}, seed={seed}")

    # 1. Generate W1 (16×K binary weights)
    print(f"📊 Creating W1 matrix (16×{k})...")
    w1 = np.random.randint(-127, 128, (16, k), dtype=np.int16)
    w1_path = f'data/w1_16x{k}.bin'
    w1.tofile(w1_path)
    print(f"   ✓ Saved to {w1_path} ({w1.nbytes / 1024 / 1024:.1f} MB)")

    # 2. Generate W2 (16×16 JSON weights)
    print("📊 Creating W2 matrix (16×16)...")
    w2 = np.random.randint(-10, 11, (16, 16)).tolist()
    w2_path = 'data/w2_16x16.json'
    with open(w2_path, 'w') as f:
        json.dump(w2, f, indent=2)
    print(f"   ✓ Saved to {w2_path}")

    # 3. Generate input vector x0 (K elements)
    print(f"📊 Creating input vector x0 (length {k})...")
    # Use smaller values to avoid overflow in computation
    x0 = np.random.randint(-10, 11, k).tolist()
    x0_path = f'data/x0_{k}.json'
    with open(x0_path, 'w') as f:
        json.dump(x0, f)
    print(f"   ✓ Saved to {x0_path} ({len(json.dumps(x0)) / 1024:.1f} KB)")

    # 4. Create a sample script to run inference
    script_path = 'run_inference.sh'
    with open(script_path, 'w') as f:
        f.write(f"""#!/bin/bash
# Auto-generated script to run inference with K={k}

echo "🚀 Running Nova POC inference with K={k}"
echo "   This will compute: y2 = W2 · (W1 · x0) with quantization"
echo ""

# Build the project if needed
cargo build --release -p nova_poc

# Run inference with the generated data
./target/release/nova_poc infer \\
    --k {k} \\
    --tile-k {min(4096, k)} \\
    --weights1-path {w1_path} \\
    --weights2-path {w2_path} \\
    --x0-path {x0_path} \\
    --scale-num 3 \\
    --freivalds-rounds 32 \\
    --out data/run_k{k}.json

echo ""
echo "✅ Inference complete! Results saved to data/run_k{k}.json"
echo ""
echo "Next steps:"
echo "  1. Setup keys (one-time): ./target/release/nova_poc setup --out-dir keys"
echo "  2. Generate proof: ./target/release/nova_poc prove data/run_k{k}.json --pk-path keys/pk.bin --out-dir data/out"
echo "  3. Verify proof:  ./target/release/nova_poc verify data/run_k{k}.json --weights1-path {w1_path}"
echo "  Or verify without Freivalds: ./target/release/nova_poc verify data/run_k{k}.json --skip-freivalds"
""")
    os.chmod(script_path, 0o755)
    print(f"   ✓ Created run script: {script_path}")

    print("\n✅ Test data generation complete!")
    print(f"\n📝 Summary:")
    print(f"   • W1 dimensions: 16×{k} ({16*k} weights)")
    print(f"   • W2 dimensions: 16×16 (256 weights)")
    print(f"   • Input vector: {k} elements")
    print(f"   • Tile size: {min(4096, k)} (for streaming)")

    print(f"\n🎯 To run inference:")
    print(f"   ./run_inference.sh")
    print(f"\n   Or manually:")
    print(f"   cargo run --release -p nova_poc -- infer \\")
    print(f"       --k {k} --tile-k {min(4096, k)} \\")
    print(f"       --weights1-path {w1_path} \\")
    print(f"       --weights2-path {w2_path} \\")
    print(f"       --x0-path {x0_path} \\")
    print(f"       --out data/run_k{k}.json")

    return w1_path, w2_path, x0_path

def main():
    parser = argparse.ArgumentParser(
        description='Generate test data for Nova POC'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=50240,
        help='Width K of large layer W1 (default: 50240)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Validate K
    if args.k < 16:
        print("❌ Error: K must be at least 16")
        return 1

    if args.k % 16 != 0:
        print(f"⚠️  Warning: K={args.k} is not divisible by 16, may affect performance")

    generate_test_data(args.k, args.seed)
    return 0

if __name__ == '__main__':
    exit(main())