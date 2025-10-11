# zkCUDA Proof of Concept

This package demonstrates matrix multiplication proofs using Polyhedra's zkCUDA framework.

## ⚠️ Important: CPU vs GPU Execution

**Current Status**: The code runs on **CPU only** despite using zkCUDA syntax.

### What is zkCUDA?

zkCUDA is a **programming model** (like CUDA syntax) that compiles to zero-knowledge circuits. It does NOT automatically use GPU acceleration.

- **ExpanderCompilerCollection's zkCUDA** - CUDA-like API that compiles to Expander circuit IR
- **Execution backend** - Uses Expander's CPU-based GKR prover by default
- **No GPU integration** - zkCUDA circuits currently execute on CPU

### What About GPU Acceleration?

Expander **does** have GPU support, but it's separate:

1. **Expander CUDA sumcheck** - Standalone C++/CUDA implementation
2. **Located at**: `~/Expander/sumcheck/cuda/`
3. **Built successfully** on the remote CUDA machine (943KB binary)
4. **Performance**: ~0.39s for 2^20 elements on GPU
5. **Not integrated** with ExpanderCompilerCollection's zkCUDA framework

### Architecture Diagram

```
┌─────────────────────────────────────┐
│   Our cuda_poc Rust Code            │
│   (uses zkCUDA #[kernel] macros)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  ExpanderCompilerCollection          │
│  (zkCUDA framework)                  │
│  - Compiles kernels to circuits      │
│  - CUDA-like syntax only             │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Expander GKR Prover (CPU)           │
│  - Executes circuit on CPU           │
│  - No GPU backend in Rust API        │
└──────────────────────────────────────┘

Separate, unintegrated:
┌──────────────────────────────────────┐
│  Expander CUDA Sumcheck (GPU)        │
│  - Standalone C++/CUDA binary        │
│  - ~/Expander/sumcheck/cuda/         │
│  - Works on GPU, but not accessible  │
│    from zkCUDA framework             │
└──────────────────────────────────────┘
```

## Performance

### Current Performance (CPU)
- **Matrix size**: 64×32 × 32×64 (fixed)
- **Proving time**: ~115ms per iteration
- **Throughput**: ~600K ops/sec
- **Verification**: ~108ms

### Running Examples

```bash
# Basic demo
cargo run -p cuda_poc --example demo --release

# Stress test (100 iterations)
cargo run -p cuda_poc --example stress_test --release 100
```

## GPU Acceleration Status

### ✅ What Works
- zkCUDA syntax compiles successfully
- Matrix multiplication proofs work correctly
- CUDA-like programming model with `#[kernel]` macros
- Expander's standalone CUDA sumcheck binary works on GPU

### ❌ What Doesn't Work
- zkCUDA framework does NOT use GPU
- No Rust API integration with CUDA backend
- No runtime flag to enable GPU mode
- Matrix dimensions are compile-time hardcoded (64×32 × 32×64)

### 🔧 GPU Build Details (for reference)

The standalone CUDA sumcheck library was successfully built:

```bash
# On remote machine
cd ~/Expander/sumcheck/cuda
nvcc -O3 -arch=sm_86 -std=c++17 -Iinclude -Iicicle \
     -DuseM31ext3 -o sumcheck.bin src/sumcheck_cuda.cu

# Test GPU mode
./sumcheck.bin -m gpu -p 20
# Result: 0.392s for 2^20 elements on RTX 4090
```

**Note**: Used sm_86 (Ampere) instead of sm_89 (Ada) due to CUDA 11.5 limitations. RTX 4090 runs it via forward compatibility.

## Future GPU Integration

To actually use GPU with our zkCUDA code would require:

1. **Create FFI bindings** to Expander's CUDA sumcheck library
2. **Modify ExpanderCompilerCollection** to support GPU backend
3. **Runtime configuration** to switch between CPU/GPU modes

This is non-trivial and would require significant development effort.

## Comparison with legacy_matrix_poc

| Feature | cuda_poc (zkCUDA) | legacy_matrix_poc |
|---------|-------------------|-------------------|
| Execution | CPU (despite name) | CPU |
| Throughput | 600K ops/sec | 13K ops/sec |
| Matrix Size | Fixed (64×32 × 32×64) | Variable |
| Programming Model | CUDA-like | Standard Rust |
| GPU Support | No | No |

## Conclusion

**zkCUDA** is a clean, CUDA-inspired programming model for zero-knowledge circuits, but it does **not** provide GPU acceleration out of the box. The name refers to the programming style, not the execution backend.

For actual GPU acceleration, you would need to integrate Expander's separate CUDA sumcheck implementation, which is currently a standalone system.
