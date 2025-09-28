# 📚 Legacy Matrix POC

**⚠️ This is the legacy implementation. For active development, use [`expander_matrix_poc`](../expander_matrix_poc/) instead.**

## Migration Notice

This implementation has been superseded by `expander_matrix_poc` which provides:

- **70× faster performance** (34µs vs 2.4ms for 32×64 matrices)
- **Comprehensive security testing** with attack vector analysis
- **Modern architecture** designed for the Expander SDK
- **Production-ready features** with proper error handling and testing

## Historical Reference

This directory contains the original matrix multiplication proof implementation using:
- Arkworks libraries (ark-bn254, ark-groth16)
- Traditional circuit compilation approach
- Groth16 proof system

## Performance Comparison

| Implementation | 32×64 Matrix | Architecture | Status |
|---------------|--------------|--------------|--------|
| **legacy_matrix_poc** | ~2.4ms | Arkworks + Groth16 | 📚 Legacy |
| **expander_matrix_poc** | ~34µs | Expander SDK + GKR | ✅ **Active** |

## Usage (Deprecated)

```bash
# Don't use this - use expander_matrix_poc instead
cargo run -p legacy_matrix_poc

# Use this instead:
cargo run -p expander_matrix_poc --example matrix_proof_demo
```

## Migration Guide

See [`expander_matrix_poc/README.md`](../expander_matrix_poc/README.md) for the modern implementation with:
- Simple API: `MatrixProofSystem::new(m, k)`
- Fast proving: `system.prove(&weights, &input, &output)`
- Instant verification: `system.verify(&proof, &input, &output)`
- Comprehensive benchmarks and security testing

---

**For all new development, use `expander_matrix_poc`.**