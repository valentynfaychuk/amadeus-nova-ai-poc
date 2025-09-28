# 🗑️ REMOVED: zk_gkr

**This legacy GKR library has been removed from the workspace.**

## Why This Was Removed

1. **Performance**: 70× slower than modern Expander SDK implementation
2. **Superseded**: `expander_matrix_poc` provides the same functionality with massive performance improvements
3. **Maintenance**: No longer needed - Expander SDK is the path forward
4. **Architecture**: Traditional approach replaced by state-of-the-art GKR implementation

## Performance Data (Final Benchmarks)

| Matrix Size | zk_gkr Performance | expander_matrix_poc | Improvement |
|-------------|-------------------|-------------------|-------------|
| 4×8 | 136µs | 687ns | **198× faster** |
| 32×64 | 2.38ms | 34µs | **70× faster** |

## Migration Completed

All functionality moved to `expander_matrix_poc`:

```rust
// OLD (zk_gkr) - No longer available
use zk_gkr::{SumCheckProver, SumCheckVerifier};

// NEW (expander_matrix_poc) - Use this
use expander_matrix_poc::MatrixProofSystem;
let mut system = MatrixProofSystem::new(m, k)?;
let proof = system.prove(&weights, &input, &output)?;
```

## Removal Date

Removed: September 28, 2024
Reason: Superseded by superior Expander SDK implementation

See `expander_matrix_poc/` for the modern, high-performance replacement.