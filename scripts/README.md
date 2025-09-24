# Scripts Documentation

This directory contains tools for benchmarking, performance analysis, and security testing of the Nova AI GKR implementation:

- `gkr_attack_suite.sh` - Comprehensive security testing suite
- `gkr_stress_test.sh` - Extended testing under more stress
- `benchmark.py` - Automated benchmarking with CSV & PNG output

## Setup

Install Python dependencies in a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas matplotlib numpy psutil
```

## Benchmarking & Performance

### `benchmark.py`
Comprehensive benchmarking and visualization tool that runs GKR performance tests with optional hardware acceleration and generates detailed analysis plots.

#### Standard Benchmarking
```bash
# Run complete benchmark with visualization (generates benchmark_results.csv and benchmark_results.png)
python3 scripts/benchmark.py --sizes 4096,8192,16384,32768 --repeats 3

# Large-scale production benchmark (generates production_bench.csv and production_bench.png)
python3 scripts/benchmark.py --sizes 4096,8192,16384,32768,50204 --repeats 5 --output production_bench
```

#### Accelerated Benchmarking
```bash
# CPU AVX2/AVX-512 accelerated benchmark (requires --features accel_cpu_avx)
python3 scripts/benchmark.py --avx --sizes 4096,8192,16384,32768 --repeats 3 --output avx_benchmark

# CUDA GPU accelerated benchmark (requires --features accel_cuda)
python3 scripts/benchmark.py --cuda --sizes 4096,8192,16384,32768 --repeats 3 --output cuda_benchmark

# CPU acceleration with custom thread count
python3 scripts/benchmark.py --avx --accel-threads 8 --sizes 4096,8192,16384 --output avx_8threads

# CUDA acceleration with specific GPU device
python3 scripts/benchmark.py --cuda --accel-device-id 0 --sizes 4096,8192,16384 --output cuda_gpu0
```

**Features:**
- **Hardware Acceleration Support**: CPU AVX2/AVX-512 and CUDA GPU backends
- **Baseline Comparison**: Pure matrix multiplication timing vs GKR proving
- **Overhead Analysis**: Shows exact ZK proving cost (e.g., "15.2× slower than baseline")
- Automated nova_poc benchmark execution with acceleration flags
- Real-time performance analysis and complexity fitting
- Automatic generation of both CSV data and PNG visualization
- Comprehensive visualization with 5 detailed plots
- System information tracking
- Production-scale testing support

**Output Files:**
- **name.csv**: Timing, memory, and proof size metrics
- **name.png**: Enhanced performance visualization with:
  - **GKR vs Baseline Scaling**: Comparison of GKR proving time vs pure matrix multiplication
  - **Verification Time**: Constant ~0.4ms verification across all matrix sizes
  - **Proof Size Growth**: Logarithmic scaling patterns with matrix size
  - **Scaling Comparison**: Normalized view of GKR, baseline, and proof size scaling
  - **Performance Summary Table**: Shows Infer time, Prove+Infer time with overhead ratios

## Security Testing

### `gkr_attack_suite.sh`
Comprehensive 7-phase security testing framework for GKR implementation.

```bash
# Run complete attack suite
./scripts/gkr_attack_suite.sh

# Run specific phases
./scripts/gkr_attack_suite.sh --phase 1  # Mathematical attacks only
```

**Attack Categories:**
1. **Mathematical Polynomial Forgery** - Sum-check manipulation
2. **Cryptographic Binding Bypass** - Commitment substitution
3. **Transcript Manipulation** - Fiat-Shamir attacks
4. **Field Overflow Attacks** - Extreme value edge cases
5. **Binary Format Exploitation** - Proof corruption
6. **Cross-Component Attacks** - Multi-vector exploitation
7. **Advanced Coordinated Attacks** - Combined techniques

### `gkr_stress_test.sh`
Advanced stress testing with 6 sophisticated attack scenarios.

```bash
# Run comprehensive stress tests
./scripts/gkr_stress_test.sh
```

**Attack Categories:**
1. **Binary Corruption Attack** - Strategic byte manipulation
2. **Proof Substitution Attack** - Public input consistency validation
3. **Field Overflow Attack** - Extreme value edge cases
4. **Zero-Knowledge Bypass Attack** - Information leakage analysis
5. **Deterministic Challenge Attack** - Challenge predictability tests
6. **Massive Scale Attack** - System stability under extreme load

## Usage Examples

### Quick Examples

```bash
# Standard complete performance analysis
python3 scripts/benchmark.py --sizes 4096,8192,16384,32768,50204 --repeats 5 --output comprehensive_benchmark

# CPU AVX accelerated performance comparison
python3 scripts/benchmark.py --avx --sizes 4096,8192,16384,32768 --repeats 5 --output avx_comparison

# CUDA GPU accelerated performance comparison
python3 scripts/benchmark.py --cuda --sizes 4096,8192,16384,32768 --repeats 5 --output cuda_comparison

# Security validation
./scripts/gkr_attack_suite.sh && ./scripts/gkr_stress_test.sh
```

## Notes

- Large matrix sizes (K>32768) may require significant time and memory
- Security testing generates temporary files that are automatically cleaned up
- Ensure nova_poc binary is built appropriately before running scripts:
  - **Standard**: `cargo build --release`
  - **CPU AVX**: `cargo build --release --features accel_cpu_avx`
  - **CUDA**: `cargo build --release --features accel_cuda`
  - **All Acceleration**: `cargo build --release --features accel_all`
- AVX acceleration requires x86_64 CPU with AVX2 support
- CUDA acceleration requires NVIDIA GPU with CUDA drivers installed
- Benchmark script automatically falls back to available backends if acceleration features aren't built
