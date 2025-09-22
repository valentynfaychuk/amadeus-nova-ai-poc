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
Comprehensive benchmarking and visualization tool that runs GKR performance tests and generates detailed analysis plots.

```bash
# Run complete benchmark with visualization (generates benchmark_results.csv and benchmark_results.png)
python3 scripts/benchmark.py --sizes 4096,8192,16384,32768 --repeats 3

# Large-scale production benchmark (generates production_bench.csv and production_bench.png)
python3 scripts/benchmark.py --sizes 4096,8192,16384,32768,50204 --repeats 5 --output production_bench
```

**Features:**
- Automated nova_poc benchmark execution
- Real-time performance analysis and complexity fitting
- Automatic generation of both CSV data and PNG visualization
- Comprehensive visualization with 5 detailed plots
- System information tracking
- Production-scale testing support

**Output Files:**
- **name.csv**: Timing, memory, and proof size metrics
- **name.png**: Comprehensive performance visualization with:
  - Proving time scaling analysis with O(K^n) complexity fitting
  - Verification time characteristics (constant-time verification)
  - Proof size growth patterns (logarithmic scaling)
  - Normalized scaling comparison
  - Performance summary table

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
# Complete performance analysis
python3 scripts/benchmark.py --sizes 4096,8192,16384,32768,50204 --repeats 5 --output comprehensive_benchmark

# Security validation
./scripts/gkr_attack_suite.sh && ./scripts/gkr_stress_test.sh
```

## Notes

- Large matrix sizes (K>32768) may require significant time and memory
- Security testing generates temporary files that are automatically cleaned up
- Ensure nova_poc binary is built with `cargo build --release` before running scripts
