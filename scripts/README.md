# Scripts Documentation

This directory contains tools for benchmarking, performance analysis, and security testing of the Nova AI GKR implementation.

## Benchmarking & Performance

### `bench.py`
Comprehensive benchmarking harness for Nova POC with detailed system profiling.

```bash
# Benchmark multiple matrix sizes with GKR mode
python3 scripts/bench.py --grid K=1024,2048,4096,8192 --modes gkr --repeats 3

# Custom configuration
python3 scripts/bench.py \
    --grid K=4096,8192,16384 \
    --tile-k 1024,2048 \
    --rounds 16,32 \
    --threads 1,auto \
    --modes gkr \
    --out custom_benchmark.csv
```

**Features:**
- System profiling (CPU, memory, I/O)
- Multiple configuration sweeps
- Git commit tracking
- Detailed CSV output with timing/memory/proof size metrics

### `plot_comprehensive_benchmarks.py`
Generates detailed performance visualization from benchmark data.

```bash
# Generate plots from comprehensive benchmark data
python3 scripts/plot_comprehensive_benchmarks.py
```

**Outputs:**
- Proving time scaling analysis
- Verification time characteristics
- Proof size growth patterns
- Performance efficiency metrics
- Summary tables and statistics

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
Focused stress testing with 6 specific attack categories.

```bash
# Run stress tests
./scripts/gkr_stress_test.sh
```

**Test Categories:**
- Polynomial coefficient manipulation
- Hash collision attempts
- Challenge prediction attacks
- Evaluation bypass attempts
- Binary corruption scenarios
- Performance under load

### `binary_validation_attack.sh`
Demonstrates and tests binary proof validation vulnerabilities.

```bash
# Test binary validation security
./scripts/binary_validation_attack.sh
```

**Validation Tests:**
- Trailing data injection
- Padding corruption
- Multiple corruption vectors
- Strategic data corruption
- Vulnerability mapping

### `run_attacks.sh`
Orchestrates multiple attack scenarios and generates comprehensive reports.

```bash
# Run all attacks with detailed reporting
./scripts/run_attacks.sh
```

**Features:**
- Automated attack execution
- Vulnerability discovery
- Security report generation
- Fix validation testing

## Usage Examples

### Complete Performance Analysis
```bash
# 1. Generate comprehensive benchmarks
python3 scripts/bench.py --grid K=1024,2048,4096,8192,16384,32768 --modes gkr --repeats 5

# 2. Create visualization
python3 scripts/plot_comprehensive_benchmarks.py

# 3. View results
open benchmark_plots/gkr_comprehensive_performance.png
```

### Security Validation Workflow
```bash
# 1. Run comprehensive attack suite
./scripts/gkr_attack_suite.sh > security_report.txt

# 2. Run specific vulnerability tests
./scripts/binary_validation_attack.sh

# 3. Stress test performance under attack
./scripts/gkr_stress_test.sh

# 4. Generate complete security analysis
./scripts/run_attacks.sh
```

### Custom Benchmark Configuration
```bash
# Production-scale benchmarking
python3 scripts/bench.py \
    --grid K=8192,16384,32768,65536 \
    --modes gkr \
    --repeats 10 \
    --out production_benchmark.csv \
    --workdir /tmp/bench_workspace
```

## Script Dependencies

**Python Scripts:**
- `pandas` - Data analysis
- `matplotlib` - Visualization
- `numpy` - Numerical computing
- `psutil` - System profiling

**Shell Scripts:**
- Standard Unix utilities (`dd`, `sha256sum`, etc.)
- Compiled `nova_poc` binary
- Rust toolchain for building

## Output Files

**Benchmarking:**
- `*.csv` - Benchmark data with timing/memory/proof metrics
- `benchmark_plots/*.png` - Performance visualization plots
- System profiling data with git commit tracking

**Security Testing:**
- Attack logs with vulnerability discovery reports
- Binary corruption test results
- Performance under attack measurements
- Security analysis summaries

## Performance Notes

- **Benchmarking**: Large matrix sizes (K>32768) may require significant time/memory
- **Security Testing**: Attack suites can generate many temporary files
- **Plotting**: Visualization scripts require matplotlib backend (may need X11 forwarding for remote systems)

## Contributing

When adding new scripts:
1. Include comprehensive help/usage information
2. Follow existing naming conventions
3. Add entry to this README with usage examples
4. Test with various input sizes and edge cases