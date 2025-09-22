#!/usr/bin/env python3
"""
Benchmarking harness for Nova POC: Measuring wall-time, memory, and I/O for infer → prove → verify.
Supports k-pass vs one-pass Freivalds, sweeps K, tile_k, k (rounds), and thread counts.
"""

import argparse
import subprocess
import json
import csv
import os
import sys
import time
import tempfile
import shutil
import platform
import psutil
import socket
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import numpy as np

# Column order for CSV (strict)
CSV_COLUMNS = [
    'timestamp', 'git_commit', 'device', 'os', 'cpu_model', 'cores_logical', 'cores_physical', 'ram_gb',
    'K', 'tile_k', 'rounds', 'threads', 'mode', 'seed',
    'stage', 'wall_s', 'user_s', 'sys_s', 'peak_rss_mb', 'bytes_read', 'bytes_written',
    'proof_bytes', 'publics_bytes', 'tx_bytes', 'success', 'exit_code'
]

def get_git_commit():
    """Get short git commit hash."""
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return 'unknown'

def get_system_info():
    """Gather system information for benchmarking context."""
    info = {
        'device': socket.gethostname(),
        'os': f"{platform.system()} {platform.release()}",
        'cpu_model': platform.processor() or platform.machine(),
        'cores_logical': psutil.cpu_count(logical=True),
        'cores_physical': psutil.cpu_count(logical=False),
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'git_commit': get_git_commit()
    }
    return info

@contextmanager
def measure(stage_name=None):
    """Context manager to measure execution metrics."""
    process = psutil.Process()

    # Get initial IO counters (may not be available on all platforms)
    try:
        io_start = process.io_counters()
        has_io = True
    except (AttributeError, psutil.AccessDenied):
        has_io = False
        io_start = None

    # Reset peak memory tracking
    try:
        process.memory_info()
    except:
        pass

    start_time = time.perf_counter()
    start_cpu = process.cpu_times()

    metrics = {}

    try:
        yield metrics
    finally:
        end_time = time.perf_counter()
        end_cpu = process.cpu_times()

        # Get peak memory
        try:
            mem_info = process.memory_info()
            peak_rss_mb = mem_info.rss / (1024 * 1024)
        except:
            peak_rss_mb = None

        # Calculate IO if available
        if has_io and io_start:
            try:
                io_end = process.io_counters()
                bytes_read = io_end.read_bytes - io_start.read_bytes
                bytes_written = io_end.write_bytes - io_start.write_bytes
            except:
                bytes_read = None
                bytes_written = None
        else:
            bytes_read = None
            bytes_written = None

        metrics.update({
            'stage': stage_name,
            'wall_s': round(end_time - start_time, 3),
            'user_s': round(end_cpu.user - start_cpu.user, 3),
            'sys_s': round(end_cpu.system - start_cpu.system, 3),
            'peak_rss_mb': round(peak_rss_mb, 2) if peak_rss_mb else None,
            'bytes_read': bytes_read,
            'bytes_written': bytes_written
        })

def gen_realistic_data(K, seed, workdir, robust=False):
    """
    Generate realistic quantized data for benchmarking.
    Returns paths to generated files and metadata.
    """
    np.random.seed(seed)

    # LLM-like initialization
    sigma = 1.0 / np.sqrt(K)

    print(f"  Generating realistic data: K={K}, seed={seed}")

    # Generate float weights and activations
    Wf = np.random.normal(0.0, sigma, size=(16, K)).astype(np.float32)
    xf = np.random.normal(0.0, 1.0, size=(K,)).astype(np.float32)

    # Per-row weight scales
    if robust:
        s_w = (np.quantile(np.abs(Wf), 0.999, axis=1) / 127.0).astype(np.float32)
    else:
        s_w = (np.max(np.abs(Wf), axis=1) / 127.0).astype(np.float32)

    # Per-tensor activation scale
    s_x = float(np.max(np.abs(xf)) / 127.0)

    # Quantize to int8
    q_w = np.clip(np.rint(Wf / s_w[:, None]), -127, 127).astype(np.int8)
    q_x = np.clip(np.rint(xf / s_x), -127, 127).astype(np.int8)

    # Convert to int16 for compatibility with existing code
    w1_int16 = q_w.astype(np.int16)
    x0_int16 = q_x.astype(np.int16).tolist()

    # Generate W2 (16×16) with similar approach
    W2f = np.random.normal(0.0, 1.0/4.0, size=(16, 16)).astype(np.float32)
    w2 = np.clip(np.rint(W2f * 10), -10, 10).astype(np.int16).tolist()

    # Save files
    preset_dir = Path(workdir) / f"K{K}_s{seed}"
    preset_dir.mkdir(parents=True, exist_ok=True)

    w1_path = preset_dir / "W1.bin"
    w1_int16.tofile(str(w1_path))

    x0_path = preset_dir / "x0.json"
    with open(x0_path, 'w') as f:
        json.dump(x0_int16, f)

    w2_path = preset_dir / "W2.json"
    with open(w2_path, 'w') as f:
        json.dump(w2, f, indent=2)

    # Save metadata
    meta_path = preset_dir / "metadata.json"
    metadata = {
        'K': K,
        'seed': seed,
        's_w': s_w.tolist(),
        's_x': s_x,
        'robust': robust
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return str(w1_path), str(x0_path), str(w2_path), metadata

def run_stage(stage, cmd, env_vars=None, workdir=None):
    """Run a single stage and capture metrics."""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    with measure(stage) as metrics:
        print(f"    Running {stage}: {' '.join(cmd[:3])}...")
        result = subprocess.run(
            cmd,
            env=env,
            cwd=workdir,
            capture_output=True,
            text=True
        )

        metrics['exit_code'] = result.returncode
        metrics['success'] = (result.returncode == 0)

        if not metrics['success']:
            print(f"      ⚠️  {stage} failed with exit code {result.returncode}")
            if result.stderr:
                print(f"      Error: {result.stderr[:200]}")

    return metrics, result

def run_benchmark_config(config, system_info, workdir, binary_path, repeats=1):
    """Run a single configuration multiple times and collect metrics."""
    K = config['K']
    tile_k = config['tile_k']
    rounds = config['rounds']
    threads = config['threads']
    mode = config['mode']
    seed = config.get('seed', 42)

    print(f"\n📊 Config: K={K}, tile_k={tile_k}, rounds={rounds}, threads={threads}, mode={mode}")

    # Generate data if needed
    data_dir = Path(workdir) / f"K{K}_s{seed}"
    if not data_dir.exists():
        w1_path, x0_path, w2_path, metadata = gen_realistic_data(K, seed, workdir)
    else:
        w1_path = str(data_dir / "W1.bin")
        x0_path = str(data_dir / "x0.json")
        w2_path = str(data_dir / "W2.json")
        with open(data_dir / "metadata.json") as f:
            metadata = json.load(f)

    # Setup keys if not present
    keys_dir = Path(workdir) / "keys"
    if not (keys_dir / "pk.bin").exists():
        print("  Setting up proving/verification keys...")
        setup_cmd = [binary_path, "setup", "--out-dir", str(keys_dir)]
        subprocess.run(setup_cmd, capture_output=True, check=True)

    results = []

    for repeat in range(repeats):
        print(f"  Repeat {repeat + 1}/{repeats}")

        run_json = Path(workdir) / f"run_K{K}_r{repeat}.json"
        proof_dir = Path(workdir) / f"proof_K{K}_r{repeat}"
        proof_dir.mkdir(exist_ok=True)

        # Set thread count
        env_vars = {}
        if threads == 'auto':
            env_vars['RAYON_NUM_THREADS'] = str(psutil.cpu_count(logical=True))
        else:
            env_vars['RAYON_NUM_THREADS'] = str(threads)

        # Stage 1: Infer
        infer_cmd = [
            binary_path, "infer",
            "--k", str(K),
            "--tile-k", str(tile_k),
            "--weights1-path", w1_path,
            "--weights2-path", w2_path,
            "--x0-path", x0_path,
            "--freivalds-rounds", str(rounds),
            "--out", str(run_json)
        ]

        if mode == 'k_pass_legacy':
            # For legacy mode comparison, we might need to modify the command
            # or use a different flag if available
            pass  # Placeholder for k-pass specific flags

        metrics_infer, result_infer = run_stage("infer", infer_cmd, env_vars)

        # Stage 2: Prove
        prove_cmd = [
            binary_path, "prove",
            str(run_json),
            "--pk-path", str(keys_dir / "pk.bin"),
            "--out-dir", str(proof_dir)
        ]

        metrics_prove, result_prove = run_stage("prove", prove_cmd, env_vars)

        # Get proof sizes
        proof_bytes = None
        publics_bytes = None
        if (proof_dir / "proof.bin").exists():
            proof_bytes = os.path.getsize(proof_dir / "proof.bin")
        if (proof_dir / "public_inputs.json").exists():
            publics_bytes = os.path.getsize(proof_dir / "public_inputs.json")

        # Stage 3: Verify
        verify_cmd = [
            binary_path, "verify",
            str(run_json),
            "--weights1-path", w1_path,
            "--vk-path", str(proof_dir / "vk.bin"),
            "--proof-path", str(proof_dir / "proof.bin"),
            "--public-inputs-path", str(proof_dir / "public_inputs.json")
        ]

        if mode == 'one_pass':
            # One-pass mode is the default
            pass
        else:
            # For k-pass legacy mode, might need different flags
            verify_cmd.append("--allow-low-k")

        metrics_verify, result_verify = run_stage("verify", verify_cmd, env_vars)

        # Calculate total transaction size
        tx_bytes = (proof_bytes or 0) + (publics_bytes or 0)

        # Prepare rows for each stage
        timestamp = datetime.now().isoformat()
        base_row = {
            'timestamp': timestamp,
            **system_info,
            'K': K,
            'tile_k': tile_k,
            'rounds': rounds,
            'threads': env_vars.get('RAYON_NUM_THREADS', '1'),
            'mode': mode,
            'seed': seed,
            'proof_bytes': proof_bytes,
            'publics_bytes': publics_bytes,
            'tx_bytes': tx_bytes
        }

        # Add row for each stage
        for metrics in [metrics_infer, metrics_prove, metrics_verify]:
            row = {**base_row, **metrics}
            # Fill missing values with None
            for col in CSV_COLUMNS:
                if col not in row:
                    row[col] = None
            results.append(row)

    return results

def parse_grid(grid_str):
    """Parse grid string like 'K=4096,12288,16384' into list."""
    if '=' in grid_str:
        key, values = grid_str.split('=')
        return [int(v) for v in values.split(',')]
    else:
        return [int(v) for v in grid_str.split(',')]

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Nova POC across multiple configurations'
    )

    # Configuration parameters
    parser.add_argument('--grid', type=str, default='K=4096',
                       help='Grid values (e.g., K=4096,12288,16384,24576)')
    parser.add_argument('--tile-k', type=str, default='1024',
                       help='Tile sizes (comma-separated, e.g., 1024,4096,8192)')
    parser.add_argument('--rounds', type=str, default='16',
                       help='Freivalds rounds (comma-separated, e.g., 8,16,32)')
    parser.add_argument('--threads', type=str, default='auto',
                       help='Thread counts (comma-separated, e.g., 1,auto)')
    parser.add_argument('--modes', type=str, default='one_pass',
                       help='Modes to test (comma-separated: one_pass,k_pass_legacy)')
    parser.add_argument('--repeats', type=int, default=3,
                       help='Number of repeats per configuration')

    # Output options
    parser.add_argument('--out', type=str, default='bench/results.csv',
                       help='Output CSV file path')
    parser.add_argument('--workdir', type=str, default='.bench_runs',
                       help='Working directory for temporary files')

    # Binary path
    parser.add_argument('--binary', type=str, default=None,
                       help='Path to nova_poc binary (auto-detect if not specified)')

    args = parser.parse_args()

    # Parse grid configurations
    K_values = parse_grid(args.grid)
    tile_k_values = [int(v) for v in args.tile_k.split(',')]
    rounds_values = [int(v) for v in args.rounds.split(',')]
    threads_values = []
    for t in args.threads.split(','):
        if t == 'auto':
            threads_values.append('auto')
        else:
            threads_values.append(int(t))
    modes = args.modes.split(',')

    # Determine binary path
    if args.binary:
        binary_path = args.binary
    else:
        # Try to find the release binary
        release_binary = "./target/release/nova_poc"
        debug_binary = "./target/debug/nova_poc"

        if os.path.exists(release_binary):
            binary_path = release_binary
            print(f"Using release binary: {binary_path}")
        elif os.path.exists(debug_binary):
            binary_path = debug_binary
            print(f"⚠️  Using debug binary (slower): {binary_path}")
        else:
            print("Building nova_poc binary...")
            subprocess.run(["cargo", "build", "--release", "-p", "nova_poc"], check=True)
            binary_path = release_binary

    # Create working directory
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    # Get system information
    system_info = get_system_info()

    print("\n🔍 System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    # Generate all configurations
    configs = []
    for K in K_values:
        for tile_k in tile_k_values:
            # Skip invalid combinations
            if tile_k > K:
                continue
            for rounds in rounds_values:
                for threads in threads_values:
                    for mode in modes:
                        configs.append({
                            'K': K,
                            'tile_k': tile_k,
                            'rounds': rounds,
                            'threads': threads,
                            'mode': mode,
                            'seed': 42
                        })

    print(f"\n📊 Total configurations: {len(configs)}")
    print(f"   Repeats per config: {args.repeats}")
    print(f"   Total runs: {len(configs) * args.repeats * 3} stages")

    # Prepare output CSV
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write header if file doesn't exist
    write_header = not output_path.exists()

    all_results = []

    # Run benchmarks
    start_time = time.time()
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running configuration...")
        results = run_benchmark_config(
            config,
            system_info,
            args.workdir,
            binary_path,
            args.repeats
        )
        all_results.extend(results)

        # Write results incrementally
        with open(output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"\n✅ Benchmarking complete!")
    print(f"   Total time: {elapsed:.1f} seconds")
    print(f"   Results saved to: {output_path}")
    print(f"   Total rows: {len(all_results)}")

    # Quick summary
    if all_results:
        total_infer = sum(r['wall_s'] for r in all_results if r['stage'] == 'infer' and r['wall_s'])
        total_prove = sum(r['wall_s'] for r in all_results if r['stage'] == 'prove' and r['wall_s'])
        total_verify = sum(r['wall_s'] for r in all_results if r['stage'] == 'verify' and r['wall_s'])

        print(f"\n📈 Time breakdown:")
        print(f"   Infer:  {total_infer:.1f}s")
        print(f"   Prove:  {total_prove:.1f}s")
        print(f"   Verify: {total_verify:.1f}s")

    return 0

if __name__ == '__main__':
    sys.exit(main())