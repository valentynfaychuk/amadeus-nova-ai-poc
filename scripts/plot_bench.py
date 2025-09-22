#!/usr/bin/env python3
"""
Plotting script for Nova POC benchmark results.
Generates visualizations from CSV output of bench.py.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
import sys

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

def load_results(csv_path):
    """Load and preprocess benchmark results."""
    df = pd.read_csv(csv_path)

    # Convert numeric columns
    numeric_cols = ['K', 'tile_k', 'rounds', 'wall_s', 'user_s', 'sys_s',
                   'peak_rss_mb', 'proof_bytes', 'publics_bytes', 'tx_bytes']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle threads column (convert 'auto' to actual number)
    if 'threads' in df.columns:
        df['threads'] = df['threads'].replace('auto', df['cores_logical'].iloc[0])
        df['threads'] = pd.to_numeric(df['threads'], errors='coerce')

    return df

def plot_wall_time_vs_k(df, output_dir):
    """Plot wall time vs K, grouped by rounds and colored by mode."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Wall Time vs Matrix Width (K)', fontsize=16, fontweight='bold')

    stages = ['infer', 'prove', 'verify', 'all']

    for idx, stage in enumerate(stages):
        ax = axes[idx // 2, idx % 2]

        if stage == 'all':
            # Combined view
            stage_df = df.groupby(['K', 'rounds', 'mode'])['wall_s'].sum().reset_index()
            title = 'Total (All Stages)'
        else:
            stage_df = df[df['stage'] == stage].copy()
            title = f'{stage.capitalize()} Stage'

        # Group by rounds
        rounds_values = sorted(stage_df['rounds'].unique())

        # Colors for different modes
        mode_colors = {'one_pass': 'blue', 'k_pass_legacy': 'red'}

        for rounds in rounds_values:
            rounds_df = stage_df[stage_df['rounds'] == rounds]

            for mode in rounds_df['mode'].unique():
                mode_df = rounds_df[rounds_df['mode'] == mode]

                # Group by K and average across repeats
                grouped = mode_df.groupby('K')['wall_s'].agg(['mean', 'std']).reset_index()

                label = f'k={rounds}, {mode}'
                color = mode_colors.get(mode, 'gray')

                # Plot with error bars
                ax.errorbar(grouped['K'], grouped['mean'], yerr=grouped['std'],
                           marker='o', label=label, color=color,
                           alpha=0.7 if mode == 'k_pass_legacy' else 1.0,
                           linestyle='--' if mode == 'k_pass_legacy' else '-')

        ax.set_xlabel('K (Matrix Width)', fontsize=11)
        ax.set_ylabel('Wall Time (seconds)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Log scale if values span multiple orders of magnitude
        if stage_df['wall_s'].max() / stage_df['wall_s'].min() > 100:
            ax.set_yscale('log')

    plt.tight_layout()
    output_path = output_dir / 'wall_time_vs_k.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_stage_breakdown(df, output_dir):
    """Create stacked bar charts showing time breakdown per stage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Stage Time Breakdown by Configuration', fontsize=16, fontweight='bold')

    # Group configurations
    df['config'] = df['K'].astype(str) + '_k' + df['rounds'].astype(str) + '_' + df['mode']

    for idx, metric in enumerate(['wall_s', 'peak_rss_mb']):
        ax = axes[idx]

        # Pivot to get stages as columns
        pivot_df = df.pivot_table(
            index='config',
            columns='stage',
            values=metric,
            aggfunc='mean'
        )

        # Create stacked bar chart
        pivot_df.plot(kind='bar', stacked=True, ax=ax,
                     color=['#1f77b4', '#ff7f0e', '#2ca02c'])

        if metric == 'wall_s':
            ax.set_ylabel('Wall Time (seconds)', fontsize=11)
            ax.set_title('Execution Time', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel('Peak RSS (MB)', fontsize=11)
            ax.set_title('Memory Usage', fontsize=12, fontweight='bold')

        ax.set_xlabel('Configuration', fontsize=11)
        ax.legend(title='Stage', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Add grid
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'stage_breakdown.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_speedup_analysis(df, output_dir):
    """Analyze speedup of one-pass vs k-pass modes."""
    # Filter for verify stage only
    verify_df = df[df['stage'] == 'verify'].copy()

    if 'k_pass_legacy' not in verify_df['mode'].values or 'one_pass' not in verify_df['mode'].values:
        print("  Skipping speedup analysis (need both one_pass and k_pass_legacy modes)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('One-Pass vs K-Pass Freivalds Performance', fontsize=16, fontweight='bold')

    # Calculate speedup
    one_pass = verify_df[verify_df['mode'] == 'one_pass']
    k_pass = verify_df[verify_df['mode'] == 'k_pass_legacy']

    # Merge on common keys
    merge_keys = ['K', 'tile_k', 'rounds', 'threads']
    merged = pd.merge(
        one_pass[merge_keys + ['wall_s']],
        k_pass[merge_keys + ['wall_s']],
        on=merge_keys,
        suffixes=('_one', '_kpass')
    )

    merged['speedup'] = merged['wall_s_kpass'] / merged['wall_s_one']

    # Plot 1: Speedup vs K
    ax1 = axes[0]
    for rounds in sorted(merged['rounds'].unique()):
        rounds_df = merged[merged['rounds'] == rounds]
        grouped = rounds_df.groupby('K')['speedup'].agg(['mean', 'std']).reset_index()

        ax1.errorbar(grouped['K'], grouped['mean'], yerr=grouped['std'],
                    marker='o', label=f'k={rounds} rounds')

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    ax1.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10× speedup')
    ax1.set_xlabel('K (Matrix Width)', fontsize=11)
    ax1.set_ylabel('Speedup Factor', fontsize=11)
    ax1.set_title('Verification Speedup: One-Pass vs K-Pass', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Absolute times comparison
    ax2 = axes[1]

    k_values = sorted(verify_df['K'].unique())
    x_pos = np.arange(len(k_values))
    width = 0.35

    one_pass_times = []
    k_pass_times = []

    for k in k_values:
        one_time = one_pass[one_pass['K'] == k]['wall_s'].mean()
        k_time = k_pass[k_pass['K'] == k]['wall_s'].mean()
        one_pass_times.append(one_time if not pd.isna(one_time) else 0)
        k_pass_times.append(k_time if not pd.isna(k_time) else 0)

    bars1 = ax2.bar(x_pos - width/2, one_pass_times, width, label='One-Pass', color='blue')
    bars2 = ax2.bar(x_pos + width/2, k_pass_times, width, label='K-Pass Legacy', color='red')

    ax2.set_xlabel('K (Matrix Width)', fontsize=11)
    ax2.set_ylabel('Verification Time (seconds)', fontsize=11)
    ax2.set_title('Absolute Verification Times', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(k_values)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = output_dir / 'speedup_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_proof_sizes(df, output_dir):
    """Analyze proof and public input sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Proof and Transaction Sizes', fontsize=16, fontweight='bold')

    # Get unique configurations
    prove_df = df[df['stage'] == 'prove'].copy()
    prove_df['config'] = 'K=' + prove_df['K'].astype(str) + ', k=' + prove_df['rounds'].astype(str)

    # Plot 1: Proof components
    ax1 = axes[0]

    configs = prove_df['config'].unique()[:10]  # Limit to 10 configs for readability
    proof_data = []
    public_data = []

    for config in configs:
        config_df = prove_df[prove_df['config'] == config]
        proof_data.append(config_df['proof_bytes'].mean() if 'proof_bytes' in config_df else 0)
        public_data.append(config_df['publics_bytes'].mean() if 'publics_bytes' in config_df else 0)

    x_pos = np.arange(len(configs))
    width = 0.35

    bars1 = ax1.bar(x_pos - width/2, proof_data, width, label='Proof', color='#1f77b4')
    bars2 = ax1.bar(x_pos + width/2, public_data, width, label='Public Inputs', color='#ff7f0e')

    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('Size (bytes)', fontsize=11)
    ax1.set_title('Proof Component Sizes', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add 1KB line
    ax1.axhline(y=1024, color='red', linestyle='--', alpha=0.5, label='1 KB')

    # Plot 2: Total transaction size vs K
    ax2 = axes[1]

    tx_df = prove_df.groupby('K')['tx_bytes'].agg(['mean', 'std']).reset_index()

    ax2.errorbar(tx_df['K'], tx_df['mean'], yerr=tx_df['std'],
                marker='o', color='green', linewidth=2, markersize=8)

    ax2.axhline(y=1024, color='red', linestyle='--', alpha=0.5, label='1 KB limit')
    ax2.fill_between(tx_df['K'], 0, 1024, alpha=0.2, color='green', label='Target range')

    ax2.set_xlabel('K (Matrix Width)', fontsize=11)
    ax2.set_ylabel('Transaction Size (bytes)', fontsize=11)
    ax2.set_title('Total Transaction Size vs Matrix Width', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Add annotation if any values exceed 1KB
    if tx_df['mean'].max() > 1024:
        ax2.annotate('⚠️ Exceeds 1KB',
                    xy=(tx_df.loc[tx_df['mean'].idxmax(), 'K'],
                        tx_df['mean'].max()),
                    xytext=(10, 10), textcoords='offset points',
                    ha='left', fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    output_path = output_dir / 'proof_sizes.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_thread_scaling(df, output_dir):
    """Analyze performance scaling with thread count."""
    if df['threads'].nunique() <= 1:
        print("  Skipping thread scaling analysis (need multiple thread counts)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Thread Scaling Analysis', fontsize=16, fontweight='bold')

    # Focus on infer stage (most compute-intensive)
    infer_df = df[df['stage'] == 'infer'].copy()

    # Plot 1: Wall time vs threads for different K values
    ax1 = axes[0]

    for k in sorted(infer_df['K'].unique()):
        k_df = infer_df[infer_df['K'] == k]
        grouped = k_df.groupby('threads')['wall_s'].agg(['mean', 'std']).reset_index()

        ax1.errorbar(grouped['threads'], grouped['mean'], yerr=grouped['std'],
                    marker='o', label=f'K={k}')

    ax1.set_xlabel('Thread Count', fontsize=11)
    ax1.set_ylabel('Wall Time (seconds)', fontsize=11)
    ax1.set_title('Inference Time vs Thread Count', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Efficiency (speedup / threads)
    ax2 = axes[1]

    for k in sorted(infer_df['K'].unique()):
        k_df = infer_df[infer_df['K'] == k]

        # Get single-threaded baseline
        baseline = k_df[k_df['threads'] == 1]['wall_s'].mean()

        if pd.isna(baseline) or baseline == 0:
            continue

        grouped = k_df.groupby('threads')['wall_s'].mean().reset_index()
        grouped['efficiency'] = (baseline / grouped['wall_s']) / grouped['threads'] * 100

        ax2.plot(grouped['threads'], grouped['efficiency'],
                marker='o', label=f'K={k}')

    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Perfect scaling')
    ax2.set_xlabel('Thread Count', fontsize=11)
    ax2.set_ylabel('Efficiency (%)', fontsize=11)
    ax2.set_title('Parallel Efficiency', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 120])

    plt.tight_layout()
    output_path = output_dir / 'thread_scaling.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def generate_summary_report(df, output_dir):
    """Generate a text summary report of benchmark results."""
    report_path = output_dir / 'benchmark_summary.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("NOVA POC BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")

        # System info
        f.write("System Information:\n")
        f.write("-" * 40 + "\n")
        first_row = df.iloc[0]
        f.write(f"Device: {first_row['device']}\n")
        f.write(f"OS: {first_row['os']}\n")
        f.write(f"CPU: {first_row['cpu_model']}\n")
        f.write(f"Cores: {first_row['cores_physical']} physical, {first_row['cores_logical']} logical\n")
        f.write(f"RAM: {first_row['ram_gb']} GB\n")
        f.write(f"Git Commit: {first_row['git_commit']}\n")
        f.write("\n")

        # Configuration summary
        f.write("Configurations Tested:\n")
        f.write("-" * 40 + "\n")
        f.write(f"K values: {sorted(df['K'].unique())}\n")
        f.write(f"Tile sizes: {sorted(df['tile_k'].unique())}\n")
        f.write(f"Freivalds rounds: {sorted(df['rounds'].unique())}\n")
        f.write(f"Thread counts: {sorted(df['threads'].unique())}\n")
        f.write(f"Modes: {sorted(df['mode'].unique())}\n")
        f.write("\n")

        # Performance summary
        f.write("Performance Summary:\n")
        f.write("-" * 40 + "\n")

        for stage in ['infer', 'prove', 'verify']:
            stage_df = df[df['stage'] == stage]
            f.write(f"\n{stage.upper()} Stage:\n")
            f.write(f"  Wall time: {stage_df['wall_s'].min():.3f} - {stage_df['wall_s'].max():.3f} seconds\n")
            f.write(f"  Peak RSS: {stage_df['peak_rss_mb'].min():.1f} - {stage_df['peak_rss_mb'].max():.1f} MB\n")

            if stage == 'verify' and 'one_pass' in df['mode'].values and 'k_pass_legacy' in df['mode'].values:
                one_pass = stage_df[stage_df['mode'] == 'one_pass']['wall_s'].mean()
                k_pass = stage_df[stage_df['mode'] == 'k_pass_legacy']['wall_s'].mean()
                if not pd.isna(k_pass) and k_pass > 0:
                    speedup = k_pass / one_pass
                    f.write(f"  One-pass speedup: {speedup:.1f}× faster than k-pass\n")

        # Proof sizes
        f.write("\nProof Sizes:\n")
        f.write("-" * 40 + "\n")
        prove_df = df[df['stage'] == 'prove']
        if not prove_df.empty:
            f.write(f"Proof bytes: {prove_df['proof_bytes'].mean():.0f} ± {prove_df['proof_bytes'].std():.0f}\n")
            f.write(f"Public inputs: {prove_df['publics_bytes'].mean():.0f} ± {prove_df['publics_bytes'].std():.0f}\n")
            f.write(f"Total TX: {prove_df['tx_bytes'].mean():.0f} ± {prove_df['tx_bytes'].std():.0f}\n")

            if prove_df['tx_bytes'].max() < 1024:
                f.write("✅ All transactions under 1 KB limit\n")
            else:
                f.write(f"⚠️  Max transaction size: {prove_df['tx_bytes'].max():.0f} bytes\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"  Saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate plots from Nova POC benchmark results'
    )
    parser.add_argument('--in', dest='input', type=str, required=True,
                       help='Input CSV file from bench.py')
    parser.add_argument('--outdir', type=str, default='bench/plots',
                       help='Output directory for plots')

    args = parser.parse_args()

    # Load results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1

    print(f"📊 Loading benchmark results from: {input_path}")
    df = load_results(input_path)

    if df.empty:
        print("❌ Error: No data found in CSV file")
        return 1

    print(f"  Loaded {len(df)} rows, {df['K'].nunique()} K values, {df['stage'].nunique()} stages")

    # Create output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📈 Generating plots in: {output_dir}")

    # Generate plots
    try:
        plot_wall_time_vs_k(df, output_dir)
        plot_stage_breakdown(df, output_dir)
        plot_speedup_analysis(df, output_dir)
        plot_proof_sizes(df, output_dir)
        plot_thread_scaling(df, output_dir)
        generate_summary_report(df, output_dir)
    except Exception as e:
        print(f"⚠️  Error generating plots: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n✅ Plot generation complete!")
    print(f"   View plots in: {output_dir}")

    return 0

if __name__ == '__main__':
    sys.exit(main())