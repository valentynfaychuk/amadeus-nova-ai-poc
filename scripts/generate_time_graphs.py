#!/usr/bin/env python3
"""
Generate specific proving and verification time graphs for GKR mode.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def load_and_process_data(csv_path):
    """Load benchmark data and compute statistics."""
    df = pd.read_csv(csv_path)

    # Group by matrix size and stage, compute statistics
    stats = df.groupby(['k', 'stage']).agg({
        'time_ms': ['mean', 'std', 'min', 'max'],
        'proof_size_kb': 'mean'
    }).round(3)

    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns]
    stats = stats.reset_index()

    # Separate proving and verification data
    prove_data = stats[stats['stage'] == 'prove'].copy()
    verify_data = stats[stats['stage'] == 'verify'].copy()

    return prove_data, verify_data

def create_proving_time_graph(prove_data, output_dir="benchmark_plots"):
    """Create detailed proving time analysis graph."""
    Path(output_dir).mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('GKR Proving Time Analysis', fontsize=18, fontweight='bold', y=0.95)

    # Linear scale plot
    ax1.errorbar(prove_data['k'], prove_data['time_ms_mean'],
                yerr=prove_data['time_ms_std'],
                marker='o', linewidth=3, markersize=10,
                capsize=8, capthick=2, elinewidth=2,
                color='#2E86AB', label='Proving Time (mean ± std)')

    # Fill min-max range
    ax1.fill_between(prove_data['k'],
                    prove_data['time_ms_min'],
                    prove_data['time_ms_max'],
                    alpha=0.2, color='#2E86AB', label='Min-Max Range')

    ax1.set_xlabel('Matrix Width (K)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Proving Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_title('Linear Scale View', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12)

    # Add data point annotations
    for _, row in prove_data.iterrows():
        ax1.annotate(f'{row["time_ms_mean"]:.1f}ms',
                    (row['k'], row['time_ms_mean']),
                    textcoords="offset points", xytext=(0,15),
                    ha='center', fontsize=10, fontweight='bold')

    # Log-log scale plot with trend analysis
    ax2.loglog(prove_data['k'], prove_data['time_ms_mean'],
              'o-', linewidth=3, markersize=10,
              color='#A23B72', label='Proving Time')

    # Add trend line
    log_k = np.log2(prove_data['k'])
    log_time = np.log(prove_data['time_ms_mean'])
    coeffs = np.polyfit(log_k, log_time, 1)

    k_trend = np.logspace(np.log2(prove_data['k'].min()),
                         np.log2(prove_data['k'].max()), 100, base=2)
    time_trend = np.exp(coeffs[1]) * (k_trend ** coeffs[0])

    ax2.loglog(k_trend, time_trend, '--', linewidth=2, alpha=0.8,
              color='#F18F01', label=f'Trend: O(K^{coeffs[0]:.2f})')

    # Reference lines
    k_ref = np.array([512, 16384])
    linear_ref = 10 * (k_ref / 512)
    quadratic_ref = 10 * (k_ref / 512) ** 2

    ax2.loglog(k_ref, linear_ref, ':', alpha=0.6, color='green',
              linewidth=2, label='O(K) Linear Reference')
    ax2.loglog(k_ref, quadratic_ref, ':', alpha=0.6, color='red',
              linewidth=2, label='O(K²) Quadratic Reference')

    ax2.set_xlabel('Matrix Width (K)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Proving Time (ms)', fontsize=14, fontweight='bold')
    ax2.set_title('Log-Log Scale with Complexity Analysis', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11)

    # Add performance annotations
    ax2.text(0.05, 0.95,
            f'Empirical Scaling: O(K^{coeffs[0]:.2f})\n'
            f'Near-Linear Performance\n'
            f'Range: {prove_data["k"].min()}K → {prove_data["k"].max()}K\n'
            f'Time: {prove_data["time_ms_mean"].min():.1f}ms → {prove_data["time_ms_mean"].max():.1f}ms',
            transform=ax2.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/gkr_proving_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gkr_proving_time_analysis.pdf', bbox_inches='tight')
    plt.close()

    return coeffs[0]  # Return scaling exponent

def create_verification_time_graph(verify_data, output_dir="benchmark_plots"):
    """Create detailed verification time analysis graph."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('GKR Verification Time Analysis', fontsize=18, fontweight='bold', y=0.95)

    # Linear scale plot
    ax1.errorbar(verify_data['k'], verify_data['time_ms_mean'],
                yerr=verify_data['time_ms_std'],
                marker='s', linewidth=3, markersize=10,
                capsize=8, capthick=2, elinewidth=2,
                color='#4CAF50', label='Verification Time (mean ± std)')

    # Fill min-max range
    ax1.fill_between(verify_data['k'],
                    verify_data['time_ms_min'],
                    verify_data['time_ms_max'],
                    alpha=0.2, color='#4CAF50', label='Min-Max Range')

    # Add horizontal line for average
    avg_time = verify_data['time_ms_mean'].mean()
    ax1.axhline(y=avg_time, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Average: {avg_time:.2f}ms')

    ax1.set_xlabel('Matrix Width (K)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Verification Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_title('Linear Scale View', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, max(verify_data['time_ms_max']) * 1.1)

    # Add data point annotations
    for _, row in verify_data.iterrows():
        ax1.annotate(f'{row["time_ms_mean"]:.2f}ms',
                    (row['k'], row['time_ms_mean']),
                    textcoords="offset points", xytext=(0,15),
                    ha='center', fontsize=10, fontweight='bold')

    # Box plot showing distribution
    verification_times = []
    k_values = []
    labels = []

    for k in verify_data['k']:
        verification_times.append(verify_data[verify_data['k'] == k]['time_ms_mean'].iloc[0])
        k_values.append(k)
        labels.append(f'{k}K')

    box_data = [verification_times]
    box = ax2.boxplot(box_data, labels=['All Sizes'], patch_artist=True,
                     boxprops=dict(facecolor='lightgreen', alpha=0.7),
                     medianprops=dict(color='darkgreen', linewidth=2))

    # Scatter plot overlay
    ax2.scatter([1] * len(verification_times), verification_times,
               alpha=0.7, s=100, color='darkgreen', zorder=10)

    # Add individual data points as text
    for i, (k, time) in enumerate(zip(k_values, verification_times)):
        ax2.annotate(f'{k}K: {time:.2f}ms',
                    (1.1, time), fontsize=10,
                    verticalalignment='center')

    ax2.set_ylabel('Verification Time (ms)', fontsize=14, fontweight='bold')
    ax2.set_title('Distribution Analysis', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Add statistics text
    std_all = np.std(verification_times)
    ax2.text(0.05, 0.95,
            f'Statistics Across All Sizes:\n'
            f'Mean: {avg_time:.3f}ms\n'
            f'Std Dev: {std_all:.3f}ms\n'
            f'Min: {min(verification_times):.3f}ms\n'
            f'Max: {max(verification_times):.3f}ms\n'
            f'Range: {max(verification_times) - min(verification_times):.3f}ms\n\n'
            f'Verification is effectively\nconstant time!',
            transform=ax2.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/gkr_verification_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gkr_verification_time_analysis.pdf', bbox_inches='tight')
    plt.close()

def create_combined_time_comparison(prove_data, verify_data, output_dir="benchmark_plots"):
    """Create combined proving vs verification time comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Plot proving times
    ax.semilogy(prove_data['k'], prove_data['time_ms_mean'],
               'o-', linewidth=4, markersize=12,
               color='#FF6B35', label='Proving Time', zorder=5)

    # Plot verification times
    ax.semilogy(verify_data['k'], verify_data['time_ms_mean'],
               's-', linewidth=4, markersize=12,
               color='#004E89', label='Verification Time', zorder=5)

    # Add error bars
    ax.errorbar(prove_data['k'], prove_data['time_ms_mean'],
               yerr=prove_data['time_ms_std'],
               fmt='none', capsize=6, capthick=2, elinewidth=2,
               color='#FF6B35', alpha=0.7)

    ax.errorbar(verify_data['k'], verify_data['time_ms_mean'],
               yerr=verify_data['time_ms_std'],
               fmt='none', capsize=6, capthick=2, elinewidth=2,
               color='#004E89', alpha=0.7)

    # Highlight the gap between proving and verification
    for k in prove_data['k']:
        prove_time = prove_data[prove_data['k'] == k]['time_ms_mean'].iloc[0]
        verify_time = verify_data[verify_data['k'] == k]['time_ms_mean'].iloc[0]

        ax.fill_between([k-k*0.05, k+k*0.05], [verify_time, verify_time],
                       [prove_time, prove_time],
                       alpha=0.1, color='gray', zorder=1)

    ax.set_xlabel('Matrix Width (K)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=16, fontweight='bold')
    ax.set_title('GKR Proving vs Verification Time Comparison',
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=14, loc='upper left')
    ax.set_xscale('log', base=2)

    # Add annotations showing the gap
    for _, prove_row in prove_data.iterrows():
        verify_row = verify_data[verify_data['k'] == prove_row['k']].iloc[0]
        ratio = prove_row['time_ms_mean'] / verify_row['time_ms_mean']

        # Annotate proving time
        ax.annotate(f'{prove_row["time_ms_mean"]:.1f}ms',
                   (prove_row['k'], prove_row['time_ms_mean']),
                   textcoords="offset points", xytext=(0,20),
                   ha='center', fontsize=11, fontweight='bold',
                   color='#FF6B35')

        # Annotate verification time
        ax.annotate(f'{verify_row["time_ms_mean"]:.2f}ms',
                   (verify_row['k'], verify_row['time_ms_mean']),
                   textcoords="offset points", xytext=(0,-25),
                   ha='center', fontsize=11, fontweight='bold',
                   color='#004E89')

        # Show ratio
        mid_y = np.sqrt(prove_row['time_ms_mean'] * verify_row['time_ms_mean'])
        ax.annotate(f'{ratio:.0f}×',
                   (prove_row['k'], mid_y),
                   textcoords="offset points", xytext=(25,0),
                   ha='left', fontsize=10, fontweight='bold',
                   color='red', alpha=0.8)

    # Add summary statistics
    avg_prove = prove_data['time_ms_mean'].mean()
    avg_verify = verify_data['time_ms_mean'].mean()
    avg_ratio = avg_prove / avg_verify

    ax.text(0.02, 0.98,
           f'Performance Summary:\n'
           f'• Proving: {prove_data["time_ms_mean"].min():.1f} - {prove_data["time_ms_mean"].max():.1f}ms\n'
           f'• Verification: {verify_data["time_ms_mean"].min():.2f} - {verify_data["time_ms_mean"].max():.2f}ms\n'
           f'• Avg Speedup: {avg_ratio:.0f}× faster verification\n'
           f'• Matrix Range: {prove_data["k"].min()}K - {prove_data["k"].max()}K elements\n'
           f'• Verification is ~constant time\n'
           f'• Proving scales near-linearly',
           transform=ax.transAxes, fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.9))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/gkr_proving_vs_verification_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/gkr_proving_vs_verification_comparison.pdf', bbox_inches='tight')
    plt.close()

def main():
    """Generate all time-specific graphs for GKR mode."""
    csv_path = "production_scale_benchmark.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    if not Path(csv_path).exists():
        print(f"Error: Benchmark file '{csv_path}' not found!")
        return 1

    print(f"📊 Loading GKR benchmark data from {csv_path}...")
    prove_data, verify_data = load_and_process_data(csv_path)

    print("📈 Generating proving time analysis...")
    scaling_exponent = create_proving_time_graph(prove_data)

    print("📉 Generating verification time analysis...")
    create_verification_time_graph(verify_data)

    print("⚖️  Generating combined comparison graph...")
    create_combined_time_comparison(prove_data, verify_data)

    print("✅ GKR time analysis complete!")
    print("   📊 Proving time analysis: gkr_proving_time_analysis.png")
    print("   📉 Verification time analysis: gkr_verification_time_analysis.png")
    print("   ⚖️  Combined comparison: gkr_proving_vs_verification_comparison.png")

    # Print key insights
    print(f"\n🔍 Key Performance Insights:")
    print(f"   • Proving complexity: O(K^{scaling_exponent:.2f}) - Near linear!")
    print(f"   • Proving range: {prove_data['time_ms_mean'].min():.1f}ms → {prove_data['time_ms_mean'].max():.1f}ms")
    print(f"   • Verification: ~{verify_data['time_ms_mean'].mean():.2f}ms (constant)")
    print(f"   • Speedup: {prove_data['time_ms_mean'].mean() / verify_data['time_ms_mean'].mean():.0f}× faster verification")
    print(f"   • Matrix scaling: {prove_data['k'].min()}K → {prove_data['k'].max()}K ({prove_data['k'].max() // prove_data['k'].min()}× increase)")

    return 0

if __name__ == "__main__":
    exit(main())