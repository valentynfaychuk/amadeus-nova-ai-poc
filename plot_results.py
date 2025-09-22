#!/usr/bin/env python3
"""
Plot GKR benchmarking results showing scalability of proof generation time
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read benchmark results
df = pd.read_csv('benchmark_results.csv')

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Proving time vs Matrix size
ax1.plot(df['K'], df['prove_time_ms'], 'bo-', linewidth=2, markersize=8, label='GKR Proving Time')
ax1.set_xlabel('Matrix Width (K)', fontsize=12)
ax1.set_ylabel('Proving Time (ms)', fontsize=12)
ax1.set_title('GKR Proving Time Scaling\n(16×K Matrix-Vector Multiplication)', fontsize=14, pad=20)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log', base=2)

# Add trend line
log_k = np.log2(df['K'])
log_time = np.log2(df['prove_time_ms'])
slope, intercept = np.polyfit(log_k, log_time, 1)
trend_k = np.logspace(np.log2(df['K'].min()), np.log2(df['K'].max()), 100, base=2)
trend_time = 2**(slope * np.log2(trend_k) + intercept)
ax1.plot(trend_k, trend_time, 'r--', alpha=0.7, label=f'Trend: O(K^{slope:.2f})')
ax1.legend()

# Add performance annotations
for i, row in df.iterrows():
    ax1.annotate(f'{row["prove_time_ms"]}ms',
                (row['K'], row['prove_time_ms']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, alpha=0.8)

# Plot 2: Complexity analysis
ax2.bar(range(len(df)), df['prove_time_ms'], color='steelblue', alpha=0.7)
ax2.set_xlabel('Matrix Size', fontsize=12)
ax2.set_ylabel('Proving Time (ms)', fontsize=12)
ax2.set_title('GKR Performance by Matrix Size', fontsize=14, pad=20)
ax2.set_xticks(range(len(df)))
ax2.set_xticklabels([f'16×{k}' for k in df['K']], rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (k, time) in enumerate(zip(df['K'], df['prove_time_ms'])):
    ax2.text(i, time + 2, f'{time}ms', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# Save the plot
plt.savefig('gkr_scaling_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('gkr_scaling_analysis.pdf', bbox_inches='tight')

print("Performance Analysis Summary:")
print("=" * 50)
print(f"Matrix sizes tested: {df['K'].tolist()}")
print(f"Proving times (ms): {df['prove_time_ms'].tolist()}")
print(f"Proof sizes (KB): {df['proof_size_kb'].tolist()}")
print(f"\nScaling factor: ~{slope:.2f} (O(K^{slope:.2f}))")
print(f"Proof size is constant: {df['proof_size_kb'].nunique() == 1}")

# Calculate efficiency metrics
print(f"\nEfficiency Metrics:")
for _, row in df.iterrows():
    ops_per_sec = (16 * row['K'] * 1000) / row['prove_time_ms']  # matrix ops per second
    print(f"  K={row['K']:4d}: {ops_per_sec:,.0f} matrix elements/sec")

print(f"\nGenerated plots: gkr_scaling_analysis.png, gkr_scaling_analysis.pdf")