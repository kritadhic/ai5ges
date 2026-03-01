"""
Phase 1: Exploratory Data Analysis and Visualization - Netop 5G Dataset
========================================================================
This script generates comprehensive visualizations for the preprocessed Netop dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create plots directory
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

print("="*80)
print("NETOP 5G NETWORK - EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load processed data from processed_data directory
print("\nLoading processed data...")
df = pd.read_csv('processed_data/netop_processed.csv')
df['Time'] = pd.to_datetime(df['Time'])
print(f"Loaded {len(df)} records")
print(f"Date range: {df['Time'].min()} to {df['Time'].max()}")

# =============================================================================
# 1. TIME SERIES PLOTS
# =============================================================================
print("\n[1/8] Creating time series plots...")

# Plot 1: Energy consumption over time
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Hourly average
hourly_energy = df.groupby(df['Time'].dt.floor('H'))['Energy'].mean()
axes[0].plot(hourly_energy.index, hourly_energy.values, linewidth=1.5, color='#2ecc71')
axes[0].set_title('Average Energy Consumption Over Time (Hourly)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Energy (normalized)')
axes[0].grid(True, alpha=0.3)

# Daily average
daily_energy = df.groupby(df['Time'].dt.date)['Energy'].mean()
axes[1].bar(range(len(daily_energy)), daily_energy.values, color='#3498db', alpha=0.7)
axes[1].set_title('Average Energy Consumption by Day', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Energy (normalized)')
axes[1].set_xticks(range(len(daily_energy)))
axes[1].set_xticklabels([str(d) for d in daily_energy.index], rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/01_energy_time_series.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 01_energy_time_series.png")

# Plot 2: Load over time
fig, ax = plt.subplots(figsize=(14, 6))
hourly_load = df.groupby(df['Time'].dt.floor('H'))['load'].mean()
ax.plot(hourly_load.index, hourly_load.values, linewidth=1.5, color='#e74c3c')
ax.set_title('Average Cell Load Over Time (Hourly)', fontsize=14, fontweight='bold')
ax.set_xlabel('Time')
ax.set_ylabel('Load (ratio)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/02_load_time_series.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 02_load_time_series.png")

# =============================================================================
# 2. HOURLY PATTERNS
# =============================================================================
print("\n[2/8] Creating hourly pattern analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Energy by hour of day
hourly_stats = df.groupby('hour_of_day').agg({
    'Energy': ['mean', 'std'],
    'load': ['mean', 'std']
})

axes[0].errorbar(hourly_stats.index, hourly_stats[('Energy', 'mean')],
                 yerr=hourly_stats[('Energy', 'std')],
                 marker='o', capsize=3, linewidth=2, markersize=6, color='#2ecc71')
axes[0].set_title('Energy Consumption by Hour of Day', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Energy (normalized)')
axes[0].grid(True, alpha=0.3)
axes[0].axvspan(8, 10, alpha=0.2, color='red', label='Morning Peak')
axes[0].axvspan(17, 19, alpha=0.2, color='orange', label='Evening Peak')
axes[0].legend()

# Load by hour of day
axes[1].errorbar(hourly_stats.index, hourly_stats[('load', 'mean')],
                 yerr=hourly_stats[('load', 'std')],
                 marker='s', capsize=3, linewidth=2, markersize=6, color='#e74c3c')
axes[1].set_title('Cell Load by Hour of Day', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Load (ratio)')
axes[1].grid(True, alpha=0.3)
axes[1].axvspan(8, 10, alpha=0.2, color='red', label='Morning Peak')
axes[1].axvspan(17, 19, alpha=0.2, color='orange', label='Evening Peak')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/03_hourly_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 03_hourly_patterns.png")

# =============================================================================
# 3. DISTRIBUTION PLOTS
# =============================================================================
print("\n[3/8] Creating distribution plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Energy distribution
axes[0, 0].hist(df['Energy'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Energy Consumption Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Energy (normalized)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['Energy'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Energy"].mean():.2f}')
axes[0, 0].axvline(df['Energy'].median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {df["Energy"].median():.2f}')
axes[0, 0].legend()

# Load distribution
axes[0, 1].hist(df['load'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Cell Load Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Load (ratio)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(df['load'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["load"].mean():.2f}')
axes[0, 1].axvline(df['load'].median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {df["load"].median():.2f}')
axes[0, 1].legend()

# Box plots
sns.boxplot(data=df[['Energy', 'load']], ax=axes[1, 0], palette=['#2ecc71', '#e74c3c'])
axes[1, 0].set_title('Box Plots: Energy and Load', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Normalized Value')

# Energy saving mode distribution
axes[1, 1].hist(df['EnergySavingMode'], bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Energy Saving Mode Activation Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Energy Saving Mode Intensity')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/04_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 04_distributions.png")

# =============================================================================
# 4. CORRELATION ANALYSIS
# =============================================================================
print("\n[4/8] Creating correlation heatmap...")

# Select numerical columns for correlation
corr_cols = ['load', 'Energy', 'TXpower', 'Bandwidth', 'Frequency',
             'EnergySavingMode', 'hour_of_day', 'is_peak_hour', 'is_night_time']
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 05_correlation_heatmap.png")

# =============================================================================
# 5. HARDWARE ANALYSIS
# =============================================================================
print("\n[5/8] Creating hardware analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Energy by RU Type
rutype_energy = df.groupby('RUType')['Energy'].agg(['mean', 'std', 'count'])
rutype_energy = rutype_energy.sort_values('mean', ascending=False)
axes[0, 0].bar(range(len(rutype_energy)), rutype_energy['mean'],
               yerr=rutype_energy['std'], capsize=3, color='#3498db', alpha=0.7)
axes[0, 0].set_title('Average Energy by RU Type', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('RU Type')
axes[0, 0].set_ylabel('Energy (normalized)')
axes[0, 0].set_xticks(range(len(rutype_energy)))
axes[0, 0].set_xticklabels(rutype_energy.index, rotation=45, ha='right')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Load by RU Type
rutype_load = df.groupby('RUType')['load'].mean().sort_values(ascending=False)
axes[0, 1].bar(range(len(rutype_load)), rutype_load.values, color='#e74c3c', alpha=0.7)
axes[0, 1].set_title('Average Load by RU Type', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('RU Type')
axes[0, 1].set_ylabel('Load (ratio)')
axes[0, 1].set_xticks(range(len(rutype_load)))
axes[0, 1].set_xticklabels(rutype_load.index, rotation=45, ha='right')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Energy by Mode
sns.boxplot(data=df, x='Mode', y='Energy', ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Energy Distribution by Mode', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Energy (normalized)')

# Bandwidth vs Energy
bandwidth_energy = df.groupby('Bandwidth')['Energy'].mean()
axes[1, 1].bar(bandwidth_energy.index, bandwidth_energy.values, color='#1abc9c', alpha=0.7)
axes[1, 1].set_title('Average Energy by Bandwidth', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Bandwidth')
axes[1, 1].set_ylabel('Energy (normalized)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/06_hardware_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 06_hardware_analysis.png")

# =============================================================================
# 6. ENERGY SAVING MODE ANALYSIS
# =============================================================================
print("\n[6/8] Creating energy saving mode analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Energy consumption vs ES Mode
es_active = df[df['EnergySavingMode'] > 0]
es_inactive = df[df['EnergySavingMode'] == 0]

axes[0].hist([es_inactive['Energy'], es_active['Energy']], bins=30,
             label=['ES Inactive', 'ES Active'], color=['#e74c3c', '#2ecc71'], alpha=0.6)
axes[0].set_title('Energy Distribution: ES Mode Active vs Inactive', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Energy (normalized)')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Scatter: Load vs Energy, colored by ES Mode
scatter = axes[1].scatter(df['load'], df['Energy'], c=df['EnergySavingMode'],
                         cmap='viridis', alpha=0.5, s=10)
axes[1].set_title('Load vs Energy (colored by ES Mode intensity)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Load (ratio)')
axes[1].set_ylabel('Energy (normalized)')
plt.colorbar(scatter, ax=axes[1], label='ES Mode Intensity')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/07_energy_saving_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 07_energy_saving_analysis.png")

# =============================================================================
# 7. PEAK VS OFF-PEAK COMPARISON
# =============================================================================
print("\n[7/8] Creating peak vs off-peak comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Energy: Peak vs Off-peak
peak_data = df[df['is_peak_hour'] == 1]['Energy']
offpeak_data = df[df['is_peak_hour'] == 0]['Energy']

axes[0].boxplot([offpeak_data, peak_data], labels=['Off-Peak', 'Peak'],
                patch_artist=True,
                boxprops=dict(facecolor='#3498db', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[0].set_title('Energy Consumption: Peak vs Off-Peak Hours', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Energy (normalized)')
axes[0].grid(True, alpha=0.3, axis='y')

# Load: Peak vs Off-peak
peak_load = df[df['is_peak_hour'] == 1]['load']
offpeak_load = df[df['is_peak_hour'] == 0]['load']

axes[1].boxplot([offpeak_load, peak_load], labels=['Off-Peak', 'Peak'],
                patch_artist=True,
                boxprops=dict(facecolor='#e74c3c', alpha=0.7),
                medianprops=dict(color='blue', linewidth=2))
axes[1].set_title('Cell Load: Peak vs Off-Peak Hours', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Load (ratio)')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/08_peak_vs_offpeak.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 08_peak_vs_offpeak.png")

# =============================================================================
# 8. LAGGED FEATURE VISUALIZATION
# =============================================================================
print("\n[8/8] Creating lagged feature analysis...")

# Sample a few base stations for clarity
sample_bs = df['BS'].unique()[:3]
df_sample = df[df['BS'].isin(sample_bs)].copy()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Load with lags
for bs in sample_bs:
    bs_data = df_sample[df_sample['BS'] == bs].sort_values('Time').head(72)  # 3 days
    axes[0].plot(bs_data['Time'], bs_data['load'], label=f'{bs} - Current', linewidth=2)
    axes[0].plot(bs_data['Time'], bs_data['load_lag1'], label=f'{bs} - Lag 1h',
                linestyle='--', alpha=0.7)

axes[0].set_title('Load: Current vs Lagged Values (Sample Base Stations)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Load (ratio)')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0].grid(True, alpha=0.3)

# Energy with lags
for bs in sample_bs:
    bs_data = df_sample[df_sample['BS'] == bs].sort_values('Time').head(72)
    axes[1].plot(bs_data['Time'], bs_data['Energy'], label=f'{bs} - Current', linewidth=2)
    axes[1].plot(bs_data['Time'], bs_data['energy_lag1'], label=f'{bs} - Lag 1h',
                linestyle='--', alpha=0.7)

axes[1].set_title('Energy: Current vs Lagged Values (Sample Base Stations)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Energy (normalized)')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/09_lagged_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 09_lagged_features.png")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "="*80)
print("EDA VISUALIZATION COMPLETE")
print("="*80)
print(f"\n✓ Generated 9 visualization files in '{PLOTS_DIR}/' directory")
print("\nVisualization Summary:")
print("  1. Energy consumption time series (hourly & daily)")
print("  2. Cell load time series")
print("  3. Hourly patterns with peak hour identification")
print("  4. Distribution plots (energy, load, ES mode)")
print("  5. Correlation heatmap")
print("  6. Hardware analysis (RU type, mode, bandwidth)")
print("  7. Energy saving mode analysis")
print("  8. Peak vs off-peak comparison")
print("  9. Lagged feature visualization")
print("\n" + "="*80)
