"""
Phase 1: Exploratory Data Analysis and Visualization - FIB LAB 5G Dataset
==========================================================================
This script generates comprehensive visualizations for the preprocessed 5G dataset.
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
PLOTS_DIR = 'plots_5g'
os.makedirs(PLOTS_DIR, exist_ok=True)

print("="*80)
print("FIB LAB 5G NETWORK - EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load processed data (sample for memory efficiency)
print("\nLoading processed data...")
print("Note: Loading a sample of data for visualization (every 10th row)...")
df = pd.read_csv('processed_data/fiblab_5g_all_processed.csv', skiprows=lambda i: i % 10 != 0 and i != 0)
df['DateTime'] = pd.to_datetime(df['DateTime'])
print(f"Loaded {len(df)} sampled records")
print(f"Original dataset: ~2M records")

# =============================================================================
# 1. TIME SERIES PLOTS
# =============================================================================
print("\n[1/11] Creating time series plots...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Energy consumption over time
hourly_energy = df.groupby(df['DateTime'].dt.floor('H'))['Total_energy'].mean()
axes[0].plot(hourly_energy.index, hourly_energy.values, linewidth=1.5, color='#2ecc71')
axes[0].set_title('Average Total Energy Consumption Over Time (5G)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Total Energy (W)')
axes[0].grid(True, alpha=0.3)

# PRB usage over time
hourly_prb = df.groupby(df['DateTime'].dt.floor('H'))['PRB_usage_ratio'].mean()
axes[1].plot(hourly_prb.index, hourly_prb.values, linewidth=1.5, color='#e74c3c')
axes[1].set_title('Average PRB Usage Ratio Over Time (5G)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('PRB Usage (%)')
axes[1].grid(True, alpha=0.3)

# Number of users over time
hourly_users = df.groupby(df['DateTime'].dt.floor('H'))['Num_users'].mean()
axes[2].plot(hourly_users.index, hourly_users.values, linewidth=1.5, color='#3498db')
axes[2].set_title('Average Number of Users Over Time (5G)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Number of Users')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/01_time_series.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 01_time_series.png")

# =============================================================================
# 2. WEEKDAY VS WEEKEND COMPARISON
# =============================================================================
print("\n[2/11] Creating weekday vs weekend comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

weekday_data = df[df['day_type'] == 'Weekday']
weekend_data = df[df['day_type'] == 'Weekend']

# Energy
axes[0, 0].hist([weekday_data['Total_energy'], weekend_data['Total_energy']],
                bins=50, label=['Weekday', 'Weekend'], color=['#3498db', '#e74c3c'], alpha=0.6)
axes[0, 0].set_title('Total Energy: Weekday vs Weekend', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Total Energy (W)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# PRB usage
axes[0, 1].boxplot([weekday_data['PRB_usage_ratio'], weekend_data['PRB_usage_ratio']],
                    labels=['Weekday', 'Weekend'], patch_artist=True,
                    boxprops=dict(alpha=0.7))
axes[0, 1].set_title('PRB Usage: Weekday vs Weekend', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('PRB Usage (%)')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Traffic
axes[1, 0].hist([weekday_data['Traffic_volume'], weekend_data['Traffic_volume']],
                bins=50, label=['Weekday', 'Weekend'], color=['#2ecc71', '#f39c12'], alpha=0.6)
axes[1, 0].set_title('Traffic Volume: Weekday vs Weekend', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Traffic Volume (KByte)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].set_xlim(0, df['Traffic_volume'].quantile(0.95))

# Users
axes[1, 1].boxplot([weekday_data['Num_users'], weekend_data['Num_users']],
                    labels=['Weekday', 'Weekend'], patch_artist=True,
                    boxprops=dict(facecolor='#9b59b6', alpha=0.7))
axes[1, 1].set_title('Number of Users: Weekday vs Weekend', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Number of Users')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/02_weekday_vs_weekend.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 02_weekday_vs_weekend.png")

# =============================================================================
# 3. HOURLY PATTERNS
# =============================================================================
print("\n[3/11] Creating hourly pattern analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

hourly_stats = df.groupby('hour_of_day').agg({
    'Total_energy': 'mean',
    'PRB_usage_ratio': 'mean',
    'Traffic_volume': 'mean',
    'Num_users': 'mean'
})

# Energy by hour
axes[0, 0].plot(hourly_stats.index, hourly_stats['Total_energy'], marker='o',
                linewidth=2, markersize=6, color='#2ecc71')
axes[0, 0].set_title('Average Energy by Hour of Day', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Hour of Day')
axes[0, 0].set_ylabel('Total Energy (W)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvspan(8, 10, alpha=0.2, color='red', label='Morning Peak')
axes[0, 0].axvspan(17, 19, alpha=0.2, color='orange', label='Evening Peak')
axes[0, 0].legend()

# PRB usage by hour
axes[0, 1].plot(hourly_stats.index, hourly_stats['PRB_usage_ratio'], marker='s',
                linewidth=2, markersize=6, color='#e74c3c')
axes[0, 1].set_title('Average PRB Usage by Hour of Day', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('PRB Usage (%)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvspan(8, 10, alpha=0.2, color='red')
axes[0, 1].axvspan(17, 19, alpha=0.2, color='orange')

# Traffic by hour
axes[1, 0].plot(hourly_stats.index, hourly_stats['Traffic_volume'], marker='^',
                linewidth=2, markersize=6, color='#3498db')
axes[1, 0].set_title('Average Traffic Volume by Hour of Day', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Traffic Volume (KByte)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvspan(8, 10, alpha=0.2, color='red')
axes[1, 0].axvspan(17, 19, alpha=0.2, color='orange')

# Users by hour
axes[1, 1].plot(hourly_stats.index, hourly_stats['Num_users'], marker='D',
                linewidth=2, markersize=6, color='#9b59b6')
axes[1, 1].set_title('Average Number of Users by Hour of Day', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Hour of Day')
axes[1, 1].set_ylabel('Number of Users')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvspan(8, 10, alpha=0.2, color='red')
axes[1, 1].axvspan(17, 19, alpha=0.2, color='orange')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/03_hourly_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 03_hourly_patterns.png")

# =============================================================================
# 4. DISTRIBUTION PLOTS
# =============================================================================
print("\n[4/11] Creating distribution plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# BBU Energy
axes[0, 0].hist(df['BBU_energy'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('BBU Energy Distribution', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('BBU Energy (W)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['BBU_energy'].mean(), color='red', linestyle='--', linewidth=2)

# RRU Energy
axes[0, 1].hist(df['RRU_energy'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('RRU Energy Distribution', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('RRU Energy (W)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(df['RRU_energy'].mean(), color='red', linestyle='--', linewidth=2)

# Total Energy
axes[0, 2].hist(df['Total_energy'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[0, 2].set_title('Total Energy Distribution', fontsize=11, fontweight='bold')
axes[0, 2].set_xlabel('Total Energy (W)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].axvline(df['Total_energy'].mean(), color='red', linestyle='--', linewidth=2)

# PRB Usage
axes[1, 0].hist(df['PRB_usage_ratio'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('PRB Usage Distribution', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('PRB Usage (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(df['PRB_usage_ratio'].mean(), color='red', linestyle='--', linewidth=2)

# Traffic Volume (log scale)
axes[1, 1].hist(np.log10(df['Traffic_volume'] + 1), bins=50, color='#f39c12', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Traffic Volume Distribution (log scale)', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('log10(Traffic Volume)')
axes[1, 1].set_ylabel('Frequency')

# Number of Users
axes[1, 2].hist(df['Num_users'], bins=50, color='#1abc9c', alpha=0.7, edgecolor='black')
axes[1, 2].set_title('Number of Users Distribution', fontsize=11, fontweight='bold')
axes[1, 2].set_xlabel('Number of Users')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].axvline(df['Num_users'].mean(), color='red', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/04_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 04_distributions.png")

# =============================================================================
# 5. CORRELATION ANALYSIS
# =============================================================================
print("\n[5/11] Creating correlation heatmap...")

corr_cols = ['PRB_usage_ratio', 'Traffic_volume', 'Num_users',
             'BBU_energy', 'RRU_energy', 'Total_energy',
             'hour_of_day', 'is_peak_hour', 'is_weekend',
             'energy_saving_intensity']
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Heatmap (5G Network)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 05_correlation_heatmap.png")

# =============================================================================
# 6. ENERGY COMPONENT ANALYSIS
# =============================================================================
print("\n[6/11] Creating energy component analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BBU vs RRU Energy scatter
sample = df.sample(min(10000, len(df)))
axes[0].scatter(sample['BBU_energy'], sample['RRU_energy'], alpha=0.3, s=10, color='#3498db')
axes[0].set_title('BBU Energy vs RRU Energy (5G)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('BBU Energy (W)')
axes[0].set_ylabel('RRU Energy (W)')
axes[0].grid(True, alpha=0.3)

# Energy component breakdown
energy_components = df[['BBU_energy', 'RRU_energy']].mean()
axes[1].pie(energy_components, labels=['BBU', 'RRU'], autopct='%1.1f%%',
            colors=['#3498db', '#2ecc71'], startangle=90)
axes[1].set_title('Average Energy Component Breakdown (5G)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/06_energy_components.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 06_energy_components.png")

# =============================================================================
# 7. ENERGY SAVING MODE ANALYSIS (5G SPECIFIC)
# =============================================================================
print("\n[7/11] Creating energy saving mode analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Channel shutdown time distribution
axes[0, 0].hist(df['Channel_shutdown_time'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Channel Shutdown Time Distribution', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Channel Shutdown Time (ms)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_xlim(0, df['Channel_shutdown_time'].quantile(0.95))

# Carrier shutdown time distribution
axes[0, 1].hist(df['Carrier_shutdown_time'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Carrier Shutdown Time Distribution', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Carrier Shutdown Time (ms)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_xlim(0, df['Carrier_shutdown_time'].quantile(0.95))

# Deep sleep time distribution
axes[1, 0].hist(df['Deep_sleep_time'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Deep Sleep Time Distribution', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Deep Sleep Time (ms)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_xlim(0, df['Deep_sleep_time'].quantile(0.95))

# Energy saving intensity
axes[1, 1].hist(df['energy_saving_intensity'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Energy Saving Intensity Distribution', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Energy Saving Intensity (hours)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/07_energy_saving_modes.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 07_energy_saving_modes.png")

# =============================================================================
# 8. ENERGY SAVING EFFECTIVENESS
# =============================================================================
print("\n[8/11] Creating energy saving effectiveness analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Energy vs ES intensity
es_active = df[df['energy_saving_intensity'] > 0]
es_inactive = df[df['energy_saving_intensity'] == 0]

axes[0].hist([es_inactive['Total_energy'], es_active['Total_energy']],
             bins=40, label=['ES Inactive', 'ES Active'],
             color=['#e74c3c', '#2ecc71'], alpha=0.6)
axes[0].set_title('Energy Distribution: ES Active vs Inactive', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Total Energy (W)')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Scatter: PRB usage vs Energy, colored by ES intensity
sample = df.sample(min(10000, len(df)))
scatter = axes[1].scatter(sample['PRB_usage_ratio'], sample['Total_energy'],
                         c=sample['energy_saving_intensity'], cmap='viridis',
                         alpha=0.5, s=20)
axes[1].set_title('PRB Usage vs Energy (colored by ES intensity)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('PRB Usage (%)')
axes[1].set_ylabel('Total Energy (W)')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1], label='ES Intensity (hours)')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/08_energy_saving_effectiveness.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 08_energy_saving_effectiveness.png")

# =============================================================================
# 9. EFFICIENCY METRICS
# =============================================================================
print("\n[9/11] Creating efficiency metrics analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Traffic per PRB
axes[0, 0].hist(df['traffic_per_prb'][df['traffic_per_prb'] < df['traffic_per_prb'].quantile(0.95)],
                bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Traffic per PRB Distribution', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Traffic per PRB (KByte/%)')
axes[0, 0].set_ylabel('Frequency')

# Traffic per User
axes[0, 1].hist(df['traffic_per_user'][df['traffic_per_user'] < df['traffic_per_user'].quantile(0.95)],
                bins=50, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Traffic per User Distribution', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Traffic per User (KByte)')
axes[0, 1].set_ylabel('Frequency')

# Energy per User
axes[1, 0].hist(df['energy_per_user'][df['energy_per_user'] < df['energy_per_user'].quantile(0.95)],
                bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Energy per User Distribution', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Energy per User (W)')
axes[1, 0].set_ylabel('Frequency')

# Energy Efficiency
axes[1, 1].hist(df['energy_efficiency'][df['energy_efficiency'] < df['energy_efficiency'].quantile(0.95)],
                bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Energy Efficiency Distribution', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Energy Efficiency (KByte/W)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/09_efficiency_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 09_efficiency_metrics.png")

# =============================================================================
# 10. PEAK VS OFF-PEAK COMPARISON
# =============================================================================
print("\n[10/11] Creating peak vs off-peak comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

peak_data = df[df['is_peak_hour'] == 1]
offpeak_data = df[df['is_peak_hour'] == 0]

# Energy
axes[0, 0].boxplot([offpeak_data['Total_energy'], peak_data['Total_energy']],
                    labels=['Off-Peak', 'Peak'], patch_artist=True,
                    boxprops=dict(facecolor='#3498db', alpha=0.7))
axes[0, 0].set_title('Total Energy: Peak vs Off-Peak', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Total Energy (W)')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# PRB Usage
axes[0, 1].boxplot([offpeak_data['PRB_usage_ratio'], peak_data['PRB_usage_ratio']],
                    labels=['Off-Peak', 'Peak'], patch_artist=True,
                    boxprops=dict(facecolor='#e74c3c', alpha=0.7))
axes[0, 1].set_title('PRB Usage: Peak vs Off-Peak', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('PRB Usage (%)')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Traffic
axes[1, 0].boxplot([offpeak_data['Traffic_volume'], peak_data['Traffic_volume']],
                    labels=['Off-Peak', 'Peak'], patch_artist=True,
                    boxprops=dict(facecolor='#2ecc71', alpha=0.7))
axes[1, 0].set_title('Traffic Volume: Peak vs Off-Peak', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Traffic Volume (KByte)')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# ES Intensity
axes[1, 1].boxplot([offpeak_data['energy_saving_intensity'], peak_data['energy_saving_intensity']],
                    labels=['Off-Peak', 'Peak'], patch_artist=True,
                    boxprops=dict(facecolor='#9b59b6', alpha=0.7))
axes[1, 1].set_title('ES Intensity: Peak vs Off-Peak', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('ES Intensity (hours)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/10_peak_vs_offpeak.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 10_peak_vs_offpeak.png")

# =============================================================================
# 11. SUMMARY STATISTICS
# =============================================================================
print("\n[11/11] Creating summary statistics visualization...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Overall statistics
ax1 = fig.add_subplot(gs[0, :])
summary_data = {
    'Metric': ['Avg Energy (W)', 'Avg PRB (%)', 'Avg Traffic (MB)',
               'Avg Users', 'Avg ES Intensity (ms)'],
    'Value': [df['Total_energy'].mean(), df['PRB_usage_ratio'].mean(),
              df['Traffic_volume'].mean()/1024, df['Num_users'].mean(),
              (df['Channel_shutdown_time'] + df['Carrier_shutdown_time'] + df['Deep_sleep_time']).mean()]
}
bars = ax1.bar(summary_data['Metric'], summary_data['Value'], color='#3498db', alpha=0.7)
ax1.set_title('Key Performance Indicators - 5G Network', fontsize=14, fontweight='bold')
ax1.set_ylabel('Value')
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, summary_data['Value']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

# Weekday vs Weekend summary
ax2 = fig.add_subplot(gs[1, 0])
weekday_energy = weekday_data['Total_energy'].mean()
weekend_energy = weekend_data['Total_energy'].mean()
ax2.bar(['Weekday', 'Weekend'], [weekday_energy, weekend_energy],
        color=['#2ecc71', '#e74c3c'], alpha=0.7)
ax2.set_title('Average Energy: Weekday vs Weekend', fontsize=12, fontweight='bold')
ax2.set_ylabel('Total Energy (W)')
ax2.grid(True, alpha=0.3, axis='y')

# Peak vs Off-Peak summary
ax3 = fig.add_subplot(gs[1, 1])
peak_energy = peak_data['Total_energy'].mean()
offpeak_energy = offpeak_data['Total_energy'].mean()
ax3.bar(['Off-Peak', 'Peak'], [offpeak_energy, peak_energy],
        color=['#3498db', '#f39c12'], alpha=0.7)
ax3.set_title('Average Energy: Peak vs Off-Peak', fontsize=12, fontweight='bold')
ax3.set_ylabel('Total Energy (W)')
ax3.grid(True, alpha=0.3, axis='y')

# Data coverage
ax4 = fig.add_subplot(gs[2, :])
coverage_text = f"""
Dataset Coverage Summary:
- Total Records: {len(df):,} (sampled from ~2M)
- Unique Base Stations: {df['BaseStationID'].nunique():,}
- Unique Cells: {df['CellID'].nunique():,}
- Weekday Records: {len(weekday_data):,}
- Weekend Records: {len(weekend_data):,}
- Date Range: {df['DateTime'].min()} to {df['DateTime'].max()}
- ES Active Records: {len(es_active):,} ({100*len(es_active)/len(df):.1f}%)
"""
ax4.text(0.1, 0.5, coverage_text, fontsize=11, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax4.axis('off')

plt.savefig(f'{PLOTS_DIR}/11_summary_statistics.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 11_summary_statistics.png")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "="*80)
print("5G EDA VISUALIZATION COMPLETE")
print("="*80)
print(f"\n✓ Generated 11 visualization files in '{PLOTS_DIR}/' directory")
print("\nVisualization Summary:")
print("  1. Time series (energy, PRB, users)")
print("  2. Weekday vs weekend comparison")
print("  3. Hourly patterns")
print("  4. Distribution plots")
print("  5. Correlation heatmap")
print("  6. Energy component analysis (BBU vs RRU)")
print("  7. Energy saving modes (channel/carrier/deep sleep)")
print("  8. Energy saving effectiveness")
print("  9. Efficiency metrics")
print(" 10. Peak vs off-peak comparison")
print(" 11. Summary statistics")
print("\n" + "="*80)
