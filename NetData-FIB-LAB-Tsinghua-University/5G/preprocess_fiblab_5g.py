"""
Phase 1: Data Preprocessing for FIB LAB 5G Network Dataset
===========================================================
This script performs data acquisition, inspection, cleaning, and feature engineering for 5G data.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("FIB LAB 5G NETWORK - DATA PREPROCESSING")
print("="*80)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n[1/6] Loading 5G data files...")

datasets = {}
files = {
    'Weekday': 'raw_data/Performance_5G_Weekday.txt',
    'Weekend': 'raw_data/Performance_5G_Weekend.txt'
}

for key, filename in files.items():
    print(f"  Loading {filename}...")
    df = pd.read_csv(filename)
    datasets[key] = df
    print(f"    ✓ {key}: {df.shape[0]} rows, {df.shape[1]} columns")

# =============================================================================
# 2. INITIAL DATA INSPECTION
# =============================================================================
print("\n[2/6] Initial data inspection...")

for key, df in datasets.items():
    print(f"\n--- {key} Dataset ---")
    print(f"Shape: {df.shape}")
    print(df.head(3))
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print("\nBasic statistics:")
    print(df.describe())

# =============================================================================
# 3. DATA CLEANING
# =============================================================================
print("\n[3/6] Data cleaning...")

cleaned_datasets = {}

for key, df in datasets.items():
    print(f"\n  Processing {key}...")
    df_clean = df.copy()

    # Fix duplicate column names (Channel Shutdown Time appears twice in 5G data)
    print("    - Fixing duplicate column names...")
    cols = df_clean.columns.tolist()

    if len(cols) == 11:
        # Rename to standard format
        df_clean.columns = [
            'BaseStationID',
            'CellID',
            'Timestamp',
            'PRB_usage_ratio',
            'Traffic_volume',
            'Num_users',
            'BBU_energy',
            'RRU_energy',
            'Channel_shutdown_time',
            'Carrier_shutdown_time',
            'Deep_sleep_time'
        ]
        print("      Renamed columns to standard format")

    # Add metadata
    df_clean['network_type'] = '5G'
    df_clean['day_type'] = key

    # Convert timestamp to proper datetime
    print("    - Converting timestamp to datetime...")
    base_date = pd.Timestamp('2024-01-08')  # Monday for Weekday
    if key == 'Weekend':
        base_date = pd.Timestamp('2024-01-13')  # Saturday for Weekend

    df_clean['Time'] = pd.to_datetime(df_clean['Timestamp'], format='%H:%M', errors='coerce').dt.time
    df_clean['DateTime'] = df_clean['Time'].apply(
        lambda t: pd.Timestamp.combine(base_date.date(), t) if pd.notna(t) else pd.NaT
    )

    # Check for invalid values
    print("    - Checking for invalid values...")
    invalid_checks = {
        'Negative PRB': (df_clean['PRB_usage_ratio'] < 0).sum(),
        'PRB > 100': (df_clean['PRB_usage_ratio'] > 100).sum(),
        'Negative Traffic': (df_clean['Traffic_volume'] < 0).sum(),
        'Negative Users': (df_clean['Num_users'] < 0).sum(),
        'Negative BBU': (df_clean['BBU_energy'] < 0).sum(),
        'Negative RRU': (df_clean['RRU_energy'] < 0).sum(),
    }
    for check, count in invalid_checks.items():
        if count > 0:
            print(f"      Warning: {count} {check} values")

    # Handle missing values
    print("    - Handling missing values...")
    original_rows = len(df_clean)
    df_clean = df_clean.sort_values(['BaseStationID', 'CellID', 'DateTime'])

    numeric_cols = ['PRB_usage_ratio', 'Traffic_volume', 'Num_users',
                   'BBU_energy', 'RRU_energy', 'Channel_shutdown_time',
                   'Carrier_shutdown_time', 'Deep_sleep_time']

    for col in numeric_cols:
        df_clean[col] = df_clean.groupby(['BaseStationID', 'CellID'])[col].fillna(method='ffill').fillna(method='bfill')

    df_clean = df_clean.dropna(subset=['DateTime', 'PRB_usage_ratio', 'BBU_energy', 'RRU_energy'])
    print(f"      {original_rows - len(df_clean)} rows removed")

    # Calculate total energy
    print("    - Calculating total energy...")
    df_clean['Total_energy'] = df_clean['BBU_energy'] + df_clean['RRU_energy']

    cleaned_datasets[key] = df_clean
    print(f"    ✓ Cleaned {key}: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
print("\n[4/6] Feature engineering...")

for key, df in cleaned_datasets.items():
    print(f"\n  Processing {key}...")

    # Time-based features
    print("    - Creating time-based features...")
    df['hour_of_day'] = df['DateTime'].dt.hour
    df['minute_of_hour'] = df['DateTime'].dt.minute
    df['time_of_day_minutes'] = df['hour_of_day'] * 60 + df['minute_of_hour']
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = df['hour_of_day'].apply(lambda x: 1 if (8 <= x <= 10) or (17 <= x <= 19) else 0)
    df['is_night_time'] = df['hour_of_day'].apply(lambda x: 1 if (22 <= x <= 23) or (0 <= x <= 6) else 0)

    # Efficiency metrics
    print("    - Creating efficiency metrics...")
    df['traffic_per_prb'] = df['Traffic_volume'] / (df['PRB_usage_ratio'] + 0.01)
    df['traffic_per_user'] = df['Traffic_volume'] / (df['Num_users'] + 0.01)
    df['energy_per_user'] = df['Total_energy'] / (df['Num_users'] + 0.01)
    df['energy_efficiency'] = df['Traffic_volume'] / (df['Total_energy'] + 0.01)

    # Energy saving intensity (total shutdown/sleep time)
    print("    - Creating energy saving intensity metric...")
    df['energy_saving_intensity'] = (
        df['Channel_shutdown_time'] +
        df['Carrier_shutdown_time'] +
        df['Deep_sleep_time']
    ) / 3600000  # Convert milliseconds to hours

    # Lagged features
    print("    - Creating lagged features...")
    df = df.sort_values(['BaseStationID', 'CellID', 'DateTime'])
    for col in ['PRB_usage_ratio', 'Traffic_volume', 'Num_users', 'Total_energy']:
        df[f'{col}_lag1'] = df.groupby(['BaseStationID', 'CellID'])[col].shift(1)
        df[f'{col}_lag2'] = df.groupby(['BaseStationID', 'CellID'])[col].shift(2)

    # Rolling statistics (3-period window ~ 1.5 hours for 30-min intervals)
    print("    - Creating rolling statistics...")
    for col in ['PRB_usage_ratio', 'Traffic_volume', 'Num_users', 'Total_energy']:
        df[f'{col}_rolling_mean'] = df.groupby(['BaseStationID', 'CellID'])[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df[f'{col}_rolling_std'] = df.groupby(['BaseStationID', 'CellID'])[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )

    cleaned_datasets[key] = df
    print(f"    ✓ Created features for {key}: now {df.shape[1]} columns")

# =============================================================================
# 5. COMBINE AND SAVE
# =============================================================================
print("\n[5/6] Combining and organizing datasets...")

# Save individual processed files
for key, df in cleaned_datasets.items():
    filename = f"processed_data/5g_{key.lower()}_processed.csv"
    df.to_csv(filename, index=False)
    print(f"  ✓ Saved: {filename} ({df.shape[0]} rows)")

# Combine both weekday and weekend
print("\n  Combining all 5G datasets...")
df_5g_all = pd.concat(cleaned_datasets.values(), ignore_index=True)
print(f"  ✓ Combined 5G dataset: {df_5g_all.shape[0]} rows, {df_5g_all.shape[1]} columns")

# =============================================================================
# 6. SAVE FINAL PROCESSED DATA
# =============================================================================
print("\n[6/6] Saving final processed data...")

df_5g_all.to_csv('processed_data/fiblab_5g_all_processed.csv', index=False)
print(f"  ✓ Saved: processed_data/fiblab_5g_all_processed.csv ({df_5g_all.shape[0]} rows)")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "="*80)
print("5G PREPROCESSING SUMMARY")
print("="*80)

print("\nDataset Breakdown:")
print(f"  - Weekday: {cleaned_datasets['Weekday'].shape[0]} rows")
print(f"  - Weekend: {cleaned_datasets['Weekend'].shape[0]} rows")
print(f"  - Total: {df_5g_all.shape[0]} rows")

print(f"\nColumns: {df_5g_all.shape[1]}")
print(f"\nData Quality:")
print(f"  - Missing values: {df_5g_all.isnull().sum().sum()}")
print(f"  - Duplicate rows: {df_5g_all.duplicated().sum()}")
print(f"  - Unique base stations: {df_5g_all['BaseStationID'].nunique()}")
print(f"  - Unique cells: {df_5g_all['CellID'].nunique()}")

print("\nKey Statistics:")
print(df_5g_all[['PRB_usage_ratio', 'Traffic_volume', 'Num_users',
                 'BBU_energy', 'RRU_energy', 'Total_energy']].describe())

print("\nEnergy Efficiency Metrics:")
print(df_5g_all[['traffic_per_prb', 'traffic_per_user',
                 'energy_per_user', 'energy_efficiency']].describe())

print("\nEnergy Saving Statistics:")
print(df_5g_all[['Channel_shutdown_time', 'Carrier_shutdown_time',
                 'Deep_sleep_time', 'energy_saving_intensity']].describe())

print("\n" + "="*80)
print("✓ 5G Preprocessing complete!")
print("="*80)
