"""
Phase 1: Data Preprocessing for Netop 5G Network Energy Consumption Dataset
============================================================================
This script performs data acquisition, inspection, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("="*80)
print("NETOP 5G NETWORK ENERGY CONSUMPTION - DATA PREPROCESSING")
print("="*80)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n[1/6] Loading data files...")

try:
    # Load the three CSV files from raw_data directory
    bsinfo = pd.read_csv('raw_data/BSinfo.csv')
    clstat = pd.read_csv('raw_data/CLstat.csv')
    ecstat = pd.read_csv('raw_data/ECstat.csv')

    print(f"✓ BSinfo: {bsinfo.shape[0]} rows, {bsinfo.shape[1]} columns")
    print(f"✓ CLstat: {clstat.shape[0]} rows, {clstat.shape[1]} columns")
    print(f"✓ ECstat: {ecstat.shape[0]} rows, {ecstat.shape[1]} columns")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# =============================================================================
# 2. INITIAL DATA INSPECTION
# =============================================================================
print("\n[2/6] Initial data inspection...")

print("\n--- BSinfo (Base Station Information) ---")
print(bsinfo.head())
print("\nData types:")
print(bsinfo.dtypes)
print(f"\nMissing values:\n{bsinfo.isnull().sum()}")
print(f"\nBasic statistics:\n{bsinfo.describe()}")
print(f"\nUnique values:")
print(f"  - Base Stations: {bsinfo['BS'].nunique()}")
print(f"  - RU Types: {bsinfo['RUType'].nunique()}")
print(f"  - Modes: {bsinfo['Mode'].nunique()}")

print("\n--- CLstat (Cell-Level Statistics) ---")
print(clstat.head())
print("\nData types:")
print(clstat.dtypes)
print(f"\nMissing values:\n{clstat.isnull().sum()}")
print(f"\nLoad statistics:\n{clstat['load'].describe()}")
print(f"\nDate range: {clstat['Time'].min()} to {clstat['Time'].max()}")

print("\n--- ECstat (Energy Consumption) ---")
print(ecstat.head())
print("\nData types:")
print(ecstat.dtypes)
print(f"\nMissing values:\n{ecstat.isnull().sum()}")
print(f"\nEnergy statistics:\n{ecstat['Energy'].describe()}")
print(f"\nDate range: {ecstat['Time'].min()} to {ecstat['Time'].max()}")

# =============================================================================
# 3. DATA CLEANING
# =============================================================================
print("\n[3/6] Data cleaning...")

# Convert time columns to datetime
print("  - Converting time columns to datetime...")
clstat['Time'] = pd.to_datetime(clstat['Time'])
ecstat['Time'] = pd.to_datetime(ecstat['Time'])

# Check for negative or invalid values
print("  - Checking for invalid values...")
if (bsinfo['TXpower'] < 0).any():
    print(f"    Warning: Found {(bsinfo['TXpower'] < 0).sum()} negative TXpower values")
if (clstat['load'] < 0).any() or (clstat['load'] > 1).any():
    print(f"    Warning: Found {((clstat['load'] < 0) | (clstat['load'] > 1)).sum()} invalid load values")
if (ecstat['Energy'] < 0).any():
    print(f"    Warning: Found {(ecstat['Energy'] < 0).sum()} negative energy values")

# Handle missing values
print("  - Handling missing values...")
original_rows_cl = len(clstat)
original_rows_ec = len(ecstat)

# For numerical columns in clstat, use forward fill then backward fill
clstat = clstat.sort_values(['BS', 'CellName', 'Time'])
clstat = clstat.fillna(method='ffill').fillna(method='bfill')

# For ecstat, use forward fill then backward fill
ecstat = ecstat.sort_values(['BS', 'Time'])
ecstat = ecstat.fillna(method='ffill').fillna(method='bfill')

# Remove any remaining rows with missing values
clstat = clstat.dropna()
ecstat = ecstat.dropna()

print(f"    CLstat: {original_rows_cl - len(clstat)} rows removed")
print(f"    ECstat: {original_rows_ec - len(ecstat)} rows removed")

# Aggregate energy saving modes into a single metric
print("  - Aggregating energy saving modes...")
esmode_cols = [col for col in clstat.columns if col.startswith('ESMode')]
if esmode_cols:
    # Sum all energy saving mode intensities
    clstat['EnergySavingMode'] = clstat[esmode_cols].sum(axis=1)
    print(f"    Combined {len(esmode_cols)} energy saving mode columns")

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
print("\n[4/6] Feature engineering...")

# Time-based features for CLstat
print("  - Creating time-based features...")
clstat['hour_of_day'] = clstat['Time'].dt.hour
clstat['day_of_week'] = clstat['Time'].dt.dayofweek
clstat['day_of_month'] = clstat['Time'].dt.day
clstat['is_weekend'] = (clstat['day_of_week'] >= 5).astype(int)
clstat['is_peak_hour'] = clstat['hour_of_day'].apply(
    lambda x: 1 if (8 <= x <= 10) or (17 <= x <= 19) else 0
)
clstat['is_night_time'] = clstat['hour_of_day'].apply(
    lambda x: 1 if (22 <= x <= 23) or (0 <= x <= 6) else 0
)

# Time-based features for ECstat
ecstat['hour_of_day'] = ecstat['Time'].dt.hour
ecstat['day_of_week'] = ecstat['Time'].dt.dayofweek
ecstat['day_of_month'] = ecstat['Time'].dt.day
ecstat['is_weekend'] = (ecstat['day_of_week'] >= 5).astype(int)
ecstat['is_peak_hour'] = ecstat['hour_of_day'].apply(
    lambda x: 1 if (8 <= x <= 10) or (17 <= x <= 19) else 0
)
ecstat['is_night_time'] = ecstat['hour_of_day'].apply(
    lambda x: 1 if (22 <= x <= 23) or (0 <= x <= 6) else 0
)

# Create lagged features for load (t-1, t-24 hours)
print("  - Creating lagged features for load...")
clstat = clstat.sort_values(['BS', 'CellName', 'Time'])
clstat['load_lag1'] = clstat.groupby(['BS', 'CellName'])['load'].shift(1)
clstat['load_lag24'] = clstat.groupby(['BS', 'CellName'])['load'].shift(24)

# Create lagged features for energy (t-1, t-24 hours)
print("  - Creating lagged features for energy...")
ecstat = ecstat.sort_values(['BS', 'Time'])
ecstat['energy_lag1'] = ecstat.groupby('BS')['Energy'].shift(1)
ecstat['energy_lag24'] = ecstat.groupby('BS')['Energy'].shift(24)

# Rolling statistics
print("  - Creating rolling statistics...")
clstat['load_rolling_mean_3h'] = clstat.groupby(['BS', 'CellName'])['load'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
clstat['load_rolling_std_3h'] = clstat.groupby(['BS', 'CellName'])['load'].transform(
    lambda x: x.rolling(window=3, min_periods=1).std()
)

ecstat['energy_rolling_mean_3h'] = ecstat.groupby('BS')['Energy'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
ecstat['energy_rolling_std_3h'] = ecstat.groupby('BS')['Energy'].transform(
    lambda x: x.rolling(window=3, min_periods=1).std()
)

print(f"  ✓ Created {len([c for c in clstat.columns if c not in clstat.columns[:10]])} new features for CLstat")
print(f"  ✓ Created {len([c for c in ecstat.columns if c not in ecstat.columns[:3]])} new features for ECstat")

# =============================================================================
# 5. DATA MERGING
# =============================================================================
print("\n[5/6] Merging datasets...")

# First, merge CLstat with BSinfo
print("  - Merging CLstat with BSinfo...")
df_merged = pd.merge(clstat, bsinfo, on=['BS', 'CellName'], how='left')
print(f"    Result: {df_merged.shape[0]} rows, {df_merged.shape[1]} columns")

# Then, merge with ECstat (on Time and BS)
print("  - Merging with ECstat...")
df_final = pd.merge(df_merged, ecstat, on=['Time', 'BS'], how='left', suffixes=('', '_ec'))
print(f"    Result: {df_final.shape[0]} rows, {df_final.shape[1]} columns")

# Remove duplicate columns from merge
duplicate_cols = [col for col in df_final.columns if col.endswith('_ec') and col[:-3] in df_final.columns]
if duplicate_cols:
    for col in duplicate_cols:
        base_col = col[:-3]
        if base_col not in ['Energy']:  # Keep energy-related columns
            df_final[base_col] = df_final[base_col].fillna(df_final[col])
    df_final = df_final.drop(columns=duplicate_cols)
    print(f"    Resolved {len(duplicate_cols)} duplicate columns")

# Remove rows with missing values after merge
rows_before = len(df_final)
df_final = df_final.dropna()
print(f"    Removed {rows_before - len(df_final)} rows with missing values")

print(f"\n  ✓ Final merged dataset: {df_final.shape[0]} rows, {df_final.shape[1]} columns")

# =============================================================================
# 6. SAVE PROCESSED DATA
# =============================================================================
print("\n[6/6] Saving processed data...")

# Save the merged dataset to processed_data directory
df_final.to_csv('processed_data/netop_processed.csv', index=False)
print(f"  ✓ Saved: processed_data/netop_processed.csv ({df_final.shape[0]} rows)")

# Save individual processed files as well
clstat.to_csv('processed_data/clstat_processed.csv', index=False)
ecstat.to_csv('processed_data/ecstat_processed.csv', index=False)
print(f"  ✓ Saved: processed_data/clstat_processed.csv ({clstat.shape[0]} rows)")
print(f"  ✓ Saved: processed_data/ecstat_processed.csv ({ecstat.shape[0]} rows)")

# =============================================================================
# 7. SUMMARY REPORT
# =============================================================================
print("\n" + "="*80)
print("PREPROCESSING SUMMARY")
print("="*80)
print(f"\nFinal Dataset Shape: {df_final.shape}")
print(f"\nColumns: {list(df_final.columns)}")
print(f"\nData Quality:")
print(f"  - Missing values: {df_final.isnull().sum().sum()}")
print(f"  - Duplicate rows: {df_final.duplicated().sum()}")
print(f"  - Date range: {df_final['Time'].min()} to {df_final['Time'].max()}")
print(f"  - Number of base stations: {df_final['BS'].nunique()}")
print(f"  - Number of cells: {df_final['CellName'].nunique()}")
print(f"  - Total measurements: {len(df_final)}")

print("\nKey Statistics:")
print(df_final[['load', 'Energy', 'TXpower', 'Bandwidth']].describe())

print("\nEnergy Saving Mode Statistics:")
if 'EnergySavingMode' in df_final.columns:
    print(df_final['EnergySavingMode'].describe())

print("\n" + "="*80)
print("✓ Preprocessing complete!")
print("="*80)
