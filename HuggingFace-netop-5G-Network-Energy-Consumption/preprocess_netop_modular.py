"""
Netop 5G Network Energy Consumption - Modular Preprocessing Script
===================================================================
Run sections by uncommenting the function calls at the bottom.
Each section can be run independently for inspection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================
def load_data():
    """Load the three CSV files from raw_data directory."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    bsinfo = pd.read_csv('raw_data/BSinfo.csv')
    clstat = pd.read_csv('raw_data/CLstat.csv')
    ecstat = pd.read_csv('raw_data/ECstat.csv')

    print(f"✓ BSinfo: {bsinfo.shape[0]:,} rows × {bsinfo.shape[1]} columns")
    print(f"✓ CLstat: {clstat.shape[0]:,} rows × {clstat.shape[1]} columns")
    print(f"✓ ECstat: {ecstat.shape[0]:,} rows × {ecstat.shape[1]} columns")

    return bsinfo, clstat, ecstat


# =============================================================================
# SECTION 2: DATA INSPECTION
# =============================================================================
def inspect_bsinfo(bsinfo):
    """Inspect BSinfo dataset."""
    print("\n" + "="*80)
    print("BSinfo - Base Station Information")
    print("="*80)
    print("\nFirst 5 rows:")
    print(bsinfo.head())
    print("\nData types:")
    print(bsinfo.dtypes)
    print(f"\nMissing values:\n{bsinfo.isnull().sum()}")
    print(f"\nBasic statistics:\n{bsinfo.describe()}")
    print(f"\nUnique values:")
    print(f"  - Base Stations: {bsinfo['BS'].nunique()}")
    print(f"  - RU Types: {bsinfo['RUType'].nunique()}")
    print(f"  - Modes: {bsinfo['Mode'].nunique()}")
    print(f"\n  RU Type distribution:")
    print(bsinfo['RUType'].value_counts())


def inspect_clstat(clstat):
    """Inspect CLstat dataset."""
    print("\n" + "="*80)
    print("CLstat - Cell-Level Statistics")
    print("="*80)
    print("\nFirst 5 rows:")
    print(clstat.head())
    print("\nData types:")
    print(clstat.dtypes)
    missing = clstat.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values:\n{missing[missing > 0]}")
    print(f"\nLoad statistics:\n{clstat['load'].describe()}")
    print(f"\nDate range: {clstat['Time'].min()} to {clstat['Time'].max()}")
    print(f"\nUnique base stations: {clstat['BS'].nunique()}")
    print(f"Unique cells: {clstat['CellName'].nunique()}")


def inspect_ecstat(ecstat):
    """Inspect ECstat dataset."""
    print("\n" + "="*80)
    print("ECstat - Energy Consumption")
    print("="*80)
    print("\nFirst 5 rows:")
    print(ecstat.head())
    print("\nData types:")
    print(ecstat.dtypes)
    print(f"\nMissing values:\n{ecstat.isnull().sum()}")
    print(f"\nEnergy statistics:\n{ecstat['Energy'].describe()}")
    print(f"\nDate range: {ecstat['Time'].min()} to {ecstat['Time'].max()}")
    print(f"\nUnique base stations: {ecstat['BS'].nunique()}")
    print(f"Unique (Time, BS) combinations: {ecstat[['Time', 'BS']].drop_duplicates().shape[0]:,}")


# =============================================================================
# SECTION 3: DATA CLEANING
# =============================================================================
def clean_data(bsinfo, clstat, ecstat):
    """Clean datasets: convert timestamps, handle missing values."""
    print("\n" + "="*80)
    print("DATA CLEANING")
    print("="*80)

    # Convert time columns
    print("\n⏰ Converting time columns to datetime...")
    clstat['Time'] = pd.to_datetime(clstat['Time'])
    ecstat['Time'] = pd.to_datetime(ecstat['Time'])
    print("  ✓ Time columns converted")

    # Check for invalid values
    print("\n🔍 Checking for invalid values...")
    issues = []
    if (bsinfo['TXpower'] < 0).any():
        issues.append(f"  ⚠️  {(bsinfo['TXpower'] < 0).sum()} negative TXpower values")
    if (clstat['load'] < 0).any() or (clstat['load'] > 1).any():
        issues.append(f"  ⚠️  {((clstat['load'] < 0) | (clstat['load'] > 1)).sum()} invalid load values")
    if (ecstat['Energy'] < 0).any():
        issues.append(f"  ⚠️  {(ecstat['Energy'] < 0).sum()} negative energy values")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✓ No invalid values found")

    # Sort data by time (needed for feature engineering later)
    print("\n📋 Sorting data by time...")
    clstat = clstat.sort_values(['BS', 'CellName', 'Time'])
    ecstat = ecstat.sort_values(['BS', 'Time'])
    print("  ✓ Data sorted")
    print(f"    CLstat: {len(clstat):,} rows")
    print(f"    ECstat: {len(ecstat):,} rows")

    # Keep energy saving mode columns as separate features
    print("\n⚡ Energy saving mode columns...")
    esmode_cols = [col for col in clstat.columns if col.startswith('ESMode')]
    if esmode_cols:
        print(f"  ✓ Keeping {len(esmode_cols)} ESMode columns as separate features: {esmode_cols}")
    else:
        print("  ℹ️  No energy saving mode columns found")

    print("\n✓ Data cleaning complete!")
    print(f"  CLstat: {len(clstat):,} rows × {clstat.shape[1]} columns")
    print(f"  ECstat: {len(ecstat):,} rows × {ecstat.shape[1]} columns")
    print("\n  Note: No missing value handling needed (raw data has no missing values)")

    return bsinfo, clstat, ecstat


# =============================================================================
# SECTION 4: FEATURE ENGINEERING - TIME-BASED
# =============================================================================
def add_time_features(clstat, ecstat):
    """Add time-based features: hour, day, weekend, peak hours."""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING - Time-Based Features")
    print("="*80)

    # CLstat time features
    print("\n⏰ Creating time features for CLstat...")
    clstat['hour_of_day'] = clstat['Time'].dt.hour
    clstat['day_of_week'] = clstat['Time'].dt.dayofweek
    clstat['is_weekend'] = (clstat['day_of_week'] >= 5).astype(int)
    clstat['is_peak_hour'] = clstat['hour_of_day'].apply(
        lambda x: 1 if (8 <= x <= 10) or (17 <= x <= 19) else 0
    )
    clstat['is_night_time'] = clstat['hour_of_day'].apply(
        lambda x: 1 if (22 <= x <= 23) or (0 <= x <= 6) else 0
    )
    print("  ✓ Added 5 time features")

    # ECstat time features
    print("\n⏰ Creating time features for ECstat...")
    ecstat['hour_of_day'] = ecstat['Time'].dt.hour
    ecstat['day_of_week'] = ecstat['Time'].dt.dayofweek
    ecstat['is_weekend'] = (ecstat['day_of_week'] >= 5).astype(int)
    ecstat['is_peak_hour'] = ecstat['hour_of_day'].apply(
        lambda x: 1 if (8 <= x <= 10) or (17 <= x <= 19) else 0
    )
    ecstat['is_night_time'] = ecstat['hour_of_day'].apply(
        lambda x: 1 if (22 <= x <= 23) or (0 <= x <= 6) else 0
    )
    print("  ✓ Added 5 time features")

    print(f"\n✓ Time features complete!")
    print(f"  CLstat: {clstat.shape[1]} columns")
    print(f"  ECstat: {ecstat.shape[1]} columns")
    print(f"\n  Note: day_of_month excluded (only 7 days of data - not meaningful)")

    return clstat, ecstat


# =============================================================================
# SECTION 5: FEATURE ENGINEERING - LAGGED FEATURES
# =============================================================================
def add_lagged_features(clstat, ecstat):
    """Add lagged features: t-1 and t-24 hours."""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING - Lagged Features")
    print("="*80)

    # CLstat lagged features
    print("\n📊 Creating lagged features for load...")
    clstat = clstat.sort_values(['BS', 'CellName', 'Time'])
    clstat['load_lag1'] = clstat.groupby(['BS', 'CellName'])['load'].shift(1)
    clstat['load_lag24'] = clstat.groupby(['BS', 'CellName'])['load'].shift(24)
    print(f"  ✓ Created load_lag1 and load_lag24")
    print(f"    NaN in load_lag1: {clstat['load_lag1'].isnull().sum():,}")
    print(f"    NaN in load_lag24: {clstat['load_lag24'].isnull().sum():,}")

    # ECstat lagged features
    print("\n⚡ Creating lagged features for energy...")
    ecstat = ecstat.sort_values(['BS', 'Time'])
    ecstat['energy_lag1'] = ecstat.groupby('BS')['Energy'].shift(1)
    ecstat['energy_lag24'] = ecstat.groupby('BS')['Energy'].shift(24)
    print(f"  ✓ Created energy_lag1 and energy_lag24")
    print(f"    NaN in energy_lag1: {ecstat['energy_lag1'].isnull().sum():,}")
    print(f"    NaN in energy_lag24: {ecstat['energy_lag24'].isnull().sum():,}")

    print(f"\n✓ Lagged features complete!")
    print(f"  CLstat: {clstat.shape[1]} columns")
    print(f"  ECstat: {ecstat.shape[1]} columns")

    return clstat, ecstat


# =============================================================================
# SECTION 6: FEATURE ENGINEERING - ROLLING STATISTICS
# =============================================================================
def add_rolling_features(clstat, ecstat):
    """Add rolling statistics: 3-hour windows."""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING - Rolling Statistics")
    print("="*80)

    # CLstat rolling features
    print("\n📊 Creating rolling statistics for load (3-hour window)...")
    clstat['load_rolling_mean_3h'] = clstat.groupby(['BS', 'CellName'])['load'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    clstat['load_rolling_std_3h'] = clstat.groupby(['BS', 'CellName'])['load'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std()
    )
    print("  ✓ Created load_rolling_mean_3h and load_rolling_std_3h")

    # ECstat rolling features
    print("\n⚡ Creating rolling statistics for energy (3-hour window)...")
    ecstat['energy_rolling_mean_3h'] = ecstat.groupby('BS')['Energy'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    ecstat['energy_rolling_std_3h'] = ecstat.groupby('BS')['Energy'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std()
    )
    print("  ✓ Created energy_rolling_mean_3h and energy_rolling_std_3h")

    print(f"\n✓ Rolling statistics complete!")
    print(f"  CLstat: {clstat.shape[1]} columns")
    print(f"  ECstat: {ecstat.shape[1]} columns")

    return clstat, ecstat


# =============================================================================
# SECTION 7: DATA MERGING
# =============================================================================
def merge_datasets(clstat, bsinfo, ecstat):
    """Merge CLstat + BSinfo + ECstat."""
    print("\n" + "="*80)
    print("DATA MERGING")
    print("="*80)

    # Step 1: CLstat + BSinfo
    print("\n🔗 Step 1: Merging CLstat + BSinfo...")
    print(f"  CLstat: {clstat.shape}")
    print(f"  BSinfo: {bsinfo.shape}")
    df_merged = pd.merge(clstat, bsinfo, on=['BS', 'CellName'], how='left')
    print(f"  ✓ Result: {df_merged.shape}")
    print(f"    Rows with NaN: {df_merged.isnull().any(axis=1).sum():,}")

    # Step 2: Result + ECstat
    print("\n🔗 Step 2: Merging Result + ECstat...")
    print(f"  Current: {df_merged.shape}")
    print(f"  ECstat: {ecstat.shape}")
    df_final = pd.merge(df_merged, ecstat, on=['Time', 'BS'], how='left', suffixes=('', '_ec'))
    print(f"  ✓ Result: {df_final.shape}")
    print(f"    Rows with NaN: {df_final.isnull().any(axis=1).sum():,}")

    # Handle duplicate columns
    duplicate_cols = [col for col in df_final.columns if col.endswith('_ec') and col[:-3] in df_final.columns]
    if duplicate_cols:
        print(f"\n🔧 Resolving {len(duplicate_cols)} duplicate columns...")
        for col in duplicate_cols:
            base_col = col[:-3]
            if base_col not in ['Energy']:
                df_final[base_col] = df_final[base_col].fillna(df_final[col])
        df_final = df_final.drop(columns=duplicate_cols)
        print("  ✓ Duplicates resolved")

    # Remove missing values
    print("\n🧹 Removing rows with missing values...")
    rows_before = len(df_final)
    df_final = df_final.dropna()
    rows_removed = rows_before - len(df_final)
    print(f"  Rows removed: {rows_removed:,} ({100*rows_removed/rows_before:.1f}%)")
    print(f"  Rows retained: {len(df_final):,} ({100*len(df_final)/rows_before:.1f}%)")

    print(f"\n✓ Final dataset: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")

    return df_final


# =============================================================================
# SECTION 8: SAVE PROCESSED DATA
# =============================================================================
def save_data(df_final, clstat, ecstat):
    """Save processed datasets."""
    print("\n" + "="*80)
    print("SAVING PROCESSED DATA")
    print("="*80)

    df_final.to_csv('processed_data/netop_processed.csv', index=False)
    print(f"  ✓ Saved: netop_processed.csv ({df_final.shape[0]:,} rows)")

    clstat.to_csv('processed_data/clstat_processed.csv', index=False)
    print(f"  ✓ Saved: clstat_processed.csv ({clstat.shape[0]:,} rows)")

    ecstat.to_csv('processed_data/ecstat_processed.csv', index=False)
    print(f"  ✓ Saved: ecstat_processed.csv ({ecstat.shape[0]:,} rows)")

    print("\n✓ All files saved successfully!")


# =============================================================================
# SECTION 9: SUMMARY REPORT
# =============================================================================
def print_summary(df_final):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)

    print(f"\n📊 Final Dataset: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")

    print("\n📋 Columns:")
    for i, col in enumerate(df_final.columns, 1):
        print(f"  {i:2d}. {col}")

    print("\n🎯 Data Quality:")
    print(f"  - Missing values: {df_final.isnull().sum().sum()}")
    print(f"  - Duplicate rows: {df_final.duplicated().sum()}")
    print(f"  - Date range: {df_final['Time'].min()} to {df_final['Time'].max()}")
    print(f"  - Base stations: {df_final['BS'].nunique()}")
    print(f"  - Cells: {df_final['CellName'].nunique()}")

    print("\n📈 Key Statistics:")
    print(df_final[['load', 'Energy', 'TXpower', 'Bandwidth']].describe())

    # Show ESMode statistics if available
    esmode_cols = [col for col in df_final.columns if col.startswith('ESMode')]
    if esmode_cols:
        print("\n⚡ Energy Saving Mode Statistics:")
        print(f"  Found {len(esmode_cols)} ESMode columns: {esmode_cols}")
        for col in esmode_cols:
            active_pct = 100 * (df_final[col] > 0).mean()
            if active_pct > 0:
                print(f"  - {col}: Active {active_pct:.2f}% of the time")

    print("\n" + "="*80)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # UNCOMMENT THE SECTIONS YOU WANT TO RUN

    # Load data
    bsinfo, clstat, ecstat = load_data()

    # Inspect data (comment out if not needed)
    inspect_bsinfo(bsinfo)
    inspect_clstat(clstat)
    inspect_ecstat(ecstat)

    # Clean data
    bsinfo, clstat, ecstat = clean_data(bsinfo, clstat, ecstat)

    # Feature engineering
    clstat, ecstat = add_time_features(clstat, ecstat)
    clstat, ecstat = add_lagged_features(clstat, ecstat)
    clstat, ecstat = add_rolling_features(clstat, ecstat)

    # Merge datasets
    df_final = merge_datasets(clstat, bsinfo, ecstat)

    # Save data
    save_data(df_final, clstat, ecstat)

    # Print summary
    print_summary(df_final)
