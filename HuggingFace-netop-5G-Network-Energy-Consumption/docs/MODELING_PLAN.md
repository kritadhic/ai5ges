# Energy Consumption Prediction - Modeling Plan

**Objective:** Predict energy consumption for 5G base stations

**Date:** March 1, 2026
**Dataset:** Netop 5G Network (7 days, ~200K samples after preprocessing)

---

## Use Cases

1. **Real-time forecasting:** Predict energy consumption for the next hour
2. **What-if analysis:** Predict energy consumption given load/configuration changes

---

## Modeling Strategy

Explore three approaches in sequence, comparing performance at each step:

1. **Time Series Models** - Baseline univariate/simple multivariate
2. **Traditional ML** - Feature-rich gradient boosting models
3. **Deep Learning** - Sequence-based neural networks

---

# PHASE 1: Data Preparation

## 1.1 Create Multiple Preprocessing Variants

We create **3 preprocessing variants** to compare the value of feature engineering:

### **Variant 1: Baseline - Raw Features Only**
- **Purpose:** ML baseline without any feature engineering
- **Features:** Raw features only (17 columns)
- **File:** `netop_ml_baseline.csv`
- **Columns:**
  ```python
  ['Time', 'BS', 'CellName', 'load',
   'ESMode1', 'ESMode2', 'ESMode3', 'ESMode4', 'ESMode5', 'ESMode6',
   'RUType', 'Mode', 'Frequency', 'Bandwidth', 'Antennas', 'TXpower',
   'Energy']  # Target
  ```
- **Use Case:** Test if raw features alone can predict energy

### **Variant 2: Time - Raw + Time Features**
- **Purpose:** ML with basic time features
- **Features:** Raw + time features (22 columns)
- **File:** `netop_ml_time.csv`
- **Additional Columns:**
  ```python
  ['hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour', 'is_night_time']
  ```
- **Use Case:** Test value of temporal features (daily/weekly patterns)

### **Variant 3: Full - Everything**
- **Purpose:** ML with all engineered features
- **Features:** Raw + time + lag + rolling features (30 columns)
- **File:** `netop_ml_full.csv`
- **Additional Columns:**
  ```python
  ['load_lag1', 'load_lag24', 'energy_lag1', 'energy_lag24',
   'load_rolling_mean_3h', 'load_rolling_std_3h',
   'energy_rolling_mean_3h', 'energy_rolling_std_3h']
  ```
- **Use Case:** Test value of lag and rolling statistics

**Note:** Time series models (ARIMA/Prophet) and deep learning models (LSTM/GRU) will be explored later with separate data preparation steps.

---

# PHASE 2: Traditional ML - Initial Focus

**Priority:** Start with traditional ML models using the 3 variants to understand feature importance and baseline performance.

Time series models (ARIMA/Prophet) and deep learning models (LSTM/GRU) will be explored in later phases after establishing ML baselines.

---

## 2.1 Experiment 1: Baseline (Raw Features Only)

**Data:** `netop_ml_baseline.csv` (17 columns)

**Models:**
1. Linear Regression (sanity check)
2. Random Forest
3. XGBoost
4. LightGBM

**Features:**
- Raw: load, TXpower, Bandwidth, ESMode1-6, RUType, Mode, Frequency, Antennas
- Target: Energy

**Implementation:**
```python
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

# Load baseline data
df = pd.read_csv('processed_data/netop_ml_baseline.csv')

# Features (exclude Time, BS, CellName, Energy)
feature_cols = ['load', 'TXpower', 'Bandwidth', 'ESMode1', 'ESMode2',
                'ESMode3', 'ESMode4', 'ESMode5', 'ESMode6',
                'RUType', 'Mode', 'Frequency', 'Antennas']
X = df[feature_cols]
y = df['Energy']

# Time-aware split
tscv = TimeSeriesSplit(n_splits=3)
for train_idx, val_idx in tscv.split(X):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[val_idx])

    # Evaluate
    mae = mean_absolute_error(y.iloc[val_idx], pred)
    print(f"Fold MAE: {mae:.2f} kW")
```

**Questions to Answer:**
- Can raw features predict energy reasonably well?
- Which raw features are most important?
- What's the baseline performance to beat?

---

## 2.2 Experiment 2: Time Features Added

**Data:** `netop_ml_time.csv` (22 columns)

**Models:** Same as Experiment 1

**Additional Features:**
- hour_of_day, day_of_week, is_weekend, is_peak_hour, is_night_time

**Implementation:** Same structure, just use `netop_ml_time.csv`

**Questions to Answer:**
- How much do time features improve performance?
- Are daily patterns (hour_of_day) more important than weekly (day_of_week)?
- Is the improvement significant enough to justify the extra features?

**Expected Improvement:** 10-25% reduction in MAE

---

## 2.3 Experiment 3: Full Features (Lag + Rolling)

**Data:** `netop_ml_full.csv` (30 columns)

**Models:** Same as Experiment 1

**Additional Features:**
- load_lag1, load_lag24, energy_lag1, energy_lag24
- load_rolling_mean_3h, load_rolling_std_3h
- energy_rolling_mean_3h, energy_rolling_std_3h

**Implementation:** Same structure, just use `netop_ml_full.csv`

**Questions to Answer:**
- Do lag features (load_lag1, load_lag24) help?
- Do rolling statistics (3-hour averages) help?
- What's the cost/benefit of engineered features?

**Expected Improvement:** 5-15% additional reduction in MAE over time features

**Trade-off:**
- ✅ Better performance
- ❌ Requires historical data for inference
- ❌ Less flexible for what-if analysis

---

# PHASE 3: Time Series Forecasting (Future Work)

**Data Preparation:**
- Create `netop_timeseries.csv` with Time, BS, Energy columns
- One time series per base station

**Models to Explore:**
- ARIMA/SARIMA (univariate)
- Facebook Prophet (with load as external regressor)

**Limitations:**
- ❌ Univariate (doesn't leverage all features)
- ❌ Can't do what-if analysis easily
- ❌ Requires separate model per base station (1,217 models)

**Decision:** Explore after establishing ML baselines in Phase 2

---

# PHASE 4: Deep Learning (Future Work)

**Data Preparation:**
- Create sequence data: `netop_sequences.npz`
- Structure: Past 24 hours → Predict next hour
- Format: (samples, 24 timesteps, features)

**Models to Explore:**
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Optional: Transformer (if LSTM/GRU underperform)

**Advantages:**
- ✅ Learns temporal patterns automatically
- ✅ No need for manual lag features

**Challenges:**
- ⚠️ Requires more data (7 days may be limited)
- ⚠️ Longer training time
- ⚠️ Harder to interpret

**Decision:** Explore after ML baselines show promise

---

# PHASE 5: Model Comparison & Analysis

## 5.1 Performance Comparison Table

### **Table: Feature Set Comparison**

| Variant | Features | Columns | Use Case | Expected Performance |
|---------|----------|---------|----------|---------------------|
| Baseline | Raw only | 17 | What-if analysis | Baseline |
| Time | Raw + Time | 22 | Real-time + What-if | +10-25% improvement |
| Full | Raw + Time + Lag | 30 | Real-time (best) | +5-15% additional |

### **Table: Model Type Comparison** (to be filled after experiments)

| Model Type | Data Variant | MAE (kW) | RMSE (kW) | MAPE (%) | R² | Train Time | Inference |
|------------|--------------|----------|-----------|----------|-----|------------|-----------|
| XGBoost | Baseline | TBD | TBD | TBD | TBD | ~X min | ~X ms |
| XGBoost | Time | TBD | TBD | TBD | TBD | ~X min | ~X ms |
| XGBoost | Full | TBD | TBD | TBD | TBD | ~X min | ~X ms |
| LightGBM | Baseline | TBD | TBD | TBD | TBD | ~X min | ~X ms |
| LightGBM | Time | TBD | TBD | TBD | TBD | ~X min | ~X ms |
| LightGBM | Full | TBD | TBD | TBD | TBD | ~X min | ~X ms |
| Prophet | Timeseries | TBD | TBD | TBD | TBD | ~X min | ~X ms |
| LSTM | Sequences | TBD | TBD | TBD | TBD | ~X min | ~X ms |

---

## 5.2 Feature Importance Analysis

**For Traditional ML (XGBoost/LightGBM):**

Questions to answer:
1. Which features are most important for energy prediction?
2. Are ESMode columns significant?
3. Do lag features (load_lag24) rank in top 10?
4. Is day_of_week important? (weekend vs weekday patterns)

**Expected Top Features:**
1. load (current)
2. load_lag24 (yesterday same hour)
3. hour_of_day (daily patterns)
4. TXpower (base station power)
5. load_lag1 (recent trend)

---

## 5.3 Ablation Study

Test the incremental value of feature groups:

```python
feature_groups = {
    '1. Core': ['load', 'TXpower', 'Bandwidth'],
    '2. + Hardware': core + ['RUType', 'Mode', 'Frequency', 'Antennas'],
    '3. + ESModes': hardware + ['ESMode1', 'ESMode2', ...],
    '4. + Time': esmodes + ['hour_of_day', 'day_of_week', 'is_weekend', ...],
    '5. + Lag': time + ['load_lag1', 'load_lag24', 'energy_lag1', 'energy_lag24'],
    '6. + Rolling': lag + ['load_rolling_mean_3h', 'load_rolling_std_3h', ...]
}

# Train XGBoost with each feature set
for name, features in feature_groups.items():
    model.fit(X[features], y)
    mae = evaluate(model)
    print(f"{name}: MAE = {mae:.2f} kW")
```

**Goal:** Understand the marginal value of each feature group

---

## 5.4 Error Analysis

**Questions to Answer:**
1. Are errors higher during peak hours?
2. Are certain base stations harder to predict?
3. Is there systematic bias (over/under prediction)?
4. Do residuals show temporal patterns?

**Visualizations:**
- Residuals over time
- Residuals by hour of day
- Residuals by base station
- Actual vs Predicted scatter plot

---

# Implementation Timeline (Updated)

## Week 1-2: Data Preparation & Traditional ML

✅ **Phase 1: Data Preparation** (Days 1-2)
- [x] Run `preprocess_netop.ipynb` to generate 3 variants
- [x] Verify data quality of all 3 CSVs
- [x] Document feature lists

**Phase 2: Traditional ML Experiments** (Days 3-10)
- [ ] Day 3-4: Experiment 1 - Baseline (raw features)
- [ ] Day 5-6: Experiment 2 - Time features added
- [ ] Day 7-8: Experiment 3 - Full features (lag + rolling)
- [ ] Day 9: Feature importance & ablation study
- [ ] Day 10: Compare all ML models, document results

## Week 3: Advanced Models (Time Series & Deep Learning)

**Phase 3: Time Series** (Days 11-13) - Optional
- [ ] Day 11: Prepare time series data
- [ ] Day 12: ARIMA/Prophet experiments
- [ ] Day 13: Compare with ML results

**Phase 4: Deep Learning** (Days 14-17) - Optional
- [ ] Day 14: Prepare sequence data
- [ ] Day 15-16: LSTM/GRU experiments
- [ ] Day 17: Compare with ML results

## Week 4: Analysis & Documentation

**Phase 5: Comparison & Final Analysis** (Days 18-21)
- [ ] Day 18: Comprehensive comparison table
- [ ] Day 19: Error analysis & visualizations
- [ ] Day 20: Feature importance insights
- [ ] Day 21: Final recommendations & documentation

---

# Deliverables

## 1. Preprocessed Datasets ✅

- [x] `netop_ml_baseline.csv` - Raw features (17 cols)
- [x] `netop_ml_time.csv` - Raw + time features (22 cols)
- [x] `netop_ml_full.csv` - Full features (30 cols)
- [ ] `netop_timeseries.csv` - For ARIMA/Prophet (future)
- [ ] `netop_sequences.npz` - For LSTM/GRU (future)

## 2. Trained Models

**Traditional ML:**
- [ ] `models/xgboost_baseline.json`
- [ ] `models/xgboost_time.json`
- [ ] `models/xgboost_full.json`
- [ ] `models/lightgbm_baseline.txt`
- [ ] `models/lightgbm_time.txt`
- [ ] `models/lightgbm_full.txt`
- [ ] `models/scaler_*.pkl` (StandardScaler for each variant)

**Advanced (Optional):**
- [ ] `models/prophet_*.pkl`
- [ ] `models/lstm_*.h5`
- [ ] `models/gru_*.h5`

## 3. Notebooks

- [ ] `01_traditional_ml_experiments.ipynb` - All 3 variants
- [ ] `02_feature_importance_analysis.ipynb` - Deep dive
- [ ] `03_model_comparison.ipynb` - Compare all approaches
- [ ] `04_error_analysis.ipynb` - Residual analysis
- [ ] `05_timeseries_models.ipynb` - ARIMA/Prophet (optional)
- [ ] `06_deep_learning_models.ipynb` - LSTM/GRU (optional)

## 4. Reports & Documentation

- [x] `MODELING_PLAN.md` - This document
- [ ] `RESULTS_SUMMARY.md` - Final comparison table
- [ ] `FEATURE_IMPORTANCE.md` - Feature analysis insights
- [ ] `ERROR_ANALYSIS.md` - Prediction error patterns
- [ ] `RECOMMENDATIONS.md` - Final model recommendation

## 5. Production Scripts (Optional)

- [ ] `predict_energy.py` - Inference script for best model
- [ ] Support for both real-time and what-if modes

---

# Success Criteria

## Minimum Viable Product (MVP)

1. **Baseline Performance:**
   - Traditional ML (XGBoost/LightGBM) achieves R² > 0.7
   - MAE < 20% of mean energy consumption
   - Clear feature importance ranking

2. **Feature Engineering Value:**
   - Quantify improvement from time features (Baseline → Time)
   - Quantify improvement from lag/rolling features (Time → Full)
   - Document trade-offs (performance vs inference requirements)

3. **Model Comparison:**
   - At least 3 variants tested (Baseline, Time, Full)
   - At least 2 algorithms tested (XGBoost, LightGBM)
   - Clear documentation of which model/features work best

## Stretch Goals

4. **Time Series Models:**
   - ARIMA/Prophet baseline for comparison
   - Understand limitations for this use case

5. **Deep Learning:**
   - LSTM/GRU experiments
   - Compare with traditional ML (likely comparable or worse on small data)

6. **Production Deployment:**
   - Inference script ready for deployment
   - Model serialization and loading
   - Support for both use cases (real-time + what-if)

---

# Next Steps

## Immediate Actions

1. **Run the preprocessing notebook** - Generate the 3 CSV files
2. **Create modeling notebook** - Start with Experiment 1 (Baseline)
3. **Set up project structure:**
   ```
   HuggingFace-netop-5G-Network-Energy-Consumption/
   ├── processed_data/
   │   ├── netop_ml_baseline.csv  ✅
   │   ├── netop_ml_time.csv      ✅
   │   └── netop_ml_full.csv      ✅
   ├── models/
   │   └── (trained models will go here)
   ├── notebooks/
   │   └── (analysis notebooks will go here)
   └── results/
       └── (comparison tables, plots will go here)
   ```

## Ready to Start?

**Status:** ✅ Plan Updated - Ready for Implementation
**Next:** Run `preprocess_netop.ipynb` to generate the 3 CSV variants

