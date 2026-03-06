# Traditional ML Experiments - Comprehensive Analysis

**Date:** March 6, 2026
**Project:** 5G Network Energy Consumption Prediction
**Evaluation:** Phase 2 - Traditional Machine Learning Models

---

## Executive Summary

Three experiments were conducted to measure the value of feature engineering for energy consumption prediction. All experiments used proper temporal train-test split (80/20) with TimeSeriesSplit cross-validation to respect the time series nature of the data.

### Key Findings

1. **Baseline features are strong** - Raw operational features (load, hardware config) achieve R² = 0.88
2. **Time features add moderate value** - 6% MAE improvement over baseline
3. **Load lag features add NO value** - No improvement over time features
4. **Total improvement: 5.97%** - Below 10-25% target, but data leakage was prevented
5. **Best production model: Time Features (Experiment 2)** - Simplest model with best performance

---

## Experiment Results

### Experiment 1: Baseline (Raw Features Only)

**Data:** `netop_ml_baseline.csv` (13 features)
- **Features:** load, ESMode1-6, RUType, Mode, Frequency, Bandwidth, Antennas, TXpower

**Best Model:** LightGBM
- **Test R²:** 0.8837
- **Test MAE:** 3.34 W
- **Test RMSE:** 4.79 W
- **Test MAPE:** 12.29%

**Status:** ✓ PASS success criteria (R² > 0.7, MAE < 20% of mean)

**Key Features (Consensus):**
1. load (most important)
2. Antennas
3. Frequency
4. RUType_Type7
5. TXpower

---

### Experiment 2: Time Features

**Data:** `netop_ml_time.csv` (18 features = 13 raw + 5 time)
- **Added Features:** hour_of_day, day_of_week, is_weekend, is_peak_hour, is_night_time

**Best Model:** LightGBM
- **Test R²:** 0.8942
- **Test MAE:** 3.14 W
- **Test RMSE:** 4.57 W
- **Test MAPE:** 11.06%

**Improvement over Baseline:**
- **MAE:** 6.00% reduction (3.34 W → 3.14 W)
- **R²:** 1.18% increase (0.8837 → 0.8942)

**Status:** ✗ BELOW TARGET (expected 10-25%, achieved 6%)

**Time Feature Importance:**
- hour_of_day: Most important time feature
- day_of_week: Secondary importance
- is_night_time, is_peak_hour, is_weekend: Minor importance

**Analysis:** Time features provide some value by capturing daily/weekly patterns, but the baseline features (especially current load) already capture most of the variance in energy consumption.

---

### Experiment 3: Full Features (Load Lags + Time)

**Data:** `netop_ml_full.csv` (22 features = 13 raw + 5 time + 4 load lags)
- **Added Features:** load_lag1, load_lag24, load_rolling_mean_3h, load_rolling_std_3h
- **EXCLUDED (data leakage):** energy_lag1, energy_lag24, energy_rolling_mean_3h, energy_rolling_std_3h

**Best Model:** LightGBM
- **Test R²:** 0.8937
- **Test MAE:** 3.14 W
- **Test RMSE:** 4.58 W
- **Test MAPE:** 11.02%

**Improvement over Time Features:**
- **MAE:** -0.04% (3.14 W → 3.14 W) - **NO IMPROVEMENT**
- **R²:** -0.05% (0.8942 → 0.8937) - **SLIGHT DEGRADATION**

**Total Improvement (Baseline → Full):**
- **MAE:** 5.97% reduction (3.34 W → 3.14 W)
- **R²:** 1.13% increase (0.8837 → 0.8937)

**Status:** ✗ BELOW TARGET (expected 10-25%, achieved 5.97%)

**Load Lag Feature Importance:** Load lag features ranked LOW in importance, confirming they don't add predictive value.

**Critical Insight:** Historical load patterns do NOT improve energy prediction beyond current load. This makes sense because:
- Energy consumption is primarily determined by CURRENT load and hardware configuration
- Historical patterns are already reflected in the current load value
- Network energy consumption responds immediately to load changes

---

## Data Leakage Prevention

### Problem Identified
The `netop_ml_full.csv` dataset contains energy lag features that cause data leakage in the test set:
- `energy_lag1` - Previous hour's energy (would contain future test values)
- `energy_lag24` - 24 hours ago energy (would contain future test values)
- `energy_rolling_mean_3h` - 3-hour rolling mean of energy (would contain future test values)
- `energy_rolling_std_3h` - 3-hour rolling std of energy (would contain future test values)

### Solution Implemented
Experiment 3 explicitly excludes these 4 features to prevent data leakage:
```python
exclude_cols = ['Time', 'BS', 'CellName', 'Energy',
                'energy_lag1', 'energy_lag24',
                'energy_rolling_mean_3h', 'energy_rolling_std_3h']
```

### Why This Matters
In production deployment:
- We CANNOT use past energy values to predict current energy
- Only past/current load values and current hardware config are available
- Using energy lags would artificially inflate test performance but fail in production

This is why **Experiment 3 correctly excluded energy lag features**, even though they exist in the dataset.

---

## Model Comparison

### Test Set Performance (All Experiments)

| Model | Experiment | Features | Test MAE (W) | Test R² | Improvement |
|-------|-----------|----------|--------------|---------|-------------|
| LightGBM | Baseline | 13 raw | 3.34 | 0.8837 | - |
| LightGBM | Time | 13 raw + 5 time | 3.14 | 0.8942 | 6.00% |
| LightGBM | Full | 13 raw + 5 time + 4 load lags | 3.14 | 0.8937 | 5.97% |

### Key Insights

1. **Time features → Full features: NO improvement**
   - Adding load lag features provides no additional value
   - Performance is essentially identical (3.14 W in both cases)

2. **All improvement comes from time features**
   - Baseline → Time: 6% improvement
   - Time → Full: 0% improvement

3. **Best model: LightGBM with Time Features (Experiment 2)**
   - Simplest feature set (18 features vs 22)
   - Best test performance (R² = 0.8942)
   - No unnecessary complexity from lag features

---

## Recommendations

### 1. Use Time Features Model for Production ✓

**Recommendation:** Deploy the **LightGBM model from Experiment 2** (Time Features)

**Rationale:**
- Best test performance (R² = 0.8942, MAE = 3.14 W)
- Simpler than Full Features (18 vs 22 features)
- No lag features needed → simpler preprocessing
- Meets success criteria (R² > 0.7, MAE < 20% of mean)

**Production Model:**
- File: `models/lightgbm_time.pkl`
- Features: 13 raw + 5 time features
- Expected MAE: ~3.14 W (11% MAPE)

### 2. Update Preprocessing Pipeline ⚠️

**Recommendation:** Modify `preprocess_netop.ipynb` to ONLY generate time features, not lag features

**Changes needed:**
- Keep: `hour_of_day`, `day_of_week`, `is_weekend`, `is_peak_hour`, `is_night_time`
- Remove: Load lag generation (`load_lag1`, `load_lag24`, rolling stats)
- Remove: Energy lag generation (already excluded due to data leakage)

**Benefits:**
- Faster preprocessing (no rolling window calculations)
- Simpler pipeline (fewer features to track)
- Same predictive performance (load lags don't help)

### 3. Understanding Why Performance is Below Target

**Expected:** 10-25% MAE improvement
**Achieved:** 6% MAE improvement

**Reasons:**
1. **Strong baseline features**
   - Current load is highly predictive of current energy
   - Hardware configuration (Antennas, Frequency) captures the rest
   - Little room for improvement from temporal features

2. **Linear energy-load relationship**
   - Energy consumption scales predictably with load
   - Complex temporal patterns don't add much value

3. **Data leakage prevention**
   - Correctly excluding energy lags prevents cheating
   - Real production performance is captured accurately

**Conclusion:** The 6% improvement is realistic and production-ready, even if below initial expectations.

---

## Next Steps

### Immediate Actions

1. **Deploy Time Features Model**
   - Use `models/lightgbm_time.pkl` for production inference
   - Expected MAE: ~3.14 W (11% MAPE on test set)

2. **Simplify Preprocessing**
   - Update `preprocess_netop.ipynb` to remove lag feature generation
   - Keep only: raw features + time features

3. **Document Production Pipeline**
   - Input: Raw data with Time, load, hardware config
   - Feature engineering: Extract time features only
   - Model: LightGBM with 18 features
   - Output: Energy prediction (W)

### Future Investigations (Optional)

If further improvement is desired:

1. **Try non-linear time features**
   - Cyclical encoding (sin/cos) for hour and day
   - Interaction terms between time and load

2. **Weather integration**
   - Temperature affects cooling requirements
   - May explain additional variance

3. **Cell-specific models**
   - Different cells may have different energy patterns
   - Train separate models per cell or cell cluster

4. **Ensemble methods**
   - Combine multiple models for marginal gains
   - May push closer to 10% improvement target

---

## Technical Details

### Evaluation Methodology

**Train-Test Split:**
- Temporal 80/20 split (no shuffling)
- Training: First 58,055 samples
- Test: Last 14,514 samples (held out)

**Cross-Validation:**
- TimeSeriesSplit with 5 folds
- Only on training set (test set never seen)
- Proper time series evaluation

**Metrics:**
- MAE (Mean Absolute Error) - primary metric
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

### Models Evaluated

All experiments tested 4 models:
1. Linear Regression (sanity check)
2. Random Forest
3. XGBoost
4. LightGBM (best performer)

Hyperparameters:
```python
LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

---

## Files Generated

### Notebooks
- `scripts/traditional_ml_baseline.ipynb` - Experiment 1 (✓ completed)
- `scripts/traditional_ml_time.ipynb` - Experiment 2 (✓ completed)
- `scripts/traditional_ml_full.ipynb` - Experiment 3 (✓ completed)

### Results
- `results/traditional_ml_baseline_results.csv` - Baseline test results
- `results/traditional_ml_time_results.csv` - Time features test results
- `results/traditional_ml_full_results.csv` - Full features test results
- `results/time_features_importance.csv` - Feature importance rankings

### Models
- `models/lightgbm_baseline.pkl` - Baseline model
- `models/lightgbm_time.pkl` - **Production model (recommended)** ⭐
- `models/lightgbm_full.pkl` - Full features model (not recommended)

---

## Conclusion

The traditional ML experiments successfully:
1. ✓ Established strong baseline performance (R² = 0.88)
2. ✓ Identified time features as valuable (6% improvement)
3. ✓ Prevented data leakage by excluding energy lag features
4. ✓ Discovered load lags don't help (0% improvement)
5. ✓ Recommended production model: LightGBM with Time Features

**Production Model:** `models/lightgbm_time.pkl`
- **Test R²:** 0.8942
- **Test MAE:** 3.14 W (11% MAPE)
- **Features:** 13 raw + 5 time = 18 features
- **Status:** Ready for deployment

**Next Phase:** Consider deep learning models (LSTM, Transformer) if further improvement is required, but the current model meets success criteria and is production-ready.
