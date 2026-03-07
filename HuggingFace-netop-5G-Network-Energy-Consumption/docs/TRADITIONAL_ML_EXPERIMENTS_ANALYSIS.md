# Traditional ML Experiments - Comprehensive Analysis

**Date:** March 6, 2026
**Project:** 5G Network Energy Consumption Prediction
**Evaluation:** Phase 2 - Traditional Machine Learning Models

---

## Executive Summary

Four experiments were completed to measure the value of feature engineering and hyperparameter optimization for energy consumption prediction. All experiments used proper temporal train-test split (80/20) with TimeSeriesSplit cross-validation to respect the time series nature of the data.

### Key Findings

1. **Baseline features are strong** - Raw operational features (load, hardware config) achieve R² = 0.88
2. **Time features add moderate value** - 6% MAE improvement over baseline
3. **Load lag features add NO value** - No improvement over time features
4. **Hyperparameter tuning yields minimal gains** - Only 0.87% additional improvement
5. **Total improvement: 6.87%** - Below 10-25% target, but data leakage was prevented
6. **Feature set is the limiting factor** - Not model configuration or hyperparameters
7. **Best production model: Time Features (Experiment 2)** - Simplest model with comparable performance (only 0.87% behind tuned version)

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
| LightGBM | Hyperparameter Tuned | 13 raw + 5 time (optimized) | 3.11 | 0.8982 | 6.87% |

### Key Insights

1. **Time features → Full features: NO improvement**
   - Adding load lag features provides no additional value
   - Performance is essentially identical (3.14 W in both cases)

2. **All improvement comes from time features**
   - Baseline → Time: 6% improvement
   - Time → Full: 0% improvement

3. **Hyperparameter tuning: MINIMAL improvement**
   - Time → Tuned: 0.87% improvement (3.14 W → 3.11 W)
   - Default hyperparameters were already near-optimal
   - Indicates feature set is the limiting factor, not model configuration

4. **Best model: LightGBM with Time Features + Tuned Hyperparameters (Experiment 4)**
   - Test performance (R² = 0.8982, MAE = 3.11 W)
   - Total improvement: 6.87% over baseline
   - Complexity vs. gain trade-off: Minimal gain for tuning effort

---

## Experiment 4: Hyperparameter Tuning

**Objective:** Optimize LightGBM and XGBoost hyperparameters to extract maximum performance from existing features.

**Data:** `netop_ml_time.csv` (same as Experiment 2)
- **Features:** 13 raw + 5 time = 18 features
- **Method:** Optuna Bayesian Optimization (100 trials per model)
- **Cross-Validation:** TimeSeriesSplit with 5 folds

### Optimization Details

**LightGBM Hyperparameter Search Space:**
- `n_estimators`: [100, 500]
- `max_depth`: [4, 10]
- `learning_rate`: [0.01, 0.2] (log scale)
- `num_leaves`: [31, 127]
- `min_child_samples`: [5, 50]
- `subsample`: [0.6, 1.0]
- `colsample_bytree`: [0.6, 1.0]
- `reg_alpha`: [0, 1.0] (L1 regularization)
- `reg_lambda`: [0, 1.0] (L2 regularization)

**XGBoost Hyperparameter Search Space:**
- `n_estimators`: [100, 500]
- `max_depth`: [3, 10]
- `learning_rate`: [0.01, 0.2] (log scale)
- `min_child_weight`: [1, 10]
- `gamma`: [0, 0.5]
- `subsample`: [0.6, 1.0]
- `colsample_bytree`: [0.6, 1.0]
- `reg_alpha`: [0, 1.0]
- `reg_lambda`: [0, 1.0]

### Results

**LightGBM:**
- **Baseline (Exp 2):** MAE = 3.1401 W, R² = 0.8942
- **Tuned (Exp 4):** MAE = 3.1127 W, R² = 0.8982
- **Improvement:** 0.87% MAE reduction
- **Status:** ✗ BELOW TARGET (target: 1% minimum, 2-3% desired)

**XGBoost:**
- **Baseline (Exp 2):** MAE = 3.1461 W, R² = 0.8916
- **Tuned (Exp 4):** MAE = 3.1575 W, R² = 0.8960
- **Improvement:** -0.36% MAE increase (degraded)
- **Status:** ✗ DEGRADED

### Best Hyperparameters Found

**LightGBM Optimal Configuration:**
(Saved in `results/best_hyperparameters.json`)

### Analysis

**Why Hyperparameter Tuning Had Minimal Impact:**

1. **Default hyperparameters were already well-optimized**
   - LightGBM and XGBoost have excellent defaults for regression
   - 100 trials of Bayesian optimization found only marginally better configurations

2. **Strong baseline features limit optimization gains**
   - Current `load` is highly predictive (accounts for most variance)
   - Hardware configuration features are straightforward
   - No complex interactions requiring sophisticated model configurations

3. **Straightforward energy-load relationship**
   - Energy consumption scales predictably with load
   - No benefit from increased model complexity (deeper trees, more leaves)
   - Regularization doesn't help when overfitting isn't the issue

4. **XGBoost degradation suggests overfitting to CV folds**
   - Tuned parameters performed slightly worse on test set
   - Indicates default parameters generalize better

### Conclusion

**Key Finding:** The **feature set is the limiting factor**, not model hyperparameters.

Further improvements must come from:
- Different feature engineering approaches (segmentation)
- Alternative model architectures (ensemble, deep learning)
- NOT from hyperparameter optimization of gradient boosting models

**Recommendation:**
- **For simplicity:** Use Experiment 2 default LightGBM model (3.14 W)
- **For marginal gain:** Use Experiment 4 tuned LightGBM model (3.11 W)
- **Trade-off:** 0.87% improvement not worth the tuning complexity in production

---

## Recommendations

### 1. Production Model Selection ✓

**Two Options Available:**

**Option A: Simplicity (RECOMMENDED)**
- **Model:** LightGBM from Experiment 2 (default hyperparameters)
- **File:** `models/lightgbm_time.pkl`
- **Performance:** R² = 0.8942, MAE = 3.14 W (11% MAPE)
- **Rationale:**
  - Simpler deployment (no custom hyperparameters)
  - Easier to retrain and maintain
  - Only 0.87% worse than tuned model
  - Meets all success criteria

**Option B: Marginal Performance Gain**
- **Model:** LightGBM from Experiment 4 (tuned hyperparameters)
- **File:** `models/lightgbm_time_ht.pkl`
- **Performance:** R² = 0.8982, MAE = 3.11 W (10.9% MAPE)
- **Rationale:**
  - Best absolute performance (6.87% improvement over baseline)
  - 0.87% better than default model
  - Requires saving/loading custom hyperparameters
  - Added complexity for minimal gain

**Recommendation:** Use **Option A (Experiment 2 default model)** for production deployment unless the 0.03W improvement is critical.

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
**Achieved:** 6.87% MAE improvement (with hyperparameter tuning)

**Reasons:**
1. **Strong baseline features**
   - Current load is highly predictive of current energy
   - Hardware configuration (Antennas, Frequency) captures the rest
   - Little room for improvement from temporal features or hyperparameter tuning

2. **Linear energy-load relationship**
   - Energy consumption scales predictably with load
   - Complex temporal patterns don't add much value

3. **Data leakage prevention**
   - Correctly excluding energy lags prevents cheating
   - Real production performance is captured accurately

4. **Near-optimal default hyperparameters**
   - Hyperparameter tuning added only 0.87% improvement
   - Indicates feature set, not model configuration, is the limiting factor

**Conclusion:** The 6.87% improvement is realistic and production-ready, even if below initial expectations. Further gains must come from segmentation (Experiment 6) or alternative approaches.

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

### Active Investigations

**Current Focus:**

1. **🔵 Segmented Models (Experiment 6 - READY FOR IMPLEMENTATION)**
   - Strong evidence: 54% performance gap on Cell1, CV=0.80 for BS variance
   - Expected improvement: 3-12% MAE reduction
   - Two segmentation dimensions:
     - **Cell-level:** 2 cells (Cell0 vs Cell1)
     - **BS-level:** 816 base stations clustered into k groups (k=3, 5, 7)
   - 7 strategies planned: cell-specific (2 models), BS clustering (3 variants), hierarchical, mixed effects
   - See "Experiment 6: Segmented Models" section for detailed design

**Skipped:**

2. **⏭️ Ensemble Methods (Experiment 5 - SKIPPED)**
   - Rationale: Hyperparameter tuning showed < 1% gain, indicating feature set limitation
   - Ensemble unlikely to provide significant improvement
   - Segmentation addresses different optimization dimension

### Future Investigations (Post Experiment 6)

If Experiment 6 doesn't achieve 10% target improvement:

1. **Try non-linear time features**
   - Cyclical encoding (sin/cos) for hour and day
   - Interaction terms between time and load

2. **Weather integration**
   - Temperature affects cooling requirements
   - May explain additional variance

3. **Deep learning models**
   - LSTM/GRU for temporal patterns
   - Attention mechanisms
   - Graph neural networks for BS relationships

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

**Experiments 1-3:** Tested 4 models with default hyperparameters
1. Linear Regression (sanity check)
2. Random Forest
3. XGBoost
4. LightGBM (best performer)

**Default Hyperparameters (Experiments 1-3):**
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

**Experiment 4:** Optimized LightGBM and XGBoost using Optuna
- 100 trials per model
- Bayesian optimization across 9 hyperparameters
- Best hyperparameters saved in `results/best_hyperparameters.json`

---

## Files Generated

### Notebooks
- `scripts/traditional_ml_baseline.ipynb` - Experiment 1 (✓ completed)
- `scripts/traditional_ml_time.ipynb` - Experiment 2 (✓ completed)
- `scripts/traditional_ml_full.ipynb` - Experiment 3 (✓ completed)
- `scripts/traditional_ml_time_ht.ipynb` - Experiment 4 (✓ completed)

### Results
- `results/traditional_ml_baseline_results.csv` - Baseline test results
- `results/traditional_ml_time_results.csv` - Time features test results
- `results/traditional_ml_full_results.csv` - Full features test results
- `results/traditional_ml_time_ht_results.csv` - Hyperparameter tuned test results
- `results/time_features_importance.csv` - Feature importance rankings
- `results/best_hyperparameters.json` - Optimal hyperparameters (Optuna)
- `results/lightgbm_tuning_history.csv` - LightGBM optimization trials
- `results/xgboost_tuning_history.csv` - XGBoost optimization trials
- `results/hyperparameter_tuning_improvement.csv` - Improvement summary

### Models
- `models/lightgbm_baseline.pkl` - Baseline model
- `models/lightgbm_time.pkl` - **Production model Option A (recommended)** ⭐
- `models/lightgbm_full.pkl` - Full features model (not recommended)
- `models/lightgbm_time_ht.pkl` - **Production model Option B (tuned)**
- `models/xgboost_time_ht.pkl` - Tuned XGBoost (for reference)

---

## Conclusion

The traditional ML experiments (1-4) successfully:
1. ✓ Established strong baseline performance (R² = 0.88)
2. ✓ Identified time features as valuable (6% improvement)
3. ✓ Prevented data leakage by excluding energy lag features
4. ✓ Discovered load lags don't help (0% improvement)
5. ✓ Tested hyperparameter optimization (0.87% additional gain)
6. ✓ **Identified feature set as the limiting factor, not model configuration**

**Key Insight from Experiment 4:**
Hyperparameter tuning provided minimal improvement (0.87%), indicating that:
- Default LightGBM hyperparameters are already near-optimal for this problem
- The strong baseline features (especially `load`) limit optimization gains
- Further improvements must come from different approaches (segmentation, deep learning)

**Production Model Options:**

**Option A (Recommended):** `models/lightgbm_time.pkl`
- **Test R²:** 0.8942
- **Test MAE:** 3.14 W (11% MAPE)
- **Features:** 13 raw + 5 time = 18 features
- **Benefits:** Simplicity, easier maintenance, default hyperparameters

**Option B (Marginal Gain):** `models/lightgbm_time_ht.pkl`
- **Test R²:** 0.8982
- **Test MAE:** 3.11 W (10.9% MAPE)
- **Features:** 13 raw + 5 time = 18 features (optimized hyperparameters)
- **Benefits:** 0.87% better performance, 6.87% total improvement over baseline

**Status:** Both models ready for deployment. Recommend Option A for simplicity.

---

## Planned Experiments: Remaining Work

While Experiment 4 (Hyperparameter Tuning) has been completed with minimal gains (0.87%), two additional experiments remain to maximize traditional ML performance before considering deep learning models.

**Status:**
- ✅ **Experiment 4 (Hyperparameter Tuning):** COMPLETE - Achieved 0.87% improvement, below 1% threshold
- ⏭️ **Experiment 5 (Ensemble Methods):** SKIPPED - Per decision rules (Exp 4 < 1% improvement)
- 🔵 **Experiment 6 (Segmented Models):** NEXT - Strong evidence supporting 3-12% improvement potential

---

## Experiment 5: Ensemble Methods (SKIPPED)

**Status:** ⏭️ SKIPPED per decision rules - Experiment 4 showed < 1% improvement threshold

**Rationale for Skipping:**
- Experiment 4 (Hyperparameter Tuning) achieved only 0.87% improvement (below 1% threshold)
- Default hyperparameters already near-optimal, indicating feature set is limiting factor
- Ensemble methods unlikely to provide significant improvement if tuning didn't
- Better to test segmentation approach (Experiment 6) which addresses a different dimension

**Original Objective:** Combine multiple models to leverage their complementary strengths and reduce prediction variance.

**Notebook:** `scripts/traditional_ml_time_ensemble.ipynb` (NOT IMPLEMENTED)

**Expected Improvement:** 1-2% MAE reduction over best single model

### Design Specifications (For Reference Only)

#### 5.1 Input Selection

**Base this experiment on the BEST performing model from:**
1. ✅ **Option A:** Experiment 4 (Hyperparameter Tuning) - if improvement > 1%
2. ⚠️ **Option B:** Experiment 2 (Time Features, no tuning) - if Experiment 4 fails to improve

**Decision Rule:**
```
if MAE_exp4 < MAE_exp2 * 0.99:  # At least 1% improvement
    use Experiment 4 tuned models
else:
    use Experiment 2 default models
```

#### 5.2 Data Source
- **Input:** `processed_data/netop_ml_time.csv`
- **Features:** 13 raw + 5 time = 18 features
- **Train-Test Split:** Temporal 80/20 (consistent with all experiments)

#### 5.3 Ensemble Strategies

**Strategy 1: Simple Averaging**

Average predictions from top 3 models:
```python
# Get predictions from each model
pred_lgbm = lgbm_pipeline.predict(X_test)
pred_xgb = xgb_pipeline.predict(X_test)
pred_rf = rf_pipeline.predict(X_test)

# Simple average
pred_ensemble_avg = (pred_lgbm + pred_xgb + pred_rf) / 3
```

**Strategy 2: Weighted Averaging**

Weight models by inverse MAE on validation set:
```python
# Calculate weights (inverse of validation MAE)
weights = np.array([1/mae_lgbm, 1/mae_xgb, 1/mae_rf])
weights = weights / weights.sum()  # Normalize to sum to 1

# Weighted average
pred_ensemble_weighted = (
    weights[0] * pred_lgbm +
    weights[1] * pred_xgb +
    weights[2] * pred_rf
)
```

**Strategy 3: Stacking (Meta-Model)**

Train a meta-model on out-of-fold predictions:
```python
from sklearn.linear_model import Ridge

# Step 1: Generate out-of-fold predictions on training set
tscv = TimeSeriesSplit(n_splits=5)
oof_preds = np.zeros((len(X_train_full), 3))  # 3 base models

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train_full)):
    # Train base models on this fold
    lgbm_fold = clone(lgbm_pipeline).fit(X_train_full.iloc[train_idx], y_train_full.iloc[train_idx])
    xgb_fold = clone(xgb_pipeline).fit(X_train_full.iloc[train_idx], y_train_full.iloc[train_idx])
    rf_fold = clone(rf_pipeline).fit(X_train_full.iloc[train_idx], y_train_full.iloc[train_idx])

    # Predict on validation fold
    oof_preds[val_idx, 0] = lgbm_fold.predict(X_train_full.iloc[val_idx])
    oof_preds[val_idx, 1] = xgb_fold.predict(X_train_full.iloc[val_idx])
    oof_preds[val_idx, 2] = rf_fold.predict(X_train_full.iloc[val_idx])

# Step 2: Train meta-model on out-of-fold predictions
meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_preds, y_train_full)

# Step 3: Generate test predictions
test_preds = np.column_stack([
    lgbm_pipeline.predict(X_test),
    xgb_pipeline.predict(X_test),
    rf_pipeline.predict(X_test)
])

pred_ensemble_stack = meta_model.predict(test_preds)

print(f"Meta-model weights: {meta_model.coef_}")
```

**Strategy 4: Voting Regressor (Sklearn)**

Use sklearn's built-in VotingRegressor:
```python
from sklearn.ensemble import VotingRegressor

ensemble_voting = VotingRegressor(
    estimators=[
        ('lgbm', lgbm_pipeline),
        ('xgb', xgb_pipeline),
        ('rf', rf_pipeline)
    ],
    weights=None  # Equal weights, or specify [w1, w2, w3]
)

ensemble_voting.fit(X_train_full, y_train_full)
pred_ensemble_voting = ensemble_voting.predict(X_test)
```

#### 5.4 Evaluation Protocol

Compare all 4 ensemble strategies + base models:

| Model | Test MAE | Test R² | Improvement |
|-------|----------|---------|-------------|
| LightGBM (base) | 3.14 W | 0.8942 | - |
| XGBoost (base) | 3.15 W | 0.8916 | - |
| Random Forest (base) | 3.25 W | 0.8787 | - |
| **Ensemble: Simple Avg** | ? | ? | ? |
| **Ensemble: Weighted Avg** | ? | ? | ? |
| **Ensemble: Stacking** | ? | ? | ? |
| **Ensemble: Voting** | ? | ? | ? |

Select the best-performing ensemble method.

#### 5.5 Success Criteria

- **Minimum:** 0.5% MAE improvement over best base model
- **Target:** 1-2% MAE improvement
- **Ensemble diversity:** Base models should have low correlation (<0.95)

#### 5.6 Diversity Analysis

Measure prediction correlation between base models:
```python
import seaborn as sns

# Correlation matrix of predictions
pred_df = pd.DataFrame({
    'LightGBM': pred_lgbm,
    'XGBoost': pred_xgb,
    'RandomForest': pred_rf
})

corr_matrix = pred_df.corr()
print("Prediction Correlation Matrix:")
print(corr_matrix)

# Visualize
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0.9)
plt.title('Base Model Prediction Correlation\n(Lower = More Diverse)')
plt.show()
```

**Interpretation:**
- Correlation < 0.90: Excellent diversity (ensemble will help)
- Correlation 0.90-0.95: Good diversity (ensemble may help)
- Correlation > 0.95: Low diversity (ensemble unlikely to help)

#### 5.7 Outputs

**Results:**
- `results/traditional_ml_time_ensemble_results.csv` - All ensemble strategies + base models

**Models:**
- `models/ensemble_stacking.pkl` - Best ensemble model (if stacking wins)
- `models/ensemble_voting.pkl` - Voting ensemble (for comparison)

**Analysis:**
- `results/ensemble_diversity_analysis.csv` - Prediction correlations
- `results/ensemble_weights.json` - Optimal weights for each strategy

**Visualizations:**
- Prediction correlation heatmap
- Ensemble weight comparison (bar plot)
- Error distribution comparison (violin plot)

#### 5.8 Implementation Notes

1. **Model Selection:**
   - Only include models with CV MAE within 5% of best model
   - Exclude poorly performing models (e.g., Linear Regression)

2. **Overfitting Prevention:**
   - Use out-of-fold predictions for stacking
   - Apply regularization to meta-model (Ridge, not OLS)
   - Monitor train vs test gap

3. **Computational Efficiency:**
   - Stacking is most expensive (requires 5-fold training)
   - Simple/weighted averaging is fast (just average predictions)

4. **Weight Interpretation:**
   - If one model gets weight ≈ 1.0, ensemble adds no value
   - Balanced weights indicate complementary models

---

## Experiment 6: Segmented Models (NEXT)

**Status:** 🔵 NEXT - Ready for implementation with strong evidence supporting the approach

**Objective:** Train specialized models for different network segments to capture location-specific and configuration-specific energy patterns.

**Notebook:** `scripts/traditional_ml_time_cell.ipynb` (TO BE IMPLEMENTED)

**Segmentation Approach:** The dataset has two orthogonal dimensions for segmentation:
1. **Cell-level (2 cells):** Cell0 (97.8% of data) vs Cell1 (2.2% of data, 54% worse performance)
2. **Base Station-level (816 BSs):** Cluster similar BSs into k homogeneous groups (k=3, 5, or 7)

**Strategies to Test:** 7 approaches ranging from simple (2 cell-specific models) to complex (hierarchical with fallbacks)

**Expected Improvement (Evidence-Based):**
- **Conservative:** 3-5% MAE reduction (expected test MAE: 2.98-3.05 W)
- **Target:** 9-12% MAE reduction (expected test MAE: 2.76-2.86 W)
- **Theoretical Maximum:** 36-42% MAE reduction (if all high-error BSs reach median level)

**Evidence Strength:** ✅ STRONG (CV=0.80, 54% performance gap on Cell1, 204 high-error BSs)

### Rationale

The current global model is trained on all network elements together (2 cells × 816 base stations). However, evidence shows significant heterogeneity:

**Cell-Level Heterogeneity:**
- Only 2 cells in the dataset (Cell0: 97.8%, Cell1: 2.2%)
- Cell1 has 54% worse performance (MAE 4.19W vs 2.72W)
- Fundamentally different energy-load relationships (correlation: 0.67 vs 0.30)

**Base Station Heterogeneity:**
- 816 unique base stations with massive performance variance (CV=0.80)
- Error range: 0.19W to 18.2W (100× difference!)
- Top 25% worst BSs (204 stations) contribute 53% of total error

Specialized models can capture these segment-specific patterns that the global model misses.

### Evidence for Segmentation Approach

**Analysis Performed:** Before proceeding with Experiment 6, a comprehensive analysis was conducted to validate whether cell/BS-specific models would actually improve performance. This evidence-based approach prevents wasted effort on segmentation if the global model already performs uniformly well.

#### Cell-Level Analysis

**Pattern Differences:**

| Metric | Cell0 (97.8%) | Cell1 (2.2%) | Difference |
|--------|---------------|--------------|------------|
| **Mean Energy** | 28.50 W | 34.89 W | +22% higher |
| **Mean Load** | 0.258 | 0.128 | -50% lower |
| **Energy-Load Correlation** | 0.67 | 0.30 | -55% weaker |
| **Hardware Configs** | 7 RUTypes, 3 Antenna configs | 3 RUTypes, 1 Antenna config | Different profiles |

**Statistical Significance:** T-test comparing Cell0 vs Cell1 energy distributions: t=-18.07, **p < 0.001** (highly significant difference)

**The Paradox:** Cell1 has **lower load** but **higher energy**, with much weaker load-energy correlation. This indicates fundamentally different energy consumption characteristics that the global model cannot capture.

#### Performance Gap Analysis

**Current Model Performance by Cell:**

| Cell | MAE | R² | Mean Error | Status |
|------|-----|-----|------------|--------|
| **Overall** | 2.75 W | 0.919 | - | Global model |
| **Cell0** | 2.72 W | 0.923 | +0.06 W | Good fit |
| **Cell1** | **4.19 W** | **0.635** | **+4.00 W** | ❌ **54% worse MAE** |

**Key Finding:** The global model has a **systematic +4W underprediction bias** on Cell1, with R² degrading from 0.92 to 0.63. This is because:
1. Cell1 represents only 2.2% of training data
2. The model learned Cell0's pattern (load-dominant)
3. Cell1's weak load correlation (0.30) breaks this assumption
4. Result: Consistent underprediction for Cell1 samples

#### Base Station Variance Analysis

**BS-Level Performance Statistics:**

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| **Mean BS MAE** | 2.72 W | Average across 816 BSs |
| **Std of BS MAE** | 2.18 W | High variance |
| **Coefficient of Variation** | **0.80** | ✓ **Very high (>0.2 threshold)** |
| **Min BS MAE** | 0.19 W | Best BS (B_423) |
| **Max BS MAE** | 18.20 W | Worst BS (B_745) |
| **Range** | 18.0 W | **100× difference!** |

**High-Error Base Stations:**
- **Top 25% worst BSs:** 204 BSs with average MAE = 5.76 W
- **Sample coverage:** 18,084 samples (24.9% of dataset)
- **Contribution to error:** 1.46 W of total 2.75 W MAE

**Top 10 Worst BSs:** B_745 (18.2W), B_105 (18.2W), B_85 (15.1W), B_382 (14.3W), B_97 (12.2W), B_728 (10.4W), B_223 (10.2W), B_43 (9.2W), B_471 (9.0W), B_499 (9.0W)

**Top 10 Best BSs:** B_423 (0.19W), B_679 (0.23W), B_424 (0.23W), B_495 (0.26W), B_800 (0.26W)

#### Improvement Potential Analysis

**Scenario 1: Cell-Specific Models (Conservative)**
- Improve Cell1 performance to Cell0 level (4.19 → 2.72 W)
- **Expected improvement:** 1.17%
- **Expected test MAE:** 3.10 W (down from 3.14 W)
- ✓ Meets 1% minimum threshold

**Scenario 2: BS Clustering - Conservative (Top 25% worst BSs)**
- Bring 204 worst BSs (MAE 5.76W) to median level (1.90W)
- **Expected improvement:** 9-12%
- **Expected test MAE:** 2.76-2.86 W
- ✓ ✓ Significantly exceeds 3% target, approaches 10% minimum success

**Scenario 3: BS Clustering - Aggressive (Top 50% worst BSs)**
- Improve 408 BSs (50% of all BSs) to median level
- **Theoretical maximum improvement:** 36-42%
- **Expected test MAE:** 1.83-2.01 W
- ✓ ✓ ✓ Would exceed 10% minimum success criteria

**Realistic Expectation:**
- Capture 25-33% of theoretical maximum
- Achieve **3-5% actual improvement** (conservative estimate)
- Expected test MAE: **2.98-3.05 W**

#### Conclusion from Evidence

**✅ STRONG EVIDENCE for proceeding with Experiment 6:**

1. **Cell-level:** Statistically significant differences (p<0.001), 54% performance gap
2. **BS-level:** Massive variance (CV=0.80), 100× error range between best/worst BSs
3. **Systematic errors:** 204 high-error BSs contribute 53% of total error
4. **Improvement potential:** 9-42% theoretical, 3-5% realistic
5. **Threshold validation:** Even conservative estimates exceed 1% minimum

**Key Insight:** The global model performs excellently on Cell0/low-error BSs but catastrophically on Cell1/high-error BSs. Segmentation directly addresses this heterogeneity.

**Strategy:** Prioritize BS clustering over cell-only models, as BS-level variance (CV=0.80) offers larger improvement potential than cell-level differences alone.

### Dataset Structure and Segmentation Dimensions

From `processed_data/netop_ml_time.csv` - 72,569 total samples:

**Dimension 1: Cell-Level (2 cells)**
- **Cell0:** 70,989 samples (97.8%) - Dominant cell with good model performance
- **Cell1:** 1,580 samples (2.2%) - Minority cell with 54% worse performance

**Dimension 2: Base Station-Level (816 base stations)**
- **Total BSs:** 816 unique stations distributed across 2 cells
- **Samples per BS:** 70-212 samples (mean: 89, median: 87)
- **Data sufficiency:** All BSs have ≥ 70 samples (adequate for training)

**Performance Heterogeneity:**
- **Cell-level gap:** Cell1 MAE (4.19W) is 54% worse than Cell0 (2.72W)
- **BS-level variance:** CV = 0.80 (very high, threshold is 0.2)
- **BS error range:** 0.19W to 18.2W (100× difference!)
- **High-error BSs:** 204 BSs (25%) have MAE > 4.9W, contributing 53% of total error

**Segmentation Strategy Rationale:**
The dataset has two orthogonal dimensions for segmentation:
1. **Cell-specific models (2 models):** Address the cell-level performance gap
2. **BS clustering (k models):** Address BS-level heterogeneity by grouping similar stations
3. **Hierarchical/Mixed:** Combine both dimensions for maximum flexibility

### Design Specifications

#### 6.1 Input Selection

**Baseline Model:** Experiment 2 (Time Features) - Default LightGBM
- **Rationale:**
  - Experiment 5 (Ensemble) was skipped per decision rules
  - Experiment 4 (Hyperparameter Tuning) achieved only 0.87% improvement
  - Simplicity is preferred when improvement is marginal
  - Exp 2 model: MAE = 3.14 W, R² = 0.8942

**Decision Made (per roadmap):**
```python
# Exp 4 improvement was 0.87% < 1% threshold
# Therefore: Skip Exp 5, use Exp 2 default model for Exp 6
base_model = lightgbm_time.pkl  # Experiment 2
baseline_test_mae = 3.14  # W
```

**Why not use Exp 4 tuned model?**
- Minimal gain (0.87%) doesn't justify added complexity
- If segmentation works, hyperparameter tuning can be applied afterward
- Start with simpler baseline to isolate segmentation benefit

#### 6.2 Segmentation Strategies

The experiment will test four complementary segmentation approaches, progressing from simple to complex:

**Strategy 1: Cell-Specific Models (2 models)**

Train one model per cell (2 cells total):
```python
# Cell0 model (97.8% of data)
df_cell0 = df_time[df_time['CellName'] == 'Cell0']
model_cell0 = clone(best_model).fit(X_cell0, y_cell0)

# Cell1 model (2.2% of data)
df_cell1 = df_time[df_time['CellName'] == 'Cell1']
model_cell1 = clone(best_model).fit(X_cell1, y_cell1)

# Prediction
if cell_name == 'Cell0':
    prediction = model_cell0.predict(X_new)
else:
    prediction = model_cell1.predict(X_new)
```

**Strategy 2: BS Clustering Models (k models for 816 base stations)**

Cluster the 816 base stations into k homogeneous groups and train one model per cluster:
```python
# Step 1: Extract BS features (aggregate statistics)
bs_features = df_time.groupby('BS').agg({
    'load': ['mean', 'std'],
    'Energy': ['mean', 'std'],
    'Frequency': 'mean',
    'Antennas': 'mean',
    'Bandwidth': 'mean',
    'TXpower': 'mean'
}).reset_index()

# Step 2: Cluster base stations (K-Means)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_bs = bs_features.drop('BS', axis=1)
scaler = StandardScaler()
X_bs_scaled = scaler.fit_transform(X_bs)

# Cluster 816 BSs into k groups (will test k=3, 5, 7)
kmeans = KMeans(n_clusters=5, random_state=42)  # Example: k=5
bs_clusters = kmeans.fit_predict(X_bs_scaled)

bs_features['Cluster'] = bs_clusters  # Each BS assigned to one of k clusters

# Step 3: Assign clusters to original data
df_time_clustered = df_time.merge(
    bs_features[['BS', 'Cluster']],
    on='BS',
    how='left'
)

# Step 4: Train model per cluster
models_by_cluster = {}
for cluster_id in range(5):
    df_cluster = df_time_clustered[df_time_clustered['Cluster'] == cluster_id]
    X_cluster = df_cluster[feature_cols]
    y_cluster = df_cluster['Energy']

    models_by_cluster[cluster_id] = clone(best_model).fit(X_cluster, y_cluster)

# Prediction
cluster_id = get_cluster_id(bs_name)  # Lookup BS cluster
prediction = models_by_cluster[cluster_id].predict(X_new)
```

**Strategy 3: Hierarchical Models (Cell → Cluster → Global)**

Use hierarchical approach with fallback:
```python
def predict_hierarchical(X_new, cell_name, bs_name):
    # Level 1: Try BS-specific model (if BS has > 100 samples)
    if bs_name in bs_specific_models:
        return bs_specific_models[bs_name].predict(X_new)

    # Level 2: Try Cell-specific model
    elif cell_name in cell_models:
        return cell_models[cell_name].predict(X_new)

    # Level 3: Fallback to global model
    else:
        return global_model.predict(X_new)
```

**Strategy 4: Mixed Effects Model (Global + Local Adjustments)**

Train global model + local correction terms:
```python
# Step 1: Train global model
global_model.fit(X_train, y_train)
global_preds = global_model.predict(X_train)

# Step 2: Calculate residuals per BS
residuals = y_train - global_preds
bs_residual_mean = df_train.groupby('BS')['residuals'].mean()

# Step 3: Prediction with adjustment
global_pred = global_model.predict(X_new)
bs_adjustment = bs_residual_mean.get(bs_name, 0)  # Default to 0 if BS unseen
final_pred = global_pred + bs_adjustment
```

#### 6.3 Optimal Cluster Selection

Determine optimal number of clusters using elbow method:
```python
from sklearn.metrics import silhouette_score

inertias = []
silhouettes = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_bs_scaled)

    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_bs_scaled, clusters))

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(2, 11), inertias, marker='o')
axes[0].set_xlabel('Number of Clusters')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True)

axes[1].plot(range(2, 11), silhouettes, marker='o')
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

#### 6.4 Evaluation Protocol

**Train-Test Split:** CRITICAL - must be temporal within each cell/cluster

```python
# CORRECT: Temporal split within each segment
for cell in ['Cell0', 'Cell1']:
    df_cell = df_time[df_time['CellName'] == cell]
    split_idx = int(len(df_cell) * 0.8)

    train_cell = df_cell.iloc[:split_idx]
    test_cell = df_cell.iloc[split_idx:]

    # Train model on this cell's training data
    # Evaluate on this cell's test data
```

**Metrics:**
- Overall MAE/R² (weighted by sample size)
- Per-cell MAE/R² (to identify which segments benefit)
- Improvement vs global model

#### 6.5 Comparison Matrix

**Evaluation Approach:**
- **Overall MAE/R²:** Performance across all test samples (weighted average)
- **Cell0/Cell1 MAE:** Performance on samples from each cell (to assess whether segmentation resolves the Cell1 imbalance)
- **Improvement:** MAE reduction vs global baseline

**Note on BS Clustering:** These strategies cluster 816 base stations into k homogeneous groups (k=3, 5, or 7). Each sample is predicted by the model corresponding to its base station's cluster. Cell0/Cell1 MAE breakdown shows whether BS clustering helps resolve the cell-level performance gap.

| Strategy | Segmentation | # Models | Overall MAE | Overall R² | Cell0 MAE | Cell1 MAE | Improvement |
|----------|--------------|----------|-------------|------------|-----------|-----------|-------------|
| **Global (baseline)** | None | 1 | 3.14 W | 0.8942 | 2.72 W | 4.19 W | - |
| **Cell-Specific** | By cell | 2 | ? | ? | ? | ? | ? |
| **BS Clustering (k=3)** | 816 BSs → 3 groups | 3 | ? | ? | ? | ? | ? |
| **BS Clustering (k=5)** | 816 BSs → 5 groups | 5 | ? | ? | ? | ? | ? |
| **BS Clustering (k=7)** | 816 BSs → 7 groups | 7 | ? | ? | ? | ? | ? |
| **Hierarchical** | BS → Cell → Global | Variable | ? | ? | ? | ? | ? |
| **Mixed Effects** | Global + BS adjustments | 1 + 816 | ? | ? | ? | ? | ? |

#### 6.6 Success Criteria

- **Minimum:** 1% MAE improvement over global model
- **Target:** 2-3% MAE improvement
- **Stretch:** 5% MAE improvement

**Segment-Specific Success:**
- At least ONE segment shows > 3% improvement
- No segment degrades by > 2% (complexity not worth it otherwise)

#### 6.7 Model Complexity Analysis

Track model complexity vs performance trade-off:
```python
complexity_df = pd.DataFrame({
    'Strategy': ['Global', 'Cell-Specific', 'BS-Cluster-3', 'BS-Cluster-5', 'Hierarchical'],
    'Num_Models': [1, 2, 3, 5, 8],
    'Test_MAE': [3.14, ?, ?, ?, ?],
    'Training_Time': [120, ?, ?, ?, ?],
    'Storage_MB': [5, 10, 15, 25, 40]
})

# Plot complexity vs performance
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(complexity_df['Num_Models'], complexity_df['Test_MAE'], s=100)
ax.set_xlabel('Number of Models (Complexity)')
ax.set_ylabel('Test MAE (W)')
ax.set_title('Model Complexity vs Performance Trade-off')
ax.invert_yaxis()  # Lower MAE is better

for i, row in complexity_df.iterrows():
    ax.annotate(row['Strategy'], (row['Num_Models'], row['Test_MAE']))

plt.grid(True, alpha=0.3)
plt.show()
```

**Decision Rule:**
- Only adopt cell/BS-specific approach if improvement justifies complexity
- If improvement < 1%, stick with global model

#### 6.8 Outputs

**Results:**
- `results/traditional_ml_time_cell_results.csv` - All segmentation strategies
- `results/cell_specific_performance.csv` - Per-cell/cluster breakdown

**Models:**
- `models/cell_specific/model_cell0.pkl`
- `models/cell_specific/model_cell1.pkl`
- `models/bs_clusters/model_cluster_0.pkl` (if clustering approach wins)
- `models/bs_clusters/model_cluster_1.pkl`
- ... (one model per cluster)

**Metadata:**
- `models/bs_clusters/bs_to_cluster_mapping.json` - BS → Cluster lookup
- `models/bs_clusters/cluster_scaler.pkl` - Scaler for clustering features

**Analysis:**
- `results/cluster_characteristics.csv` - Average features per cluster
- `results/complexity_analysis.csv` - Models, training time, storage size

**Visualizations:**
- BS clustering visualization (PCA 2D projection)
- Per-cluster performance comparison
- Complexity vs performance trade-off plot

#### 6.9 Implementation Notes

1. **Data Leakage Prevention:**
   - ALWAYS use temporal split within each segment
   - Do NOT cluster based on test set information
   - Freeze cluster assignments before train-test split

2. **Small Sample Handling:**
   - Cell1 has only 1,580 samples
   - Monitor for overfitting on small segments
   - Consider using global model for segments with < 500 samples

3. **Production Deployment:**
   - Requires BS/Cell lookup at inference time
   - More complex deployment than single global model
   - Document model selection logic clearly

4. **Cold Start Problem:**
   - New BS not seen in training: use global model
   - New BS in existing cluster: use cluster model
   - Document fallback strategy

---

## Overall Experiment Roadmap

### Sequential Execution Plan

```
Experiment 2 (✅ COMPLETE)
   └─> Time Features Baseline: MAE = 3.14 W, R² = 0.8942
           │
           ├─> Experiment 4: Hyperparameter Tuning (✅ COMPLETE)
           │      └─> Result: MAE = 3.11 W (0.87% improvement)
           │             │
           │             └─> ❌ BELOW 1% THRESHOLD ──> Skip Exp 5
           │
           ├─X Experiment 5: Ensemble Methods (⏭️ SKIPPED per decision rule)
           │      └─> Rationale: Exp 4 showed < 1% gain, tuning hit ceiling
           │
           └─> Experiment 6: Segmented Models (🔵 NEXT)
                  └─> Goal: MAE < 3.00 W (3% improvement needed)
                  └─> Baseline: Use Exp 2 default model (3.14 W)
                  └─> Test 2 cell-specific + 3 BS clustering strategies
                         │
                         ├─> If MAE < 3.00 W ──> DEPLOY SEGMENTED MODEL
                         ├─> If 3.00 ≤ MAE < 3.11 W ──> DEPLOY EXP 2 (simpler)
                         └─> If MAE ≥ 3.11 W ──> CONSIDER DEEP LEARNING
```

### Performance Targets Summary

| Experiment | MAE Target | R² Target | Improvement | Cumulative | Status |
|------------|------------|-----------|-------------|------------|--------|
| Exp 1: Baseline | 3.34 W | 0.8837 | - | - | ✅ COMPLETE |
| Exp 2: Time Features | 3.14 W | 0.8942 | 6.0% | 6.0% | ✅ COMPLETE |
| Exp 3: Full Features | 3.14 W | 0.8937 | 0.0% | 6.0% | ✅ COMPLETE |
| Exp 4: Hyperparameter Tuning | 3.11 W | 0.8982 | 0.87% | 6.87% | ✅ COMPLETE (⚠️ Below 1% target) |
| **Exp 5: Ensemble Methods** | **< 3.08 W** | **> 0.898** | **1%** | **9%** | ⏭️ SKIP (per decision rules) |
| **Exp 6: Segmented Models** | **< 3.00 W** | **> 0.905** | **3%** | **12%** | 🔵 NEXT |

### Success Criteria

**Minimum Success:** Achieve MAE < 3.00 W (10.2% total improvement over baseline)

**Target Success:** Achieve MAE < 2.95 W (11.7% total improvement over baseline)

**Stretch Success:** Achieve MAE < 2.85 W (14.7% total improvement over baseline)

### Decision Points

1. **After Experiment 4:** ✅ **EXECUTED**
   - **Result:** Improvement = 0.87% (< 1% threshold)
   - **Decision:** Skip Experiment 5 (Ensemble), proceed directly to Experiment 6
   - **Rationale:**
     - Hyperparameter tuning showed minimal gains (below 1% target)
     - Default hyperparameters already near-optimal
     - Ensemble unlikely to provide significant improvement if tuning didn't
     - Better to test segmentation approach (different dimension of optimization)

2. **After Experiment 5:** ⏭️ **SKIPPED**
   - Experiment 5 was skipped per decision rule #1
   - Proceed directly to Experiment 6 using Experiment 2 default model as baseline

3. **After Experiment 6:** 🔵 **PENDING**
   - If segmented models add ≥ 1% value: Deploy segmented approach
   - If segmented models add < 1% value: Deploy Exp 2 default model (3.14 W)
   - Select best segmentation strategy (cell-specific vs BS clustering vs hybrid)
   - Document complexity vs. performance trade-off

4. **Final Decision:** 🔵 **PENDING**
   - If Exp 6 achieves MAE < 3.00 W (10% improvement): Deploy traditional ML
   - If Exp 6 fails to reach MAE < 3.00 W: Consider deep learning models
   - Current best: 3.11 W (6.87% improvement) - need 3% more for 10% target

---

## Implementation Timeline

**Completed:**
1. ✅ **Experiment 4 (Hyperparameter Tuning):** Completed - 6 hours implementation + 3 hours computation
   - Result: 0.87% improvement (below 1% target)
   - Decision: Skip Experiment 5, proceed to Experiment 6

**Skipped:**
2. ⏭️ **Experiment 5 (Ensemble Methods):** Skipped per decision rule (Exp 4 < 1% improvement)

**Remaining:**
3. 🔵 **Experiment 6 (Segmented Models):** ~3-4 hours implementation + 2-3 hours computation
   - Use Experiment 2 default model as baseline
   - Test 2 cell-specific + 3 BS clustering + 2 hybrid strategies (7 total approaches)

**Total Actual Effort (so far):** 6 hours implementation + 3 hours computation

**Estimated Remaining:** 3-4 hours implementation + 2-3 hours computation

---

## Key Design Principles

1. **Consistency:** All experiments use same data, same temporal split, same evaluation metrics
2. **Fairness:** Always compare against proper baseline (Exp 2 Time Features)
3. **Transparency:** Document all decisions, parameters, and trade-offs
4. **Reproducibility:** Set random seeds, save all results and models
5. **Practicality:** Consider production complexity vs performance gain trade-off
6. **Sequential Improvement:** Each experiment builds on the best previous result
7. **Fail-Safe:** Always fall back to simpler model if complex approach doesn't help
8. **Decision-Driven:** Use performance thresholds to decide whether to proceed with next experiment

---

## Current Status

**Experiments Status:**
- ✅ **Completed:** Experiments 1-4 (Baseline, Time Features, Full Features, Hyperparameter Tuning)
- ⏭️ **Skipped:** Experiment 5 (Ensemble Methods) - per decision rules
- 🔵 **Next:** Experiment 6 (Segmented Models) - strong evidence for 3-12% improvement

**Best Model So Far:**
- **Option A (Recommended):** Experiment 2 default LightGBM (MAE = 3.14 W, R² = 0.8942)
  - Simplest model, easiest to maintain
  - Meets all success criteria
- **Option B (Marginal Gain):** Experiment 4 tuned LightGBM (MAE = 3.11 W, R² = 0.8982)
  - 0.87% better performance
  - Added complexity for minimal gain

**Key Insights:**
1. **Feature set is the limiting factor** - Not model hyperparameters
2. **Default hyperparameters are already near-optimal** - Tuning yielded only 0.87% gain
3. **Segmentation shows promise** - Cell1 has 54% worse performance, 204 high-error BSs identified
4. **Next step is critical** - Need 3% improvement from Experiment 6 to reach 10% target

**Next Action:** Implement Experiment 6 (Segmented Models) to test cell-level and BS-level segmentation approaches.

**Target:** Achieve MAE < 3.00 W (10% total improvement over baseline) to meet minimum success criteria.
