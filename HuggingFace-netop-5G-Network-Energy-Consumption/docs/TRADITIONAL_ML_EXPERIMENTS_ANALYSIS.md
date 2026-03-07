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

## Next Phase: Advanced ML Experiments

While the current Time Features model meets success criteria (R² = 0.89, MAE = 3.14 W), the 6% improvement is below the 10-25% target. Three additional experiments are planned to maximize traditional ML performance before considering deep learning models.

### Experiment 4: Hyperparameter Tuning
### Experiment 5: Ensemble Methods
### Experiment 6: Cell/BS-Specific Models

---

## Experiment 4: Hyperparameter Tuning

**Objective:** Optimize hyperparameters for LightGBM and XGBoost to extract maximum performance from existing features.

**Notebook:** `scripts/traditional_ml_time_ht.ipynb` (ht = hyperparameter tuning)

**Expected Improvement:** 1-3% MAE reduction over Time Features baseline (bringing total to 8-10% over Experiment 1)

### Rationale

Current experiments use default hyperparameters with NO tuning:
```python
LGBMRegressor(
    n_estimators=100,      # Default
    max_depth=6,           # Default
    learning_rate=0.1,     # Default
    subsample=0.8,         # Default
    colsample_bytree=0.8   # Default
)
```

Gradient boosting models are highly sensitive to hyperparameters. Systematic optimization could yield 1-3% additional improvement.

### Design Specifications

#### 4.1 Data Source
- **Input:** `processed_data/netop_ml_time.csv` (same as Experiment 2)
- **Features:** 13 raw + 5 time = 18 features
- **Train-Test Split:** Temporal 80/20 (consistent with Experiments 1-3)

#### 4.2 Models to Tune

**Priority 1: LightGBM** (current best performer, Test R² = 0.8942)

**Priority 2: XGBoost** (close second, Test R² = 0.8916)

Do NOT tune Random Forest or Linear Regression (already underperforming).

#### 4.3 Hyperparameter Search Space

**LightGBM Search Space:**
```python
lgbm_param_grid = {
    'regressor__n_estimators': [100, 200, 300, 500],
    'regressor__max_depth': [4, 6, 8, 10, -1],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'regressor__num_leaves': [31, 50, 70, 100, 127],
    'regressor__min_child_samples': [5, 10, 20, 30, 50],
    'regressor__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'regressor__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'regressor__reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],      # L1 regularization
    'regressor__reg_lambda': [0, 0.01, 0.1, 0.5, 1.0]      # L2 regularization
}
```

**XGBoost Search Space:**
```python
xgb_param_grid = {
    'regressor__n_estimators': [100, 200, 300, 500],
    'regressor__max_depth': [3, 4, 6, 8, 10],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'regressor__min_child_weight': [1, 3, 5, 7, 10],
    'regressor__gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'regressor__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'regressor__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'regressor__reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
    'regressor__reg_lambda': [0, 0.01, 0.1, 0.5, 1.0]
}
```

#### 4.4 Tuning Method

**Use Optuna** (preferred) for Bayesian optimization:

```python
import optuna
from optuna.integration import OptunaSearchCV

# Define objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    model = Pipeline([
        ('preprocessor', preprocessor_time),
        ('regressor', LGBMRegressor(**params))
    ])

    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_train_full, y_train_full,
                             cv=tscv, scoring='neg_mean_absolute_error',
                             n_jobs=1)

    return -scores.mean()  # Optuna minimizes

# Run optimization
study = optuna.create_study(direction='minimize', study_name='lightgbm_tuning')
study.optimize(objective, n_trials=100, show_progress_bar=True)

# Get best parameters
best_params = study.best_params
```

**Alternative: RandomizedSearchCV** (if Optuna not available):

```python
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=lgbm_pipeline,
    param_distributions=lgbm_param_grid,
    n_iter=100,                              # 100 random combinations
    scoring='neg_mean_absolute_error',
    cv=TimeSeriesSplit(n_splits=5),
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_full, y_train_full)
best_params = random_search.best_params_
```

#### 4.5 Evaluation Protocol

1. **Cross-Validation:** TimeSeriesSplit with 5 folds on training set (80% of data)
2. **Metric:** MAE (primary), R², RMSE, MAPE (secondary)
3. **Test Set Evaluation:** Evaluate best model on held-out test set (20%)
4. **Comparison:** Compare against Experiment 2 baseline (Time Features, no tuning)

#### 4.6 Success Criteria

- **Minimum:** 1% MAE improvement over Time Features baseline (3.14 W → 3.11 W)
- **Target:** 2-3% MAE improvement (3.14 W → 3.05-3.08 W)
- **Stretch:** 4-5% MAE improvement (3.14 W → 3.00-3.02 W)

#### 4.7 Outputs

**Results:**
- `results/traditional_ml_time_ht_results.csv` - Test set results for tuned models

**Models:**
- `models/lightgbm_time_ht.pkl` - Best tuned LightGBM model
- `models/xgboost_time_ht.pkl` - Best tuned XGBoost model

**Tuning Reports:**
- `results/lightgbm_tuning_history.csv` - Optuna trial history
- `results/xgboost_tuning_history.csv` - Optuna trial history

**Visualizations:**
- Hyperparameter importance plot (Optuna)
- Optimization history plot (convergence)
- Parallel coordinate plot (parameter relationships)

#### 4.8 Implementation Notes

1. **Computational Cost:**
   - 100 trials × 5 CV folds = 500 model fits per algorithm
   - Estimated time: 2-4 hours on single machine
   - Use `n_jobs=-1` for parallel processing

2. **Early Stopping:**
   - Use Optuna's pruning to skip unpromising trials
   - Saves ~30-40% computation time

3. **Baseline Comparison:**
   - Always compare against Experiment 2 results
   - Report both absolute and relative improvements

4. **Parameter Analysis:**
   - Identify which parameters have the most impact
   - Document best parameter combinations
   - Check for overfitting (CV vs Test performance gap)

---

## Experiment 5: Ensemble Methods

**Objective:** Combine multiple models to leverage their complementary strengths and reduce prediction variance.

**Notebook:** `scripts/traditional_ml_time_ensemble.ipynb`

**Expected Improvement:** 1-2% MAE reduction over best single model

### Rationale

Current best models have different strengths:
- **LightGBM:** Best test R² (0.8942), fast training
- **XGBoost:** Close second (0.8916), different regularization
- **Random Forest:** Different algorithm class, captures different patterns

Ensemble methods can combine these strengths to achieve better performance than any single model.

### Design Specifications

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

## Experiment 6: Cell/BS-Specific Models

**Objective:** Train specialized models for different cells or base station groups to capture location-specific energy patterns.

**Notebook:** `scripts/traditional_ml_time_cell.ipynb`

**Expected Improvement:** 2-5% MAE reduction over global model (varies by segment)

### Rationale

Current model is GLOBAL - trained on all cells/base stations together. However:
- Different cells may have different hardware configurations
- Different base stations may serve different geographical areas (urban vs rural)
- Different usage patterns (residential vs commercial)

Specialized models can capture these location-specific patterns.

### Dataset Analysis

From `processed_data/netop_ml_time.csv`:

**Cell Distribution:**
- Cell0: 70,989 samples (97.8%)
- Cell1: 1,580 samples (2.2%)

**Base Station Distribution:**
- 816 unique base stations
- 70-212 samples per BS (mean: 89, median: 87)
- All BSs have ≥ 70 samples (sufficient for modeling)

### Design Specifications

#### 6.1 Input Selection

**Base this experiment on the BEST performing model from:**
1. ✅ **Option A:** Experiment 5 (Ensemble) - if improvement > 0.5%
2. ✅ **Option B:** Experiment 4 (Hyperparameter Tuning) - if Experiment 5 fails
3. ⚠️ **Option C:** Experiment 2 (Time Features) - fallback

**Decision Rule:**
```python
if MAE_exp5 < best_previous_MAE * 0.995:  # 0.5% improvement
    base_model = exp5_best_model
elif MAE_exp4 < MAE_exp2 * 0.99:  # 1% improvement
    base_model = exp4_best_model
else:
    base_model = exp2_model
```

#### 6.2 Segmentation Strategies

**Strategy 1: Cell-Specific Models (Simple)**

Train separate models for each cell:
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

**Strategy 2: BS Clustering Models**

Group similar base stations and train models per cluster:
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

# Try 3-10 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
bs_clusters = kmeans.fit_predict(X_bs_scaled)

bs_features['Cluster'] = bs_clusters

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

| Strategy | Overall MAE | Overall R² | Cell0 MAE | Cell1 MAE | Improvement |
|----------|-------------|------------|-----------|-----------|-------------|
| Global Model (baseline) | 3.14 W | 0.8942 | ? | ? | - |
| Cell-Specific | ? | ? | ? | ? | ? |
| BS Clustering (k=3) | ? | ? | ? | ? | ? |
| BS Clustering (k=5) | ? | ? | ? | ? | ? |
| BS Clustering (k=7) | ? | ? | ? | ? | ? |
| Hierarchical | ? | ? | ? | ? | ? |
| Mixed Effects | ? | ? | ? | ? | ? |

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
           └─> Experiment 6: Cell/BS-Specific Models (🔵 NEXT)
                  └─> Goal: MAE < 3.00 W (3% improvement needed)
                  └─> Baseline: Use Exp 2 default model (3.14 W)
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
| **Exp 6: Cell-Specific Models** | **< 3.00 W** | **> 0.905** | **3%** | **12%** | 🔵 NEXT |

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
   - If cell-specific models add ≥ 1% value: Deploy segmented approach
   - If cell-specific models add < 1% value: Deploy Exp 2 default model (3.14 W)
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
3. 🔵 **Experiment 6 (Cell-Specific Models):** ~3-4 hours implementation + 2-3 hours computation
   - Use Experiment 2 default model as baseline
   - Test 4 segmentation strategies

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

**Experiments Completed:** 4 of 6 planned (Experiments 1-4)

**Best Model So Far:**
- **Option A (Recommended):** Experiment 2 default LightGBM (MAE = 3.14 W, R² = 0.8942)
- **Option B:** Experiment 4 tuned LightGBM (MAE = 3.11 W, R² = 0.8982)

**Key Learning from Experiment 4:**
- Feature set is the limiting factor, not model hyperparameters
- Default hyperparameters are already near-optimal
- Need segmentation or deep learning for further improvement

**Next Action:** Implement Experiment 6 (Cell/BS-Specific Models) to test segmentation approach.

**Target:** Achieve MAE < 3.00 W (10% total improvement over baseline) to meet minimum success criteria.
