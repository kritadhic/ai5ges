  **Join Analysis Summary**

  **Original Data:**

  - **CLstat:** 125,575 rows (cell-level load data)
  - **ECstat:** 92,629 rows (energy consumption data)
  - **BSinfo:** 1,217 rows (base station metadata)

  **Join Results:**

  **Step 1: CLstat + BSinfo (on BS + CellName)**
  - **✓ No data loss:** 125,575 rows → 125,575 rows
  - All CLstat records matched perfectly with BSinfo (as expected for static metadata)

  **Step 2: Result + ECstat (on Time + BS)**
  - **✓ No rows dropped in join:** 125,575 rows → 125,575 rows (LEFT join preserves all)
  - **⚠️ But created NaN values:** 27,491 rows have missing Energy values (21.9%)

  **Why NaN Values Occurred:**

  **1. Unmatched Time+BS combinations (27,491 rows - 21.9%)**
    - CLstat has 118,768 unique (Time, BS) combinations
    - ECstat has only 92,629 unique (Time, BS) combinations
    - Gap: 26,139 combinations exist in CLstat but not in ECstat
    - These create NaN in the Energy column
  **2. Lagged features creating NaN**
    - load_lag24: 29,208 NaN (23.3%) - first 24 hours per cell group
    - energy_lag24: 50,600 NaN (40.3%) - first 24 hours per BS group + unmatchedrows

  **Final Data Loss After dropna():**

  72,569 rows retained (57.8% of original)
  53,006 rows dropped (42.2% of original)

  **Breakdown of Dropped Rows:**

  **- ~27,491 rows: Missing ECstat data (Time+BS not in ECstat)**
  **- ~25,515 rows: Additional rows with NaN from 24-hour lag features**

  **Conclusion:**

  **Yes, significant data fell out of the join!** The LEFT join itself preserved all rows, but the subsequent dropna() removed:
  1. Records where ECstat didn't have matching Time+BS data
  2. Records where lagged features (especially 24-hour lags) created NaN values

  This is a 42% data loss, which is significant but expected given:
  - ECstat has less temporal coverage than CLstat
  - Lag features inherently create NaN for initial time periods