<!-- README for the rsi_calibration package -->

# RSI Calibration Workflows

The `rsi_calibration` package gathers experimental pipelines for calibrating the RSI indicator and using it to classify drop/flat/rise movements. Below you will find a short guide and an overview of the available cases.

## 1. Data preparation (required before running any case)

1. **Download TSLA data from Yahoo Finance**  
   Run the script:
   ```bash
   python rsi_calibration/fetch_tsla_data.py
   ```
   By default it stores 10 years of daily quotes in `rsi_calibration/data/tsla_yf_dataset.csv`.  
   If you want to save the file elsewhere:
   ```bash
   python -c "from rsi_calibration.fetch_tsla_data import fetch_tsla; fetch_tsla(output_path='path/to/file.csv')"
   ```

2. **Optional:** Feed that file into other pipelines (for example by copying it to `data/ml_input/ml_dataset.csv`) when you want to work with a fresh date range. Otherwise the standard cases will use the current dataset from `data/ml_input`.

## 2. Available cases

### Case 1: `case_1_tsla_rsi_baseline.py`
- Builds a simple feature set based only on RSI (14 periods by default).
- Uses `rsi_calibration/data/tsla_yf_dataset.csv` by default; switch to a different dataset with the `--dataset` flag.
- Creates drop/flat/rise labels based on the neutral band (`DEFAULT_NEUTRAL_BAND`).
- Splits the data chronologically into train/test (80/20).
- Trains 10 classifiers (LR, SVC, KNN, GaussianNB, RandomForest, ExtraTrees, GradientBoosting, AdaBoost, DecisionTree, MLP) on the same split.
- Saves:
  - A dataset snapshot (`*_rsi_dataset.csv`) marking which rows are in train or test.
  - Classification metrics per class (`*_metrics.csv`): Brier score, log loss, ROC AUC (when available), mutual information, accuracy, macro precision/recall/F1.
  - A PNG results table when the `great_tables` library (and a working local WebDriver) is available.
- Note: some metrics (for example AUC) may show `NaN` when the selected test slice contains only one label.

### Case 2: `case_2_tsla_target_distribution.py`
- Analyses the drop/flat/rise label distribution for the same baseline RSI feed.
- Summarises the observation counts and percentage share per class (`*_target_distribution.csv`).
- Renders a table (PNG + HTML) with a header and interpretation notes (requires `great_tables`).
- Generates a Seaborn-style pie chart (`*_target_distribution_pie.png`) showing class shares.
- Uses the same input parameters (ticker, neutral band, dataset) as Case 1, so comparisons stay straightforward.

### Case 3: `case_3_tsla_rsi_multihorizon.py`
- Compares how well RSI-14 classifies drop/flat/rise for several forward-return horizons (5/14/21 days by default).
- Keeps the chronological train/test split and reuses the model roster from Case 1.
- Stores detailed metrics per horizon (`*_metrics.csv`), a data snapshot (`*_dataset_snapshot.csv`), and an accuracy summary (`*_horizon_summary.csv` plus PNG/HTML table).

### Case 4: `case_4_tsla_rsi_heatmap.py`
- Builds a heatmap showing the average future return by RSI bucket and horizon (5/14/21 days by default, with emphasis on the 21-day target).
- Outputs include the pivot table (`*_heatmap_data.csv`), observation counts per bucket (`*_bucket_counts.csv`), a PNG/HTML table, and a heatmap PNG.
- Helps you quickly check, for example, whether RSI < 25 leads to positive 21-day results.

### Case 6: `case_6_tsla_rsi_period_bias.py`
- Compares the mean bias of future returns for different RSI periods across 1/7/14/21-day horizons.
- Produces summaries in wide and long formats (`*_bias_summary.csv`, `*_bias_long.csv`), a ranking of the best and weakest periods (`*_bias_rankings.csv`), and prints the top/bottom period for each horizon. Display-ready GT table versions are also saved as CSV (`*_bias_table_display.csv`, `*_bias_rankings_display.csv`).
- When `great_tables` is installed, additional PNG/HTML tables with RSI period comparisons and rankings are generated.
- Forces deterministic computations by setting `np.random.seed(42)`.

### Case 7: `case_7_tsla_rsi_period_model_comparison.py`
- Reuses the Case 1/3 model roster to compare performance for RSI 5/14/21/50 across 1/7/14/21-day targets.
- Outputs complete metrics (`*_metrics.csv`), a dataset snapshot (`*_dataset_snapshot.csv`), a list of top model+RSI+target combinations (`*_best_combos.csv`), and pivoted accuracy/model tables (`*_best_accuracy_pivot.csv`, `*_best_accuracy_display.csv`, `*_best_model_display.csv`, `*_horizon_top_models.csv`).
- When `great_tables` is available, exports PNG/HTML tables with accuracy matrices and model rankings for every configuration.

### Case 8: `case_8_tsla_rsi50_bucket_patterns.py`
- Focuses on RSI-50 and the 21-day horizon, reusing the same RSI buckets as Case 5.
- Saves the detailed drop/rise breakdown for the 21-day horizon (`*_rsi50_h21_patterns.csv`), a combined view (`*_rsi50_combined_patterns.csv`), and a long format (`*_rsi50_long.csv`).
- With `great_tables` installed, produces a PNG/HTML table for the buckets and highlights key statistics (for example, the largest drop/rise share).

### Case 9: `case_9_tsla_rsi_bucket_distribution.py`
- Builds the observation distribution in classic RSI buckets (<30, [30, 70), >=70) for periods 7/14/21/50 and horizons 1/5/14/21 days.
- Saves the full long view (`*_bucket_counts_long.csv`) and per-period pivots with counts and shares (`*_rsi*_count_pivot.csv`, `*_rsi*_share_pivot.csv`).
- When `great_tables` is enabled, renders PNG/HTML tables showing count and share matrices; logs also print the observation totals per horizon.

### (Optional) `case_1_metrics_table_demo.py`
- Helper renderer that turns any `*_metrics.csv` file into an additional PNG+HTML table with a header and interpretation notes.
- Also writes a column glossary (`*_metrics_glossary.csv/html`) explaining the meaning of each metric.
- Example run:
  ```bash
  python rsi_calibration/case_1_metrics_table_demo.py \
      --metrics rsi_calibration/outputs/case_1/tsla_case_1_metrics.csv
  ```

## 3. Practical notes

- All cases keep a consistent train/test split so you can compare results after any modifications.
- When extending the package, consider adding new case scripts using the `case_2_...`, `case_3_...` pattern and documenting each experiment in this README.
- If you experiment with different models or metrics, make sure the train/test records stay separate - ideally change only the pipeline while leaving the data split untouched.

Happy calibrating!
