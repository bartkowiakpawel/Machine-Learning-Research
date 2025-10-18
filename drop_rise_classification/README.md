# Drop Rise Classification

This package collects our experiments on predicting three-state price movements: drop, flat, or rise. Each case script represents a different modelling approach, from single-ticker baselines to multi-ticker time-series pipelines.

What lives here:
- Reusable dataset preparation utilities that create consistent train/test splits and engineered features.
- Classification and regression scenarios that compare classical machine learning models on Tesla and other tickers.
- Timeseries-aware workflows that respect chronological ordering and rolling retraining constraints.
- Outputs with datasets, metrics, and visual snapshots saved under `outputs/` for later review.

Use this package when you want to benchmark classification strategies or reuse prepared labels in other research tracks. Detailed instructions for specific experiments sit inside the case files.
