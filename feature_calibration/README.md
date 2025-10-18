# Feature Calibration

This package mirrors the workflow of `rsi_calibration`, but we focus on ranking and calibrating the most informative features using feature importance (FI) and, above all, mutual information (MI). The goal is to understand which engineered signals generalise best across different modelling horizons.

Early roadmap:
- Re-run the RSI calibration experiments with MI-driven feature selection to see how the rankings change when the target signal is not an oscillator.
- Compare top features under MI vs. tree-based feature importance and document stability across rolling windows.
- Generate the same set of outputs as the RSI package (datasets, metrics, tables, plots) so that results stay comparable across research tracks.
- Track experiments in dedicated case scripts (for example `case_1_tsla_mi_baseline.py`, `case_2_multiticker_mi_screening.py`) as we expand beyond RSI features.

Practical notes:
- The package ships with its own `fetch_tsla_data.py` so it can download Yahoo Finance data without importing helpers from other modules.
- Use the `data/` folder for raw inputs and `outputs/` for experiment artefacts, mirroring the conventions used in other research packages.
- Detailed instructions for each experiment will live inside the case files once they are implemented.

This package will grow alongside the rest of the learning plan, giving us a dedicated space to explore MI-first calibration strategies.
