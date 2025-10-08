"""Case study 5: compare RFECV-selected features with exhaustive combinations using RobustScaler."""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Iterable, Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from config import ML_INPUT_DIR
from core.shared_utils import (
    clean_feature_matrix,
    load_ml_dataset,
    prepare_feature_matrix,
    resolve_output_dir,
    save_dataframe,
    write_case_metadata,
)
from feature_selection_methods.settings import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_FEATURES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TARGET,
    DEFAULT_TARGET_SOURCE,
    FEATURE_SELECTION_CASES,
)

CASE_ID = "case_5"
CASE_CONFIG = FEATURE_SELECTION_CASES[CASE_ID]
CASE_NAME = CASE_CONFIG.name
CASE_METHOD = CASE_CONFIG.method
MAX_SAMPLE_SIZE = 2500


def _filter_tickers(df: pd.DataFrame, tickers: Sequence[str] | None) -> pd.DataFrame:
    if not tickers:
        return df
    upper = {ticker.upper() for ticker in tickers}
    return df[df["ticker"].astype(str).str.upper().isin(upper)].copy()


def _build_feature_combinations(features: Sequence[str]) -> Iterable[tuple[str, ...]]:
    for r in range(1, len(features) + 1):
        for combo in itertools.combinations(features, r):
            yield combo


def _pipeline_feature_importances(estimator: Pipeline) -> np.ndarray:
    model = estimator.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        raise AttributeError("Underlying model does not expose feature_importances_.")
    return model.feature_importances_


def _evaluate_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    feature_subset: Sequence[str],
    cv: TimeSeriesSplit,
    estimator: Pipeline,
) -> dict[str, float | int | str]:
    mae_scores: list[float] = []
    rmse_scores: list[float] = []
    r2_scores: list[float] = []

    for train_idx, test_idx in cv.split(X):
        X_train = X.iloc[train_idx, :].loc[:, feature_subset]
        X_val = X.iloc[test_idx, :].loc[:, feature_subset]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[test_idx]

        model = clone(estimator)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        mae_scores.append(mean_absolute_error(y_val, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        r2_scores.append(r2_score(y_val, preds))

    mae_arr = np.asarray(mae_scores)
    rmse_arr = np.asarray(rmse_scores)
    r2_arr = np.asarray(r2_scores)

    return {
        "feature_set": ";".join(feature_subset),
        "n_features": int(len(feature_subset)),
        "mean_mae": float(mae_arr.mean()),
        "std_mae": float(mae_arr.std(ddof=0)),
        "mean_rmse": float(rmse_arr.mean()),
        "std_rmse": float(rmse_arr.std(ddof=0)),
        "mean_r2": float(r2_arr.mean()),
        "std_r2": float(r2_arr.std(ddof=0)),
    }


def run_case(
    *,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    tickers: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
    target: str = DEFAULT_TARGET,
    target_source: str | None = DEFAULT_TARGET_SOURCE,
    output_root: Path | str | None = None,
) -> Path:
    """Compare RFECV-selected features with exhaustive combinations using RobustScaler preprocessing."""

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    dataset_path = ML_INPUT_DIR / dataset_filename
    selected_tickers = list(tickers) if tickers is not None else list(CASE_CONFIG.tickers or ())
    feature_list = list(features) if features is not None else list(DEFAULT_FEATURES)

    print(f"=== Running {CASE_ID}: {CASE_NAME} ===")
    print(f"Method: {CASE_METHOD}")
    print(f"Dataset: {dataset_path}")
    print(f"Tickers: {', '.join(selected_tickers) if selected_tickers else 'ALL'}")
    print(f"Features ({len(feature_list)}): {', '.join(feature_list)}")

    df = load_ml_dataset(dataset_path)
    df = _filter_tickers(df, selected_tickers)

    if df.empty:
        print("No rows available after filtering tickers; aborting case.")
        return case_output_dir

    available_columns = set(df.columns)
    missing_features = [feature for feature in feature_list if feature not in available_columns]
    if missing_features:
        print("Warning: dropping missing features - " + ", ".join(missing_features))
        feature_list = [feature for feature in feature_list if feature in available_columns]

    if not feature_list:
        print("No usable features remain after dropping missing columns; aborting case.")
        return case_output_dir

    matrix = prepare_feature_matrix(df, feature_list, target, target_source)
    matrix = clean_feature_matrix(matrix)

    if matrix.empty:
        print("No clean rows available after preprocessing; aborting case.")
        return case_output_dir

    if len(matrix) > MAX_SAMPLE_SIZE:
        matrix = matrix.tail(MAX_SAMPLE_SIZE)
        print(f"Sample truncated to the most recent {MAX_SAMPLE_SIZE} rows for runtime control.")

    X = matrix[feature_list]
    y = matrix[target]

    n_splits = min(5, max(3, len(matrix) // 120))
    if n_splits < 3:
        n_splits = 3

    if n_splits >= len(matrix):
        print("Insufficient observations for cross-validation; aborting case.")
        return case_output_dir

    cv = TimeSeriesSplit(n_splits=n_splits)
    base_estimator = GradientBoostingRegressor(random_state=42)
    estimator = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("model", base_estimator),
        ]
    )

    print(f"Running RFECV (RobustScaler + GradientBoosting) over {len(feature_list)} features with {n_splits} splits...")
    rfecv_start = time.perf_counter()
    selector = RFECV(
        estimator=estimator,
        step=1,
        min_features_to_select=1,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=None,
        importance_getter=_pipeline_feature_importances,
    )
    selector.fit(X, y)
    rfecv_duration = time.perf_counter() - rfecv_start
    print(f"RFECV completed in {rfecv_duration:.2f} seconds.")

    selected_features = [feature for feature, keep in zip(feature_list, selector.support_) if keep]

    if not selected_features:
        print("RFECV did not retain any features; defaulting to best ranked feature.")
        min_rank = selector.ranking_.min()
        selected_features = [
            feature for feature, rank in zip(feature_list, selector.ranking_) if rank == min_rank
        ]

    rfecv_metrics = _evaluate_feature_set(X, y, selected_features, cv, estimator)
    rfecv_metrics["selection_method"] = "RFECV_Robust"
    rfecv_metrics["optimal_feature_count"] = int(getattr(selector, "n_features_", len(selected_features)))

    rfecv_df = pd.DataFrame([rfecv_metrics])
    rfecv_csv = save_dataframe(rfecv_df, case_output_dir, "case_5_rfecv_metrics.csv")
    print(f"RFECV feature metrics saved to: {rfecv_csv}")

    print("Evaluating all feature combinations (brute force) with RobustScaler...")
    brute_start = time.perf_counter()
    combo_results: list[dict[str, float | int | str]] = []
    for combo in _build_feature_combinations(feature_list):
        metrics = _evaluate_feature_set(X, y, combo, cv, estimator)
        metrics["selection_method"] = "brute_force_robust"
        combo_results.append(metrics)
    brute_duration = time.perf_counter() - brute_start
    print(f"Brute-force evaluation completed in {brute_duration:.2f} seconds.")

    combos_df = pd.DataFrame(combo_results)
    combos_df.sort_values("mean_mae", inplace=True)
    combos_csv = save_dataframe(combos_df, case_output_dir, "case_5_bruteforce_metrics.csv")
    print(f"Brute-force combination metrics saved to: {combos_csv}")

    top_bruteforce = combos_df.iloc[0]
    comparison_df = pd.DataFrame(
        [
            {
                "selection_method": "RFECV_Robust",
                "feature_set": rfecv_metrics["feature_set"],
                "mean_mae": rfecv_metrics["mean_mae"],
                "mean_rmse": rfecv_metrics["mean_rmse"],
                "mean_r2": rfecv_metrics["mean_r2"],
            },
            {
                "selection_method": "Brute-force best (Robust, MAE)",
                "feature_set": top_bruteforce["feature_set"],
                "mean_mae": top_bruteforce["mean_mae"],
                "mean_rmse": top_bruteforce["mean_rmse"],
                "mean_r2": top_bruteforce["mean_r2"],
            },
        ]
    )

    comparison_path = save_dataframe(comparison_df, case_output_dir, "case_5_method_comparison.csv")
    print(f"Method comparison summary saved to: {comparison_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = comparison_df["selection_method"]
    mae_values = comparison_df["mean_mae"]
    rmse_values = comparison_df["mean_rmse"]
    r2_values = comparison_df["mean_r2"]

    x = np.arange(len(methods))
    width = 0.25
    ax.bar(x - width, mae_values, width=width, label="MAE")
    ax.bar(x, rmse_values, width=width, label="RMSE")
    ax.bar(x + width, r2_values, width=width, label="R2")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("RFECV vs brute-force (RobustScaler) feature selection metrics")
    ax.legend()
    fig.tight_layout()
    comparison_plot_path = case_output_dir / "case_5_method_comparison.png"
    fig.savefig(comparison_plot_path, dpi=300)
    plt.close(fig)
    print(f"Method comparison plot saved to: {comparison_plot_path}")

    metadata_path = write_case_metadata(
        case_dir=case_output_dir,
        case_id=CASE_ID,
        case_name=CASE_NAME,
        package="feature_selection_methods",
        dataset_path=dataset_path,
        tickers=selected_tickers,
        features=feature_list,
        target=target,
        models=("RobustScaler+GradientBoostingRegressor",),
        extras={
            "method": CASE_METHOD,
            "cv_splits": int(n_splits),
            "rfecv_selected_feature_count": int(len(selected_features)),
            "bruteforce_combination_count": int(len(combo_results)),
            "rfecv_runtime_seconds": float(rfecv_duration),
            "bruteforce_runtime_seconds": float(brute_duration),
            "rfecv_metrics": {
                "mean_mae": rfecv_metrics["mean_mae"],
                "mean_rmse": rfecv_metrics["mean_rmse"],
                "mean_r2": rfecv_metrics["mean_r2"],
            },
            "best_bruteforce_metrics": {
                "mean_mae": float(top_bruteforce["mean_mae"]),
                "mean_rmse": float(top_bruteforce["mean_rmse"]),
                "mean_r2": float(top_bruteforce["mean_r2"]),
            },
            "missing_features": missing_features,
            "target_source": target_source,
        },
    )
    print(f"Metadata written to: {metadata_path}")

    return case_output_dir


if __name__ == "__main__":
    run_case()
