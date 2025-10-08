"""Case study 2: wrapper-based feature selection with RFECV."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit

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

CASE_ID = "case_2"
CASE_CONFIG = FEATURE_SELECTION_CASES[CASE_ID]
CASE_NAME = CASE_CONFIG.name
CASE_METHOD = CASE_CONFIG.method
TOP_K = CASE_CONFIG.top_k or len(DEFAULT_FEATURES)
MAX_SAMPLE_SIZE = 2500


def _filter_tickers(df: pd.DataFrame, tickers: Sequence[str] | None) -> pd.DataFrame:
    if not tickers:
        return df
    upper = {ticker.upper() for ticker in tickers}
    return df[df["ticker"].astype(str).str.upper().isin(upper)].copy()


def run_case(
    *,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    tickers: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
    target: str = DEFAULT_TARGET,
    target_source: str | None = DEFAULT_TARGET_SOURCE,
    output_root: Path | str | None = None,
) -> Path:
    """Select features with recursive feature elimination and cross-validation."""

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
        print(f"Sample truncated to the most recent {MAX_SAMPLE_SIZE} rows for RFECV runtime control.")

    X = matrix[feature_list]
    y = matrix[target]

    n_splits = min(5, max(2, len(matrix) // 100))  # basic safeguard for time-series split
    if n_splits < 2:
        n_splits = 2

    if n_splits >= len(matrix):
        print("Insufficient observations for cross-validation; aborting case.")
        return case_output_dir

    cv = TimeSeriesSplit(n_splits=n_splits)
    estimator = GradientBoostingRegressor(random_state=42)
    half_point = max(1, len(feature_list) // 2) if len(feature_list) > 1 else 1
    min_features_to_select = max(1, min(TOP_K, half_point))
    selector = RFECV(
        estimator,
        step=1,
        min_features_to_select=min_features_to_select,
        cv=cv,
        n_jobs=None,
        scoring="neg_mean_squared_error",
    )

    print(f"Fitting RFECV with {n_splits} time-series splits...")
    selector.fit(X, y)

    ranking = pd.DataFrame(
        {
            "feature": feature_list,
            "ranking": selector.ranking_,
            "selected": selector.support_,
        }
    )
    ranking.sort_values(["selected", "ranking"], ascending=[False, True], inplace=True)
    ranking.reset_index(drop=True, inplace=True)

    ranking_path = save_dataframe(ranking, case_output_dir, "rfecv_feature_ranking.csv")
    print(f"RFECV ranking saved to: {ranking_path}")

    selected_features = ranking.loc[ranking["selected"], "feature"].tolist()
    selected_path = save_dataframe(
        pd.DataFrame({"feature": selected_features}),
        case_output_dir,
        "rfecv_selected_features.csv",
    )
    if selected_features:
        print(f"Selected features ({len(selected_features)}): {', '.join(selected_features)}")
    else:
        print("Selected features: none retained by RFECV.")
    print(f"Selected feature list saved to: {selected_path}")

    if hasattr(selector, "cv_results_"):
        cv_results_dict: dict[str, np.ndarray] = {}
        for key, values in selector.cv_results_.items():
            arr = np.asarray(values)
            if arr.ndim == 0:
                cv_results_dict[key] = np.repeat(arr, len(ranking))
                continue
            if arr.ndim == 1:
                cv_results_dict[key] = arr
                continue
            flattened = arr.reshape(arr.shape[0], -1)
            column_count = flattened.shape[1]
            for idx in range(column_count):
                suffix = f"_{idx}" if column_count > 1 else ""
                cv_results_dict[f"{key}{suffix}"] = flattened[:, idx]
        cv_results = pd.DataFrame(cv_results_dict)
        cv_results_path = save_dataframe(cv_results, case_output_dir, "rfecv_cv_results.csv")
        print(f"Cross-validation diagnostics saved to: {cv_results_path}")

    metadata_path = write_case_metadata(
        case_dir=case_output_dir,
        case_id=CASE_ID,
        case_name=CASE_NAME,
        package="feature_selection_methods",
        dataset_path=dataset_path,
        tickers=selected_tickers,
        features=feature_list,
        target=target,
        models=("GradientBoostingRegressor",),
        extras={
            "method": CASE_METHOD,
            "cv_splits": int(n_splits),
            "selected_feature_count": int(len(selected_features)),
            "min_features_to_select": int(selector.min_features_to_select),
            "optimal_feature_count": int(getattr(selector, "n_features_", len(selected_features))),
            "total_features_considered": int(len(feature_list)),
            "missing_features": missing_features,
            "target_source": target_source,
        },
    )
    print(f"Metadata written to: {metadata_path}")

    return case_output_dir


if __name__ == "__main__":
    run_case()
