"""Case study 3: embedded feature selection using LassoCV."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

CASE_ID = "case_3"
CASE_CONFIG = FEATURE_SELECTION_CASES[CASE_ID]
CASE_NAME = CASE_CONFIG.name
CASE_METHOD = CASE_CONFIG.method
TOP_K = CASE_CONFIG.top_k or len(DEFAULT_FEATURES)
MAX_SAMPLE_SIZE = 5000


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
    """Identify sparse, high-signal features via LassoCV coefficients."""

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
        print(f"Sample truncated to the most recent {MAX_SAMPLE_SIZE} rows for LassoCV runtime control.")

    X = matrix[feature_list]
    y = matrix[target]

    n_splits = min(5, max(3, len(matrix) // 120))
    if n_splits < 3:
        n_splits = 3

    if n_splits >= len(matrix):
        print("Insufficient observations for cross-validation; aborting case.")
        return case_output_dir

    cv = TimeSeriesSplit(n_splits=n_splits)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lasso",
                LassoCV(
                    cv=cv,
                    random_state=42,
                    n_alphas=100,
                    max_iter=10000,
                    n_jobs=None,
                ),
            ),
        ]
    )

    print(f"Fitting LassoCV with {n_splits} time-series splits...")
    pipeline.fit(X, y)

    lasso: LassoCV = pipeline.named_steps["lasso"]
    coefficients = pd.Series(lasso.coef_, index=feature_list)
    abs_coefficients = coefficients.abs()

    ranking = pd.DataFrame(
        {
            "feature": feature_list,
            "coefficient": coefficients,
            "abs_coefficient": abs_coefficients,
        }
    )
    ranking.sort_values("abs_coefficient", ascending=False, inplace=True)
    ranking.reset_index(drop=True, inplace=True)

    ranking_path = save_dataframe(ranking, case_output_dir, "lasso_feature_coefficients.csv")
    print(f"Lasso coefficient ranking saved to: {ranking_path}")

    adjusted_top_k = min(TOP_K, len(ranking))
    top_features = ranking.head(adjusted_top_k).copy()
    top_features_path = save_dataframe(top_features, case_output_dir, "lasso_top_features.csv")
    print(f"Top {adjusted_top_k} features by |coefficient| saved to: {top_features_path}")

    non_zero = ranking[ranking["abs_coefficient"] > 0]
    selected_features = non_zero["feature"].tolist()
    selected_df = non_zero.copy()
    selected_path = save_dataframe(selected_df, case_output_dir, "lasso_selected_features.csv")
    if selected_features:
        print(f"Selected features ({len(selected_features)}): {', '.join(selected_features)}")
    else:
        print("Selected features: none retained by Lasso.")
    print(f"Selected feature details saved to: {selected_path}")

    mse_df = pd.DataFrame(
        lasso.mse_path_.T,
        columns=[f"fold_{idx + 1}" for idx in range(lasso.mse_path_.shape[1])],
    )
    mse_df.insert(0, "alpha", lasso.alphas_)
    mse_path = save_dataframe(mse_df, case_output_dir, "lasso_mse_path.csv")
    print(f"Lasso cross-validation MSE path saved to: {mse_path}")

    metadata_path = write_case_metadata(
        case_dir=case_output_dir,
        case_id=CASE_ID,
        case_name=CASE_NAME,
        package="feature_selection_methods",
        dataset_path=dataset_path,
        tickers=selected_tickers,
        features=feature_list,
        target=target,
        models=("LassoCV",),
        extras={
            "method": CASE_METHOD,
            "cv_splits": n_splits,
            "alpha": float(lasso.alpha_),
            "selected_feature_count": len(selected_features),
            "top_k": adjusted_top_k,
            "missing_features": missing_features,
            "target_source": target_source,
        },
    )
    print(f"Metadata written to: {metadata_path}")

    return case_output_dir


if __name__ == "__main__":
    run_case()
