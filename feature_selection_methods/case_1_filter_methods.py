"""Case study 1: filter-based feature ranking with mutual information."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.feature_selection import f_regression, mutual_info_regression

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
    DEFAULT_TICKERS,
    FEATURE_SELECTION_CASES,
)

CASE_ID = "case_1"
CASE_CONFIG = FEATURE_SELECTION_CASES[CASE_ID]
CASE_NAME = CASE_CONFIG.name
CASE_METHOD = CASE_CONFIG.method
TOP_K = CASE_CONFIG.top_k or len(DEFAULT_FEATURES)


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
    """Rank engineered features using mutual information and correlation filters."""

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    dataset_path = ML_INPUT_DIR / dataset_filename
    selected_tickers = list(tickers) if tickers is not None else list(CASE_CONFIG.tickers or DEFAULT_TICKERS)
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

    X = matrix[feature_list]
    y = matrix[target]

    print(f"Working sample size: {len(matrix)} rows")

    mi_scores = mutual_info_regression(X, y, random_state=42)
    f_scores, p_values = f_regression(X, y)
    correlations = X.corrwith(y)

    ranking = pd.DataFrame(
        {
            "feature": feature_list,
            "mutual_information": mi_scores,
            "f_score": f_scores,
            "f_pvalue": p_values,
            "pearson_correlation": correlations,
            "abs_correlation": correlations.abs(),
        }
    )
    ranking["rank_mutual_information"] = ranking["mutual_information"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking["rank_abs_correlation"] = ranking["abs_correlation"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking["rank_f_score"] = ranking["f_score"].rank(ascending=False, method="min").astype(int)
    ranking.sort_values("rank_mutual_information", inplace=True)
    ranking.reset_index(drop=True, inplace=True)

    ranking_path = save_dataframe(ranking, case_output_dir, "feature_ranking_filter_method.csv")
    print(f"Full ranking saved to: {ranking_path}")

    adjusted_top_k = min(TOP_K, len(ranking))
    top_features = ranking.nsmallest(adjusted_top_k, "rank_mutual_information").copy()
    top_features_path = save_dataframe(top_features, case_output_dir, "top_features_filter_method.csv")
    print(f"Top {adjusted_top_k} features saved to: {top_features_path}")

    metadata_path = write_case_metadata(
        case_dir=case_output_dir,
        case_id=CASE_ID,
        case_name=CASE_NAME,
        package="feature_selection_methods",
        dataset_path=dataset_path,
        tickers=selected_tickers,
        features=feature_list,
        target=target,
        models=("SelectKBest",),
        extras={
            "method": CASE_METHOD,
            "top_k": adjusted_top_k,
            "sample_size": int(len(matrix)),
            "missing_features": missing_features,
            "target_source": target_source,
        },
    )
    print(f"Metadata written to: {metadata_path}")

    return case_output_dir


if __name__ == "__main__":
    run_case()
