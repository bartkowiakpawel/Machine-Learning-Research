"""Case study 3: hybrid feature scaling for TSLA."""

from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler

from config import FEATURE_SCALING_CASES, ML_INPUT_DIR, MODEL_DICT
from core.model_evaluation import plot_learning_curves_for_models
from core.visualization import plot_features_distribution_grid

CASE_ID = "case_3"
CASE_NAME = "Hybrid scaling for TSLA"
DEFAULT_OUTPUT_ROOT = Path("feature_scaling") / "outputs"
CASE_CONFIG = FEATURE_SCALING_CASES.get(CASE_ID, {})
CASE_FEATURES = list(CASE_CONFIG.get("features", []))
CASE_TECH_FEATURES = list(CASE_CONFIG.get("technical_features", []))
CASE_FUND_FEATURES = list(CASE_CONFIG.get("fundamental_features", []))
CASE_TARGET = CASE_CONFIG.get("target")
CASE_TARGET_SOURCE = CASE_CONFIG.get("target_source", CASE_TARGET)
CSV_KWARGS = {"index": False, "sep": ",", "decimal": "."}


def _resolve_output_dir(output_root: Path | str | None) -> Path:
    root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    case_dir = root / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _load_ml_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"ML dataset not found at {dataset_path}. Run the ML preparation pipeline first."
        )

    df = pd.read_csv(dataset_path)
    if "ticker" not in df.columns:
        raise KeyError("ML dataset must contain a 'ticker' column")

    return df


def _filter_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    ticker_upper = ticker.upper()
    result = df[df["ticker"].astype(str).str.upper() == ticker_upper].copy()
    return result


def _prepare_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str | None,
    target_source: str | None,
) -> pd.DataFrame:
    if not feature_columns:
        raise ValueError("Feature list for the case study is empty.")

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        missing_fmt = ", ".join(missing_features)
        raise KeyError(f"Missing feature columns in dataset: {missing_fmt}")

    working = df.copy()

    if target_column:
        if target_column not in working.columns:
            if target_source and target_source in working.columns:
                working[target_column] = working[target_source]
            else:
                raise KeyError(
                    "Target column not found and no fallback target_source available in dataset."
                )
        selected_columns = feature_columns + [target_column]
    else:
        selected_columns = feature_columns

    return working.loc[:, selected_columns]


def _clean_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sanitized = df.replace([np.inf, -np.inf], np.nan)
    return sanitized.dropna()


def _save_dataframe(df: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    output_path = output_dir / filename
    df.to_csv(output_path, **CSV_KWARGS)
    return output_path


def _run_learning_workflow(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    title_suffix: str,
    output_dir: Path,
    show_plots: bool,
) -> tuple[pd.DataFrame, Path | None, Path | None]:
    return plot_learning_curves_for_models(
        MODEL_DICT,
        X,
        y,
        main_title=f"Learning curves for {title_suffix}",
        output_dir=output_dir,
        show=show_plots,
        feature_distribution=None,
        original_feature_distribution=None,
    )


def _transform_hybrid(
    X_raw: pd.DataFrame,
    technical_features: list[str],
    fundamental_features: list[str],
) -> tuple[pd.DataFrame, ColumnTransformer]:
    preprocessor = ColumnTransformer(
        transformers=[
            ("yeo", PowerTransformer(method="yeo-johnson"), technical_features),
            ("scaler", StandardScaler(), fundamental_features),
        ],
        remainder="drop",
    )
    transformed = preprocessor.fit_transform(X_raw)
    transformed_df = pd.DataFrame(transformed, columns=technical_features + fundamental_features, index=X_raw.index)
    return transformed_df, preprocessor


def run_case(
    *,
    ticker: str = "TSLA",
    dataset_filename: str = "ml_dataset.csv",
    output_root: Path | str | None = None,
    show_plots: bool = False,
) -> Path:
    case_output_dir = _resolve_output_dir(output_root)

    dataset_path = ML_INPUT_DIR / dataset_filename
    print(f"\n=== Running {CASE_ID}: {CASE_NAME} ===")
    print(f"Expecting ML dataset at: {dataset_path}")

    data = _load_ml_dataset(dataset_path)
    subset = _filter_ticker(data, ticker)
    print(f"Rows for ticker '{ticker}': {len(subset)}")

    if subset.empty:
        print("No data available for the ticker; aborting case study.")
        return case_output_dir

    if not CASE_FEATURES:
        raise ValueError("CASE_FEATURES configuration is empty for case_3.")
    if CASE_TARGET is None:
        raise ValueError("CASE_TARGET configuration is missing for case_3.")

    feature_matrix = _prepare_feature_matrix(subset, CASE_FEATURES, CASE_TARGET, CASE_TARGET_SOURCE)
    feature_matrix = _clean_feature_matrix(feature_matrix)
    if feature_matrix.empty:
        print("Feature matrix empty after cleaning; aborting case study.")
        return case_output_dir

    raw_path = _save_dataframe(
        feature_matrix,
        case_output_dir,
        f"{ticker.lower()}_case3_feature_matrix_raw.csv",
    )
    print(f"Raw feature matrix saved to: {raw_path}")

    raw_summary_path = _save_dataframe(
        feature_matrix.describe(include="all").transpose(),
        case_output_dir,
        f"{ticker.lower()}_case3_feature_matrix_raw_summary.csv",
    )
    print(f"Raw summary saved to: {raw_summary_path}")

    X_raw = feature_matrix[CASE_FEATURES]
    y_raw = feature_matrix[CASE_TARGET]

    print("\n--- Learning curves on raw features ---")
    raw_results, raw_curve_path, _ = _run_learning_workflow(
        X_raw,
        y_raw,
        title_suffix=f"{ticker} (case3 raw)",
        output_dir=case_output_dir,
        show_plots=show_plots,
    )
    raw_results_path = _save_dataframe(
        raw_results,
        case_output_dir,
        f"{ticker.lower()}_case3_learning_curve_raw.csv",
    )
    print(f"Raw learning curve diagnostics saved to: {raw_results_path}")
    if raw_curve_path:
        print(f"Raw learning curve figure saved to: {raw_curve_path}")

    print("\n--- Applying hybrid scaling (Yeo-Johnson + StandardScaler) ---")
    X_transformed, preprocessor = _transform_hybrid(
        X_raw,
        CASE_TECH_FEATURES,
        CASE_FUND_FEATURES,
    )
    transformed_matrix = X_transformed.copy()
    transformed_matrix[CASE_TARGET] = y_raw.values

    transformed_path = _save_dataframe(
        transformed_matrix,
        case_output_dir,
        f"{ticker.lower()}_case3_feature_matrix_hybrid.csv",
    )
    print(f"Hybrid feature matrix saved to: {transformed_path}")

    transformed_summary_path = _save_dataframe(
        transformed_matrix.describe(include="all").transpose(),
        case_output_dir,
        f"{ticker.lower()}_case3_feature_matrix_hybrid_summary.csv",
    )
    print(f"Hybrid summary saved to: {transformed_summary_path}")

    print("\n--- Learning curves on hybrid features ---")
    transformed_results, transformed_curve_path, _ = _run_learning_workflow(
        X_transformed,
        y_raw,
        title_suffix=f"{ticker} (case3 hybrid)",
        output_dir=case_output_dir,
        show_plots=show_plots,
    )
    transformed_results_path = _save_dataframe(
        transformed_results,
        case_output_dir,
        f"{ticker.lower()}_case3_learning_curve_hybrid.csv",
    )
    print(f"Hybrid learning curve diagnostics saved to: {transformed_results_path}")
    if transformed_curve_path:
        print(f"Hybrid learning curve figure saved to: {transformed_curve_path}")

    distribution_path = plot_features_distribution_grid(
        X_transformed,
        X_raw,
        title=f"{ticker} features: hybrid vs raw",
        output_dir=case_output_dir,
    )
    print(f"Feature distribution comparison saved to: {distribution_path}")

    comparison_results = pd.concat(
        [
            raw_results.assign(dataset="raw"),
            transformed_results.assign(dataset="hybrid"),
        ],
        ignore_index=True,
    )
    comparison_path = _save_dataframe(
        comparison_results,
        case_output_dir,
        f"{ticker.lower()}_case3_learning_curve_comparison.csv",
    )
    print(f"Learning curve comparison saved to: {comparison_path}")

    return case_output_dir


if __name__ == "__main__":
    run_case(show_plots=True)
