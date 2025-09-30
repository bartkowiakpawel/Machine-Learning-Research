"""Case study 7: Yeo-Johnson plus polynomial features for TSLA."""

from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.pipeline import Pipeline

from config import ML_INPUT_DIR, MODEL_DICT
from core.prediction_validation import evaluate_last_n_day_predictions
from core.visualization import plot_features_distribution_grid
from core.shared_utils import (
    clean_feature_matrix,
    filter_ticker,
    load_ml_dataset,
    prepare_feature_matrix,
    resolve_output_dir,
    run_learning_workflow,
    save_dataframe,
)
from feature_scaling.settings import FEATURE_SCALING_CASES

CASE_ID = "case_7"
CASE_NAME = "TSLA Yeo-Johnson + polynomial"
DEFAULT_OUTPUT_ROOT = Path("feature_scaling") / "outputs"
CASE_CONFIG = FEATURE_SCALING_CASES.get(CASE_ID, {})
CASE_FEATURES = list(CASE_CONFIG.get("features", []))
CASE_TARGET = CASE_CONFIG.get("target")
CASE_TARGET_SOURCE = CASE_CONFIG.get("target_source", CASE_TARGET)
POLYNOMIAL_DEGREE = int(CASE_CONFIG.get("polynomial_degree", 2))

LAST_N_DAYS = 15


def _run_last_n_validation(
    X_features: pd.DataFrame,
    y_target: pd.Series,
    *,
    metadata: pd.DataFrame | None,
    dataset_label: str,
    output_dir: Path,
    plot_prefix: str,
    model_store_dir: Path,
    preprocessor=None,
) -> None:
    if metadata is None:
        return

    try:
        result = evaluate_last_n_day_predictions(
            MODEL_DICT,
            X_features,
            y_target,
            metadata=metadata,
            dataset_label=dataset_label,
            n_days=LAST_N_DAYS,
            ticker_column="ticker",
            date_column="Date",
            output_dir=output_dir,
            plot_prefix=plot_prefix,
            n_plot_columns=3,
            show_plot=False,
            preprocessor=preprocessor,
            model_store_dir=model_store_dir,
        )
    except ValueError as exc:
        print(f"Skipping last-{LAST_N_DAYS}-day validation for {dataset_label}: {exc}")
        return

    global_path = save_dataframe(
        result.model_metrics,
        output_dir,
        f"{plot_prefix}_last_{LAST_N_DAYS}_day_metrics.csv",
    )
    print(f"{dataset_label} last-{LAST_N_DAYS}-day global metrics saved to: {global_path}")

    ticker_path = save_dataframe(
        result.per_ticker_metrics,
        output_dir,
        f"{plot_prefix}_last_{LAST_N_DAYS}_day_metrics_by_ticker.csv",
    )
    print(f"{dataset_label} last-{LAST_N_DAYS}-day per-ticker metrics saved to: {ticker_path}")

    predictions_path = save_dataframe(
        result.predictions,
        output_dir,
        f"{plot_prefix}_last_{LAST_N_DAYS}_day_predictions.csv",
    )
    print(f"{dataset_label} last-{LAST_N_DAYS}-day predictions saved to: {predictions_path}")


def run_case(
    *,
    ticker: str = "TSLA",
    dataset_filename: str = "ml_dataset.csv",
    output_root: Path | str | None = None,
    show_plots: bool = False,
) -> Path:
    """Run Yeo-Johnson plus polynomial workflow for a single ticker."""

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    last_n_output_dir = case_output_dir / "last_n_day_validation"
    model_store_dir = case_output_dir / "models"

    dataset_path = ML_INPUT_DIR / dataset_filename
    print(f"=== Running {CASE_ID}: {CASE_NAME} ===")
    print(f"Expecting ML dataset at: {dataset_path}")

    data = load_ml_dataset(dataset_path)
    subset = filter_ticker(data, ticker)
    print(f"Rows for ticker '{ticker}': {len(subset)}")

    if subset.empty:
        print("No data available for the ticker; aborting case study.")
        return case_output_dir

    if not CASE_FEATURES:
        raise ValueError("CASE_FEATURES configuration is empty for case_7.")
    if CASE_TARGET is None:
        raise ValueError("CASE_TARGET configuration is missing for case_7.")

    feature_matrix = prepare_feature_matrix(subset, CASE_FEATURES, CASE_TARGET, CASE_TARGET_SOURCE)
    rows_before = len(feature_matrix)
    feature_matrix = clean_feature_matrix(feature_matrix)
    rows_after = len(feature_matrix)
    dropped = rows_before - rows_after
    if dropped > 0:
        print(f"Removed {dropped} rows with NaN/inf values from feature matrix.")

    if feature_matrix.empty:
        print("Feature matrix empty after cleaning; aborting case study.")
        return case_output_dir

    metadata_for_validation = None
    if {"Date", "ticker"}.issubset(subset.columns):
        metadata_for_validation = subset.loc[feature_matrix.index, ["Date", "ticker"]].copy()
    else:
        print("Skipping last-N-day validation because required 'Date' or 'ticker' columns are missing.")

    raw_matrix_path = save_dataframe(
        feature_matrix,
        case_output_dir,
        f"{ticker.lower()}_case7_feature_matrix_raw.csv",
    )
    print(f"Raw feature matrix saved to: {raw_matrix_path}")

    raw_summary_path = save_dataframe(
        feature_matrix.describe(include="all").transpose(),
        case_output_dir,
        f"{ticker.lower()}_case7_feature_matrix_raw_summary.csv",
    )
    print(f"Raw summary saved to: {raw_summary_path}")

    X_raw = feature_matrix[CASE_FEATURES]
    y_raw = feature_matrix[CASE_TARGET]

    print("--- Applying Yeo-Johnson transformation ---")
    yeo_transformer = PowerTransformer(method="yeo-johnson")
    X_yeojohnson_array = yeo_transformer.fit_transform(X_raw)
    X_yeojohnson = pd.DataFrame(X_yeojohnson_array, columns=X_raw.columns, index=X_raw.index)

    yeojohnson_matrix = X_yeojohnson.copy()
    yeojohnson_matrix[CASE_TARGET] = y_raw.values

    yeojohnson_matrix_path = save_dataframe(
        yeojohnson_matrix,
        case_output_dir,
        f"{ticker.lower()}_case7_feature_matrix_yeojohnson.csv",
    )
    print(f"Yeo-Johnson feature matrix saved to: {yeojohnson_matrix_path}")

    yeojohnson_summary_path = save_dataframe(
        yeojohnson_matrix.describe(include="all").transpose(),
        case_output_dir,
        f"{ticker.lower()}_case7_feature_matrix_yeojohnson_summary.csv",
    )
    print(f"Yeo-Johnson summary saved to: {yeojohnson_summary_path}")

    print("--- Learning curves on Yeo-Johnson features ---")
    yeojohnson_results, yeojohnson_curve_path, _ = run_learning_workflow(
        MODEL_DICT,
        X_yeojohnson,
        y_raw,
        title_suffix=f"{ticker} (case7 Yeo-Johnson)",
        output_dir=case_output_dir,
        show_plots=show_plots,
    )
    yeojohnson_results_path = save_dataframe(
        yeojohnson_results,
        case_output_dir,
        f"{ticker.lower()}_case7_learning_curve_yeojohnson.csv",
    )
    print(f"Yeo-Johnson learning curve diagnostics saved to: {yeojohnson_results_path}")
    if yeojohnson_curve_path:
        print(f"Yeo-Johnson learning curve figure saved to: {yeojohnson_curve_path}")

    if metadata_for_validation is not None:
        _run_last_n_validation(
            X_raw,
            y_raw,
            metadata=metadata_for_validation,
            dataset_label=f"{ticker} case7 yeojohnson",
            output_dir=last_n_output_dir,
            plot_prefix=f"{ticker.lower()}_case7_yeojohnson",
            model_store_dir=model_store_dir,
            preprocessor=PowerTransformer(method="yeo-johnson"),
        )

    print("--- Applying polynomial expansion on Yeo-Johnson features ---")
    polynomial_transformer = PolynomialFeatures(
        degree=POLYNOMIAL_DEGREE,
        include_bias=False,
    )
    X_polynomial_array = polynomial_transformer.fit_transform(X_yeojohnson)
    feature_names = polynomial_transformer.get_feature_names_out(X_yeojohnson.columns)
    X_polynomial = pd.DataFrame(
        X_polynomial_array,
        columns=feature_names,
        index=X_yeojohnson.index,
    )

    polynomial_matrix = X_polynomial.copy()
    polynomial_matrix[CASE_TARGET] = y_raw.values

    polynomial_matrix_path = save_dataframe(
        polynomial_matrix,
        case_output_dir,
        f"{ticker.lower()}_case7_feature_matrix_polynomial.csv",
    )
    print(f"Polynomial feature matrix saved to: {polynomial_matrix_path}")

    polynomial_summary_path = save_dataframe(
        polynomial_matrix.describe(include="all").transpose(),
        case_output_dir,
        f"{ticker.lower()}_case7_feature_matrix_polynomial_summary.csv",
    )
    print(f"Polynomial summary saved to: {polynomial_summary_path}")

    print("--- Learning curves on polynomial (Yeo-Johnson) features ---")
    polynomial_results, polynomial_curve_path, _ = run_learning_workflow(
        MODEL_DICT,
        X_polynomial,
        y_raw,
        title_suffix=f"{ticker} (case7 Yeo-Johnson polynomial)",
        output_dir=case_output_dir,
        show_plots=show_plots,
    )
    polynomial_results_path = save_dataframe(
        polynomial_results,
        case_output_dir,
        f"{ticker.lower()}_case7_learning_curve_polynomial.csv",
    )
    print(f"Polynomial learning curve diagnostics saved to: {polynomial_results_path}")
    if polynomial_curve_path:
        print(f"Polynomial learning curve figure saved to: {polynomial_curve_path}")

    if metadata_for_validation is not None:
        poly_pipeline = Pipeline(
            steps=[
                ("yeojohnson", PowerTransformer(method="yeo-johnson")),
                (
                    "polynomial",
                    PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False),
                ),
            ]
        )
        _run_last_n_validation(
            X_raw,
            y_raw,
            metadata=metadata_for_validation,
            dataset_label=f"{ticker} case7 yeojohnson polynomial",
            output_dir=last_n_output_dir,
            plot_prefix=f"{ticker.lower()}_case7_polynomial",
            model_store_dir=model_store_dir,
            preprocessor=poly_pipeline,
        )

    try:
        comparison_original_cols = X_polynomial.loc[:, X_yeojohnson.columns]
    except KeyError:
        comparison_original_cols = X_polynomial.loc[:, CASE_FEATURES]

    polynomial_distribution_path = plot_features_distribution_grid(
        comparison_original_cols,
        X_yeojohnson,
        title=f"{ticker} features: case7 Polynomial (Yeo-Johnson) vs Yeo-Johnson",
        output_dir=case_output_dir,
    )
    print(f"Polynomial feature distribution chart saved to: {polynomial_distribution_path}")

    comparison_results = pd.concat(
        [
            yeojohnson_results.assign(dataset="case7_yeojohnson"),
            polynomial_results.assign(dataset="case7_yeojohnson_polynomial"),
        ],
        ignore_index=True,
    )
    comparison_path = save_dataframe(
        comparison_results,
        case_output_dir,
        f"{ticker.lower()}_case7_learning_curve_comparison.csv",
    )
    print(f"Learning curve comparison saved to: {comparison_path}")

    return case_output_dir


if __name__ == "__main__":
    run_case(show_plots=True)

