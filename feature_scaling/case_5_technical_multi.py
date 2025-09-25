"""Case study 5: technical feature scaling for multiple tickers."""

from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.preprocessing import PowerTransformer

from config import DEFAULT_STOCKS, ML_INPUT_DIR, MODEL_DICT
from core.prediction_validation import PredictionValidationResult, evaluate_last_n_day_predictions
from core.visualization import plot_features_distribution_grid
from feature_scaling._shared import (
    clean_feature_matrix,
    filter_ticker,
    load_ml_dataset,
    prepare_feature_matrix,
    resolve_output_dir,
    run_learning_workflow,
    save_dataframe,
)
from feature_scaling.settings import FEATURE_SCALING_CASES

CASE_ID = "case_5"
CASE_NAME = "Technical features for multiple tickers"
DEFAULT_OUTPUT_ROOT = Path("feature_scaling") / "outputs"
CASE_CONFIG = FEATURE_SCALING_CASES.get(CASE_ID, {})
CASE_FEATURES = list(CASE_CONFIG.get("features", []))
CASE_TARGET = CASE_CONFIG.get("target")
CASE_TARGET_SOURCE = CASE_CONFIG.get("target_source", CASE_TARGET)
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
) -> PredictionValidationResult | None:
    if metadata is None:
        return None

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
        return None

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

    return result


def _run_for_ticker(
    *,
    ticker: str,
    data: pd.DataFrame,
    output_root: Path,
    show_plots: bool,
) -> None:
    print(f">>> Processing ticker {ticker}")
    subset = filter_ticker(data, ticker)
    print(f"Rows for ticker '{ticker}': {len(subset)}")

    if subset.empty:
        print("No data available for the ticker; skipping.")
        return

    if not CASE_FEATURES:
        raise ValueError("CASE_FEATURES configuration is empty for case_5.")
    if CASE_TARGET is None:
        raise ValueError("CASE_TARGET configuration is missing for case_5.")

    ticker_dir = output_root / ticker.lower()
    model_store_dir = ticker_dir / "models"
    last_n_output_dir = ticker_dir / "last_n_day_validation"
    ticker_dir.mkdir(parents=True, exist_ok=True)

    feature_matrix = prepare_feature_matrix(subset, CASE_FEATURES, CASE_TARGET, CASE_TARGET_SOURCE)
    rows_before = len(feature_matrix)
    feature_matrix = clean_feature_matrix(feature_matrix)
    rows_after = len(feature_matrix)
    dropped = rows_before - rows_after
    if dropped > 0:
        print(f"Removed {dropped} rows with NaN/inf values from feature matrix.")

    if feature_matrix.empty:
        print("Feature matrix empty after cleaning; skipping ticker.")
        return

    metadata_for_validation = None
    if {"Date", "ticker"}.issubset(subset.columns):
        metadata_for_validation = subset.loc[feature_matrix.index, ["Date", "ticker"]].copy()
    else:
        print("Skipping last-N-day validation because required 'Date' or 'ticker' columns are missing.")

    raw_matrix_path = save_dataframe(
        feature_matrix,
        ticker_dir,
        f"{ticker.lower()}_case5_feature_matrix_raw.csv",
    )
    print(f"Raw feature matrix saved to: {raw_matrix_path}")

    raw_summary_path = save_dataframe(
        feature_matrix.describe(include="all").transpose(),
        ticker_dir,
        f"{ticker.lower()}_case5_feature_matrix_raw_summary.csv",
    )
    print(f"Raw summary saved to: {raw_summary_path}")

    X_raw = feature_matrix[CASE_FEATURES]
    y_raw = feature_matrix[CASE_TARGET]

    print("--- Learning curves on raw features ---")
    raw_results, raw_curve_path, _ = run_learning_workflow(
        MODEL_DICT,
        X_raw,
        y_raw,
        title_suffix=f"{ticker} (case5 raw)",
        output_dir=ticker_dir,
        show_plots=show_plots,
    )
    raw_results_path = save_dataframe(
        raw_results,
        ticker_dir,
        f"{ticker.lower()}_case5_learning_curve_raw.csv",
    )
    print(f"Raw learning curve diagnostics saved to: {raw_results_path}")
    if raw_curve_path:
        print(f"Raw learning curve figure saved to: {raw_curve_path}")

    aggregated_model_metrics: list[pd.DataFrame] = []
    aggregated_ticker_metrics: list[pd.DataFrame] = []

    raw_validation = _run_last_n_validation(
        X_raw,
        y_raw,
        metadata=metadata_for_validation,
        dataset_label=f"{ticker} case5 raw",
        output_dir=last_n_output_dir,
        plot_prefix=f"{ticker.lower()}_case5_raw",
        model_store_dir=model_store_dir,
    )
    if raw_validation is not None:
        aggregated_model_metrics.append(raw_validation.model_metrics)
        aggregated_ticker_metrics.append(raw_validation.per_ticker_metrics)

    print("--- Applying Yeo-Johnson transformation ---")
    analysis_transformer = PowerTransformer(method="yeo-johnson")
    X_transformed_array = analysis_transformer.fit_transform(X_raw)
    X_transformed = pd.DataFrame(X_transformed_array, columns=X_raw.columns, index=X_raw.index)

    transformed_matrix = X_transformed.copy()
    transformed_matrix[CASE_TARGET] = y_raw.values

    transformed_matrix_path = save_dataframe(
        transformed_matrix,
        ticker_dir,
        f"{ticker.lower()}_case5_feature_matrix_yeojohnson.csv",
    )
    print(f"Transformed feature matrix saved to: {transformed_matrix_path}")

    transformed_summary_path = save_dataframe(
        transformed_matrix.describe(include="all").transpose(),
        ticker_dir,
        f"{ticker.lower()}_case5_feature_matrix_yeojohnson_summary.csv",
    )
    print(f"Transformed summary saved to: {transformed_summary_path}")

    print("--- Learning curves on Yeo-Johnson transformed features ---")
    transformed_results, transformed_curve_path, _ = run_learning_workflow(
        MODEL_DICT,
        X_transformed,
        y_raw,
        title_suffix=f"{ticker} (case5 Yeo-Johnson)",
        output_dir=ticker_dir,
        show_plots=show_plots,
    )
    transformed_results_path = save_dataframe(
        transformed_results,
        ticker_dir,
        f"{ticker.lower()}_case5_learning_curve_yeojohnson.csv",
    )
    print(f"Transformed learning curve diagnostics saved to: {transformed_results_path}")
    if transformed_curve_path:
        print(f"Transformed learning curve figure saved to: {transformed_curve_path}")

    yeo_transformer = PowerTransformer(method="yeo-johnson")
    transformed_validation = _run_last_n_validation(
        X_raw,
        y_raw,
        metadata=metadata_for_validation,
        dataset_label=f"{ticker} case5 yeojohnson",
        output_dir=last_n_output_dir,
        plot_prefix=f"{ticker.lower()}_case5_yeojohnson",
        model_store_dir=model_store_dir,
        preprocessor=yeo_transformer,
    )
    if transformed_validation is not None:
        aggregated_model_metrics.append(transformed_validation.model_metrics)
        aggregated_ticker_metrics.append(transformed_validation.per_ticker_metrics)

    transformed_distribution_path = plot_features_distribution_grid(
        X_transformed,
        X_raw,
        title=f"{ticker} features: case5 Yeo-Johnson vs raw",
        output_dir=ticker_dir,
    )
    print(f"Transformed feature distribution chart saved to: {transformed_distribution_path}")

    comparison_results = pd.concat(
        [
            raw_results.assign(dataset="case5_raw"),
            transformed_results.assign(dataset="case5_yeojohnson"),
        ],
        ignore_index=True,
    )
    comparison_path = save_dataframe(
        comparison_results,
        ticker_dir,
        f"{ticker.lower()}_case5_learning_curve_comparison.csv",
    )
    print(f"Learning curve comparison saved to: {comparison_path}")

    if aggregated_model_metrics:
        combined_model_metrics = pd.concat(aggregated_model_metrics, ignore_index=True)
        save_dataframe(
            combined_model_metrics,
            ticker_dir,
            f"{ticker.lower()}_case5_last_{LAST_N_DAYS}_day_model_metrics.csv",
        )
    if aggregated_ticker_metrics:
        combined_ticker_metrics = pd.concat(aggregated_ticker_metrics, ignore_index=True)
        save_dataframe(
            combined_ticker_metrics,
            ticker_dir,
            f"{ticker.lower()}_case5_last_{LAST_N_DAYS}_day_metrics_by_ticker.csv",
        )


def run_case(
    *,
    dataset_filename: str = "ml_dataset.csv",
    output_root: Path | str | None = None,
    show_plots: bool = False,
) -> Path:
    """Run the technical feature scaling workflow for all configured tickers."""

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)

    dataset_path = ML_INPUT_DIR / dataset_filename
    print(f"=== Running {CASE_ID}: {CASE_NAME} ===")
    print(f"Expecting ML dataset at: {dataset_path}")

    data = load_ml_dataset(dataset_path)

    for stock in DEFAULT_STOCKS:
        ticker = stock.get("ticker")
        if not ticker:
            continue
        _run_for_ticker(
            ticker=ticker,
            data=data,
            output_root=case_output_dir,
            show_plots=show_plots,
        )

    return case_output_dir


if __name__ == "__main__":
    run_case(show_plots=True)
