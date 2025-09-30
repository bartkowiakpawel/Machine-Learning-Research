"""EDA workflow using technical features with 14-day target validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ML_INPUT_DIR, MODEL_DICT
from core.shared_utils import (
    clean_feature_matrix,
    filter_ticker,
    load_ml_dataset,
    prepare_feature_matrix,
    resolve_output_dir,
    save_dataframe,
    write_case_metadata,
)
from core.prediction_validation import evaluate_last_n_day_predictions

try:
    from .settings import (
        DEFAULT_DATASET_FILENAME,
        DEFAULT_FEATURES,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_TICKER,
        DEFAULT_TICKERS,
        get_case_config,
    )
    from .visualization import (
        build_color_map,
        plot_features_boxplot,
        plot_single_feature_boxplot,
    )
    from .reporting import export_last_n_day_prediction_reports
except ImportError:
    from eda_boxplots.settings import (
        DEFAULT_DATASET_FILENAME,
        DEFAULT_FEATURES,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_TICKER,
        DEFAULT_TICKERS,
        get_case_config,
    )
    from eda_boxplots.visualization import (
        build_color_map,
        plot_features_boxplot,
        plot_single_feature_boxplot,
    )
    from eda_boxplots.reporting import export_last_n_day_prediction_reports

CASE_ID = "case_5"
CASE_CONFIG = get_case_config(CASE_ID)
CASE_NAME = CASE_CONFIG.name

ROLLING_WINDOW = 30
TARGET_SHIFT = 14
LAST_N_DAYS = 15
TECHNICAL_FEATURE_COLUMNS = list(DEFAULT_FEATURES)


def _normalize_tickers(tickers: Sequence[str] | None) -> list[str]:
    source = tickers if tickers is not None else DEFAULT_TICKERS
    normalized: list[str] = []
    for value in source:
        ticker = str(value).strip().upper()
        if ticker and ticker not in normalized:
            normalized.append(ticker)
    if not normalized:
        normalized.append(DEFAULT_TICKER)
    return normalized


def _compute_target(series_close: pd.Series) -> pd.Series:
    target = (series_close.shift(-TARGET_SHIFT) - series_close) / series_close
    return target


def run_case(
    *,
    tickers: Sequence[str] | None = None,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    output_root: Path | str | None = None,
    show_plots: bool = False,
    yeojohnson: bool = False,
) -> Path:
    """Run case 5 using technical features only."""

    selected_tickers = _normalize_tickers(tickers)
    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    comparison_dir = case_output_dir / "comparison"
    single_plots_dir = comparison_dir / "single_feature_plots"
    validation_dir = comparison_dir / "last_n_day_validation"
    model_store_dir = case_output_dir / "models"
    for directory in (comparison_dir, single_plots_dir, validation_dir, model_store_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_path = ML_INPUT_DIR / dataset_filename
    print(f"=== Running {CASE_ID}: {CASE_NAME} ===")
    print(f"Expecting ML dataset at: {dataset_path}")
    print(f"Tickers under comparison: {', '.join(selected_tickers)}")

    data = load_ml_dataset(dataset_path)

    per_ticker_frames: dict[str, pd.DataFrame] = {}
    combined_frames: list[pd.DataFrame] = []
    model_metrics = []
    ticker_metrics = []
    aggregated_predictions: list[pd.DataFrame] = []

    for ticker in selected_tickers:
        subset = filter_ticker(data, ticker)
        print(f"Rows for ticker '{ticker}': {len(subset)}")

        if subset.empty:
            print(f"No data available for {ticker}; skipping.")
            continue

        if "Close" not in subset.columns:
            print(f"Missing 'Close' column for {ticker}; skipping.")
            continue

        technical_matrix = prepare_feature_matrix(
            subset,
            TECHNICAL_FEATURE_COLUMNS,
            target_column=None,
            target_source=None,
        )
        technical_matrix = clean_feature_matrix(technical_matrix)
        if technical_matrix.empty:
            print(f"Technical features empty after cleaning for {ticker}; skipping.")
            continue

        target_series = _compute_target(subset.loc[technical_matrix.index, "Close"])
        target_series = target_series.loc[technical_matrix.index].dropna()
        technical_matrix = technical_matrix.loc[target_series.index]

        metadata = None
        if {"Date", "ticker"}.issubset(subset.columns):
            metadata = subset.loc[target_series.index, ["Date", "ticker"]].copy()

        if technical_matrix.empty or metadata is None:
            print(f"Insufficient data for validation for {ticker}; skipping.")
            continue

        if yeojohnson:
            transformer = PowerTransformer(method="yeo-johnson")
            transformed_array = transformer.fit_transform(technical_matrix)
            technical_matrix = pd.DataFrame(
                transformed_array,
                columns=technical_matrix.columns,
                index=technical_matrix.index,
            )

        ticker_dir = case_output_dir / ticker.lower()
        ticker_dir.mkdir(parents=True, exist_ok=True)

        matrix_path = save_dataframe(
            technical_matrix,
            ticker_dir,
            f"{ticker.lower()}_{CASE_ID}_technical_features{'_yeojohnson' if yeojohnson else ''}.csv",
        )
        print(f"Technical feature matrix for {ticker} saved to: {matrix_path}")

        summary_path = save_dataframe(
            technical_matrix.describe(include="all").transpose(),
            ticker_dir,
            f"{ticker.lower()}_{CASE_ID}_technical_features_summary{'_yeojohnson' if yeojohnson else ''}.csv",
        )
        print(f"Technical feature summary for {ticker} saved to: {summary_path}")

        per_ticker_frames[ticker] = technical_matrix.copy()
        combined_frames.append(technical_matrix.assign(ticker=ticker))

        try:
            result = evaluate_last_n_day_predictions(
                MODEL_DICT,
                technical_matrix,
                target_series,
                metadata=metadata,
                dataset_label=f"{ticker} {CASE_ID} technical features",
                n_days=LAST_N_DAYS,
                ticker_column="ticker",
                date_column="Date",
                output_dir=validation_dir,
                plot_prefix=f"{ticker.lower()}_{CASE_ID}{'_yeojohnson' if yeojohnson else ''}",
                n_plot_columns=3,
                show_plot=False,
                model_store_dir=model_store_dir,
            )
        except ValueError as exc:
            print(f"Skipping validation for {ticker}: {exc}")
            continue

        predictions_frame = result.predictions.copy()
        predictions_frame.insert(0, "dataset", f"{ticker} {CASE_ID} technical features")
        if {"actual", "prediction"}.issubset(predictions_frame.columns):
            predictions_frame["prediction_diff"] = predictions_frame["prediction"] - predictions_frame["actual"]
        aggregated_predictions.append(predictions_frame)

        model_metrics.append(result.model_metrics.assign(ticker=ticker))
        ticker_metrics.append(result.per_ticker_metrics.assign(ticker=ticker))

    if not per_ticker_frames:
        print("No tickers produced usable data; aborting case.")
        return case_output_dir

    combined_df = pd.concat(combined_frames, ignore_index=True)
    ticker_bundle = "_".join(t.lower() for t in per_ticker_frames.keys())

    combined_matrix_path = save_dataframe(
        combined_df,
        comparison_dir,
        f"{ticker_bundle}_{CASE_ID}_technical_features{'_yeojohnson' if yeojohnson else ''}_combined.csv",
    )
    print(f"Combined technical feature matrix saved to: {combined_matrix_path}")

    summary = combined_df.groupby("ticker")[TECHNICAL_FEATURE_COLUMNS].describe()
    summary.columns = [f"{feature}_{stat}" for feature, stat in summary.columns]
    summary_path = save_dataframe(
        summary,
        comparison_dir,
        f"{ticker_bundle}_{CASE_ID}_technical_features_summary_by_ticker{'_yeojohnson' if yeojohnson else ''}.csv",
    )
    print(f"Per-ticker technical feature summary saved to: {summary_path}")

    color_map = build_color_map(per_ticker_frames.keys())

    comparison_plot_path = plot_features_boxplot(
        per_ticker_frames,
        TECHNICAL_FEATURE_COLUMNS,
        title=f"{CASE_ID} technical features{' (Yeo-Johnson)' if yeojohnson else ''}: {'/'.join(per_ticker_frames.keys())}",
        output_dir=comparison_dir,
        show=show_plots,
        color_map=color_map,
    )
    print(f"Technical feature boxplot grid saved to: {comparison_plot_path}")

    single_plot_records: list[dict[str, str]] = []
    for feature in TECHNICAL_FEATURE_COLUMNS:
        series_map = {
            ticker: frame[feature]
            for ticker, frame in per_ticker_frames.items()
            if feature in frame.columns
        }
        try:
            plot_path = plot_single_feature_boxplot(
                feature_name=feature,
                series_by_ticker=series_map,
                title=f"{CASE_ID} {feature} comparison{' (Yeo-Johnson)' if yeojohnson else ''}",
                output_dir=single_plots_dir,
                show=show_plots,
                color_map=color_map,
            )
        except ValueError as exc:
            print(f"Skipping feature '{feature}' single plot: {exc}")
            continue

        single_plot_records.append(
            {
                "feature": feature,
                "boxplot_path": str(plot_path),
            }
        )
        print(f"Single-feature technical comparison saved to: {plot_path}")

    if single_plot_records:
        single_index_path = save_dataframe(
            pd.DataFrame(single_plot_records),
            single_plots_dir,
            f"{ticker_bundle}_{CASE_ID}_single_technical_boxplots{'_yeojohnson' if yeojohnson else ''}.csv",
        )
        print(f"Single feature technical boxplot index saved to: {single_index_path}")
    else:
        print("No single-feature technical boxplots were generated.")

    if aggregated_predictions:
        combined_predictions = pd.concat(aggregated_predictions, ignore_index=True)
        predictions_csv_path = save_dataframe(
            combined_predictions,
            validation_dir,
            f"{ticker_bundle}_{CASE_ID}_last_{LAST_N_DAYS}_day_predictions{'_yeojohnson' if yeojohnson else ''}.csv",
        )
        print(f"Validation predictions saved to: {predictions_csv_path}")

        excel_path = validation_dir / f"{ticker_bundle}_{CASE_ID}_last_{LAST_N_DAYS}_day_predictions{'_yeojohnson' if yeojohnson else ''}.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            combined_predictions.to_excel(writer, index=False, sheet_name="predictions")
        print(f"Validation predictions Excel saved to: {excel_path}")
        report_paths = export_last_n_day_prediction_reports(
            combined_predictions,
            case_id=CASE_ID,
            case_output_dir=case_output_dir,
            ticker_bundle=ticker_bundle,
            last_n_days=LAST_N_DAYS,
            yeojohnson=yeojohnson,
        )
        pivot_report_path = report_paths.get("case_pivot_layout")
        if pivot_report_path:
            print(f"Pivot-style validation summary saved to: {pivot_report_path}")
        case_summary_excel = report_paths.get("case_summary_excel")
        if case_summary_excel:
            print(f"Case standard summary workbook saved to: {case_summary_excel}")
        shared_summary_excel = report_paths.get("standard_summary_excel")
        if shared_summary_excel:
            print(f"Shared standard summary workbook saved to: {shared_summary_excel}")

    if model_metrics:
        combined_model_metrics = pd.concat(model_metrics, ignore_index=True)
        metrics_path = save_dataframe(
            combined_model_metrics,
            validation_dir,
            f"{ticker_bundle}_{CASE_ID}_last_{LAST_N_DAYS}_day_model_metrics{'_yeojohnson' if yeojohnson else ''}.csv",
        )
        print(f"Validation model metrics saved to: {metrics_path}")

    if ticker_metrics:
        combined_ticker_metrics = pd.concat(ticker_metrics, ignore_index=True)
        ticker_metrics_path = save_dataframe(
            combined_ticker_metrics,
            validation_dir,
            f"{ticker_bundle}_{CASE_ID}_last_{LAST_N_DAYS}_day_metrics_by_ticker{'_yeojohnson' if yeojohnson else ''}.csv",
        )
        print(f"Validation per-ticker metrics saved to: {ticker_metrics_path}")

    per_ticker_dirs = sorted(t.lower() for t in per_ticker_frames)
    artifacts = {
        "comparison_dir": "comparison",
        "single_feature_plots_dir": "comparison/single_feature_plots",
        "validation_dir": "comparison/last_n_day_validation",
        "models_dir": "models",
        "per_ticker_dirs": per_ticker_dirs,
    }
    reports_dir = case_output_dir / "reports"
    if reports_dir.exists():
        artifacts["reports_dir"] = "reports"

    transformations = []
    if yeojohnson:
        transformations.append("Yeo-Johnson")

    write_case_metadata(
        case_dir=case_output_dir,
        case_id=CASE_ID,
        case_name=CASE_NAME,
        package="eda_boxplots",
        dataset_path=dataset_path,
        tickers=selected_tickers,
        features={
            "technical_features": TECHNICAL_FEATURE_COLUMNS,
        },
        target="target_14d",
        models=tuple(MODEL_DICT.keys()),
        extras={
            "last_n_day_validation": LAST_N_DAYS,
            "artifacts": artifacts,
            "transformations": transformations,
            "yeojohnson_enabled": yeojohnson,
        },
    )

    return case_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the technical-only feature boxplot workflow.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_FILENAME,
        help="Dataset filename inside the ML input directory (default: %(default)s).",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help=f"Tickers to include (default: {' '.join(DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional custom output directory root.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively while generating them.",
    )
    parser.add_argument(
        "--yeojohnson",
        action="store_true",
        help="Apply Yeo-Johnson transformation to the feature matrix before plotting/validation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_root = Path(args.output_root) if args.output_root else None

    run_case(
        tickers=args.tickers,
        dataset_filename=args.dataset,
        output_root=output_root,
        show_plots=args.show_plots,
        yeojohnson=args.yeojohnson,
    )
