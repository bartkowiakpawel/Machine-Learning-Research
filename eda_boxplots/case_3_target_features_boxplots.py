"""EDA workflow producing target-driven boxplots with 14-day horizon features.""" 

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
from feature_scaling._shared import (
    clean_feature_matrix,
    filter_ticker,
    load_ml_dataset,
    prepare_feature_matrix,
    resolve_output_dir,
    save_dataframe,
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

CASE_ID = "case_3"
CASE_CONFIG = get_case_config(CASE_ID)
CASE_NAME = CASE_CONFIG.name

ROLLING_WINDOW = 30
TARGET_SHIFT = 14
LAST_N_DAYS = 15
TARGET_FEATURE_COLUMNS = [
    "rolling_median_target_14d",
    "rolling_iqr_target_14d",
    "rolling_outlier_ratio_target_14d",
]


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


def _outlier_ratio(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if np.isclose(iqr, 0.0):
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return float(mask.mean())


def _compute_target_features(
    subset: pd.DataFrame,
    *,
    window: int,
    target_shift: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if 'Close' not in subset.columns:
        raise KeyError("Missing 'Close' column required for target feature computation.")

    working = subset.sort_values("Date").copy()
    working["target_14d"] = (working["Close"].shift(-target_shift) - working["Close"]) / working["Close"]
    target_series = working["target_14d"]

    rolling_obj = target_series.rolling(window=window, min_periods=window)
    working["rolling_median_target_14d"] = rolling_obj.median()
    working["rolling_iqr_target_14d"] = rolling_obj.quantile(0.75) - rolling_obj.quantile(0.25)
    working["rolling_outlier_ratio_target_14d"] = rolling_obj.apply(_outlier_ratio, raw=False)

    feature_frame = working[TARGET_FEATURE_COLUMNS].copy()
    combined = pd.concat([feature_frame, target_series], axis=1)
    combined = combined.dropna()

    feature_matrix = combined[TARGET_FEATURE_COLUMNS].copy()
    y_target = combined["target_14d"].copy()
    metadata_columns = [col for col in ("Date", "ticker") if col in working.columns]
    metadata = working.loc[combined.index, metadata_columns].copy()

    return feature_matrix, y_target, metadata


def run_case(
    *,
    tickers: Sequence[str] | None = None,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    features: list[str] | None = None,
    output_root: Path | str | None = None,
    show_plots: bool = False,
) -> Path:
    """Run case 3 target-driven feature boxplots for the configured tickers."""

    selected_tickers = _normalize_tickers(tickers)
    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    comparison_dir = case_output_dir / "comparison"
    single_plots_dir = comparison_dir / "single_feature_plots"
    validation_dir = comparison_dir / "last_n_day_validation"
    model_store_dir = case_output_dir / "models"
    aggregated_predictions: list[pd.DataFrame] = []
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

    for ticker in selected_tickers:
        subset = filter_ticker(data, ticker)
        print(f"Rows for ticker '{ticker}': {len(subset)}")

        if subset.empty:
            print(f"No data available for {ticker}; skipping.")
            continue

        try:
            feature_matrix, target_series, metadata = _compute_target_features(
                subset,
                window=ROLLING_WINDOW,
                target_shift=TARGET_SHIFT,
            )
        except KeyError as exc:
            print(f"Skipping {ticker} due to missing data: {exc}")
            continue

        feature_matrix = clean_feature_matrix(feature_matrix)
        mask = feature_matrix.index.intersection(target_series.index)
        feature_matrix = feature_matrix.loc[mask]
        target_series = target_series.loc[mask]
        metadata = metadata.loc[mask]

        if feature_matrix.empty:
            print(f"Feature matrix empty after cleaning for {ticker}; skipping.")
            continue

        ticker_dir = case_output_dir / ticker.lower()
        ticker_dir.mkdir(parents=True, exist_ok=True)

        matrix_path = save_dataframe(
            feature_matrix,
            ticker_dir,
            f"{ticker.lower()}_{CASE_ID}_target_features.csv",
        )
        print(f"Target feature matrix for {ticker} saved to: {matrix_path}")

        summary_path = save_dataframe(
            feature_matrix.describe(include="all").transpose(),
            ticker_dir,
            f"{ticker.lower()}_{CASE_ID}_target_features_summary.csv",
        )
        print(f"Target feature summary for {ticker} saved to: {summary_path}")

        per_ticker_frames[ticker] = feature_matrix.copy()
        combined_frames.append(feature_matrix.assign(ticker=ticker))

        if metadata is None or {"Date", "ticker"}.difference(metadata.columns):
            print(f"Metadata missing required columns for {ticker}; skipping validation.")
            continue

        try:
            result = evaluate_last_n_day_predictions(
                MODEL_DICT,
                feature_matrix,
                target_series,
                metadata=metadata,
                dataset_label=f"{ticker} {CASE_ID} target features",
                n_days=LAST_N_DAYS,
                ticker_column="ticker",
                date_column="Date",
                output_dir=validation_dir,
                plot_prefix=f"{ticker.lower()}_{CASE_ID}",
                n_plot_columns=3,
                show_plot=False,
                model_store_dir=model_store_dir,
            )
        except ValueError as exc:
            print(f"Skipping validation for {ticker}: {exc}")
            continue

        predictions_frame = result.predictions.copy()
        predictions_frame.insert(0, 'dataset', f"{ticker} {CASE_ID} target features")
        if {'actual', 'prediction'}.issubset(predictions_frame.columns):
            predictions_frame['prediction_diff'] = predictions_frame['prediction'] - predictions_frame['actual']
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
        f"{ticker_bundle}_{CASE_ID}_target_features_combined.csv",
    )
    print(f"Combined target feature matrix saved to: {combined_matrix_path}")

    summary = combined_df.groupby("ticker")[TARGET_FEATURE_COLUMNS].describe()
    summary.columns = [f"{feature}_{stat}" for feature, stat in summary.columns]
    summary_path = save_dataframe(
        summary,
        comparison_dir,
        f"{ticker_bundle}_{CASE_ID}_target_features_summary_by_ticker.csv",
    )
    print(f"Per-ticker target feature summary saved to: {summary_path}")

    color_map = build_color_map(per_ticker_frames.keys())

    comparison_plot_path = plot_features_boxplot(
        per_ticker_frames,
        TARGET_FEATURE_COLUMNS,
        title=f"case_3 target features: {'/'.join(per_ticker_frames.keys())}",
        output_dir=comparison_dir,
        show=show_plots,
        color_map=color_map,
    )
    print(f"Target feature boxplot grid saved to: {comparison_plot_path}")

    single_plot_records: list[dict[str, str]] = []
    for feature in TARGET_FEATURE_COLUMNS:
        series_map = {
            ticker: frame[feature]
            for ticker, frame in per_ticker_frames.items()
            if feature in frame.columns
        }
        try:
            plot_path = plot_single_feature_boxplot(
                feature_name=feature,
                series_by_ticker=series_map,
                title=f"case_3 {feature} comparison",
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
        print(f"Single-feature target comparison saved to: {plot_path}")

    if single_plot_records:
        single_index_path = save_dataframe(
            pd.DataFrame(single_plot_records),
            single_plots_dir,
            f"{ticker_bundle}_{CASE_ID}_single_target_boxplots.csv",
        )
        print(f"Single feature target boxplot index saved to: {single_index_path}")
    else:
        print("No single-feature target boxplots were generated.")

    if aggregated_predictions:
        combined_predictions = pd.concat(aggregated_predictions, ignore_index=True)
        predictions_csv_path = save_dataframe(
            combined_predictions,
            validation_dir,
            f"{ticker_bundle}_{CASE_ID}_last_{LAST_N_DAYS}_day_predictions.csv",
        )
        print(f"Validation predictions saved to: {predictions_csv_path}")

        excel_path = validation_dir / f"{ticker_bundle}_{CASE_ID}_last_{LAST_N_DAYS}_day_predictions.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            combined_predictions.to_excel(writer, index=False, sheet_name='predictions')
        print(f"Validation predictions Excel saved to: {excel_path}")

    if model_metrics:
        combined_model_metrics = pd.concat(model_metrics, ignore_index=True)
        metrics_path = save_dataframe(
            combined_model_metrics,
            validation_dir,
            f"{ticker_bundle}_{CASE_ID}_last_{LAST_N_DAYS}_day_model_metrics.csv",
        )
        print(f"Validation model metrics saved to: {metrics_path}")

    if ticker_metrics:
        combined_ticker_metrics = pd.concat(ticker_metrics, ignore_index=True)
        ticker_metrics_path = save_dataframe(
            combined_ticker_metrics,
            validation_dir,
            f"{ticker_bundle}_{CASE_ID}_last_{LAST_N_DAYS}_day_metrics_by_ticker.csv",
        )
        print(f"Validation per-ticker metrics saved to: {ticker_metrics_path}")

    return case_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the target-feature boxplot workflow.",
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
        "--features",
        nargs="*",
        default=None,
        help="Optional list of feature names (unused placeholder for API symmetry).",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively while generating them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_root = Path(args.output_root) if args.output_root else None

    run_case(
        tickers=args.tickers,
        dataset_filename=args.dataset,
        features=args.features,
        output_root=output_root,
        show_plots=args.show_plots,
    )
