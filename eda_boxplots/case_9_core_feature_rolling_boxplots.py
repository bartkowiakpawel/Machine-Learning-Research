"""EDA workflow combining target horizon metrics with rolling statistics of core features."""

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
from sklearn.preprocessing import PowerTransformer

from config import ML_INPUT_DIR, MODEL_DICT
from core.shared_utils import (
    clean_feature_matrix,
    filter_ticker,
    load_ml_dataset,
    resolve_output_dir,
    save_dataframe,
    write_case_metadata,
)
from core.prediction_validation import evaluate_last_n_day_predictions

try:
    from .settings import (
        DEFAULT_DATASET_FILENAME,
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

CASE_ID = "case_9"
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
CORE_FEATURE_COLUMNS = [
    "OpenCloseReturn",
    "IntradayRange",
    "rsi_14",
    "Volume_vs_MA14",
]
CORE_ROLLING_FEATURE_COLUMNS = [
    feature
    for column in CORE_FEATURE_COLUMNS
    for feature in (
        f"rolling_median_{column.lower()}_14d",
        f"rolling_iqr_{column.lower()}_14d",
        f"rolling_outlier_ratio_{column.lower()}_14d",
    )
]
ALL_FEATURE_COLUMNS = TARGET_FEATURE_COLUMNS + CORE_ROLLING_FEATURE_COLUMNS


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
    if "Close" not in subset.columns:
        raise KeyError("Missing 'Close' column required for target feature computation.")

    working = subset.sort_values("Date").copy()
    working["target_14d"] = (working["Close"].shift(-target_shift) - working["Close"]) / working["Close"]
    target_series = working["target_14d"]

    rolling_obj = target_series.rolling(window=window, min_periods=window)
    working["rolling_median_target_14d"] = rolling_obj.median()
    working["rolling_iqr_target_14d"] = rolling_obj.quantile(0.75) - rolling_obj.quantile(0.25)
    working["rolling_outlier_ratio_target_14d"] = rolling_obj.apply(_outlier_ratio, raw=False)

    feature_frame = working[TARGET_FEATURE_COLUMNS].copy()
    combined = pd.concat([feature_frame, target_series], axis=1).dropna()

    feature_matrix = combined[TARGET_FEATURE_COLUMNS].copy()
    y_target = combined["target_14d"].copy()
    metadata_columns = [col for col in ("Date", "ticker") if col in working.columns]
    metadata = working.loc[combined.index, metadata_columns].copy()

    return feature_matrix, y_target, metadata


def _compute_core_feature_rolling_statistics(
    subset: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    missing = [col for col in CORE_FEATURE_COLUMNS if col not in subset.columns]
    if missing:
        missing_fmt = ", ".join(missing)
        raise KeyError(f"Missing required core features for rolling statistics: {missing_fmt}")

    working = subset.sort_values("Date").copy()
    feature_store: dict[str, pd.Series] = {}

    for column in CORE_FEATURE_COLUMNS:
        series = pd.to_numeric(working[column], errors="coerce")
        rolling_obj = series.rolling(window=window, min_periods=window)
        base = column.lower()
        feature_store[f"rolling_median_{base}_14d"] = rolling_obj.median()
        feature_store[f"rolling_iqr_{base}_14d"] = rolling_obj.quantile(0.75) - rolling_obj.quantile(0.25)
        feature_store[f"rolling_outlier_ratio_{base}_14d"] = rolling_obj.apply(_outlier_ratio, raw=False)

    features_df = pd.DataFrame(feature_store)
    features_df = features_df.dropna()
    return features_df


def run_case(
    *,
    tickers: Sequence[str] | None = None,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    output_root: Path | str | None = None,
    show_plots: bool = False,
    yeojohnson: bool = False,
) -> Path:
    """Run case 9 combining target metrics with rolling statistics of core features."""

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

        try:
            target_features, target_series, metadata = _compute_target_features(
                subset,
                window=ROLLING_WINDOW,
                target_shift=TARGET_SHIFT,
            )
        except KeyError as exc:
            print(f"Skipping {ticker} due to missing data: {exc}")
            continue

        try:
            core_features = _compute_core_feature_rolling_statistics(
                subset,
                window=ROLLING_WINDOW,
            )
        except KeyError as exc:
            print(f"Skipping {ticker} due to missing core feature data: {exc}")
            continue

        common_index = target_features.index.intersection(core_features.index)
        if metadata is not None:
            metadata = metadata.loc[common_index]
        target_series = target_series.loc[common_index]

        combined_matrix = pd.concat(
            [
                target_features.loc[common_index],
                core_features.loc[common_index],
            ],
            axis=1,
            join="inner",
        )
        combined_matrix = combined_matrix.loc[:, ALL_FEATURE_COLUMNS]
        combined_matrix = clean_feature_matrix(combined_matrix)
        if combined_matrix.empty:
            print(f"Combined feature matrix empty after cleaning for {ticker}; skipping.")
            continue
        target_series = target_series.loc[combined_matrix.index]
        if metadata is not None:
            metadata = metadata.loc[combined_matrix.index]

        if metadata is None or metadata.empty:
            print(f"Insufficient metadata for validation for {ticker}; skipping.")
            continue

        ticker_dir = case_output_dir / ticker.lower()
        ticker_dir.mkdir(parents=True, exist_ok=True)

        matrix_path = save_dataframe(
            combined_matrix,
            ticker_dir,
            f"{ticker.lower()}_{CASE_ID}_core_feature_rolling_features{'_yeojohnson' if yeojohnson else ''}.csv",
        )
        print(f"Core rolling + target feature matrix for {ticker} saved to: {matrix_path}")

        summary_path = save_dataframe(
            combined_matrix.describe(include="all").transpose(),
            ticker_dir,
            f"{ticker.lower()}_{CASE_ID}_core_feature_rolling_features_summary{'_yeojohnson' if yeojohnson else ''}.csv",
        )
        print(f"Core rolling + target feature summary for {ticker} saved to: {summary_path}")

        if yeojohnson:
            transformer = PowerTransformer(method="yeo-johnson")
            transformed_array = transformer.fit_transform(combined_matrix)
            combined_matrix = pd.DataFrame(
                transformed_array,
                columns=combined_matrix.columns,
                index=combined_matrix.index,
            )

        per_ticker_frames[ticker] = combined_matrix[ALL_FEATURE_COLUMNS].copy()
        combined_frames.append(combined_matrix.assign(ticker=ticker))

        dataset_label = f"{ticker} {CASE_ID} core rolling features"
        if yeojohnson:
            dataset_label += " (Yeo-Johnson)"

        try:
            result = evaluate_last_n_day_predictions(
                MODEL_DICT,
                combined_matrix,
                target_series,
                metadata=metadata,
                dataset_label=dataset_label,
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
        predictions_frame.insert(0, "dataset", dataset_label)
        predictions_frame["ticker"] = ticker
        if {"actual", "prediction"}.issubset(predictions_frame.columns):
            predictions_frame["prediction_diff"] = predictions_frame["prediction"] - predictions_frame["actual"]

        model_metrics.append(result.model_metrics.assign(ticker=ticker))
        ticker_metrics.append(result.per_ticker_metrics.assign(ticker=ticker))
        aggregated_predictions.append(predictions_frame)

    if not per_ticker_frames:
        print("No tickers produced valid feature matrices; exiting.")
        return case_output_dir

    combined_df = pd.concat(combined_frames, ignore_index=True)
    ticker_bundle = "_".join(t.lower() for t in per_ticker_frames.keys())

    combined_matrix_path = save_dataframe(
        combined_df,
        comparison_dir,
        f"{ticker_bundle}_{CASE_ID}_core_feature_rolling_features{'_yeojohnson' if yeojohnson else ''}_combined.csv",
    )
    print(f"Combined core rolling + target feature matrix saved to: {combined_matrix_path}")

    summary = combined_df.groupby("ticker")[ALL_FEATURE_COLUMNS].describe()
    summary.columns = [f"{feature}_{stat}" for feature, stat in summary.columns]
    summary_path = save_dataframe(
        summary,
        comparison_dir,
        f"{ticker_bundle}_{CASE_ID}_core_feature_rolling_features_summary_by_ticker{'_yeojohnson' if yeojohnson else ''}.csv",
    )
    print(f"Per-ticker core rolling + target feature summary saved to: {summary_path}")

    color_map = build_color_map(per_ticker_frames.keys())

    comparison_plot_path = plot_features_boxplot(
        per_ticker_frames,
        ALL_FEATURE_COLUMNS,
        title=f"{CASE_ID} core rolling features{' (Yeo-Johnson)' if yeojohnson else ''}: {'/'.join(per_ticker_frames.keys())}",
        output_dir=comparison_dir,
        show=show_plots,
        color_map=color_map,
    )
    print(f"Core rolling feature boxplot grid saved to: {comparison_plot_path}")

    single_plot_records: list[dict[str, str]] = []
    for feature in ALL_FEATURE_COLUMNS:
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
        print(f"Single-feature core rolling comparison saved to: {plot_path}")

    if single_plot_records:
        single_index_path = save_dataframe(
            pd.DataFrame(single_plot_records),
            single_plots_dir,
            f"{ticker_bundle}_{CASE_ID}_single_core_feature_rolling_boxplots{'_yeojohnson' if yeojohnson else ''}.csv",
        )
        print(f"Single-feature core rolling boxplot index saved to: {single_index_path}")
    else:
        print("No single-feature core rolling boxplots were generated.")

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

    write_case_metadata(
        case_dir=case_output_dir,
        case_id=CASE_ID,
        case_name=CASE_NAME,
        package="eda_boxplots",
        dataset_path=dataset_path,
        tickers=selected_tickers,
        features={
            "target_features": TARGET_FEATURE_COLUMNS,
            "core_feature_rolling_features": CORE_ROLLING_FEATURE_COLUMNS,
            "combined_features": ALL_FEATURE_COLUMNS,
        },
        target="target_14d",
        models=tuple(MODEL_DICT.keys()),
        extras={
            "last_n_day_validation": LAST_N_DAYS,
            "artifacts": artifacts,
            "transformations": ["Yeo-Johnson"] if yeojohnson else [],
            "yeojohnson_enabled": yeojohnson,
            "target_shift_days": TARGET_SHIFT,
            "rolling_window": ROLLING_WINDOW,
        },
    )

    return case_output_dir



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the core rolling feature boxplot workflow.",
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




