"""EDA case 15: scaled technical feature boxplots with 1-day target validation (AAPL only)."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

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
        DEFAULT_OUTPUT_ROOT,
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
        get_case_config,
    )
    from eda_boxplots.visualization import (
        build_color_map,
        plot_features_boxplot,
        plot_single_feature_boxplot,
    )
    from eda_boxplots.reporting import export_last_n_day_prediction_reports

CASE_ID = "case_15"
CASE_CONFIG = get_case_config(CASE_ID)
CASE_NAME = CASE_CONFIG.name

ENFORCED_TICKER = "AAPL"
ROLLING_WINDOW = 30
TARGET_SHIFT = 1
LAST_N_DAYS = 15
LOG_VOLUME_SOURCE = "avg_volume_14"
LOG_VOLUME_FEATURE = "avg_volume_14_log"

TECHNICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "IntradayRange",
    "Volume_vs_MA14",
    LOG_VOLUME_FEATURE,
    "rsi_14",
    "rsi_50",
    "volatility_14",
    "atr_50",
    "macd_diff",
    "cci_14",
    "stoch_d_14",
    "bb_width_14",
)
TARGET_FEATURE_COLUMNS: tuple[str, ...] = (
    "rolling_median_target_1d",
    "rolling_iqr_target_1d",
    "rolling_outlier_ratio_target_1d",
)
ALL_FEATURE_COLUMNS: tuple[str, ...] = TECHNICAL_FEATURE_COLUMNS + TARGET_FEATURE_COLUMNS

FEATURE_SCALERS: OrderedDict[str, str | None] = OrderedDict(
    (
        ("IntradayRange", "StandardScaler"),
        ("Volume_vs_MA14", None),
        (LOG_VOLUME_FEATURE, "StandardScaler"),
        ("rsi_14", "MinMaxScaler"),
        ("rsi_50", "MinMaxScaler"),
        ("volatility_14", "RobustScaler"),
        ("atr_50", "StandardScaler"),
        ("macd_diff", "StandardScaler"),
        ("cci_14", "RobustScaler"),
        ("stoch_d_14", "MinMaxScaler"),
        ("bb_width_14", "RobustScaler"),
    )
)
SCALER_FACTORIES = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    None: None,
}


def _normalize_tickers(tickers: Sequence[str] | None) -> list[str]:
    requested = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
    if requested and set(requested) != {ENFORCED_TICKER}:
        print(
            f"[WARN] Case {CASE_ID} enforces ticker {ENFORCED_TICKER}; ignoring requested list: {', '.join(requested)}"
        )
    return [ENFORCED_TICKER]


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


def _compute_target_features(subset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if "Close" not in subset.columns:
        raise KeyError("Missing 'Close' column required for target feature computation.")

    working = subset.sort_values("Date").copy()
    working["target_1d"] = (working["Close"].shift(-TARGET_SHIFT) - working["Close"]) / working["Close"]
    target_series = working["target_1d"]

    rolling_obj = target_series.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
    working["rolling_median_target_1d"] = rolling_obj.median()
    working["rolling_iqr_target_1d"] = rolling_obj.quantile(0.75) - rolling_obj.quantile(0.25)
    working["rolling_outlier_ratio_target_1d"] = rolling_obj.apply(_outlier_ratio, raw=False)

    feature_frame = working[list(TARGET_FEATURE_COLUMNS)].copy()
    combined = pd.concat([feature_frame, target_series], axis=1).dropna()

    feature_matrix = combined[list(TARGET_FEATURE_COLUMNS)].copy()
    y_target = combined["target_1d"].copy()
    metadata_columns = [col for col in ("Date", "ticker") if col in working.columns]
    metadata = working.loc[combined.index, metadata_columns].copy()

    return feature_matrix, y_target, metadata


def _ensure_log_feature(subset: pd.DataFrame) -> pd.DataFrame:
    working = subset.copy()
    if LOG_VOLUME_SOURCE not in working.columns:
        raise KeyError(
            f"Expected '{LOG_VOLUME_SOURCE}' column to derive {LOG_VOLUME_FEATURE} but it is missing in dataset."
        )
    log_values = np.log1p(pd.to_numeric(working[LOG_VOLUME_SOURCE], errors="coerce").clip(lower=0.0))
    working[LOG_VOLUME_FEATURE] = log_values
    return working


def _scale_features(frame: pd.DataFrame) -> pd.DataFrame:
    scaled_columns = []
    for feature, scaler_name in FEATURE_SCALERS.items():
        if feature not in frame.columns:
            raise KeyError(f"Feature '{feature}' missing after preparation; cannot scale.")
        data = frame[[feature]].copy()
        factory = SCALER_FACTORIES.get(scaler_name)
        if factory is None:
            scaled_series = data[feature]
        else:
            scaler = factory()
            scaled_array = scaler.fit_transform(data.values)
            scaled_series = pd.Series(scaled_array.ravel(), index=data.index, name=feature)
        scaled_columns.append(scaled_series)
    return pd.concat(scaled_columns, axis=1)


def run_case(
    *,
    tickers: Sequence[str] | None = None,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    output_root: Path | str | None = None,
    show_plots: bool = False,
    yeojohnson: bool = False,
) -> Path:
    """Run the targeted scaled-feature workflow for ticker AAPL."""

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
    print(f"Enforced ticker: {ENFORCED_TICKER}")
    print(f"Dataset path: {dataset_path}")

    data = load_ml_dataset(dataset_path)

    per_ticker_frames: dict[str, pd.DataFrame] = {}
    combined_frames: list[pd.DataFrame] = []
    model_metrics = []
    ticker_metrics = []
    aggregated_predictions: list[pd.DataFrame] = []

    for ticker in selected_tickers:
        subset = filter_ticker(data, ticker)
        print(f"Rows for {ticker}: {len(subset)}")

        if subset.empty:
            print(f"No data available for {ticker}; skipping.")
            continue

        subset = _ensure_log_feature(subset)

        try:
            target_features, target_series, metadata = _compute_target_features(subset)
        except KeyError as exc:
            print(f"Skipping {ticker} due to missing data: {exc}")
            continue

        try:
            technical_matrix = prepare_feature_matrix(
                subset,
                TECHNICAL_FEATURE_COLUMNS,
                target_column=None,
                target_source=None,
            )
        except KeyError as exc:
            print(f"Skipping {ticker} due to missing technical features: {exc}")
            continue

        technical_matrix = clean_feature_matrix(technical_matrix)
        target_features = clean_feature_matrix(target_features)

        common_index = target_features.index.intersection(technical_matrix.index)
        if metadata is not None:
            metadata = metadata.loc[common_index]
        target_series = target_series.loc[common_index]

        if technical_matrix.empty or target_features.empty:
            print(f"Insufficient data after cleaning for {ticker}; skipping.")
            continue

        scaled_technical = _scale_features(technical_matrix.loc[common_index])
        combined_matrix = pd.concat([scaled_technical, target_features.loc[common_index]], axis=1, join="inner")
        combined_matrix = clean_feature_matrix(combined_matrix)
        target_series = target_series.loc[combined_matrix.index]
        if metadata is not None:
            metadata = metadata.loc[combined_matrix.index]

        if combined_matrix.empty:
            print(f"Feature matrix empty after alignment for {ticker}; skipping.")
            continue

        if yeojohnson:
            from sklearn.preprocessing import PowerTransformer

            transformer = PowerTransformer(method="yeo-johnson")
            transformed = transformer.fit_transform(combined_matrix)
            combined_matrix = pd.DataFrame(transformed, columns=combined_matrix.columns, index=combined_matrix.index)

        ticker_dir = case_output_dir / ticker.lower()
        ticker_dir.mkdir(parents=True, exist_ok=True)

        matrix_path = save_dataframe(
            combined_matrix,
            ticker_dir,
            f"{ticker.lower()}_{CASE_ID}_scaled_features{'_yeojohnson' if yeojohnson else ''}.csv",
        )
        print(f"Scaled feature matrix for {ticker} saved to: {matrix_path}")

        summary_path = save_dataframe(
            combined_matrix.describe(include="all").transpose(),
            ticker_dir,
            f"{ticker.lower()}_{CASE_ID}_scaled_features_summary{'_yeojohnson' if yeojohnson else ''}.csv",
        )
        print(f"Feature summary for {ticker} saved to: {summary_path}")

        per_ticker_frames[ticker] = combined_matrix.copy()
        combined_frames.append(combined_matrix.assign(ticker=ticker))

        if metadata is None or {"Date", "ticker"}.difference(metadata.columns):
            print(f"Metadata missing required columns for {ticker}; skipping validation.")
            continue

        try:
            result = evaluate_last_n_day_predictions(
                MODEL_DICT,
                combined_matrix,
                target_series,
                metadata=metadata,
                dataset_label=f"{ticker} {CASE_ID} scaled features",
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
        predictions_frame.insert(0, "dataset", f"{ticker} {CASE_ID} scaled features")
        if {"actual", "prediction"}.issubset(predictions_frame.columns):
            predictions_frame["prediction_diff"] = predictions_frame["prediction"] - predictions_frame["actual"]
        aggregated_predictions.append(predictions_frame)

        model_metrics.append(result.model_metrics.assign(ticker=ticker))
        ticker_metrics.append(result.per_ticker_metrics.assign(ticker=ticker))

    if not per_ticker_frames:
        print("No usable data generated; aborting case.")
        return case_output_dir

    combined_df = pd.concat(combined_frames, ignore_index=True)
    ticker_bundle = "_".join(t.lower() for t in per_ticker_frames.keys())

    combined_matrix_path = save_dataframe(
        combined_df,
        comparison_dir,
        f"{ticker_bundle}_{CASE_ID}_scaled_features{'_yeojohnson' if yeojohnson else ''}_combined.csv",
    )
    print(f"Combined scaled feature matrix saved to: {combined_matrix_path}")

    summary = combined_df.groupby("ticker")[list(ALL_FEATURE_COLUMNS)].describe()
    summary.columns = [f"{feature}_{stat}" for feature, stat in summary.columns]
    summary_path = save_dataframe(
        summary,
        comparison_dir,
        f"{ticker_bundle}_{CASE_ID}_scaled_features_summary_by_ticker{'_yeojohnson' if yeojohnson else ''}.csv",
    )
    print(f"Per-ticker summary saved to: {summary_path}")

    color_map = build_color_map(per_ticker_frames.keys())

    comparison_plot_path = plot_features_boxplot(
        {ticker: frame[list(ALL_FEATURE_COLUMNS)] for ticker, frame in per_ticker_frames.items()},
        list(ALL_FEATURE_COLUMNS),
        title=f"{CASE_ID} scaled features{' (Yeo-Johnson)' if yeojohnson else ''}: {'/'.join(per_ticker_frames.keys())}",
        output_dir=comparison_dir,
        show=show_plots,
        color_map=color_map,
    )
    print(f"Scaled feature boxplot grid saved to: {comparison_plot_path}")

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

        single_plot_records.append({"feature": feature, "boxplot_path": str(plot_path)})
        print(f"Single-feature comparison saved to: {plot_path}")

    if single_plot_records:
        single_index_path = save_dataframe(
            pd.DataFrame(single_plot_records),
            single_plots_dir,
            f"{ticker_bundle}_{CASE_ID}_single_scaled_boxplots{'_yeojohnson' if yeojohnson else ''}.csv",
        )
        print(f"Single feature boxplot index saved to: {single_index_path}")
    else:
        print("No single-feature boxplots generated.")

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
            print(f"Case summary workbook saved to: {case_summary_excel}")
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

    artifacts = {
        "comparison_dir": "comparison",
        "single_feature_plots_dir": "comparison/single_feature_plots",
        "validation_dir": "comparison/last_n_day_validation",
        "models_dir": "models",
        "per_ticker_dirs": sorted(t.lower() for t in per_ticker_frames),
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
            "technical_features": list(TECHNICAL_FEATURE_COLUMNS),
            "target_features": list(TARGET_FEATURE_COLUMNS),
            "scalers": {feature: (scaler or "None") for feature, scaler in FEATURE_SCALERS.items()},
            "target_rolling_window": ROLLING_WINDOW,
        },
        target="target_1d",
        models=tuple(MODEL_DICT.keys()),
        extras={
            "last_n_day_validation": LAST_N_DAYS,
            "artifacts": artifacts,
            "yeojohnson_enabled": yeojohnson,
            "target_shift_days": TARGET_SHIFT,
            "log_volume_source": LOG_VOLUME_SOURCE,
        },
    )

    return case_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scaled technical feature workflow (AAPL only).",
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
        help="Tickers to include (ignored; case enforces AAPL).",
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
        help="Apply Yeo-Johnson transformation after scaling.",
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



