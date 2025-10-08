"""Case 18: TSLA combined feature distribution comparisons."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api import types as ptypes

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
    from .reporting import export_last_n_day_prediction_reports
except ImportError:  # pragma: no cover - fallback for direct execution
    from eda_boxplots.reporting import export_last_n_day_prediction_reports

from eda_boxplots.settings import DEFAULT_DATASET_FILENAME, DEFAULT_OUTPUT_ROOT, DEFAULT_TICKER, DEFAULT_FEATURES, get_case_config

CASE_ID = "case_18"
CASE_CONFIG = get_case_config(CASE_ID)
CASE_NAME = CASE_CONFIG.name

ROLLING_WINDOW = 30
TARGET_SHIFT = 1
LAST_N_DAYS = 15
TARGET_COLUMN = "target_1d"
INTRADAY_ROLLING_WINDOW = 14
TARGET_FEATURE_COLUMNS = [
    "rolling_median_target_1d",
    "rolling_iqr_target_1d",
    "rolling_outlier_ratio_target_1d",
]
INTRADAY_FEATURE_COLUMNS = [
    "intradayrange_volatility",
    "intradayrange_zscore",
    "intradayrange_risk_level",
    "intradayrange_risk_level_code",
]
DEFAULT_COMBINED_FEATURES = list(DEFAULT_FEATURES) + TARGET_FEATURE_COLUMNS + INTRADAY_FEATURE_COLUMNS
TSLA_TICKER = "TSLA"
LAST_N_DIR = Path("comparison") / "last_n_day_validation"


def _normalize_ticker(candidate: str | None) -> str:
    if candidate is None:
        return TSLA_TICKER
    value = str(candidate).strip()
    if not value:
        return TSLA_TICKER
    return value.upper()


def _prepare_feature_series(df: pd.DataFrame, ticker: str, feature: str) -> pd.Series:
    subset = filter_ticker(df, ticker)
    if subset.empty:
        raise ValueError(f"No rows found for ticker '{ticker}'.")
    if feature not in subset.columns:
        raise KeyError(f"Feature '{feature}' not found in dataset.")
    series_raw = subset[feature]
    if ptypes.is_numeric_dtype(series_raw):
        series = pd.to_numeric(series_raw, errors="coerce")
    else:
        category = series_raw.astype("category")
        series = (
            category.cat.codes.replace(-1, np.nan).astype(float)
        )
    series = series.dropna()
    if series.empty:
        raise ValueError(f"No numeric data available for feature '{feature}' and ticker '{ticker}'.")
    return series


def _prepare_model_matrix(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for feature in features:
        if feature not in df.columns:
            raise KeyError(f"Feature '{feature}' missing when preparing model matrix.")
        series_raw = df[feature]
        if ptypes.is_numeric_dtype(series_raw):
            numeric = pd.to_numeric(series_raw, errors="coerce")
        else:
            category = series_raw.astype("category")
            numeric = category.cat.codes.replace(-1, np.nan).astype(float)
        columns[feature] = numeric
    return pd.DataFrame(columns, index=df.index)


def _augment_dataset(df: pd.DataFrame, *, window: int, target_shift: int) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise KeyError(
            "Dataset missing 'Close' column required to compute target-based features."
        )
    if "Date" not in df.columns:
        raise KeyError(
        "Dataset missing 'Date' column required to compute rolling target statistics."
    )

    augmented = df.copy()
    augmented["Date"] = pd.to_datetime(augmented["Date"], errors="coerce")
    if augmented["Date"].isna().any():
        raise ValueError("Date column contains values that could not be parsed to datetime.")

    for ticker, group in augmented.groupby("ticker"):
        ordered = group.sort_values("Date").copy()

        # Compute 1-day forward return target features
        target_col = ordered["Close"].shift(-target_shift)
        ordered[TARGET_COLUMN] = (target_col - ordered["Close"]) / ordered["Close"]
        rolling_obj = ordered[TARGET_COLUMN].rolling(window=window, min_periods=window)
        ordered["rolling_median_target_1d"] = rolling_obj.median()
        ordered["rolling_iqr_target_1d"] = rolling_obj.quantile(0.75) - rolling_obj.quantile(0.25)
        ordered["rolling_outlier_ratio_target_1d"] = rolling_obj.apply(
            _outlier_ratio, raw=False
        )

        # Compute IntradayRange-derived features
        if "IntradayRange" not in ordered.columns:
            raise KeyError("Dataset missing 'IntradayRange' column required for intraday features.")
        intraday = pd.to_numeric(ordered["IntradayRange"], errors="coerce")
        rolling_std = intraday.rolling(
            INTRADAY_ROLLING_WINDOW, min_periods=INTRADAY_ROLLING_WINDOW
        ).std()
        rolling_mean = intraday.rolling(
            INTRADAY_ROLLING_WINDOW, min_periods=INTRADAY_ROLLING_WINDOW
        ).mean()
        zscore = (intraday - rolling_mean) / rolling_std
        zscore = zscore.where(~np.isclose(rolling_std, 0.0), np.nan)

        risk_series = pd.Series(pd.NA, index=ordered.index, dtype="object")
        valid_idx = intraday.dropna().index
        if valid_idx.size >= 3:
            try:
                risk_values = pd.qcut(
                    intraday.loc[valid_idx],
                    q=3,
                    labels=["low_vol", "medium_vol", "high_vol"],
                    duplicates="drop",
                )
                risk_series.loc[valid_idx] = risk_values.astype("object")
            except ValueError:
                risk_series.loc[valid_idx] = "medium_vol"
        risk_cat = pd.Categorical(
            risk_series,
            categories=["low_vol", "medium_vol", "high_vol"],
            ordered=True,
        )
        risk_codes = pd.Series(risk_cat.codes, index=ordered.index).replace(-1, np.nan)

        ordered["intradayrange_volatility"] = rolling_std
        ordered["intradayrange_zscore"] = zscore
        ordered["intradayrange_risk_level"] = risk_series
        ordered["intradayrange_risk_level_code"] = risk_codes

        augmented.loc[ordered.index, [TARGET_COLUMN] + TARGET_FEATURE_COLUMNS + INTRADAY_FEATURE_COLUMNS] = ordered[
            [TARGET_COLUMN] + TARGET_FEATURE_COLUMNS + INTRADAY_FEATURE_COLUMNS
        ]
    return augmented


def _collect_last_n_day_validation(
    outputs_root: Path,
    ticker: str,
) -> list[dict[str, str]]:
    ticker_upper = ticker.upper()
    matches: list[dict[str, str]] = []
    for case_dir in sorted(outputs_root.glob("case_*")):
        validation_dir = case_dir / LAST_N_DIR
        if not validation_dir.exists():
            continue
        for csv_path in validation_dir.glob("*_last_*_day_predictions*.csv"):
            try:
                frame = pd.read_csv(csv_path, usecols=["ticker"])
            except ValueError:
                frame = pd.read_csv(csv_path)
            if frame.empty or "ticker" not in frame.columns:
                continue
            tickers = frame["ticker"].astype(str).str.upper()
            if ticker_upper not in tickers.unique():
                continue
            try:
                relative = csv_path.relative_to(outputs_root)
                path_str = relative.as_posix()
            except ValueError:
                path_str = str(csv_path)
            matches.append(
                {
                    "case_id": case_dir.name,
                    "predictions_file": path_str,
                }
            )
    return matches


def _to_relative(path: Path | str, base: Path) -> str:
    path_obj = Path(path)
    try:
        return path_obj.relative_to(base).as_posix()
    except ValueError:
        return path_obj.as_posix()


def _outlier_ratio(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if np.isclose(iqr, 0.0):
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return float(mask.mean())


def _compute_summary_stats(series: pd.Series) -> dict[str, float]:
    quantiles = np.percentile(series, [25, 50, 75])
    q1, median, q3 = map(float, quantiles)
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "min": float(series.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(series.max()),
        "iqr": float(q3 - q1),
        "skew": float(series.skew()),
        "kurtosis": float(series.kurt()),
    }


def _plot_distribution_comparison(
    series: pd.Series,
    *,
    ticker: str,
    feature: str,
    summary: dict[str, float],
    output_dir: Path,
    show: bool,
) -> Path:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))

    median = summary["median"]
    q1 = summary["q1"]
    q3 = summary["q3"]
    iqr = summary["iqr"]
    skew = summary["skew"]

    sns.boxplot(y=series, color="#86c5da", ax=axes[0])
    axes[0].set_title("Boxplot")
    axes[0].set_ylabel("Value")
    axes[0].axhline(median, ls="--", lw=1.1, color="tab:orange", label=f"median={median:.4f}")
    axes[0].legend(loc="upper right")

    bins = max(10, int(math.sqrt(summary["count"])))
    sns.histplot(series, bins=bins, kde=False, ax=axes[1], color="#5d9fc7")
    axes[1].set_title("Histogram")
    axes[1].set_xlabel("Value")
    axes[1].axvline(median, ls="--", lw=1.1, color="tab:orange")

    sns.violinplot(y=series, inner="quartile", color="#b3a2c8", ax=axes[2])
    axes[2].set_title("Violin plot")
    axes[2].set_ylabel("Value")

    suptitle = (
        f"{CASE_NAME}\n"
        f"{ticker} | {feature} | Q1={q1:.4f} | Q3={q3:.4f} | IQR={iqr:.4f} | "
        f"median={median:.4f} | skew={skew:.3f}"
    )
    fig.suptitle(suptitle, fontsize=12, y=1.05)
    fig.tight_layout()

    safe_ticker = "".join(ch if ch.isalnum() else "_" for ch in ticker).strip("_") or "ticker"
    safe_feature = "".join(ch if ch.isalnum() else "_" for ch in feature).strip("_") or "feature"
    filename = f"{safe_ticker.lower()}_{safe_feature.lower()}_{CASE_ID}_distribution.png"
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def run_case(
    *,
    ticker: str | None = None,
    features: Sequence[str] | None = None,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    output_root: Path | None = None,
    show_plots: bool = False,
) -> Path:
    dataset_path = ML_INPUT_DIR / dataset_filename
    print(f"Loading dataset from {dataset_path} ...")
    df = load_ml_dataset(dataset_path)
    df = _augment_dataset(df, window=ROLLING_WINDOW, target_shift=TARGET_SHIFT)

    selected_ticker = _normalize_ticker(ticker or TSLA_TICKER)
    selected_features = list(features) if features else list(DEFAULT_COMBINED_FEATURES)
    print(f"Selected ticker: {selected_ticker}")
    print(f"Selected features: {', '.join(selected_features)}")

    missing_features = [feature for feature in selected_features if feature not in df.columns]
    if missing_features:
        missing_fmt = ", ".join(missing_features)
        raise KeyError(f"Dataset missing required features: {missing_fmt}")

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    comparison_root = case_output_dir / "comparison"
    distribution_dir = comparison_root / "distribution_comparison"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = comparison_root / "last_n_day_validation"
    models_dir = case_output_dir / "models"

    subset = filter_ticker(df, selected_ticker).sort_values("Date").copy()
    if subset.empty:
        raise ValueError(f"No rows available for ticker '{selected_ticker}' after augmentation.")

    model_matrix_raw = _prepare_model_matrix(subset, selected_features)
    target_series = pd.to_numeric(subset[TARGET_COLUMN], errors="coerce")
    metadata_columns = [col for col in ("Date", "ticker") if col in subset.columns]
    metadata = subset.loc[:, metadata_columns].copy()
    if "ticker" not in metadata.columns:
        metadata["ticker"] = selected_ticker
    if "Date" in metadata.columns:
        metadata["Date"] = pd.to_datetime(metadata["Date"], errors="coerce")

    common_idx = model_matrix_raw.index.intersection(target_series.index)
    model_matrix_raw = model_matrix_raw.loc[common_idx]
    target_series = target_series.loc[common_idx]
    metadata = metadata.loc[common_idx]

    model_matrix = clean_feature_matrix(model_matrix_raw)
    target_series = target_series.loc[model_matrix.index]
    metadata = metadata.loc[model_matrix.index]

    valid_target_idx = target_series.dropna().index
    model_matrix = model_matrix.loc[valid_target_idx]
    target_series = target_series.loc[valid_target_idx]
    metadata = metadata.loc[valid_target_idx]

    validation_artifacts: dict[str, object] = {}
    if model_matrix.empty or len(model_matrix) <= LAST_N_DAYS:
        print("Not enough samples available for last-N-day validation; skipping.")
        validation_result = None
    else:
        validation_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        dataset_label = f"{selected_ticker} {CASE_ID} combined features"
        try:
            validation_result = evaluate_last_n_day_predictions(
                MODEL_DICT,
                model_matrix,
                target_series,
                metadata=metadata,
                dataset_label=dataset_label,
                n_days=LAST_N_DAYS,
                ticker_column="ticker",
                date_column="Date",
                output_dir=validation_dir,
                plot_prefix=f"{selected_ticker.lower()}_{CASE_ID}",
                n_plot_columns=3,
                show_plot=show_plots,
                model_store_dir=models_dir,
            )
        except ValueError as exc:
            print(f"Skipping last-N-day validation: {exc}")
            validation_result = None
    if validation_result is not None:
        ticker_slug = selected_ticker.lower()
        prefix = f"{ticker_slug}_{CASE_ID}_last_{LAST_N_DAYS}_day"

        predictions_df = validation_result.predictions.copy()
        if {"actual", "prediction"}.issubset(predictions_df.columns) and "prediction_diff" not in predictions_df.columns:
            predictions_df["prediction_diff"] = predictions_df["prediction"] - predictions_df["actual"]

        predictions_csv_path = save_dataframe(
            predictions_df,
            validation_dir,
            f"{prefix}_predictions.csv",
        )
        print(f"Validation predictions saved to: {predictions_csv_path}")

        predictions_excel_path = validation_dir / f"{prefix}_predictions.xlsx"
        with pd.ExcelWriter(predictions_excel_path) as writer:
            predictions_df.to_excel(writer, index=False, sheet_name="predictions")
        print(f"Validation predictions Excel saved to: {predictions_excel_path}")

        model_metrics_path: Path | None = None
        if not validation_result.model_metrics.empty:
            model_metrics_path = save_dataframe(
                validation_result.model_metrics,
                validation_dir,
                f"{prefix}_model_metrics.csv",
            )
            print(f"Validation model metrics saved to: {model_metrics_path}")

        ticker_metrics_path: Path | None = None
        if not validation_result.per_ticker_metrics.empty:
            ticker_metrics_path = save_dataframe(
                validation_result.per_ticker_metrics,
                validation_dir,
                f"{prefix}_metrics_by_ticker.csv",
            )
            print(f"Validation per-ticker metrics saved to: {ticker_metrics_path}")

        report_paths = export_last_n_day_prediction_reports(
            predictions_df,
            case_id=CASE_ID,
            case_output_dir=case_output_dir,
            ticker_bundle=ticker_slug,
            last_n_days=LAST_N_DAYS,
            yeojohnson=False,
        )

        validation_artifacts = {
            "predictions_csv": _to_relative(predictions_csv_path, case_output_dir),
            "predictions_excel": _to_relative(predictions_excel_path, case_output_dir),
            "model_metrics_csv": _to_relative(model_metrics_path, case_output_dir)
            if model_metrics_path is not None
            else None,
            "ticker_metrics_csv": _to_relative(ticker_metrics_path, case_output_dir)
            if ticker_metrics_path is not None
            else None,
            "plot_paths": {
                model_name: (
                    _to_relative(plot_path, case_output_dir)
                    if plot_path is not None and plot_path.exists()
                    else None
                )
                for model_name, plot_path in validation_result.plot_paths.items()
            },
            "reports": {
                key: _to_relative(path, case_output_dir)
                for key, path in report_paths.items()
                if path is not None
            },
        }

    summary_records: list[dict[str, float]] = []
    figure_paths: dict[str, Path] = {}

    for feature in selected_features:
        series = _prepare_feature_series(df, selected_ticker, feature)
        summary = _compute_summary_stats(series)
        summary_record = {"ticker": selected_ticker, "feature": feature, **summary}
        summary_records.append(summary_record)

        figure_path = _plot_distribution_comparison(
            series,
            ticker=selected_ticker,
            feature=feature,
            summary=summary,
            output_dir=distribution_dir,
            show=show_plots,
        )
        figure_paths[feature] = figure_path
        print(f"[INFO] Saved plots for {feature} to {figure_path}")

    summary_frame = pd.DataFrame(summary_records)
    summary_filename = f"{selected_ticker.lower()}_{CASE_ID}_feature_stats.csv"
    summary_path = save_dataframe(summary_frame, distribution_dir, summary_filename)
    print(f"Summary statistics saved to: {summary_path}")

    artifacts = {
        "comparison_dir": "comparison",
        "distribution_dir": "comparison/distribution_comparison",
        "summary_csv": summary_path.relative_to(case_output_dir).as_posix(),
        "figures": {
            feature: path.relative_to(case_output_dir).as_posix()
            for feature, path in figure_paths.items()
        },
    }
    if validation_artifacts:
        artifacts["validation_dir"] = "comparison/last_n_day_validation"
        artifacts["models_dir"] = "models"
        artifacts["validation_outputs"] = validation_artifacts

    validation_matches = _collect_last_n_day_validation(DEFAULT_OUTPUT_ROOT, selected_ticker)
    if validation_matches:
        print("Found last-N-day validation predictions for the selected ticker:")
        for match in validation_matches:
            print(f"  - {match['case_id']}: {match['predictions_file']}")
    else:
        print(
            "No last-N-day validation predictions found for the selected ticker under "
            f"{DEFAULT_OUTPUT_ROOT}."
        )

    write_case_metadata(
        case_dir=case_output_dir,
        case_id=CASE_ID,
        case_name=CASE_NAME,
        package="eda_boxplots",
        dataset_path=dataset_path,
        tickers=[selected_ticker],
        features={"combined_features": selected_features},
        target=TARGET_COLUMN,
        models=list(MODEL_DICT.keys()),
        extras={
            "summary_statistics": summary_records,
            "artifacts": artifacts,
            "last_n_day_validation": validation_matches,
            "model_validation": validation_artifacts,
        },
    )

    return case_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate boxplot/histogram/violin comparisons for TSLA combined features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Ticker symbol to analyze (defaults to TSLA).",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional list of features to plot (defaults to combined features).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_FILENAME,
        help="Dataset filename inside the ML input directory.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional destination root for case outputs.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively after saving.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_case(
        ticker=args.ticker,
        features=args.features,
        dataset_filename=args.dataset,
        output_root=args.output_root,
        show_plots=args.show_plots,
    )
