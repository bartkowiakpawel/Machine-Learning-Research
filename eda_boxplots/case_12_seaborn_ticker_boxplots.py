"""EDA case 12: seaborn boxplots per ticker with horizontal orientation."""

from __future__ import annotations

import argparse
import re
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

from config import ML_INPUT_DIR
from core.shared_utils import (
    filter_ticker,
    load_ml_dataset,
    resolve_output_dir,
    write_case_metadata,
)

try:
    from .settings import (
        DEFAULT_DATASET_FILENAME,
        DEFAULT_FEATURES,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_TICKER,
        DEFAULT_TICKERS,
        get_case_config,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from eda_boxplots.settings import (
        DEFAULT_DATASET_FILENAME,
        DEFAULT_FEATURES,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_TICKER,
        DEFAULT_TICKERS,
        get_case_config,
    )

CASE_ID = "case_12"
CASE_CONFIG = get_case_config(CASE_ID)
CASE_NAME = CASE_CONFIG.name

ROLLING_WINDOW = 30
TARGET_SHIFT = 1
LAST_N_DAYS = 15
FEATURE_COLUMNS = list(DEFAULT_FEATURES)
TARGET_FEATURE_COLUMNS = [
    "rolling_median_target_1d",
    "rolling_iqr_target_1d",
    "rolling_outlier_ratio_target_1d",
]
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + TARGET_FEATURE_COLUMNS


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


def _prepare_feature_frame(subset: pd.DataFrame) -> pd.DataFrame:
    if subset.empty:
        return pd.DataFrame(columns=ALL_FEATURE_COLUMNS)

    working = subset.sort_values("Date").copy()
    if "Close" not in working.columns:
        raise KeyError("Missing 'Close' column required for target feature computation.")

    working["target_1d"] = (working["Close"].shift(-TARGET_SHIFT) - working["Close"]) / working["Close"]
    target_series = working["target_1d"]

    rolling = target_series.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
    working["rolling_median_target_1d"] = rolling.median()
    working["rolling_iqr_target_1d"] = rolling.quantile(0.75) - rolling.quantile(0.25)
    working["rolling_outlier_ratio_target_1d"] = rolling.apply(_outlier_ratio, raw=False)

    available_columns = [col for col in ALL_FEATURE_COLUMNS if col in working.columns]
    missing_columns = [col for col in ALL_FEATURE_COLUMNS if col not in available_columns]
    if missing_columns:
        missing_fmt = ", ".join(missing_columns)
        raise KeyError(f"Missing feature columns required for plotting: {missing_fmt}")

    feature_frame = working.loc[:, available_columns].replace([np.inf, -np.inf], np.nan).dropna()
    if feature_frame.empty:
        return feature_frame

    ordered_columns = [col for col in ALL_FEATURE_COLUMNS if col in feature_frame.columns]
    return feature_frame.loc[:, ordered_columns]


def _plot_ticker_boxplot(
    ticker: str,
    feature_frame: pd.DataFrame,
    *,
    output_dir: Path,
    show: bool,
) -> Path:
    melted = feature_frame.melt(var_name="feature", value_name="value")
    melted = melted.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    if melted.empty:
        raise ValueError(f"No numeric data available to plot for ticker {ticker}.")

    feature_order = [feature for feature in ALL_FEATURE_COLUMNS if feature in melted["feature"].unique()]
    if not feature_order:
        feature_order = sorted(melted["feature"].unique())

    sns.set_theme(style="whitegrid")

    fig_height = max(4.0, 0.6 * len(feature_order) + 1.5)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    sns.boxplot(
        data=melted,
        x="value",
        y="feature",
        order=feature_order,
        hue="feature",
        hue_order=feature_order,
        orient="h",
        palette="Set2",
        dodge=False,
        ax=ax,
        width=0.6,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title(f"{ticker} feature distribution ({CASE_NAME})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()

    safe_ticker = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_") or "ticker"
    filename = f"{safe_ticker.lower()}_{CASE_ID}_feature_boxplot.png"
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return output_path


def run_case(
    *,
    tickers: Sequence[str] | None = None,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    output_root: Path | None = None,
    show_plots: bool = False,
) -> Path:
    dataset_path = ML_INPUT_DIR / dataset_filename
    print(f"Loading dataset from {dataset_path} ...")
    df = load_ml_dataset(dataset_path)

    selected_tickers = _normalize_tickers(tickers)
    print(f"Selected tickers: {', '.join(selected_tickers)}")

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    plots_dir = case_output_dir / "seaborn_boxplots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: dict[str, str] = {}
    for ticker in selected_tickers:
        subset = filter_ticker(df, ticker)
        if subset.empty:
            print(f"[WARN] No rows found for ticker {ticker}; skipping.")
            continue
        try:
            feature_frame = _prepare_feature_frame(subset)
        except KeyError as exc:  # pragma: no cover - logged for user visibility
            print(f"[ERROR] {exc}")
            continue
        if feature_frame.empty:
            print(f"[WARN] Feature frame empty after cleaning for ticker {ticker}; skipping.")
            continue
        try:
            plot_path = _plot_ticker_boxplot(
                ticker,
                feature_frame,
                output_dir=plots_dir,
                show=show_plots,
            )
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue
        plot_paths[ticker] = str(plot_path)
        print(f"Saved seaborn boxplot for {ticker} to {plot_path}")

    if not plot_paths:
        raise RuntimeError("No plots were generated; check dataset and feature configuration.")

    write_case_metadata(
        case_dir=case_output_dir,
        case_id=CASE_ID,
        case_name=CASE_NAME,
        package="eda_boxplots",
        dataset_path=dataset_path,
        tickers=selected_tickers,
        features={
            "technical_features": FEATURE_COLUMNS,
            "target_features": TARGET_FEATURE_COLUMNS,
            "combined_features": ALL_FEATURE_COLUMNS,
        },
        target=None,
        models=(),
        extras={
            "rolling_window": ROLLING_WINDOW,
            "target_shift_days": TARGET_SHIFT,
            "last_n_days_parameter": LAST_N_DAYS,
            "plot_paths": plot_paths,
            "plot_style": "seaborn_horizontal_boxplot_per_ticker",
        },
    )

    return case_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the seaborn per-ticker feature boxplot workflow.",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_root = Path(args.output_root) if args.output_root else None

    run_case(
        tickers=args.tickers,
        dataset_filename=args.dataset,
        output_root=output_root,
        show_plots=args.show_plots,
    )
