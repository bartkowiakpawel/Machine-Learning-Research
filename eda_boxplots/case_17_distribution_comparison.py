"""Case 17: compare boxplot, histogram, and violin plot for a single feature."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import ML_INPUT_DIR
from core.shared_utils import (
    filter_ticker,
    load_ml_dataset,
    resolve_output_dir,
    save_dataframe,
    write_case_metadata,
)

from eda_boxplots.settings import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_FEATURES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TICKER,
    get_case_config,
)

CASE_ID = "case_17"
CASE_CONFIG = get_case_config(CASE_ID)
CASE_NAME = CASE_CONFIG.name
DEFAULT_FEATURE = DEFAULT_FEATURES[1] if len(DEFAULT_FEATURES) > 1 else DEFAULT_FEATURES[0]


def _normalize_ticker(value: str | None) -> str:
    if value is None:
        return DEFAULT_TICKER
    candidate = str(value).strip()
    return candidate.upper() if candidate else DEFAULT_TICKER


def _prepare_feature_series(df: pd.DataFrame, ticker: str, feature: str) -> pd.Series:
    subset = filter_ticker(df, ticker)
    if subset.empty:
        raise ValueError(f"No rows found for ticker '{ticker}'.")
    if feature not in subset.columns:
        raise KeyError(f"Feature '{feature}' is not available in the dataset.")
    series = pd.to_numeric(subset[feature], errors="coerce").dropna()
    if series.empty:
        raise ValueError(
            f"No numeric data available for feature '{feature}' and ticker '{ticker}'."
        )
    return series


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
        f"{ticker} · {feature} | Q1={q1:.4f} · Q3={q3:.4f} · IQR={iqr:.4f} · "
        f"median={median:.4f} · skew={skew:.3f}"
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
    feature: str | None = None,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    output_root: Path | None = None,
    show_plots: bool = False,
) -> Path:
    dataset_path = ML_INPUT_DIR / dataset_filename
    print(f"Loading dataset from {dataset_path} ...")
    df = load_ml_dataset(dataset_path)

    selected_ticker = _normalize_ticker(ticker)
    selected_feature = feature or DEFAULT_FEATURE
    print(f"Selected ticker: {selected_ticker}")
    print(f"Selected feature: {selected_feature}")

    series = _prepare_feature_series(df, selected_ticker, selected_feature)
    summary_stats = _compute_summary_stats(series)

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    comparison_dir = case_output_dir / "distribution_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    summary_frame = pd.DataFrame([{"ticker": selected_ticker, "feature": selected_feature, **summary_stats}])
    summary_path = save_dataframe(
        summary_frame,
        comparison_dir,
        f"{selected_ticker.lower()}_{selected_feature.lower()}_{CASE_ID}_stats.csv",
    )
    print(f"Summary statistics saved to: {summary_path}")

    figure_path = _plot_distribution_comparison(
        series,
        ticker=selected_ticker,
        feature=selected_feature,
        summary=summary_stats,
        output_dir=comparison_dir,
        show=show_plots,
    )
    print(f"Distribution comparison figure saved to: {figure_path}")

    write_case_metadata(
        case_dir=case_output_dir,
        case_id=CASE_ID,
        case_name=CASE_NAME,
        package="eda_boxplots",
        dataset_path=dataset_path,
        tickers=[selected_ticker],
        features={"feature": selected_feature},
        target=None,
        models=[],
        extras={
            "summary_statistics": summary_stats,
            "artifacts": {
                "summary_csv": summary_path.relative_to(case_output_dir).as_posix(),
                "distribution_figure": figure_path.relative_to(case_output_dir).as_posix(),
            },
        },
    )

    return case_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare boxplot, histogram, and violin plot for a selected feature.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", default=None, help="Ticker symbol to analyze.")
    parser.add_argument("--feature", default=None, help="Feature column to plot.")
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
        feature=args.feature,
        dataset_filename=args.dataset,
        output_root=args.output_root,
        show_plots=args.show_plots,
    )
