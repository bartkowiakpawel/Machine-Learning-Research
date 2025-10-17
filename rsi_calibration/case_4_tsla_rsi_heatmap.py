"""Case 4: RSI level vs future return heatmap for TSLA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure imports work when executed as a script
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

try:
    from great_tables import GT
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    GT = None  # type: ignore[assignment]

from config import ML_INPUT_DIR
from core.shared_utils import load_ml_dataset, resolve_output_dir, save_dataframe
from rsi_calibration.case_1_tsla_rsi_baseline import (
    DEFAULT_NEUTRAL_BAND,
    DEFAULT_TICKER,
    DEFAULT_YF_DATASET_PATH,
    RSI_DEFAULT_PERIOD,
    _compute_rsi,
)

CASE_ID = "case_4"
CASE_NAME = "Case 4: RSI heatmap vs future returns"
DEFAULT_OUTPUT_ROOT = Path("rsi_calibration") / "outputs"
DEFAULT_TARGET_HORIZONS = (5, 14, 21)
RSI_BUCKET_SIZE = 5
RSI_MIN = 0
RSI_MAX = 100


def _prepare_rsi_table(
    dataset: pd.DataFrame,
    *,
    ticker: str,
    rsi_period: int,
    target_horizons: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = dataset.copy()
    working["ticker"] = working["ticker"].astype(str).str.upper()
    ticker_upper = ticker.upper()
    working = working[working["ticker"] == ticker_upper].copy()
    if working.empty:
        raise ValueError(f"No rows available for ticker {ticker_upper}.")

    if "Date" in working.columns:
        working["Date"] = pd.to_datetime(working["Date"], utc=True, errors="coerce").dt.tz_convert(None)
        working = working.sort_values("Date")

    close = pd.to_numeric(working["Close"], errors="coerce")
    rsi = _compute_rsi(close, period=rsi_period)

    bins = np.arange(RSI_MIN, RSI_MAX + RSI_BUCKET_SIZE, RSI_BUCKET_SIZE)
    labels = [f"{bins[i]}–{bins[i + 1]}" for i in range(len(bins) - 1)]

    long_records: list[dict[str, object]] = []
    count_records: list[dict[str, object]] = []
    for horizon in target_horizons:
        future_return = (close.shift(-horizon) - close) / close
        bucket = pd.cut(rsi, bins=bins, labels=labels, include_lowest=True)

        df = pd.DataFrame(
            {
                "Date": working["Date"],
                "RSI": rsi,
                "RSI_bucket": bucket,
                "future_return": future_return,
            }
        ).dropna()

        grouped = df.groupby("RSI_bucket", observed=False)["future_return"].mean().reindex(labels)
        counts = df.groupby("RSI_bucket", observed=False)["future_return"].size().reindex(labels, fill_value=0)

        for bucket_label, mean_return in grouped.items():
            long_records.append(
                {
                    "RSI_bucket": bucket_label,
                    "target_horizon": horizon,
                    "avg_future_return": mean_return,
                }
            )
        for bucket_label, count_value in counts.items():
            count_records.append(
                {
                    "RSI_bucket": bucket_label,
                    "target_horizon": horizon,
                    "count": int(count_value),
                }
            )

    long_df = pd.DataFrame(long_records)
    counts_df = pd.DataFrame(count_records)
    pivot_df = long_df.pivot(index="RSI_bucket", columns="target_horizon", values="avg_future_return")
    pivot_df = pivot_df.reindex(labels)
    return pivot_df, counts_df


def _render_heatmap(pivot_df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        center=0.0,
        linewidths=0.5,
        cbar_kws={"label": "Average future return"},
    )
    ax.set_xlabel("Target horizon (days)")
    ax.set_ylabel("RSI bucket")
    ax.set_title("TSLA RSI bucket vs average future return")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _render_table(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    if GT is None:
        print("[WARN] great_tables not installed; skipping heatmap table.")
        return
    try:
        formatted = df.reset_index().rename(columns={"index": "RSI bucket"})
        table = (
            GT(formatted)
            .tab_header(
                title=f"{CASE_NAME} – average future returns",
                subtitle="Values represent mean forward returns for each horizon (RSI period = 14).",
            )
            .tab_source_note("All values rounded to 4 decimal places.")
        )
        table.save(str(output_path))
        table.write_raw_html(output_path.with_suffix(".html"), inline_css=True, make_page=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to render table with great_tables: {exc}")


def run_case(
    *,
    dataset_filename: str | None = None,
    ticker: str = DEFAULT_TICKER,
    rsi_period: int = RSI_DEFAULT_PERIOD,
    target_horizons: Iterable[int] = DEFAULT_TARGET_HORIZONS,
    output_root: Path | None = None,
) -> Path:
    if dataset_filename is None:
        dataset_path = DEFAULT_YF_DATASET_PATH
    else:
        dataset_path = Path(dataset_filename)
        if not dataset_path.exists():
            dataset_path = ML_INPUT_DIR / dataset_filename

    dataset = load_ml_dataset(dataset_path)

    pivot_df, counts_df = _prepare_rsi_table(
        dataset,
        ticker=ticker,
        rsi_period=rsi_period,
        target_horizons=target_horizons,
    )

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    pivot_path = save_dataframe(
        pivot_df.reset_index(),
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_heatmap_data.csv",
    )
    counts_path = save_dataframe(
        counts_df,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_bucket_counts.csv",
    )

    heatmap_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_heatmap.png"
    _render_heatmap(pivot_df, heatmap_path)

    table_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_heatmap_table.png"
    _render_table(pivot_df, table_path)

    print(f"Heatmap data saved to: {pivot_path}")
    print(f"Bucket counts saved to: {counts_path}")
    if heatmap_path.exists():
        print(f"Heatmap figure saved to: {heatmap_path}")
    if table_path.exists():
        print(f"Annotated table saved to: {table_path}")

    return case_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CASE_NAME)
    parser.add_argument(
        "--dataset",
        dest="dataset_filename",
        default=None,
        help="Optional path/filename to dataset (default: local tsla_yf_dataset.csv).",
    )
    parser.add_argument(
        "--ticker",
        dest="ticker",
        default=DEFAULT_TICKER,
        help=f"Ticker symbol to analyse (default: {DEFAULT_TICKER}).",
    )
    parser.add_argument(
        "--rsi-period",
        dest="rsi_period",
        type=int,
        default=RSI_DEFAULT_PERIOD,
        help="Look-back period for RSI calculation (default: %(default)s).",
    )
    parser.add_argument(
        "--target-horizons",
        dest="target_horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_TARGET_HORIZONS),
        help="List of horizons (in days) to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default=None,
        help="Optional override for the case output root directory.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI convenience
    args = _parse_args()
    output_root = Path(args.output_root) if args.output_root else None

    run_case(
        dataset_filename=args.dataset_filename,
        ticker=args.ticker,
        rsi_period=args.rsi_period,
        target_horizons=args.target_horizons,
        output_root=output_root,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
