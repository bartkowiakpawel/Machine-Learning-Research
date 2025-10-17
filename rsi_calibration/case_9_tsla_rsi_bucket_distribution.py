"""Case 9: RSI bucket distribution counts across horizons for TSLA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

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
    DEFAULT_TICKER,
    DEFAULT_YF_DATASET_PATH,
    _compute_rsi,
)

CASE_ID = "case_9"
CASE_NAME = "Case 9: RSI bucket distribution"
DEFAULT_OUTPUT_ROOT = Path("rsi_calibration") / "outputs"
DEFAULT_TARGET_HORIZONS = (1, 5, 14, 21)
DEFAULT_RSI_PERIODS = (7, 14, 21, 50)
RSI_BUCKET_BOUNDS = (-np.inf, 30.0, 70.0, np.inf)
RSI_BUCKET_LABELS = ("<30", "[30, 70)", ">=70")


def _assign_rsi_buckets(rsi: pd.Series) -> pd.Categorical:
    """Bucketise RSI values into the canonical oversold/neutral/overbought zones."""
    return pd.cut(
        rsi,
        bins=RSI_BUCKET_BOUNDS,
        labels=RSI_BUCKET_LABELS,
        include_lowest=True,
        right=False,
    )


def _collect_bucket_counts(
    dataset: pd.DataFrame,
    *,
    ticker: str,
    rsi_periods: Sequence[int],
    target_horizons: Iterable[int],
) -> pd.DataFrame:
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

    records: list[dict[str, object]] = []
    for rsi_period in rsi_periods:
        rsi = _compute_rsi(close, period=rsi_period)
        rsi_bucket = _assign_rsi_buckets(rsi)

        for horizon in target_horizons:
            future_return = (close.shift(-horizon) - close) / close
            df = pd.DataFrame(
                {
                    "RSI": rsi,
                    "RSI_bucket": rsi_bucket,
                    "future_return": future_return,
                }
            ).dropna()
            if df.empty:
                continue

            counts = df.groupby("RSI_bucket", observed=False).size().reindex(RSI_BUCKET_LABELS, fill_value=0)
            total = counts.sum()
            shares = counts / total if total > 0 else np.nan

            for bucket_label in RSI_BUCKET_LABELS:
                records.append(
                    {
                        "rsi_period": int(rsi_period),
                        "target_horizon": int(horizon),
                        "RSI_bucket": bucket_label,
                        "count": int(counts.loc[bucket_label]),
                        "share": float(shares.loc[bucket_label]) if total > 0 else np.nan,
                        "total": int(total),
                    }
                )

    counts_df = pd.DataFrame(records)
    if counts_df.empty:
        raise ValueError("No bucket counts were produced; check RSI periods or horizons.")
    counts_df = counts_df.sort_values(["rsi_period", "target_horizon", "RSI_bucket"]).reset_index(drop=True)
    return counts_df


def _pivot_per_period(
    counts_df: pd.DataFrame,
    value_col: str,
) -> Dict[int, pd.DataFrame]:
    per_period: Dict[int, pd.DataFrame] = {}
    for period in sorted(counts_df["rsi_period"].unique()):
        subset = counts_df[counts_df["rsi_period"] == period]
        pivot = subset.pivot(index="RSI_bucket", columns="target_horizon", values=value_col)
        pivot = pivot.reindex(index=RSI_BUCKET_LABELS, columns=sorted(subset["target_horizon"].unique()))
        per_period[int(period)] = pivot
    return per_period


def _render_table(df: pd.DataFrame, output_path: Path, *, title: str, subtitle: str) -> None:
    if df.empty:
        return
    if GT is None:
        print("[WARN] great_tables not installed; skipping table rendering.")
        return
    try:
        table = (
            GT(df.reset_index().rename(columns={"index": "RSI bucket"}))
            .tab_header(title=title, subtitle=subtitle)
            .tab_source_note("Counts derived from TSLA dataset; shares rounded to 4 decimal places.")
        )
        table.save(str(output_path))
        table.write_raw_html(output_path.with_suffix(".html"), inline_css=True, make_page=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to render table with great_tables: {exc}")


def run_case(
    *,
    dataset_filename: str | None = None,
    ticker: str = DEFAULT_TICKER,
    rsi_periods: Sequence[int] = DEFAULT_RSI_PERIODS,
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
    counts_df = _collect_bucket_counts(
        dataset,
        ticker=ticker,
        rsi_periods=rsi_periods,
        target_horizons=target_horizons,
    )

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    counts_long_path = save_dataframe(
        counts_df,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_bucket_counts_long.csv",
    )
    print(f"Bucket counts (long) saved to: {counts_long_path}")

    counts_pivot = _pivot_per_period(counts_df, "count")
    shares_pivot = _pivot_per_period(counts_df, "share")

    for period, pivot_df in counts_pivot.items():
        counts_path = save_dataframe(
            pivot_df.reset_index(),
            case_output_dir,
            f"{ticker.lower()}_{CASE_ID}_rsi{period}_count_pivot.csv",
        )
        shares_df = shares_pivot[period].round(4)
        shares_path = save_dataframe(
            shares_df.reset_index(),
            case_output_dir,
            f"{ticker.lower()}_{CASE_ID}_rsi{period}_share_pivot.csv",
        )
        print(f"RSI {period}: count pivot saved to {counts_path}")
        print(f"RSI {period}: share pivot saved to {shares_path}")

        table_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_rsi{period}_count_table.png"
        _render_table(
            pivot_df.fillna(0).astype(int),
            table_path,
            title=f"{CASE_NAME} (RSI={period})",
            subtitle="Counts per RSI bucket and target horizon.",
        )
        if table_path.exists():
            print(f"RSI {period}: count table saved to {table_path}")

        share_table_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_rsi{period}_share_table.png"
        _render_table(
            shares_df,
            share_table_path,
            title=f"{CASE_NAME} (RSI={period}, shares)",
            subtitle="Share of samples per RSI bucket (rows sum to 1).",
        )
        if share_table_path.exists():
            print(f"RSI {period}: share table saved to {share_table_path}")

        totals = pivot_df.sum(axis=0)
        totals_line = ", ".join(f"h{int(h)}d={int(v)}" for h, v in totals.items())
        print(f"RSI {period}: total samples per horizon -> {totals_line}")

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
        "--rsi-periods",
        dest="rsi_periods",
        type=int,
        nargs="+",
        default=list(DEFAULT_RSI_PERIODS),
        help="List of RSI look-back periods to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--target-horizons",
        dest="target_horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_TARGET_HORIZONS),
        help="List of horizons (in days) used to derive targets (default: %(default)s).",
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
        rsi_periods=args.rsi_periods,
        target_horizons=args.target_horizons,
        output_root=output_root,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
