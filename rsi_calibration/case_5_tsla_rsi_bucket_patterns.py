"""Case 5: RSI bucket drop vs rise pattern summary for TSLA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
    RSI_DEFAULT_PERIOD,
    _compute_rsi,
)

CASE_ID = "case_5"
CASE_NAME = "Case 5: RSI bucket drop vs rise patterns"
DEFAULT_OUTPUT_ROOT = Path("rsi_calibration") / "outputs"
DEFAULT_TARGET_HORIZONS = (5, 14, 21)

RSI_BUCKET_BOUNDS = (-np.inf, 40.0, 60.0, np.inf)
RSI_BUCKET_LABELS = ("<=40", "(40, 60]", ">60")


def _assign_rsi_buckets(rsi: pd.Series) -> pd.Categorical:
    """Map RSI values into the coarse buckets used for the summary."""
    return pd.cut(
        rsi,
        bins=RSI_BUCKET_BOUNDS,
        labels=RSI_BUCKET_LABELS,
        include_lowest=True,
        right=True,
    )


def _finalise_pattern_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent ordering, totals, and share columns."""
    for col in ("drop", "rise"):
        if col not in summary.columns:
            summary[col] = 0
    summary = summary.reindex(index=RSI_BUCKET_LABELS)
    summary = summary.loc[:, ["drop", "rise"]]
    summary["total"] = summary["drop"] + summary["rise"]

    with np.errstate(divide="ignore", invalid="ignore"):
        summary["drop_pct"] = np.where(summary["total"] > 0, summary["drop"] / summary["total"], np.nan)
        summary["rise_pct"] = np.where(summary["total"] > 0, summary["rise"] / summary["total"], np.nan)

    summary[["drop", "rise", "total"]] = summary[["drop", "rise", "total"]].astype(int)
    summary[["drop_pct", "rise_pct"]] = summary[["drop_pct", "rise_pct"]].round(4)
    return summary


def _summarise_patterns(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["RSI_bucket", "pattern"])
        .size()
        .unstack(fill_value=0)
    )
    return _finalise_pattern_summary(grouped)


def _prepare_bucket_pattern_tables(
    dataset: pd.DataFrame,
    *,
    ticker: str,
    rsi_period: int,
    target_horizons: Iterable[int],
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
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
    rsi_bucket = _assign_rsi_buckets(rsi)

    per_horizon: Dict[int, pd.DataFrame] = {}
    long_records: list[pd.DataFrame] = []
    combined_counts: list[pd.DataFrame] = []

    for horizon in target_horizons:
        future_return = (close.shift(-horizon) - close) / close
        horizon_df = pd.DataFrame(
            {
                "Date": working["Date"],
                "RSI": rsi,
                "RSI_bucket": rsi_bucket,
                "future_return": future_return,
            }
        ).dropna()

        if horizon_df.empty:
            continue

        horizon_df["pattern"] = np.select(
            [horizon_df["future_return"] < 0, horizon_df["future_return"] > 0],
            ["drop", "rise"],
            default=None,
        )
        horizon_df = horizon_df.dropna(subset=["pattern"])
        if horizon_df.empty:
            continue

        summary = _summarise_patterns(horizon_df)
        per_horizon[horizon] = summary
        combined_counts.append(summary.loc[:, ["drop", "rise"]])

        horizon_summary = summary.reset_index().rename(columns={"index": "RSI_bucket"})
        horizon_summary["target_horizon"] = horizon
        long_records.append(horizon_summary)

    if not per_horizon:
        raise ValueError("No valid samples available after filtering RSI buckets and future returns.")

    combined_summary = sum(combined_counts[1:], combined_counts[0].copy())
    combined_summary = _finalise_pattern_summary(combined_summary)

    long_df = pd.concat(long_records, ignore_index=True)
    long_df = long_df.loc[:, ["RSI_bucket", "target_horizon", "drop", "rise", "total", "drop_pct", "rise_pct"]]

    combined_long = combined_summary.reset_index().rename(columns={"index": "RSI_bucket"})
    combined_long["target_horizon"] = "combined"
    combined_long = combined_long.loc[:, ["RSI_bucket", "target_horizon", "drop", "rise", "total", "drop_pct", "rise_pct"]]
    long_df = pd.concat([long_df, combined_long], ignore_index=True)

    return per_horizon, combined_summary, long_df


def _render_table(df: pd.DataFrame, output_path: Path, title: str) -> None:
    if df.empty:
        return
    if GT is None:
        print("[WARN] great_tables not installed; skipping bucket pattern table.")
        return
    try:
        formatted = df.reset_index().rename(columns={"index": "RSI bucket"})
        table = (
            GT(formatted)
            .tab_header(title=title, subtitle="Counts of future drops/rises per RSI bucket.")
            .tab_source_note("Percentages rounded to 4 decimal places. Zero future returns are excluded.")
        )
        table.save(str(output_path))
        table.write_raw_html(output_path.with_suffix(".html"), inline_css=True, make_page=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to render bucket pattern table with great_tables: {exc}")


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

    per_horizon, combined_summary, long_df = _prepare_bucket_pattern_tables(
        dataset,
        ticker=ticker,
        rsi_period=rsi_period,
        target_horizons=target_horizons,
    )

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    for horizon, summary in per_horizon.items():
        horizon_path = save_dataframe(
            summary.reset_index().rename(columns={"index": "RSI_bucket"}),
            case_output_dir,
            f"{ticker.lower()}_{CASE_ID}_h{horizon}d_patterns.csv",
        )
        print(f"Horizon {horizon} summary saved to: {horizon_path}")

        table_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_h{horizon}d_table.png"
        _render_table(summary, table_path, title=f"{CASE_NAME} (horizon = {horizon} days)")

    combined_path = save_dataframe(
        combined_summary.reset_index().rename(columns={"index": "RSI_bucket"}),
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_combined_patterns.csv",
    )
    print(f"Combined summary saved to: {combined_path}")

    long_path = save_dataframe(
        long_df,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_all_horizons_patterns.csv",
    )
    print(f"Long-format summary saved to: {long_path}")

    combined_table_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_combined_table.png"
    _render_table(combined_summary, combined_table_path, title=f"{CASE_NAME} (combined horizons)")

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
