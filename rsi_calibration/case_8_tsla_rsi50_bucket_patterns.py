"""Case 8: RSI-50 bucket drop vs rise pattern summary for TSLA (21-day horizon)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Ensure imports work when executed as a script
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

from config import ML_INPUT_DIR
from core.shared_utils import load_ml_dataset, resolve_output_dir, save_dataframe
from rsi_calibration.case_1_tsla_rsi_baseline import (
    DEFAULT_TICKER,
    DEFAULT_YF_DATASET_PATH,
)
from rsi_calibration.case_5_tsla_rsi_bucket_patterns import (
    _prepare_bucket_pattern_tables,
    _render_table,
)

CASE_ID = "case_8"
CASE_NAME = "Case 8: RSI-50 bucket patterns (21-day horizon)"
DEFAULT_OUTPUT_ROOT = Path("rsi_calibration") / "outputs"
RSI_PERIOD = 50
TARGET_HORIZONS: Iterable[int] = (21,)


def _summarise_bucket_stats(summary: pd.DataFrame) -> list[str]:
    stats: list[str] = []
    if {"drop_pct", "rise_pct"}.issubset(summary.columns):
        drop_top = summary["drop_pct"].idxmax()
        rise_top = summary["rise_pct"].idxmax()
        stats.append(
            f"Highest drop share bucket: {drop_top} ({summary.loc[drop_top, 'drop_pct']:.2%})"
        )
        stats.append(
            f"Highest rise share bucket: {rise_top} ({summary.loc[rise_top, 'rise_pct']:.2%})"
        )
    stats.append(f"Total analysed samples: {int(summary['total'].sum())}")
    return stats


def run_case(
    *,
    dataset_filename: str | None = None,
    ticker: str = DEFAULT_TICKER,
    rsi_period: int = RSI_PERIOD,
    target_horizons: Iterable[int] = TARGET_HORIZONS,
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

    if not per_horizon:
        raise ValueError("No summaries produced; verify dataset and horizon availability.")

    horizon_key = next(iter(target_horizons))
    horizon_summary = per_horizon.get(horizon_key)
    if horizon_summary is None:
        raise ValueError(f"Horizon {horizon_key} summary missing from results.")

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    horizon_path = save_dataframe(
        horizon_summary.reset_index().rename(columns={"index": "RSI_bucket"}),
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_rsi{rsi_period}_h{horizon_key}d_patterns.csv",
    )
    combined_path = save_dataframe(
        combined_summary.reset_index().rename(columns={"index": "RSI_bucket"}),
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_rsi{rsi_period}_combined_patterns.csv",
    )

    filtered_long = long_df[long_df["target_horizon"].isin(target_horizons)]
    long_path = save_dataframe(
        filtered_long,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_rsi{rsi_period}_long.csv",
    )

    print(f"Horizon summary saved to: {horizon_path}")
    print(f"Combined summary saved to: {combined_path}")
    print(f"Long-format summary saved to: {long_path}")

    table_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_rsi{rsi_period}_h{horizon_key}d_table.png"
    _render_table(
        horizon_summary,
        table_path,
        title=f"{CASE_NAME} (horizon = {horizon_key} days)",
    )
    if table_path.exists():
        print(f"Annotated table saved to: {table_path}")

    for line in _summarise_bucket_stats(horizon_summary):
        print(line)

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
        default=RSI_PERIOD,
        help="Look-back period for RSI calculation (default: %(default)s).",
    )
    parser.add_argument(
        "--target-horizons",
        dest="target_horizons",
        type=int,
        nargs="+",
        default=list(TARGET_HORIZONS),
        help="Horizon (in days) to evaluate (default: %(default)s).",
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
