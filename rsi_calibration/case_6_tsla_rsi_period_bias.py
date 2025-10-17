"""Case 6: RSI period bias comparison for TSLA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

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
)
from rsi_calibration.case_4_tsla_rsi_heatmap import _prepare_rsi_table

CASE_ID = "case_6"
CASE_NAME = "Case 6: RSI period bias comparison"
DEFAULT_OUTPUT_ROOT = Path("rsi_calibration") / "outputs"
DEFAULT_TARGET_HORIZONS = (1, 7, 14, 21)
DEFAULT_RSI_PERIODS = (5, 7, 10, 14, 21, 28, 50)
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def _compute_bias_tables(
    dataset: pd.DataFrame,
    *,
    ticker: str,
    rsi_periods: Sequence[int],
    target_horizons: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate average future returns (bias) per RSI period and horizon."""
    long_records: list[dict[str, float]] = []

    for rsi_period in rsi_periods:
        pivot_df, _ = _prepare_rsi_table(
            dataset,
            ticker=ticker,
            rsi_period=rsi_period,
            target_horizons=target_horizons,
        )
        if pivot_df.empty:
            continue

        bias_per_horizon = pivot_df.mean(axis=0, skipna=True)
        for horizon, avg_bias in bias_per_horizon.items():
            long_records.append(
                {
                    "rsi_period": int(rsi_period),
                    "target_horizon": int(horizon),
                    "avg_future_return": float(avg_bias),
                    "overall_avg_bias": float(bias_per_horizon.mean(skipna=True)),
                }
            )

    long_df = pd.DataFrame(long_records)
    if long_df.empty:
        raise ValueError("No bias records were produced; verify dataset contents.")

    long_df = long_df.sort_values(["target_horizon", "rsi_period"]).reset_index(drop=True)
    wide_df = long_df.pivot_table(
        index="rsi_period",
        columns="target_horizon",
        values="avg_future_return",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)
    wide_df["overall_avg_bias"] = wide_df.mean(axis=1, skipna=True)
    wide_df = wide_df.reset_index()

    return wide_df, long_df


def _prepare_bias_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    renamed = {}
    for col in display.columns:
        if col == "rsi_period":
            renamed[col] = "RSI period"
        elif col == "overall_avg_bias":
            renamed[col] = "Overall avg bias"
        elif isinstance(col, (int, np.integer)):
            renamed[col] = f"h{int(col)}d avg return"
    display = display.rename(columns=renamed)
    for col in display.columns:
        if col != "RSI period":
            display[col] = display[col].astype(float).round(6)
    return display


def _prepare_bias_ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    display = display.rename(
        columns={
            "target_horizon": "Horizon (days)",
            "best_rsi_period": "Best RSI period",
            "best_avg_future_return": "Best avg return",
            "worst_rsi_period": "Worst RSI period",
            "worst_avg_future_return": "Worst avg return",
        }
    )
    display["Horizon (days)"] = display["Horizon (days)"].astype(int)
    for col in ("Best avg return", "Worst avg return"):
        display[col] = display[col].astype(float).round(6)
    return display


def _render_table(df: pd.DataFrame, output_path: Path, *, title: str, subtitle: str) -> None:
    if df.empty:
        return
    if GT is None:
        print("[WARN] great_tables not installed; skipping table rendering.")
        return
    try:
        table = (
            GT(df)
            .tab_header(title=title, subtitle=subtitle)
            .tab_source_note("Values rounded to 6 decimal places. Random seed fixed at 42.")
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

    wide_df, long_df = _compute_bias_tables(
        dataset,
        ticker=ticker,
        rsi_periods=rsi_periods,
        target_horizons=target_horizons,
    )

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    wide_path = save_dataframe(
        wide_df,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_bias_summary.csv",
    )
    long_path = save_dataframe(
        long_df,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_bias_long.csv",
    )

    print(f"Bias summary saved to: {wide_path}")
    print(f"Bias records saved to: {long_path}")

    print("Overall bias per RSI period (higher is better):")
    horizon_cols = [col for col in wide_df.columns if isinstance(col, (int, np.integer))]
    for _, row in wide_df.iterrows():
        horizon_details = ", ".join(
            f"h{int(h)}d={row[h]:.6f}" for h in horizon_cols if h in row and pd.notna(row[h])
        )
        print(
            f"RSI {int(row['rsi_period']):>2}: "
            f"overall={row['overall_avg_bias']:.6f}"
            + (f" | {horizon_details}" if horizon_details else "")
        )

    ranking_records: list[dict[str, float]] = []
    for horizon in sorted(set(long_df["target_horizon"])):
        horizon_slice = long_df[long_df["target_horizon"] == horizon]
        best_idx = horizon_slice["avg_future_return"].idxmax()
        worst_idx = horizon_slice["avg_future_return"].idxmin()
        best_row = horizon_slice.loc[best_idx]
        worst_row = horizon_slice.loc[worst_idx]
        ranking_records.append(
            {
                "target_horizon": int(horizon),
                "best_rsi_period": int(best_row["rsi_period"]),
                "best_avg_future_return": float(best_row["avg_future_return"]),
                "worst_rsi_period": int(worst_row["rsi_period"]),
                "worst_avg_future_return": float(worst_row["avg_future_return"]),
            }
        )
        print(
            f"Horizon {horizon:>2}d | "
            f"best RSI={int(best_row['rsi_period'])} ({best_row['avg_future_return']:.6f}) "
            f"vs worst RSI={int(worst_row['rsi_period'])} ({worst_row['avg_future_return']:.6f})"
        )

    ranking_df = pd.DataFrame(ranking_records)
    ranking_path = save_dataframe(
        ranking_df,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_bias_rankings.csv",
    )
    print(f"Bias rankings saved to: {ranking_path}")

    bias_display_df = _prepare_bias_summary_table(wide_df)
    ranking_display_df = _prepare_bias_ranking_table(ranking_df)

    bias_display_path = save_dataframe(
        bias_display_df,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_bias_table_display.csv",
    )
    ranking_display_path = save_dataframe(
        ranking_display_df,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_bias_rankings_display.csv",
    )
    print(f"Bias display table CSV saved to: {bias_display_path}")
    print(f"Bias ranking table CSV saved to: {ranking_display_path}")

    bias_table_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_bias_table.png"
    _render_table(
        bias_display_df,
        bias_table_path,
        title=f"{CASE_NAME} (RSI vs horizons)",
        subtitle="Average future returns per RSI period and horizon.",
    )
    ranking_table_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_bias_rankings_table.png"
    _render_table(
        ranking_display_df,
        ranking_table_path,
        title=f"{CASE_NAME} (best vs worst)",
        subtitle="Top and bottom RSI periods by target horizon.",
    )
    if bias_table_path.exists():
        print(f"Bias summary table saved to: {bias_table_path}")
    if ranking_table_path.exists():
        print(f"Bias ranking table saved to: {ranking_table_path}")

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
        rsi_periods=args.rsi_periods,
        target_horizons=args.target_horizons,
        output_root=output_root,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
