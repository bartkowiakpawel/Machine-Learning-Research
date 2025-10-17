"""Case 2: TSLA target distribution analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

# Ensure imports work when executed as a script
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from great_tables import GT
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    GT = None  # type: ignore[assignment]

from rsi_calibration.case_1_tsla_rsi_baseline import (
    DEFAULT_NEUTRAL_BAND,
    DEFAULT_TICKER,
    DEFAULT_YF_DATASET_PATH,
    RSI_DEFAULT_PERIOD,
    _prepare_rsi_dataset,
)
from core.shared_utils import load_ml_dataset, resolve_output_dir, save_dataframe

CASE_ID = "case_2"
CASE_NAME = "Case 2: TSLA target distribution"
DEFAULT_OUTPUT_ROOT = Path("rsi_calibration") / "outputs"


def _summarize_target_distribution(target_class: pd.Series) -> pd.DataFrame:
    counts = target_class.value_counts(dropna=False).rename("count")
    counts = counts.reindex(target_class.cat.categories, fill_value=0)
    total = counts.sum()
    percentages = counts / total * 100.0
    summary = pd.DataFrame(
        {
            "class": counts.index.astype(str),
            "count": counts.values,
            "percentage": percentages.round(4),
        }
    )
    summary["total_samples"] = total
    return summary


def _render_table(df: pd.DataFrame, png_path: Path) -> None:
    if df.empty:
        return
    if GT is None:
        print("[WARN] great_tables not installed; skipping PNG/HTML table generation.")
        return
    try:
        table = (
            GT(df)
            .tab_header(
                title=f"{CASE_NAME} – target distribution",
                subtitle="Share of drop/flat/rise classes for TSLA RSI dataset.",
            )
            .tab_source_note(
                "Percentages rounded to 4 decimal places. "
                "Counts derived from the same RSI dataset as Case 1 (chronological ordering)."
            )
        )
        table.save(str(png_path))
        table.write_raw_html(png_path.with_suffix(".html"), inline_css=True, make_page=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to render PNG/HTML table with great_tables: {exc}")


def _render_pie_chart(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        df["count"],
        labels=df["class"],
        autopct=lambda pct: f"{pct:.2f}%",
        startangle=90,
        colors=sns.color_palette("Set2", n_colors=len(df)),
    )
    ax.set_title("TSLA target distribution (drop / flat / rise)", fontsize=14)
    ax.axis("equal")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_case(
    *,
    dataset_filename: str | None = None,
    ticker: str = DEFAULT_TICKER,
    neutral_band: float = DEFAULT_NEUTRAL_BAND,
    output_root: Path | None = None,
    rsi_period: int = RSI_DEFAULT_PERIOD,
) -> Path:
    if dataset_filename is None:
        dataset_path = DEFAULT_YF_DATASET_PATH
    else:
        dataset_path = Path(dataset_filename)
        if not dataset_path.exists():
            dataset_path = DEFAULT_YF_DATASET_PATH

    dataset = load_ml_dataset(dataset_path)

    features, target_class, target_continuous = _prepare_rsi_dataset(
        dataset,
        ticker=ticker,
        rsi_period=rsi_period,
        neutral_band=neutral_band,
    )
    summary_df = _summarize_target_distribution(target_class)

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = save_dataframe(
        summary_df,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_target_distribution.csv",
    )

    table_png_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_target_distribution_table.png"
    _render_table(summary_df, table_png_path)

    pie_chart_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_target_distribution_pie.png"
    _render_pie_chart(summary_df, pie_chart_path)

    print(f"Summary saved to: {summary_path}")
    if table_png_path.exists():
        print(f"Annotated metrics table saved to: {table_png_path}")
    if pie_chart_path.exists():
        print(f"Pie chart saved to: {pie_chart_path}")

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
        "--neutral-band",
        dest="neutral_band",
        type=float,
        default=DEFAULT_NEUTRAL_BAND,
        help="Absolute threshold around zero treated as 'flat'.",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default=None,
        help="Optional override for the case output root directory.",
    )
    parser.add_argument(
        "--rsi-period",
        dest="rsi_period",
        type=int,
        default=RSI_DEFAULT_PERIOD,
        help="Look-back period for RSI calculation.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI convenience
    args = _parse_args()
    output_root = Path(args.output_root) if args.output_root else None

    run_case(
        dataset_filename=args.dataset_filename,
        ticker=args.ticker,
        neutral_band=args.neutral_band,
        output_root=output_root,
        rsi_period=args.rsi_period,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
