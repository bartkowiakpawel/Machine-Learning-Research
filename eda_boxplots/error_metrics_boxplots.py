"""Generate boxplots of validation errors (MAE/RMSE) for EDA boxplot cases."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

if __package__ in (None, ""):
    import sys

    sys.path.append(str(PROJECT_ROOT))

from eda_boxplots.settings import DEFAULT_OUTPUT_ROOT

DEFAULT_CASES: tuple[str, ...] = ("case_4", "case_5", "case_6", "case_15")
DEFAULT_METRICS: tuple[str, ...] = ("MAE", "RMSE")
OUTPUT_SUBDIR = "error_boxplots"


def _normalise_case_ids(case_ids: Iterable[str]) -> list[str]:
    normalised: list[str] = []
    for raw in case_ids:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered.startswith("case_"):
            candidate = lowered
        elif lowered.startswith("case"):
            candidate = f"case_{lowered[4:].lstrip('_')}"
        elif lowered.isdigit():
            candidate = f"case_{lowered}"
        else:
            candidate = lowered
        if candidate and candidate not in normalised:
            normalised.append(candidate)
    return normalised


def _resolve_metric_columns(df: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    if not requested:
        requested = DEFAULT_METRICS

    mapping: dict[str, str] = {}
    for column in df.columns:
        key = column.lower()
        mapping.setdefault(key, column)
        normalized = key.replace("-", "_")
        mapping.setdefault(normalized, column)

    resolved: list[str] = []
    for metric in requested:
        if metric is None:
            continue
        key = str(metric).strip().lower()
        if not key:
            continue
        key = key.replace("-", "_")
        column = mapping.get(key)
        if column and column not in resolved:
            resolved.append(column)

    if not resolved:
        raise ValueError("None of the requested metrics are present in the metrics data.")

    return resolved


def _prepare_metrics_frame(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    working = df.copy()
    for metric in metrics:
        working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working = working.dropna(subset=list(metrics))
    if working.empty:
        raise ValueError("Metrics data is empty after dropping non-numeric entries.")
    return working


def resolve_metric_columns(df: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    """Expose metric column resolution for reuse in case workflows."""
    return _resolve_metric_columns(df, requested)


def prepare_metrics_frame(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Expose metrics frame preparation for reuse in case workflows."""
    return _prepare_metrics_frame(df, metrics)


def normalise_case_ids(case_ids: Iterable[str]) -> list[str]:
    """Expose case id normalization for reuse in case workflows."""
    return _normalise_case_ids(case_ids)


def load_metrics(outputs_root: Path, case_ids: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    missing_cases: list[str] = []

    for case_id in case_ids:
        case_dir = outputs_root / case_id / "comparison" / "last_n_day_validation"
        if not case_dir.exists():
            missing_cases.append(case_id)
            continue

        case_files = sorted(case_dir.glob("*_metrics_by_ticker.csv"))
        if not case_files:
            missing_cases.append(case_id)
            continue

        for csv_path in case_files:
            frame = pd.read_csv(csv_path)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["case_id"] = case_id
            frame["source_file"] = csv_path.name
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            "No *_metrics_by_ticker.csv files found for the requested cases under "
            f"{outputs_root}."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["ticker"] = combined["ticker"].astype(str).str.upper()
    combined["model"] = combined["model"].astype(str)
    return combined, missing_cases


def plot_error_boxplots(
    df: pd.DataFrame,
    metrics: Sequence[str],
    *,
    case_ids: Sequence[str],
    output_dir: Path,
    dpi: int = 300,
    show: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False)
    axes = axes[0]

    tickers = sorted(df["ticker"].unique())
    ordered_cases = [case for case in case_ids if case in df["case_id"].unique()]
    if not ordered_cases:
        ordered_cases = sorted(df["case_id"].unique())

    palette = sns.color_palette("Set2", n_colors=len(ordered_cases)) if ordered_cases else None

    for metric, ax in zip(metrics, axes):
        if ordered_cases:
            sns.boxplot(
                data=df,
                x="ticker",
                y=metric,
                hue="case_id",
                order=tickers,
                hue_order=ordered_cases,
                palette=palette,
                ax=ax,
            )
            ax.legend(title="Case", loc="upper right", frameon=True)
        else:
            sns.boxplot(
                data=df,
                x="ticker",
                y=metric,
                order=tickers,
                ax=ax,
            )
            if ax.legend_ is not None:
                ax.legend_.remove()

        ax.set_title(f"{metric} by ticker")
        ax.set_xlabel("Ticker")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Distribution of validation errors across models", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    cases_slug = "-".join(ordered_cases) if ordered_cases else "cases"
    metrics_slug = "-".join(metric.lower() for metric in metrics)
    output_path = output_dir / f"error_boxplots_{cases_slug}_{metrics_slug}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return output_path


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Generate MAE/RMSE boxplots per ticker from last-n-day validation metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(DEFAULT_CASES),
        help="Case identifiers to include (e.g. case_4 case_5).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric columns to plot (matched case-insensitively).",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory containing case output folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional destination for the generated figure. Defaults to <outputs-root>/error_boxplots.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Image export DPI.")
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display the plot interactively after saving.",
    )
    args = parser.parse_args(argv)

    case_ids = _normalise_case_ids(args.cases) if args.cases else list(DEFAULT_CASES)
    if not case_ids:
        case_ids = list(DEFAULT_CASES)

    outputs_root = Path(args.outputs_root)
    if not outputs_root.is_absolute():
        outputs_root = (PROJECT_ROOT / outputs_root).resolve()
    else:
        outputs_root = outputs_root.resolve()

    data, missing_cases = load_metrics(outputs_root, case_ids)

    if missing_cases:
        missing_str = ", ".join(sorted(set(missing_cases)))
        print(f"Warning: no metrics files found for: {missing_str}")

    resolved_metrics = _resolve_metric_columns(data, args.metrics)
    prepared = _prepare_metrics_frame(data, resolved_metrics)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (PROJECT_ROOT / output_dir).resolve()
        else:
            output_dir = output_dir.resolve()
    else:
        output_dir = outputs_root / OUTPUT_SUBDIR

    figure_path = plot_error_boxplots(
        prepared,
        resolved_metrics,
        case_ids=case_ids,
        output_dir=output_dir,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"Saved boxplot figure to {figure_path}")
    return figure_path


if __name__ == "__main__":
    main()
