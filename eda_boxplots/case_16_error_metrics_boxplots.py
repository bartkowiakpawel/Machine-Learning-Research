"""Case 16 workflow: visualize validation error distributions (MAE/RMSE)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

from core.shared_utils import resolve_output_dir, save_dataframe, write_case_metadata

from eda_boxplots.error_metrics_boxplots import (
    DEFAULT_CASES as DEFAULT_SOURCE_CASES,
    DEFAULT_METRICS as DEFAULT_METRIC_COLUMNS,
    OUTPUT_SUBDIR as ERROR_PLOT_SUBDIR,
    load_metrics,
    normalise_case_ids,
    plot_error_boxplots,
    prepare_metrics_frame,
    resolve_metric_columns,
)
from eda_boxplots.settings import DEFAULT_OUTPUT_ROOT, get_case_config

CASE_ID = "case_16"
CASE_CONFIG = get_case_config(CASE_ID)
CASE_NAME = CASE_CONFIG.name
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_root(path: Path | str | None, *, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    if raw.is_absolute():
        return raw.resolve()
    return (PROJECT_ROOT / raw).resolve()


def run_case(
    case_ids: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    *,
    output_root: Path | str | None = None,
    source_output_root: Path | str | None = None,
    show_plots: bool = False,
) -> Path:
    selected_case_ids = normalise_case_ids(case_ids or DEFAULT_SOURCE_CASES)
    if not selected_case_ids:
        selected_case_ids = list(DEFAULT_SOURCE_CASES)

    source_root = _resolve_root(source_output_root, default=DEFAULT_OUTPUT_ROOT)
    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    comparison_dir = case_output_dir / "comparison"
    plots_dir = comparison_dir / ERROR_PLOT_SUBDIR
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_frame, missing_cases = load_metrics(source_root, selected_case_ids)
    if missing_cases:
        missing_str = ", ".join(sorted(set(missing_cases)))
        print(f"Warning: no metrics files found for: {missing_str}")

    resolved_metrics = resolve_metric_columns(metrics_frame, metrics or DEFAULT_METRIC_COLUMNS)
    prepared_metrics = prepare_metrics_frame(metrics_frame, resolved_metrics)

    combined_metrics_path = save_dataframe(
        metrics_frame,
        comparison_dir,
        f"{CASE_ID}_raw_metrics.csv",
    )
    print(f"Combined metrics loaded from previous cases saved to: {combined_metrics_path}")

    prepared_metrics_path = save_dataframe(
        prepared_metrics,
        comparison_dir,
        f"{CASE_ID}_prepared_metrics.csv",
    )
    print(f"Filtered metrics used for plotting saved to: {prepared_metrics_path}")

    figure_path = plot_error_boxplots(
        prepared_metrics,
        resolved_metrics,
        case_ids=selected_case_ids,
        output_dir=plots_dir,
        show=show_plots,
    )
    print(f"Error metric boxplots saved to: {figure_path}")

    tickers = prepared_metrics["ticker"].unique().tolist()
    models = prepared_metrics["model"].unique().tolist()

    extras = {
        "source_case_ids": selected_case_ids,
        "missing_case_ids": sorted(set(missing_cases)),
        "metrics_plotted": resolved_metrics,
        "artifacts": {
            "raw_metrics_csv": combined_metrics_path.relative_to(case_output_dir).as_posix(),
            "prepared_metrics_csv": prepared_metrics_path.relative_to(case_output_dir).as_posix(),
            "boxplot_figure": figure_path.relative_to(case_output_dir).as_posix(),
        },
    }

    write_case_metadata(
        case_dir=case_output_dir,
        case_id=CASE_ID,
        case_name=CASE_NAME,
        package="eda_boxplots",
        dataset_path=None,
        tickers=tickers,
        features={"metrics": resolved_metrics},
        target=None,
        models=models,
        extras=extras,
    )

    return case_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MAE/RMSE boxplots from existing last-n-day validation metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Case identifiers to include when gathering metrics (e.g. case_4 case_5).",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metric columns to plot (case-insensitive). Defaults to MAE and RMSE.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional destination root for Case 16 artifacts.",
    )
    parser.add_argument(
        "--source-output-root",
        default=None,
        help="Root directory containing the metrics produced by earlier cases.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively after saving them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_case(
        case_ids=args.cases,
        metrics=args.metrics,
        output_root=args.output_root,
        source_output_root=args.source_output_root,
        show_plots=args.show_plots,
    )
