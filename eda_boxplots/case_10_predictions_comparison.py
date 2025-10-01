"""EDA workflow comparing last-N-day predictions across cases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

from config import MODEL_DICT
from core.shared_utils import (
    resolve_output_dir,
    save_dataframe,
    write_case_metadata,
)

try:
    from .settings import (
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_TICKERS,
        get_case_config,
    )
except ImportError:
    from eda_boxplots.settings import (
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_TICKERS,
        get_case_config,
    )

CASE_ID = "case_10"
CASE_CONFIG = get_case_config(CASE_ID)
DEFAULT_SOURCE_CASE_IDS: tuple[str, ...] = (
    "case_3",
    "case_4",
    "case_5",
    "case_6",
    "case_7",
    "case_8",
    "case_9",
)
LAST_N_DAYS = 15


@dataclass(frozen=True)
class CasePredictionArtifact:
    """Descriptor for a predictions CSV produced by an earlier case."""

    case_id: str
    path: Path


def _mean_absolute_percentage_error(actual: np.ndarray, prediction: np.ndarray, *, epsilon: float = 1e-8) -> float:
    """Return mean absolute percentage error in percent, ignoring near-zero actuals."""

    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(prediction, dtype=float)
    denom = np.abs(actual_arr)
    mask = denom > epsilon
    if not np.any(mask):
        return float('nan')
    mape = np.abs((actual_arr[mask] - pred_arr[mask]) / denom[mask])
    return float(np.mean(mape) * 100.0)


def _median_absolute_percentage_error(actual: np.ndarray, prediction: np.ndarray, *, epsilon: float = 1e-8) -> float:
    """Return median absolute percentage error in percent, ignoring near-zero actuals."""

    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(prediction, dtype=float)
    denom = np.abs(actual_arr)
    mask = denom > epsilon
    if not np.any(mask):
        return float('nan')
    mape = np.abs((actual_arr[mask] - pred_arr[mask]) / denom[mask]) * 100.0
    return float(np.median(mape))


def _normalize_case_ids(case_ids: Sequence[str] | None) -> tuple[str, ...]:
    if case_ids is None or not case_ids:
        return DEFAULT_SOURCE_CASE_IDS
    normalized = []
    for case_id in case_ids:
        cid = str(case_id).strip().lower()
        if not cid:
            continue
        if not cid.startswith("case_"):
            cid = f"case_{cid}"
        normalized.append(cid)
    return tuple(dict.fromkeys(normalized))


def _normalize_tickers(tickers: Sequence[str] | None) -> tuple[str, ...]:
    if tickers is None or not tickers:
        return DEFAULT_TICKERS
    normalized = []
    for ticker in tickers:
        value = str(ticker).strip().upper()
        if value:
            normalized.append(value)
    return tuple(dict.fromkeys(normalized))


def _find_predictions_file(case_output_dir: Path, case_id: str) -> CasePredictionArtifact | None:
    validation_dir = case_output_dir / "comparison" / "last_n_day_validation"
    if not validation_dir.exists():
        return None
    pattern = f"*_{case_id}_last_*_day_predictions.csv"
    matches = sorted(validation_dir.glob(pattern))
    if not matches:
        return None
    return CasePredictionArtifact(case_id=case_id, path=matches[0])


def _load_case_predictions(artifact: CasePredictionArtifact) -> pd.DataFrame:
    frame = pd.read_csv(artifact.path)
    required_columns = {"Date", "ticker", "actual", "prediction"}
    missing = required_columns.difference(frame.columns)
    if missing:
        missing_fmt = ", ".join(sorted(missing))
        raise KeyError(
            f"Predictions file for {artifact.case_id} is missing required columns: {missing_fmt}"
        )
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    aggregated = (
        frame.groupby(["ticker", "Date"], as_index=False)
        .agg(actual=("actual", "mean"), prediction=("prediction", "mean"))
        .assign(case_id=artifact.case_id)
    )
    return aggregated


def _collect_predictions(
    *,
    case_ids: Sequence[str],
    source_root: Path,
    tickers: Sequence[str],
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for case_id in case_ids:
        case_dir = source_root / case_id
        artifact = _find_predictions_file(case_dir, case_id)
        if artifact is None:
            print(f"No predictions file found for {case_id} under {case_dir}; skipping.")
            continue
        try:
            frame = _load_case_predictions(artifact)
        except KeyError as exc:
            print(f"Skipping {case_id} due to malformed predictions file: {exc}")
            continue
        frame = frame[frame["ticker"].isin(tickers)]
        if frame.empty:
            print(f"Predictions for {case_id} contain no rows for tickers {tickers}; skipping.")
            continue
        records.append(frame)
    if not records:
        raise RuntimeError("No predictions data loaded from the specified cases.")
    combined = pd.concat(records, ignore_index=True)
    combined = combined.sort_values(["ticker", "Date", "case_id"])
    return combined


def _prepare_ticker_frames(combined: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
    ticker_frames: dict[str, pd.DataFrame] = {}
    for ticker, ticker_frame in combined.groupby("ticker"):
        actual_series = (
            ticker_frame.groupby("Date")
            .agg(actual=("actual", "mean"))
            .sort_index()
        )
        prediction_pivot = (
            ticker_frame.pivot_table(
                index="Date",
                columns="case_id",
                values="prediction",
                aggfunc="mean",
            )
            .sort_index()
        )
        merged = pd.concat([actual_series, prediction_pivot], axis=1)
        merged.index.name = "Date"
        ticker_frames[ticker] = merged
    return ticker_frames


def _plot_predictions(
    *,
    ticker: str,
    frame: pd.DataFrame,
    output_dir: Path,
    case_ids: Sequence[str],
    show: bool,
):
    if frame.empty:
        print(f"No data available to plot for {ticker}; skipping plot generation.")
        return None
    fig, ax = plt.subplots(figsize=(11, 6))
    frame = frame.sort_index()
    if "actual" in frame:
        ax.plot(frame.index, frame["actual"], label="Actual", color="black", linewidth=2.0)
    colors = matplotlib.colormaps["tab10"]
    for idx, case_id in enumerate(case_ids):
        if case_id not in frame.columns:
            continue
        label = get_case_config(case_id).name
        ax.plot(
            frame.index,
            frame[case_id],
            label=label,
            linewidth=1.6,
            color=colors(idx % colors.N),
        )
    ax.set_title(f"{ticker}: Actual vs. Case Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    output_path = output_dir / f"{ticker.lower()}_{CASE_ID}_prediction_comparison.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Prediction comparison plot saved to: {output_path}")
    return output_path


def _compute_metrics(combined: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (ticker, case_id), group in combined.groupby(["ticker", "case_id"]):
        actual = group["actual"].to_numpy()
        prediction = group["prediction"].to_numpy()
        if len(actual) == 0:
            continue
        mae = float(np.mean(np.abs(prediction - actual)))
        rmse = float(np.sqrt(np.mean(np.square(prediction - actual))))
        mape = _mean_absolute_percentage_error(actual, prediction)
        median_mape = _median_absolute_percentage_error(actual, prediction)
        records.append(
            {
                "ticker": ticker,
                "case_id": case_id,
                "case_name": get_case_config(case_id).name,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "median_mape": median_mape,
                "n_points": int(len(actual)),
            }
        )
    if not records:
        return pd.DataFrame(columns=["ticker", "case_id", "case_name", "mae", "rmse", "mape", "median_mape", "n_points"])
    return pd.DataFrame(records)


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    summary = (
        metrics.groupby("case_id", as_index=False)
        .agg(
            case_name=("case_name", "first"),
            tickers_covered=("ticker", lambda values: "/".join(sorted(set(values)))),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            mean_median_mape=("median_mape", "mean"),
            total_points=("n_points", "sum"),
        )
        .sort_values("case_id")
    )
    return summary


def run_case(
    *,
    case_ids: Sequence[str] | None = None,
    tickers: Sequence[str] | None = None,
    output_root: Path | str | None = None,
    source_root: Path | str | None = None,
    show_plots: bool = False,
) -> Path:
    selected_case_ids = _normalize_case_ids(case_ids)
    selected_tickers = _normalize_tickers(tickers)

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    case_output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = case_output_dir / "comparison"
    plots_dir = comparison_dir / "prediction_plots"
    for directory in (comparison_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    resolved_source_root = Path(source_root) if source_root else case_output_dir.parent

    print(f"=== Running {CASE_ID}: {CASE_CONFIG.name} ===")
    print(f"Source cases: {', '.join(selected_case_ids)}")
    print(f"Tickers under comparison: {', '.join(selected_tickers)}")
    print(f"Using predictions source root: {resolved_source_root}")

    combined = _collect_predictions(
        case_ids=selected_case_ids,
        source_root=resolved_source_root,
        tickers=selected_tickers,
    )

    combined_path = save_dataframe(
        combined,
        comparison_dir,
        f"{'_'.join(t.lower() for t in selected_tickers)}_{CASE_ID}_predictions_long.csv",
    )
    print(f"Combined predictions (long format) saved to: {combined_path}")

    ticker_frames = _prepare_ticker_frames(combined)

    plot_records: list[dict[str, str]] = []
    for ticker, frame in ticker_frames.items():
        plot_path = _plot_predictions(
            ticker=ticker,
            frame=frame,
            output_dir=plots_dir,
            case_ids=selected_case_ids,
            show=show_plots,
        )
        if plot_path is not None:
            plot_records.append({"ticker": ticker, "plot_path": str(plot_path)})

    if plot_records:
        plots_index_path = save_dataframe(
            pd.DataFrame(plot_records),
            plots_dir,
            f"{'_'.join(t.lower() for t in selected_tickers)}_{CASE_ID}_plots_index.csv",
        )
        print(f"Plot index saved to: {plots_index_path}")

    metrics = _compute_metrics(combined)
    metrics_path = save_dataframe(
        metrics,
        comparison_dir,
        f"{'_'.join(t.lower() for t in selected_tickers)}_{CASE_ID}_prediction_metrics_by_ticker.csv",
    )
    print(f"Per-ticker prediction metrics saved to: {metrics_path}")

    summary_metrics = _summarize_metrics(metrics)
    summary_path = save_dataframe(
        summary_metrics,
        comparison_dir,
        f"{CASE_ID}_prediction_metrics_summary.csv",
    )
    print(f"Prediction metrics summary saved to: {summary_path}")

    artifacts = {
        "comparison_dir": "comparison",
        "plots_dir": "comparison/prediction_plots",
        "plots_index": plots_index_path.name if plot_records else None,
        "metrics_by_ticker": metrics_path.name,
        "metrics_summary": summary_path.name,
        "source_cases": list(selected_case_ids),
    }

    write_case_metadata(
        case_dir=case_output_dir,
        case_id=CASE_ID,
        case_name=CASE_CONFIG.name,
        package="eda_boxplots",
        dataset_path=None,
        tickers=selected_tickers,
        features=None,
        target="target_14d",
        models=tuple(MODEL_DICT.keys()),
        extras=
        {
            "last_n_day_validation": LAST_N_DAYS,
            "source_cases": list(selected_case_ids),
            "artifacts": artifacts,
            "tickers": list(selected_tickers),
        },
    )

    return case_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare last-n-day predictions from multiple EDA cases.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Subset of case identifiers to include (default: case_3 .. case_9).",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Tickers to include (default: standard package tickers).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional destination root for case 10 artifacts.",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Root directory containing source case outputs (defaults to the parent of the case 10 output).",
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
    source_root = Path(args.source_root) if args.source_root else None

    run_case(
        case_ids=args.case_ids,
        tickers=args.tickers,
        output_root=output_root,
        source_root=source_root,
        show_plots=args.show_plots,
    )
