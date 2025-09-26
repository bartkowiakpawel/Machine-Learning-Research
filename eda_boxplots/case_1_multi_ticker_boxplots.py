"""EDA workflow producing boxplots for multiple technology tickers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import pandas as pd

from config import ML_INPUT_DIR
from feature_scaling._shared import (
    clean_feature_matrix,
    filter_ticker,
    load_ml_dataset,
    prepare_feature_matrix,
    resolve_output_dir,
    save_dataframe,
)

try:
    from .settings import (
        DEFAULT_DATASET_FILENAME,
        DEFAULT_FEATURES,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_TICKER,
        DEFAULT_TICKERS,
        get_case_config,
    )
    from .visualization import (
        build_color_map,
        plot_features_boxplot,
        plot_single_feature_boxplot,
    )
except ImportError:
    from eda_boxplots.settings import (
        DEFAULT_DATASET_FILENAME,
        DEFAULT_FEATURES,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_TICKER,
        DEFAULT_TICKERS,
        get_case_config,
    )
    from eda_boxplots.visualization import (
        build_color_map,
        plot_features_boxplot,
        plot_single_feature_boxplot,
    )

CASE_ID = "case_1"
CASE_CONFIG = get_case_config(CASE_ID)
CASE_NAME = CASE_CONFIG.name


def _normalize_tickers(tickers: Sequence[str] | None) -> list[str]:
    """Return a unique, upper-case ticker list with sensible defaults."""
    source = tickers if tickers is not None else DEFAULT_TICKERS
    normalized: list[str] = []
    for value in source:
        ticker = str(value).strip().upper()
        if ticker and ticker not in normalized:
            normalized.append(ticker)
    if not normalized:
        normalized.append(DEFAULT_TICKER)
    return normalized


def run_case(
    *,
    tickers: Sequence[str] | None = None,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    features: list[str] | None = None,
    output_root: Path | str | None = None,
    show_plots: bool = False,
) -> Path:
    """Run case 1 boxplots for the configured tickers."""

    selected_tickers = _normalize_tickers(tickers)
    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    comparison_dir = case_output_dir / "comparison"
    single_plots_dir = comparison_dir / "single_feature_plots"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    single_plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = ML_INPUT_DIR / dataset_filename
    print(f"=== Running {CASE_ID}: {CASE_NAME} ===")
    print(f"Expecting ML dataset at: {dataset_path}")
    print(f"Tickers under comparison: {', '.join(selected_tickers)}")

    data = load_ml_dataset(dataset_path)

    selected_features = list(features) if features is not None else list(DEFAULT_FEATURES)
    if not selected_features:
        raise ValueError("No features configured for EDA boxplots.")

    per_ticker_frames: dict[str, pd.DataFrame] = {}
    combined_frames: list[pd.DataFrame] = []

    for ticker in selected_tickers:
        subset = filter_ticker(data, ticker)
        print(f"Rows for ticker '{ticker}': {len(subset)}")

        if subset.empty:
            print(f"No data available for {ticker}; skipping.")
            continue

        feature_matrix = prepare_feature_matrix(
            subset,
            selected_features,
            target_column=None,
            target_source=None,
        )
        rows_before = len(feature_matrix)
        feature_matrix = clean_feature_matrix(feature_matrix)
        rows_after = len(feature_matrix)
        dropped = rows_before - rows_after
        if dropped > 0:
            print(f"Removed {dropped} rows with NaN/inf values for {ticker}.")

        if feature_matrix.empty:
            print(f"Feature matrix empty after cleaning for {ticker}; skipping.")
            continue

        ticker_dir = case_output_dir / ticker.lower()
        ticker_dir.mkdir(parents=True, exist_ok=True)

        matrix_path = save_dataframe(
            feature_matrix,
            ticker_dir,
            f"{ticker.lower()}_{CASE_ID}_feature_matrix.csv",
        )
        print(f"Clean feature matrix for {ticker} saved to: {matrix_path}")

        summary_path = save_dataframe(
            feature_matrix.describe(include="all").transpose(),
            ticker_dir,
            f"{ticker.lower()}_{CASE_ID}_feature_summary.csv",
        )
        print(f"Feature summary for {ticker} saved to: {summary_path}")

        per_ticker_frames[ticker] = feature_matrix[selected_features].copy()
        combined_frames.append(per_ticker_frames[ticker].assign(ticker=ticker))

    if not per_ticker_frames:
        print("No tickers produced usable data; aborting EDA workflow.")
        return case_output_dir

    combined_df = pd.concat(combined_frames, ignore_index=True)
    ticker_bundle = "_".join(t.lower() for t in per_ticker_frames.keys())

    combined_matrix_path = save_dataframe(
        combined_df,
        comparison_dir,
        f"{ticker_bundle}_{CASE_ID}_feature_matrix_combined.csv",
    )
    print(f"Combined feature matrix saved to: {combined_matrix_path}")

    summary = combined_df.groupby("ticker")[selected_features].describe()
    summary.columns = [f"{feature}_{stat}" for feature, stat in summary.columns]
    summary_path = save_dataframe(
        summary,
        comparison_dir,
        f"{ticker_bundle}_{CASE_ID}_feature_summary_by_ticker.csv",
    )
    print(f"Per-ticker feature summary saved to: {summary_path}")

    color_map = build_color_map(per_ticker_frames.keys())

    comparison_plot_path = plot_features_boxplot(
        per_ticker_frames,
        selected_features,
        title=f"case_1 boxplots: {'/'.join(per_ticker_frames.keys())}",
        output_dir=comparison_dir,
        show=show_plots,
        color_map=color_map,
    )
    print(f"Comparison boxplot grid saved to: {comparison_plot_path}")

    single_plot_records: list[dict[str, str]] = []
    for feature in selected_features:
        series_map = {
            ticker: frame[feature]
            for ticker, frame in per_ticker_frames.items()
            if feature in frame.columns
        }
        try:
            plot_path = plot_single_feature_boxplot(
                feature_name=feature,
                series_by_ticker=series_map,
                title=f"case_1 {feature} comparison",
                output_dir=single_plots_dir,
                show=show_plots,
                color_map=color_map,
            )
        except ValueError as exc:
            print(f"Skipping feature '{feature}' single plot: {exc}")
            continue

        single_plot_records.append(
            {
                "feature": feature,
                "boxplot_path": str(plot_path),
            }
        )
        print(f"Single-feature comparison saved to: {plot_path}")

    if single_plot_records:
        single_index_path = save_dataframe(
            pd.DataFrame(single_plot_records),
            single_plots_dir,
            f"{ticker_bundle}_{CASE_ID}_single_feature_boxplots.csv",
        )
        print(f"Single feature boxplot index saved to: {single_index_path}")
    else:
        print("No single-feature boxplots were generated.")

    return case_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-ticker boxplot workflow.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_FILENAME,
        help="Dataset filename inside the ML input directory (default: %(default)s).",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help=f"Tickers to include (default: {' '.join(DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional custom output directory root.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional list of feature names to plot (defaults to configured features).",
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
    features = args.features if args.features else None

    run_case(
        tickers=args.tickers,
        dataset_filename=args.dataset,
        features=features,
        output_root=output_root,
        show_plots=args.show_plots,
    )
