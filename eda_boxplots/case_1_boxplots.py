"""EDA workflow producing boxplots for a single ticker."""

from __future__ import annotations

import re
from pathlib import Path

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from config import ML_INPUT_DIR
from feature_scaling._shared import (
    clean_feature_matrix,
    filter_ticker,
    load_ml_dataset,
    prepare_feature_matrix,
    resolve_output_dir,
    save_dataframe,
)

from .settings import (
    CASE_ID,
    CASE_NAME,
    DEFAULT_DATASET_FILENAME,
    DEFAULT_FEATURES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TICKER,
)
from .visualization import plot_features_boxplot, plot_single_feature_boxplot


_FEATURE_FILENAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize_feature_name(name: str) -> str:
    cleaned = _FEATURE_FILENAME_PATTERN.sub("_", name.strip().lower())
    cleaned = cleaned.strip("._")
    return cleaned or "feature"


def _build_boxplot_legend() -> list:
    """Return legend handles explaining the boxplot components."""

    return [
        Patch(
            facecolor="#add8e6",
            edgecolor="black",
            label="IQR (25th-75th percentile)",
        ),
        Line2D([0], [0], color="black", linewidth=2, label="Median"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1,
            label="Whiskers (±1.5 IQR)",
        ),
    ]


def run_case(
    *,
    ticker: str = DEFAULT_TICKER,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    features: list[str] | None = None,
    output_root: Path | str | None = None,
    show_plots: bool = False,
) -> Path:
    """Run the EDA boxplot workflow for a given ticker."""

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    ticker_dir = case_output_dir / ticker.lower()
    single_plots_dir = ticker_dir / "single_feature_plots"
    ticker_dir.mkdir(parents=True, exist_ok=True)
    single_plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = ML_INPUT_DIR / dataset_filename
    print(f"=== Running {CASE_ID}: {CASE_NAME} ===")
    print(f"Expecting ML dataset at: {dataset_path}")

    data = load_ml_dataset(dataset_path)
    subset = filter_ticker(data, ticker)
    print(f"Rows for ticker '{ticker}': {len(subset)}")

    if subset.empty:
        print("No data available for the ticker; aborting EDA workflow.")
        return ticker_dir

    selected_features = list(features) if features is not None else list(DEFAULT_FEATURES)
    if not selected_features:
        raise ValueError("No features configured for EDA boxplots.")

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
        print(f"Removed {dropped} rows with NaN/inf values from feature matrix.")

    if feature_matrix.empty:
        print("Feature matrix empty after cleaning; aborting EDA workflow.")
        return ticker_dir

    matrix_path = save_dataframe(
        feature_matrix,
        ticker_dir,
        f"{ticker.lower()}_{CASE_ID}_feature_matrix.csv",
    )
    print(f"Clean feature matrix saved to: {matrix_path}")

    summary_path = save_dataframe(
        feature_matrix.describe(include="all").transpose(),
        ticker_dir,
        f"{ticker.lower()}_{CASE_ID}_feature_summary.csv",
    )
    print(f"Feature summary saved to: {summary_path}")

    legend_handles = _build_boxplot_legend()
    boxplot_path = plot_features_boxplot(
        feature_matrix,
        title=f"{ticker} feature boxplots",
        output_dir=ticker_dir,
        show=show_plots,
        legend_handles=legend_handles,
    )
    print(f"Boxplot figure saved to: {boxplot_path}")

    single_plot_records: list[dict[str, str]] = []
    for column in feature_matrix.columns:
        column_series = feature_matrix[column]
        filename_component = _sanitize_feature_name(column)
        single_plot_path = plot_single_feature_boxplot(
            column_series,
            title=f"{ticker} - {column} boxplot",
            output_dir=single_plots_dir,
            show=show_plots,
            legend_handles=legend_handles,
        )
        single_plot_records.append(
            {
                "feature": column,
                "boxplot_path": str(single_plot_path),
                "filename_component": filename_component,
            }
        )
        print(f"Single-feature boxplot saved to: {single_plot_path}")

    single_index_path = save_dataframe(
        pd.DataFrame(single_plot_records),
        single_plots_dir,
        f"{ticker.lower()}_{CASE_ID}_single_boxplots.csv",
    )
    print(f"Single feature boxplot index saved to: {single_index_path}")

    return ticker_dir


__all__ = ["run_case"]
