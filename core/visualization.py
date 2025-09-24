"""Common plotting utilities for feature distributions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DEFAULT_OUTPUT_DIR = Path("feature_scaling") / "outputs"


def _validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        missing_fmt = ", ".join(missing)
        raise KeyError(f"Missing columns in DataFrame: {missing_fmt}")


def _sanitize_filename(title: str, suffix: str) -> str:
    base = re.sub(r"[^\w.-]+", "_", title.strip().lower())
    base = base.strip("._") or "plot"
    suffix = suffix.strip("._")
    if suffix:
        return f"{base}_{suffix}.png"
    return f"{base}.png"


def _save_figure(fig: plt.Figure, output_dir: Path | str | None, filename: str) -> Path:
    directory = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return filepath


def features_values_distribution(
    df: pd.DataFrame,
    original_features: pd.DataFrame,
    title: str,
    *,
    output_dir: Path | str | None = None,
) -> Path:
    """Create paired histograms for original vs transformed features and save to PNG.

    Parameters
    ----------
    df
        DataFrame containing the transformed features.
    original_features
        DataFrame with the original feature values.
    title
        Label describing the type of transformation applied.
    output_dir
        Target directory for the generated plot. Defaults to
        ``feature_scaling/outputs`` when None.

    Returns
    -------
    Path
        Location of the saved PNG file.
    """

    cols: Sequence[str] = list(original_features.columns)
    if not cols:
        raise ValueError("original_features must contain at least one column")

    _validate_columns(df, cols)

    n_rows = len(cols)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for i, col in enumerate(cols):
        orig_data = original_features[col].dropna()
        transformed_data = df[col].dropna()

        axes[i, 0].hist(orig_data, bins=50, color="skyblue", edgecolor="black")
        axes[i, 0].set_title(f"{col} - original data")

        axes[i, 1].hist(transformed_data, bins=50, color="orange", edgecolor="black")
        axes[i, 1].set_title(f"{col} - {title}")

    fig.tight_layout()

    filename = _sanitize_filename(title or "features", "comparison")
    return _save_figure(fig, output_dir, filename)


def plot_features_distribution_grid(
    df: pd.DataFrame,
    original_features: pd.DataFrame,
    title: str,
    n_features_per_row: int = 3,
    *,
    output_dir: Path | str | None = None,
) -> Path:
    """Create histogram grid for original vs transformed features and save to PNG.

    Parameters
    ----------
    df
        DataFrame containing the transformed features.
    original_features
        DataFrame with the original feature values.
    title
        Label describing the transformation.
    n_features_per_row
        How many features to display per row (each feature uses two subplots).
    output_dir
        Target directory for the generated plot. Defaults to
        ``feature_scaling/outputs`` when None.

    Returns
    -------
    Path
        Location of the saved PNG file.
    """

    if n_features_per_row <= 0:
        raise ValueError("n_features_per_row must be a positive integer")

    cols: Sequence[str] = list(original_features.columns)
    if not cols:
        raise ValueError("original_features must contain at least one column")

    _validate_columns(df, cols)

    n_features = len(cols)
    n_rows = int(np.ceil(n_features / n_features_per_row))
    n_cols = n_features_per_row * 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    flat_axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        orig_data = original_features[col].dropna()
        transformed_data = df[col].dropna()

        ax_orig = flat_axes[i * 2]
        ax_orig.hist(orig_data, bins=50, color="skyblue", edgecolor="black")
        ax_orig.set_title(f"{col} - original", fontsize=9)

        if not orig_data.empty:
            mean_orig = float(orig_data.mean())
            median_orig = float(orig_data.median())
            ax_orig.axvline(mean_orig, color="red", linestyle="--", linewidth=1, label="Mean")
            ax_orig.axvline(median_orig, color="green", linestyle="-.", linewidth=1, label="Median")
            ax_orig.legend(loc="upper right", fontsize=7, frameon=False)


        ax_trans = flat_axes[i * 2 + 1]
        ax_trans.hist(transformed_data, bins=50, color="orange", edgecolor="black")
        ax_trans.set_title(f"{col} - modified", fontsize=9)

        if not transformed_data.empty:
            mean_trans = float(transformed_data.mean())
            median_trans = float(transformed_data.median())
            ax_trans.axvline(mean_trans, color="red", linestyle="--", linewidth=1, label="Mean")
            ax_trans.axvline(median_trans, color="green", linestyle="-.", linewidth=1, label="Median")
            ax_trans.legend(loc="upper right", fontsize=7, frameon=False)

    used_slots = len(cols) * 2
    for ax in flat_axes[used_slots:]:
        fig.delaxes(ax)

    fig.suptitle(f"Features distribution: {title}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = _sanitize_filename(title or "features", "grid")
    return _save_figure(fig, output_dir, filename)
