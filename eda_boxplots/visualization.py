"""Visualization utilities for EDA boxplots."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

_DEFAULT_OUTPUT_DIR = Path("eda_boxplots") / "outputs"


def _sanitize_filename(label: str, suffix: str) -> str:
    base = re.sub(r"[^\w.-]+", "_", label.strip().lower()) or "eda_boxplots"
    base = base.strip("._")
    suffix = suffix.strip("._")
    if suffix:
        return f"{base}_{suffix}.png"
    return f"{base}.png"


def plot_features_boxplot(
    df: pd.DataFrame,
    *,
    title: str,
    output_dir: Path | str | None = None,
    figsize: Iterable[float] | None = None,
    show: bool = False,
    legend_handles: Sequence | None = None,
) -> Path:
    """Create a multi-feature boxplot figure.

    Parameters
    ----------
    df:
        Feature matrix with numeric columns to visualize.
    title:
        Title displayed on the chart.
    output_dir:
        Directory where the figure should be saved. Defaults to
        ``eda_boxplots/outputs`` when ``None``.
    figsize:
        Optional custom figure size.
    show:
        Whether to display the plot interactively.
    legend_handles:
        Optional legend handles explaining the boxplot components.
    """

    if df.empty:
        raise ValueError("DataFrame is empty; nothing to plot.")

    directory = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    n_features = len(df.columns)
    width = max(8.0, n_features * 1.5)
    size = tuple(figsize) if figsize is not None else (width, 6.0)

    fig, ax = plt.subplots(figsize=size)
    df.plot(kind="box", ax=ax, grid=True, patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)

    if legend_handles:
        ax.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper right")

    fig.tight_layout()

    filename = _sanitize_filename(title or "eda_boxplot", "boxplot")
    output_path = directory / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return output_path


def plot_single_feature_boxplot(
    series: pd.Series,
    *,
    title: str,
    output_dir: Path | str | None = None,
    show: bool = False,
    legend_handles: Sequence | None = None,
) -> Path:
    """Plot a single feature boxplot and save to disk."""

    if series.empty:
        raise ValueError("Series is empty; nothing to plot.")

    directory = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 6.0))
    ax.boxplot(series.dropna(), vert=True, patch_artist=True, labels=[series.name or "feature"])
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if legend_handles:
        ax.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper right")

    fig.tight_layout()

    filename = _sanitize_filename(title or series.name or "feature", "boxplot")
    output_path = directory / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return output_path


__all__ = ["plot_features_boxplot", "plot_single_feature_boxplot"]
