"""Visualization utilities for EDA boxplots."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_DEFAULT_OUTPUT_DIR = Path("eda_boxplots") / "outputs"


def _sanitize_filename(label: str, suffix: str) -> str:
    base = re.sub(r"[^\w.-]+", "_", label.strip().lower()) or "eda_boxplots"
    base = base.strip("._")
    suffix = suffix.strip("._")
    if suffix:
        return f"{base}_{suffix}.png"
    return f"{base}.png"


def build_color_map(tickers: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    """Return a stable color mapping for the provided tickers."""

    ordered = list(dict.fromkeys(tickers))
    if not ordered:
        return {}

    cmap = plt.get_cmap("Set2")
    if len(ordered) == 1:
        colors = [cmap(0.5)]
    else:
        positions = np.linspace(0.1, 0.9, num=len(ordered))
        colors = [cmap(pos) for pos in positions]
    return {ticker: color for ticker, color in zip(ordered, colors)}


def plot_features_boxplot(
    data_by_ticker: Mapping[str, pd.DataFrame],
    features: Sequence[str],
    *,
    title: str,
    output_dir: Path | str | None = None,
    show: bool = False,
    color_map: Mapping[str, tuple[float, float, float, float]] | None = None,
    n_cols: int = 3,
) -> Path:
    """Create a grid of boxplots comparing tickers for each feature."""

    if not data_by_ticker:
        raise ValueError("data_by_ticker must contain at least one entry")
    if not features:
        raise ValueError("features must contain at least one feature name")

    directory = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    available_tickers = [ticker for ticker, frame in data_by_ticker.items() if not frame.empty]
    if not available_tickers:
        raise ValueError("All provided ticker frames are empty; nothing to plot.")

    if color_map:
        ticker_order = [ticker for ticker in color_map if ticker in available_tickers]
    else:
        ticker_order = available_tickers
    if not ticker_order:
        ticker_order = available_tickers

    color_map = color_map or build_color_map(ticker_order)
    ticker_order = [ticker for ticker in ticker_order if ticker in color_map]

    n_features = len(features)
    n_cols = max(1, min(n_cols, n_features))
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    legend_handles = [
        Patch(facecolor=color_map[ticker], edgecolor="black", label=ticker, alpha=0.6)
        for ticker in ticker_order
    ]

    for idx, feature in enumerate(features):
        ax = axes[idx]
        box_data = []
        labels = []
        colors = []
        for ticker in ticker_order:
            frame = data_by_ticker.get(ticker)
            if frame is None or feature not in frame.columns:
                continue
            series = pd.to_numeric(frame[feature], errors="coerce").dropna()
            if series.empty:
                continue
            box_data.append(series.values)
            labels.append(ticker)
            colors.append(color_map[ticker])

        if not box_data:
            ax.set_axis_off()
            ax.set_title(f"{feature} (no data)")
            continue

        boxes = ax.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_alpha(0.6)
        ax.set_title(feature)
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        if idx == 0 and legend_handles:
            ax.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper right")

    for extra_ax in axes[n_features:]:
        fig.delaxes(extra_ax)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    filename = _sanitize_filename(title or "eda_boxplot", "comparison")
    output_path = directory / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return output_path


def plot_single_feature_boxplot(
    feature_name: str,
    series_by_ticker: Mapping[str, pd.Series],
    *,
    title: str,
    output_dir: Path | str | None = None,
    show: bool = False,
    color_map: Mapping[str, tuple[float, float, float, float]] | None = None,
    include_legend: bool = False,
) -> Path:
    """Plot a single feature boxplot comparing several tickers."""

    if not series_by_ticker:
        raise ValueError("series_by_ticker must contain at least one entry")

    directory = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    if color_map:
        ticker_order = [ticker for ticker in color_map if ticker in series_by_ticker]
    else:
        ticker_order = list(series_by_ticker.keys())
    if not ticker_order:
        ticker_order = list(series_by_ticker.keys())

    color_map = color_map or build_color_map(ticker_order)

    box_data = []
    labels = []
    colors = []
    for ticker in ticker_order:
        series = series_by_ticker.get(ticker)
        if series is None:
            continue
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            continue
        box_data.append(values.values)
        labels.append(ticker)
        colors.append(color_map[ticker])

    if not box_data:
        raise ValueError(f"No numeric data available to plot feature '{feature_name}'.")

    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    boxes = ax.boxplot(box_data, labels=labels, patch_artist=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_alpha(0.6)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if include_legend:
        handles = [
            Patch(facecolor=color_map[label], edgecolor="black", label=label, alpha=0.6)
            for label in labels
        ]
        ax.legend(handles, [h.get_label() for h in handles], loc="upper right")

    fig.tight_layout()

    filename = _sanitize_filename(f"{feature_name}_comparison", "boxplot")
    output_path = directory / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return output_path


__all__ = [
    "build_color_map",
    "plot_features_boxplot",
    "plot_single_feature_boxplot",
]
