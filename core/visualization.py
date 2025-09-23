"""Common plotting utilities for feature distributions."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        missing_fmt = ", ".join(missing)
        raise KeyError(f"Missing columns in DataFrame: {missing_fmt}")


def features_values_distribution(
    df: pd.DataFrame,
    original_features: pd.DataFrame,
    title: str,
) -> None:
    """Plot histograms comparing original and transformed feature distributions.

    Parameters
    ----------
    df
        DataFrame containing the transformed features.
    original_features
        DataFrame with the original feature values.
    title
        Label describing the type of transformation applied.
    """

    cols = list(original_features.columns)
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
        axes[i, 0].set_title(f"{col} – original data")

        axes[i, 1].hist(transformed_data, bins=50, color="orange", edgecolor="black")
        axes[i, 1].set_title(f"{col} – {title}")

    plt.tight_layout()
    plt.show()


def plot_features_distribution_grid(
    df: pd.DataFrame,
    original_features: pd.DataFrame,
    title: str,
    n_features_per_row: int = 3,
) -> None:
    """Plot a grid of histograms comparing original vs transformed features.

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
    """

    if n_features_per_row <= 0:
        raise ValueError("n_features_per_row must be a positive integer")

    cols = list(original_features.columns)
    if not cols:
        raise ValueError("original_features must contain at least one column")

    _validate_columns(df, cols)

    n_features = len(cols)
    n_rows = int(np.ceil(n_features / n_features_per_row))
    n_cols = n_features_per_row * 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        orig_data = original_features[col].dropna()
        transformed_data = df[col].dropna()

        ax_orig = axes[i * 2]
        ax_orig.hist(orig_data, bins=50, color="skyblue", edgecolor="black")
        ax_orig.set_title(f"{col} – original", fontsize=9)

        ax_trans = axes[i * 2 + 1]
        ax_trans.hist(transformed_data, bins=50, color="orange", edgecolor="black")
        ax_trans.set_title(f"{col} – {title}", fontsize=9)

    for j in range(i * 2 + 2, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Features distribution: {title}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
