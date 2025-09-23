"""Demonstration of feature scaling on the California Housing dataset."""

from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PowerTransformer

from core.visualization import plot_features_distribution_grid


def _display_dataframe(
    df: pd.DataFrame,
    label: str,
    formatter: Callable[[pd.DataFrame], None] | None = None,
) -> None:
    """Display a DataFrame head with fallback for non-notebook environments."""

    print(f"\n{label}")
    print(f"Shape: {df.shape}")

    if formatter is not None:
        formatter(df)
        return

    try:
        from IPython.display import display  # type: ignore

        display(df.head())
    except ImportError:
        print(df.head())


def run_demo(n_features_per_row: int = 3) -> None:
    """Fetch the California Housing dataset and visualise Yeo–Johnson scaling."""

    data = fetch_california_housing(as_frame=True)
    original_frame = data.frame

    _display_dataframe(original_frame, "Original data (first rows)")

    features = original_frame.drop(columns=["MedHouseVal"])

    transformer = PowerTransformer(method="yeo-johnson")
    transformed_features = transformer.fit_transform(features)

    transformed_frame = pd.DataFrame(transformed_features, columns=features.columns)

    _display_dataframe(
        transformed_frame,
        "Data after Yeo–Johnson transformation (first rows)",
    )

    plot_features_distribution_grid(
        transformed_frame,
        features,
        title="Yeo-Johnson applied for California Housing set",
        n_features_per_row=n_features_per_row,
    )


if __name__ == "__main__":
    run_demo()
