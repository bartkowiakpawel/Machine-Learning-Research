"""Shared helpers for feature scaling case studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.model_evaluation import plot_learning_curves_for_models


def resolve_output_dir(
    case_id: str,
    default_output_root: Path,
    output_root: Path | str | None,
) -> Path:
    root = Path(output_root) if output_root is not None else default_output_root
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def load_ml_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"ML dataset not found at {dataset_path}. Run the ML preparation pipeline first."
        )

    df = pd.read_csv(dataset_path)
    if "ticker" not in df.columns:
        raise KeyError("ML dataset must contain a 'ticker' column")

    return df


def filter_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    ticker_upper = ticker.upper()
    return df[df["ticker"].astype(str).str.upper() == ticker_upper].copy()


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str | None,
    target_source: str | None,
) -> pd.DataFrame:
    if not feature_columns:
        raise ValueError("Feature list for the case study is empty.")

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        missing_fmt = ", ".join(missing_features)
        raise KeyError(f"Missing feature columns in dataset: {missing_fmt}")

    working = df.copy()

    if target_column:
        if target_column not in working.columns:
            if target_source and target_source in working.columns:
                working[target_column] = working[target_source]
            else:
                raise KeyError(
                    "Target column not found and no fallback target_source available in dataset."
                )
        selected_columns = list(feature_columns) + [target_column]
    else:
        selected_columns = list(feature_columns)

    return working.loc[:, selected_columns]


def clean_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sanitized = df.replace([np.inf, -np.inf], np.nan)
    return sanitized.dropna()


DEFAULT_CSV_KWARGS = {"index": False, "sep": ",", "decimal": "."}


def save_dataframe(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str,
    *,
    csv_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    output_path = output_dir / filename
    options = dict(DEFAULT_CSV_KWARGS)
    if csv_kwargs is not None:
        options.update(csv_kwargs)
    df.to_csv(output_path, **options)
    return output_path


def run_learning_workflow(
    model_dict: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    title_suffix: str,
    output_dir: Path,
    show_plots: bool,
) -> tuple[pd.DataFrame, Path | None, Path | None]:
    return plot_learning_curves_for_models(
        model_dict,
        X,
        y,
        main_title=f"Learning curves for {title_suffix}",
        output_dir=output_dir,
        show=show_plots,
        feature_distribution=None,
        original_feature_distribution=None,
    )

