"""Shared helpers for ML case studies and EDA workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import json
from datetime import datetime, timezone

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



def write_case_metadata(
    *,
    case_dir: Path,
    case_id: str,
    case_name: str,
    package: str,
    dataset_path: Path | str | None,
    tickers: Sequence[str],
    features: Any,
    target: str | None,
    models: Sequence[str],
    extras: Mapping[str, Any] | None = None,
    filename: str = 'meta.json',
) -> Path:
    """Persist a metadata descriptor for a case workflow."""

    case_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    def _clean_sequence(values: Sequence[str]) -> list[str]:
        cleaned: list[str] = []
        seen_upper = set()
        for value in values:
            if value is None:
                continue
            normalised = str(value).strip()
            if not normalised:
                continue
            key = normalised.upper()
            if key in seen_upper:
                continue
            cleaned.append(normalised)
            seen_upper.add(key)
        return cleaned

    dataset_str = str(dataset_path) if dataset_path is not None else None
    tickers_list = _clean_sequence(tickers)
    models_list = _clean_sequence(models)

    digits = ''.join(ch for ch in case_id if ch.isdigit())
    payload: dict[str, Any] = {
        'case_id': case_id,
        'case_name': case_name,
        'package': package,
        'dataset': dataset_str,
        'tickers': tickers_list,
        'target': target,
        'models_used': models_list,
        'features': features,
        'executed_at': timestamp,
        'metadata_generated_at': timestamp,
    }
    if digits:
        payload['case_number'] = int(digits)
    if extras:
        payload.update(extras)

    output_path = case_dir / filename
    with output_path.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
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

