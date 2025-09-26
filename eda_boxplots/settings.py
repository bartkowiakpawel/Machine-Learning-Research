"""Configuration helpers for the EDA boxplots package."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from feature_scaling.settings import FEATURE_SCALING_CASES

CASE_ID = "eda_boxplots"
CASE_NAME = "EDA boxplots for technical features"

DEFAULT_TICKER: str = "TSLA"
DEFAULT_DATASET_FILENAME: str = "ml_dataset.csv"
DEFAULT_OUTPUT_ROOT = Path("eda_boxplots") / "outputs"

DEFAULT_FEATURES: Sequence[str] = tuple(
    FEATURE_SCALING_CASES.get("case_4", {}).get("features", [])
)

__all__ = [
    "CASE_ID",
    "CASE_NAME",
    "DEFAULT_TICKER",
    "DEFAULT_DATASET_FILENAME",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_FEATURES",
]
