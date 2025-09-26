"""EDA & Boxplots package entry points."""

from __future__ import annotations

from pathlib import Path

from .case_1_boxplots import run_case
from .settings import (
    CASE_ID,
    CASE_NAME,
    DEFAULT_DATASET_FILENAME,
    DEFAULT_FEATURES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TICKER,
)

__all__ = [
    "CASE_ID",
    "CASE_NAME",
    "DEFAULT_TICKER",
    "DEFAULT_DATASET_FILENAME",
    "DEFAULT_FEATURES",
    "DEFAULT_OUTPUT_ROOT",
    "run_case",
]
