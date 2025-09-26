"""Configuration helpers for the EDA boxplots package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from feature_scaling.settings import FEATURE_SCALING_CASES

DEFAULT_TICKER: str = "TSLA"
DEFAULT_TICKERS: tuple[str, ...] = ("TSLA", "AAPL", "MSFT")
DEFAULT_DATASET_FILENAME: str = "ml_dataset.csv"
DEFAULT_OUTPUT_ROOT = Path("eda_boxplots") / "outputs"
DEFAULT_FEATURES: Sequence[str] = tuple(
    FEATURE_SCALING_CASES.get("case_4", {}).get("features", [])
)

@dataclass(frozen=True)
class CaseConfig:
    """Descriptor for an EDA boxplot case."""

    case_id: str
    name: str
    description: str | None = None


_CASE_CONFIGS: Mapping[str, CaseConfig] = {
    "case_1": CaseConfig(
        case_id="case_1",
        name="Case 1: multi-ticker technical feature boxplots",
        description="Compare raw technical features across tickers via boxplots.",
    ),
    "case_2": CaseConfig(
        case_id="case_2",
        name="Case 2: Yeo-Johnson multi-ticker boxplots",
        description="Apply Yeo-Johnson transformation prior to boxplot comparison.",
    ),
}


def get_case_config(case_id: str) -> CaseConfig:
    """Return the configuration record for the requested case."""

    try:
        return _CASE_CONFIGS[case_id]
    except KeyError as exc:
        raise KeyError(f"Unknown EDA boxplots case id: {case_id}") from exc


AVAILABLE_CASE_IDS: tuple[str, ...] = tuple(_CASE_CONFIGS.keys())

__all__ = [
    "DEFAULT_TICKER",
    "DEFAULT_TICKERS",
    "DEFAULT_DATASET_FILENAME",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_FEATURES",
    "CaseConfig",
    "AVAILABLE_CASE_IDS",
    "get_case_config",
]
