"""Configuration defaults for drop/rise classification workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_TICKER: str = "TSLA"
DEFAULT_DATASET_FILENAME: str = "ml_dataset.csv"
DEFAULT_OUTPUT_ROOT = Path("drop_rise_classification") / "outputs"
DEFAULT_NEUTRAL_BAND: float = 0.002


@dataclass(frozen=True)
class CaseConfig:
    """Descriptor for a drop/rise classification case."""

    case_id: str
    name: str
    description: str | None = None
    tickers: tuple[str, ...] | None = None


_CASE_CONFIGS: Mapping[str, CaseConfig] = {
    "case_1": CaseConfig(
        case_id="case_1",
        name="Case 1: TSLA drop/rise classification baseline",
        description="Prepare TSLA classification dataset, perform RFECV feature selection, and benchmark classifiers.",
        tickers=("TSLA",),
    ),
    "case_2": CaseConfig(
        case_id="case_2",
        name="Case 2: TSLA drop/rise classification without distribution diagnostics",
        description=(
            "Run the TSLA classification pipeline while skipping distribution diagnostics to avoid plotting dependencies."
        ),
        tickers=("TSLA",),
    ),
    "case_3": CaseConfig(
        case_id="case_3",
        name="Case 3: TSLA classification with fixed features and GradientBoosting enrichments",
        description=(
            "Use the established TSLA feature set, train GradientBoosting as the primary model, "
            "and persist model-derived probability features for downstream workflows."
        ),
        tickers=("TSLA",),
    ),
    "case_4": CaseConfig(
        case_id="case_4",
        name="Case 4: TSLA regression with RFECV feature selection",
        description=(
            "Perform RFECV-based feature selection on the Case 3 enriched dataset and benchmark regression estimators."
        ),
        tickers=("TSLA",),
    ),
    "case_5": CaseConfig(
        case_id="case_5",
        name="Case 5: TSLA regression with feature screening and RFECV",
        description=(
            "Filter features using training-only screening (missing ratio, variance, mutual information) "
            "before executing RFECV and model benchmarking."
        ),
        tickers=("TSLA",),
    ),
    "case_6": CaseConfig(
        case_id="case_6",
        name="Case 6: TSLA classification with feature screening and RFECV",
        description=(
            "Apply the regression-style feature screening pipeline to the classification workflow prior to RFECV "
            "to compare with the fixed-feature Case 3 outputs."
        ),
        tickers=("TSLA",),
    ),
    "case_7": CaseConfig(
        case_id="case_7",
        name="Case 7: TSLA regression with feature screening (Case 6 dataset)",
        description=(
            "Replicate the Case 5 regression pipeline while sourcing inputs from the Case 6 classification dataset "
            "to compare downstream performance."
        ),
        tickers=("TSLA",),
    ),
    "case_8": CaseConfig(
        case_id="case_8",
        name="Case 8: TSLA classification with realistic time-series CV baseline",
        description=(
            "Time-series CV baseline without model-derived meta-features, to evaluate classification difficulty after "
            "removing leakage and using chronological slicing."
        ),
        tickers=("TSLA",),
    ),
    "case_9": CaseConfig(
        case_id="case_9",
        name="Case 9: Multi-ticker classification with time-series CV baseline",
        description=(
            "Combine classification datasets from multiple tickers before running feature screening, RFECV, "
            "and model benchmarking to evaluate the impact of a broader training universe."
        ),
        tickers=None,
    ),
    "case_10": CaseConfig(
        case_id="case_10",
        name="Case 10: Multi-ticker classification with compact feature expansion",
        description=(
            "Reuse the multi-ticker pipeline while leveraging compact feature expansion and lightweight pruning to "
            "control dimensionality before RFECV and model benchmarking."
        ),
        tickers=None,
    ),
}


def get_case_config(case_id: str) -> CaseConfig:
    try:
        return _CASE_CONFIGS[case_id]
    except KeyError as exc:  # pragma: no cover - defensive runtime guard
        raise KeyError(f"Unknown drop/rise classification case id: {case_id}") from exc


AVAILABLE_CASE_IDS: tuple[str, ...] = tuple(_CASE_CONFIGS.keys())


__all__ = [
    "AVAILABLE_CASE_IDS",
    "CaseConfig",
    "DEFAULT_DATASET_FILENAME",
    "DEFAULT_NEUTRAL_BAND",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TICKER",
    "get_case_config",
]
