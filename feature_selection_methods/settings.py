"""Configuration for the feature_selection_methods package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

DEFAULT_DATASET_FILENAME: str = "ml_dataset.csv"
DEFAULT_OUTPUT_ROOT = Path("feature_selection_methods") / "outputs"
DEFAULT_TARGET: str = "target_Close+14_pct"
DEFAULT_TARGET_SOURCE: str = "target_pct_14"
DEFAULT_TICKERS: tuple[str, ...] = ("TSLA", "AAPL", "MSFT")
DEFAULT_FEATURES: tuple[str, ...] = (
    "OpenCloseReturn",
    "IntradayRange",
    "Volume_vs_MA14",
    "avg_volume_14",
    "avg_volume_ccy_14",
    "rolling_volatility_14d",
    "rolling_skew_target_14d",
    "rolling_kurt_target_14d",
    "bb_width_14",
    "macd_diff",
)


@dataclass(frozen=True)
class FeatureSelectionCase:
    """Descriptor for a feature selection experiment."""

    case_id: str
    name: str
    method: str
    description: str
    top_k: int | None = None
    tickers: Sequence[str] | None = None
    target: str | None = DEFAULT_TARGET
    target_source: str | None = DEFAULT_TARGET_SOURCE


FEATURE_SELECTION_CASES: Mapping[str, FeatureSelectionCase] = {
    "case_1": FeatureSelectionCase(
        case_id="case_1",
        name="Case 1: filter based ranking (mutual information)",
        method="filter_mutual_information",
        description="Rank engineered features with mutual information and correlation filters.",
        top_k=10,
        tickers=DEFAULT_TICKERS,
    ),
    "case_2": FeatureSelectionCase(
        case_id="case_2",
        name="Case 2: wrapper selection with RFECV",
        method="wrapper_rfecv",
        description="Use recursive feature elimination with cross-validation and gradient boosted trees.",
        top_k=8,
        tickers=("TSLA",),
    ),
    "case_3": FeatureSelectionCase(
        case_id="case_3",
        name="Case 3: embedded selection via Lasso",
        method="embedded_lasso",
        description="Leverage LassoCV coefficients to keep sparse, high-signal features.",
        top_k=12,
        tickers=("TSLA", "AAPL"),
    ),
    "case_4": FeatureSelectionCase(
        case_id="case_4",
        name="Case 4: RFECV vs brute-force feature sets",
        method="comparison_rfecv_bruteforce",
        description="Compare GradientBoosting performance for RFECV-selected features against all feature combinations.",
        top_k=None,
        tickers=("TSLA",),
    ),
    "case_5": FeatureSelectionCase(
        case_id="case_5",
        name="Case 5: RFECV vs brute-force (Robust scaled)",
        method="comparison_rfecv_bruteforce_robust",
        description="Compare RFECV and exhaustive feature sets with GradientBoosting after RobustScaler preprocessing.",
        top_k=None,
        tickers=("TSLA",),
    ),
    "case_6": FeatureSelectionCase(
        case_id="case_6",
        name="Case 6: RFECV vs brute-force (Robust scaled Ridge)",
        method="comparison_rfecv_bruteforce_ridge",
        description="Compare RFECV and exhaustive feature sets with RobustScaler + Ridge regression.",
        top_k=None,
        tickers=("TSLA",),
    ),
}


def get_case_config(case_id: str) -> FeatureSelectionCase:
    """Return the configuration for a feature selection experiment."""

    try:
        return FEATURE_SELECTION_CASES[case_id]
    except KeyError as exc:
        raise KeyError(f"Unknown feature selection case id: {case_id}") from exc


AVAILABLE_CASE_IDS: tuple[str, ...] = tuple(FEATURE_SELECTION_CASES.keys())

__all__ = [
    "AVAILABLE_CASE_IDS",
    "DEFAULT_DATASET_FILENAME",
    "DEFAULT_FEATURES",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TARGET",
    "DEFAULT_TARGET_SOURCE",
    "DEFAULT_TICKERS",
    "FEATURE_SELECTION_CASES",
    "FeatureSelectionCase",
    "get_case_config",
]
