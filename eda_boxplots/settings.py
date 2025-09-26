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
DEFAULT_TARGET_FEATURES: Sequence[str] = (
    "rolling_median_target_14d",
    "rolling_iqr_target_14d",
    "rolling_outlier_ratio_target_14d",
)
EXTENDED_TECH_FEATURES: Sequence[str] = (
    "OpenCloseReturn",
    "IntradayRange",
    "Volume_vs_MA14",
    "sma_14",
    "ema_14",
    "wma_14",
    "stddev_14",
    "volatility_14",
    "rsi_14",
    "avg_volume_14",
    "avg_volume_ccy_14",
    "slope_14",
    "stoch_k_14",
    "stoch_d_14",
    "cci_14",
    "atr_14",
    "adx_14",
    "dmi_plus_14",
    "dmi_minus_14",
    "sma_50",
    "ema_50",
    "wma_50",
    "stddev_50",
    "volatility_50",
    "rsi_50",
    "avg_volume_50",
    "avg_volume_ccy_50",
    "slope_50",
    "stoch_k_50",
    "stoch_d_50",
    "cci_50",
    "atr_50",
    "adx_50",
    "dmi_plus_50",
    "dmi_minus_50",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_high_14",
    "bb_low_14",
    "bb_mavg_14",
    "bb_width_14",
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
    "case_3": CaseConfig(
        case_id="case_3",
        name="Case 3: target horizon feature boxplots (14-day)",
        description="Engineer rolling statistics on the 14-day forward return and assess predictive value.",
    ),
    "case_4": CaseConfig(
        case_id="case_4",
        name="Case 4: hybrid technical + target feature boxplots",
        description="Combine technical inputs with 14-day forward return statistics for validation.",
    ),
    "case_5": CaseConfig(
        case_id="case_5",
        name="Case 5: technical-only boxplots with 14-day target",
        description="Use core technical features while validating against the 14-day forward return.",
    ),
    "case_6": CaseConfig(
        case_id="case_6",
        name="Case 6: hybrid technical + 1-day target boxplots",
        description="Combine technical inputs with 1-day forward return statistics for validation.",
    ),
    "case_7": CaseConfig(
        case_id="case_7",
        name="Case 7: extended technical + 1-day target boxplots",
        description="Use an extended technical feature set together with 1-day forward return statistics.",
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
    "DEFAULT_TARGET_FEATURES",
    "EXTENDED_TECH_FEATURES",
    "CaseConfig",
    "AVAILABLE_CASE_IDS",
    "get_case_config",
]
