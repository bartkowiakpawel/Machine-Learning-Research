"""EDA & Boxplots case registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from .case_1_multi_ticker_boxplots import run_case as run_case_1
from .case_2_yeojohnson_boxplots import run_case as run_case_2
from .case_3_target_features_boxplots import run_case as run_case_3
from .case_4_hybrid_boxplots import run_case as run_case_4
from .case_5_technical_boxplots import run_case as run_case_5
from .case_6_hybrid_1d_boxplots import run_case as run_case_6
from .case_7_extended_hybrid_boxplots import run_case as run_case_7
from .case_8_extended_ohlcv_boxplots import run_case as run_case_8
from .case_9_core_feature_rolling_boxplots import run_case as run_case_9
from .case_10_predictions_comparison import run_case as run_case_10
from .case_11_hybrid_1d_boxplots_60d import run_case as run_case_11
from .settings import (
    AVAILABLE_CASE_IDS,
    DEFAULT_DATASET_FILENAME,
    DEFAULT_FEATURES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TARGET_FEATURES,
    DEFAULT_TICKER,
    DEFAULT_TICKERS,
    EXTENDED_TECH_FEATURES,
    CaseConfig,
    get_case_config,
)


@dataclass(frozen=True)
class CaseStudy:
    """Descriptor for an EDA boxplot case study."""

    case_id: str
    title: str
    runner: Callable[..., Path]


CASE_STUDIES: Tuple[CaseStudy, ...] = (
    CaseStudy(case_id="case_1", title=get_case_config("case_1").name, runner=run_case_1),
    CaseStudy(case_id="case_2", title=get_case_config("case_2").name, runner=run_case_2),
    CaseStudy(case_id="case_3", title=get_case_config("case_3").name, runner=run_case_3),
    CaseStudy(case_id="case_4", title=get_case_config("case_4").name, runner=run_case_4),
    CaseStudy(case_id="case_5", title=get_case_config("case_5").name, runner=run_case_5),
    CaseStudy(case_id="case_6", title=get_case_config("case_6").name, runner=run_case_6),
    CaseStudy(case_id="case_7", title=get_case_config("case_7").name, runner=run_case_7),
    CaseStudy(case_id="case_8", title=get_case_config("case_8").name, runner=run_case_8),
    CaseStudy(case_id="case_9", title=get_case_config("case_9").name, runner=run_case_9),
    CaseStudy(case_id="case_10", title=get_case_config("case_10").name, runner=run_case_10),
    CaseStudy(case_id="case_11", title=get_case_config("case_11").name, runner=run_case_11),
)

_CASE_RUNNERS = {case.case_id: case.runner for case in CASE_STUDIES}


def get_case_studies() -> Tuple[CaseStudy, ...]:
    """Return the registered EDA boxplot case studies."""

    return CASE_STUDIES


def run_case_by_id(case_id: str, **kwargs):
    """Execute a case study by its identifier."""

    try:
        runner = _CASE_RUNNERS[case_id]
    except KeyError as exc:
        raise KeyError(f"Unknown EDA boxplots case id: {case_id}") from exc
    return runner(**kwargs)


__all__ = [
    "CaseStudy",
    "CASE_STUDIES",
    "AVAILABLE_CASE_IDS",
    "DEFAULT_TICKER",
    "DEFAULT_TICKERS",
    "DEFAULT_DATASET_FILENAME",
    "DEFAULT_FEATURES",
    "DEFAULT_TARGET_FEATURES",
    "EXTENDED_TECH_FEATURES",
    "DEFAULT_OUTPUT_ROOT",
    "CaseConfig",
    "get_case_config",
    "get_case_studies",
    "run_case_by_id",
]
