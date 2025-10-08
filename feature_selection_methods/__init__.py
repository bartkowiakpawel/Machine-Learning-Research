"""Feature selection methods case registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from .case_1_filter_methods import CASE_ID as CASE_1_ID
from .case_1_filter_methods import CASE_NAME as CASE_1_NAME
from .case_1_filter_methods import run_case as run_case_1
from .case_2_wrapper_rfecv import CASE_ID as CASE_2_ID
from .case_2_wrapper_rfecv import CASE_NAME as CASE_2_NAME
from .case_2_wrapper_rfecv import run_case as run_case_2
from .case_3_embedded_lasso import CASE_ID as CASE_3_ID
from .case_3_embedded_lasso import CASE_NAME as CASE_3_NAME
from .case_3_embedded_lasso import run_case as run_case_3
from .settings import (
    AVAILABLE_CASE_IDS,
    DEFAULT_DATASET_FILENAME,
    DEFAULT_FEATURES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TARGET,
    DEFAULT_TARGET_SOURCE,
    DEFAULT_TICKERS,
    FEATURE_SELECTION_CASES,
    FeatureSelectionCase,
    get_case_config,
)


@dataclass(frozen=True)
class CaseStudy:
    """Descriptor for a feature selection case study."""

    case_id: str
    title: str
    runner: Callable[..., Path]


CASE_STUDIES: Tuple[CaseStudy, ...] = (
    CaseStudy(case_id=CASE_1_ID, title=CASE_1_NAME, runner=run_case_1),
    CaseStudy(case_id=CASE_2_ID, title=CASE_2_NAME, runner=run_case_2),
    CaseStudy(case_id=CASE_3_ID, title=CASE_3_NAME, runner=run_case_3),
)

_CASE_RUNNERS = {case.case_id: case.runner for case in CASE_STUDIES}


def get_case_studies() -> Tuple[CaseStudy, ...]:
    """Return the registered feature selection case studies."""

    return CASE_STUDIES


def run_case_by_id(case_id: str, **kwargs):
    """Execute a feature selection case by its identifier."""

    try:
        runner = _CASE_RUNNERS[case_id]
    except KeyError as exc:
        raise KeyError(f"Unknown feature selection case id: {case_id}") from exc
    return runner(**kwargs)


def run_all_cases() -> None:
    """Execute all registered feature selection case studies sequentially."""

    for case in CASE_STUDIES:
        print(f"\n>>> Running {case.case_id}: {case.title}")
        case.runner()


__all__ = [
    "AVAILABLE_CASE_IDS",
    "CASE_STUDIES",
    "CaseStudy",
    "DEFAULT_DATASET_FILENAME",
    "DEFAULT_FEATURES",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TARGET",
    "DEFAULT_TARGET_SOURCE",
    "DEFAULT_TICKERS",
    "FEATURE_SELECTION_CASES",
    "FeatureSelectionCase",
    "get_case_config",
    "get_case_studies",
    "run_all_cases",
    "run_case_by_id",
]
