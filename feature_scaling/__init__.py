"""Feature scaling case studies registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from .case_1_california_housing_demo import CASE_ID as CASE_1_ID
from .case_1_california_housing_demo import CASE_NAME as CASE_1_NAME
from .case_1_california_housing_demo import run_case as run_case_1
from .case_2_ml_dataset_tsla import CASE_ID as CASE_2_ID
from .case_2_ml_dataset_tsla import CASE_NAME as CASE_2_NAME
from .case_2_ml_dataset_tsla import run_case as run_case_2
from .case_3_hybrid_tsla import CASE_ID as CASE_3_ID
from .case_3_hybrid_tsla import CASE_NAME as CASE_3_NAME
from .case_3_hybrid_tsla import run_case as run_case_3
from .case_4_technical_tsla import CASE_ID as CASE_4_ID
from .case_4_technical_tsla import CASE_NAME as CASE_4_NAME
from .case_4_technical_tsla import run_case as run_case_4
from .settings import FEATURE_SCALING_CASES


@dataclass(frozen=True)
class CaseStudy:
    """Descriptor for a feature scaling case study."""

    case_id: str
    title: str
    runner: Callable[[], Path]


CASE_STUDIES: Tuple[CaseStudy, ...] = (
    CaseStudy(CASE_1_ID, CASE_1_NAME, run_case_1),
    CaseStudy(CASE_2_ID, CASE_2_NAME, run_case_2),
    CaseStudy(CASE_3_ID, CASE_3_NAME, run_case_3),
    CaseStudy(CASE_4_ID, CASE_4_NAME, run_case_4),
)



def get_case_studies() -> Tuple[CaseStudy, ...]:
    """Return the registered case studies."""

    return CASE_STUDIES



def run_all_cases() -> None:
    """Execute all registered case studies sequentially."""

    for case in CASE_STUDIES:
        print(f"\n>>> Running {case.case_id}: {case.title}")
        case.runner()


__all__ = ["CaseStudy", "CASE_STUDIES", "FEATURE_SCALING_CASES", "get_case_studies", "run_all_cases"]

