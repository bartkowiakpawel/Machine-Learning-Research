"""Compatibility layer re-exporting shared utilities for feature-scaling workflows."""

from core.shared_utils import (
    clean_feature_matrix,
    filter_ticker,
    load_ml_dataset,
    prepare_feature_matrix,
    resolve_output_dir,
    run_learning_workflow,
    save_dataframe,
)

__all__ = [
    "clean_feature_matrix",
    "filter_ticker",
    "load_ml_dataset",
    "prepare_feature_matrix",
    "resolve_output_dir",
    "run_learning_workflow",
    "save_dataframe",
]
