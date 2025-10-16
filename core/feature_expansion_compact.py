"""Compact feature expansion helpers to control feature-space growth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from core.feature_expansion import expand_base_features


_TARGET_COLUMNS = {
    "target",
    "target_pct",
    "target_pct_1d",
    "target_1d",
    "target_1d_class",
}


@dataclass(frozen=True)
class CompactExpansionResult:
    """Container describing the outcome of compact feature expansion."""

    frame: pd.DataFrame
    feature_columns: list[str]
    dropped_pre_screen: list[str]
    dropped_variance: list[str]
    dropped_mutual_info: list[str]
    config: Mapping[str, object]


def _collect_columns_by_pattern(columns: Iterable[str], patterns: Sequence[str]) -> list[str]:
    if not patterns:
        return []
    matched: list[str] = []
    for column in columns:
        for pattern in patterns:
            if pattern in column:
                matched.append(column)
                break
    return matched


def compact_expand_base_features(
    df: pd.DataFrame,
    *,
    base_features: Sequence[str],
    ticker_column: str = "ticker",
    date_column: str = "Date",
    close_column: str = "Close",
    skew_bins: Sequence[float] = (-np.inf, -0.3, 0.3, np.inf),
    skew_labels: Sequence[str] = ("left", "neutral", "right"),
    risk_quantiles: int = 2,
    risk_labels: Sequence[str] = ("low", "high"),
    drop_skew_sign_interactions: bool = True,
    drop_delta: bool = True,
    drop_ema: bool = True,
    keep_risk_level: bool = False,
    keep_risk_level_code: bool = True,
    additional_drop_patterns: Sequence[str] | None = None,
    variance_threshold: float | None = 1e-5,
    top_n_mutual_info: int | None = None,
    mutual_info_target: pd.Series | None = None,
    mutual_info_random_state: int = 0,
    exclude_from_features: Sequence[str] | None = None,
    remove_forward_targets: bool = True,
) -> CompactExpansionResult:
    """Produce a compact feature expansion using configurable pruning heuristics.

    Parameters
    ----------
    df:
        Input dataframe containing at least the ticker/date/close columns.
    base_features:
        List of base feature column names to expand.
    drop_* flags:
        Control which derived columns should be discarded after expansion.
    additional_drop_patterns:
        Optional substrings; any column containing one of them will be removed.
    variance_threshold:
        If set, apply VarianceThreshold with this value (treated in the same way
        as scikit-learn). Provide ``None`` to disable.
    top_n_mutual_info:
        If provided together with ``mutual_info_target``, keep only the top-N
        features ranked by mutual information score.
    mutual_info_target:
        Target series aligned with ``df`` used during mutual information ranking.
    exclude_from_features:
        Columns that should never be treated as model features (e.g., identifiers).

    Returns
    -------
    CompactExpansionResult
        Dataframe with retained features plus metadata about dropped columns.
    """

    expanded = expand_base_features(
        df,
        base_features=base_features,
        ticker_column=ticker_column,
        date_column=date_column,
        close_column=close_column,
        skew_bins=skew_bins,
        skew_labels=skew_labels,
        risk_quantiles=risk_quantiles,
        risk_labels=risk_labels,
    )

    drop_columns: set[str] = set()
    drop_patterns: list[str] = []

    if remove_forward_targets:
        drop_columns.update(col for col in _TARGET_COLUMNS if col in expanded.columns and col != "target_1d_class")

    if drop_skew_sign_interactions:
        drop_patterns.extend(["_x_skew_sign_pos", "_x_skew_sign_neg"])
    if drop_delta:
        drop_patterns.append("_skew_delta_")
    if drop_ema:
        drop_patterns.append("_skew_ema_")
    if not keep_risk_level:
        drop_patterns.append("_risk_level")
    if not keep_risk_level_code:
        drop_patterns.append("_risk_level_code")
    if additional_drop_patterns:
        drop_patterns.extend(additional_drop_patterns)

    pattern_matches = _collect_columns_by_pattern(expanded.columns, drop_patterns)
    drop_columns.update(pattern_matches)

    pre_screen_dropped = sorted(drop_columns)
    processed = expanded.drop(columns=pre_screen_dropped, errors="ignore")

    feature_exclude = set(exclude_from_features or ())
    feature_exclude.update(
        {
            ticker_column,
            date_column,
            close_column,
        }
    )
    feature_exclude.update(_TARGET_COLUMNS)

    candidate_features = [
        column
        for column in processed.columns
        if column not in feature_exclude and ptypes.is_numeric_dtype(processed[column])
    ]
    dropped_by_variance: list[str] = []
    dropped_by_mi: list[str] = []
    retained_features = candidate_features

    if variance_threshold is not None and candidate_features:
        vt = VarianceThreshold(threshold=variance_threshold)
        matrix = processed[candidate_features].fillna(0.0)
        vt.fit(matrix)
        support = vt.get_support()
        dropped_by_variance = [col for col, keep in zip(candidate_features, support) if not keep]
        retained_features = [col for col in candidate_features if col not in dropped_by_variance]
        processed = processed.drop(columns=dropped_by_variance, errors="ignore")

    if top_n_mutual_info is not None and top_n_mutual_info > 0 and retained_features:
        if mutual_info_target is None:
            raise ValueError("mutual_info_target must be provided when top_n_mutual_info is set.")
        top_n_mutual_info = min(top_n_mutual_info, len(retained_features))

        aligned_target = mutual_info_target.reindex(processed.index)
        target_mask = aligned_target.notna()
        if not target_mask.any():
            raise ValueError("mutual_info_target does not contain valid targets for mutual information ranking.")
        aligned_features = processed.loc[target_mask, retained_features].fillna(0.0)
        aligned_target = aligned_target.loc[target_mask].to_numpy()

        mi_scores = mutual_info_classif(
            aligned_features.to_numpy(dtype=float),
            aligned_target.astype(int),
            random_state=mutual_info_random_state,
        )
        order = np.argsort(mi_scores)[::-1]
        selected_indices = order[:top_n_mutual_info]
        selected_features = [retained_features[idx] for idx in selected_indices]
        dropped_by_mi = [feature for feature in retained_features if feature not in selected_features]
        retained_features = selected_features
        processed = processed.drop(columns=dropped_by_mi, errors="ignore")

    config: dict[str, object] = {
        "skew_bins": tuple(skew_bins),
        "skew_labels": tuple(skew_labels),
        "risk_quantiles": int(risk_quantiles),
        "risk_labels": tuple(risk_labels),
        "drop_skew_sign_interactions": drop_skew_sign_interactions,
        "drop_delta": drop_delta,
        "drop_ema": drop_ema,
        "keep_risk_level": keep_risk_level,
        "keep_risk_level_code": keep_risk_level_code,
        "variance_threshold": variance_threshold,
        "top_n_mutual_info": top_n_mutual_info,
        "mutual_info_random_state": mutual_info_random_state,
    }

    return CompactExpansionResult(
        frame=processed,
        feature_columns=list(retained_features),
        dropped_pre_screen=pre_screen_dropped,
        dropped_variance=dropped_by_variance,
        dropped_mutual_info=dropped_by_mi,
        config=config,
    )


__all__ = [
    "CompactExpansionResult",
    "compact_expand_base_features",
]
