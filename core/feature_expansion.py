"""Utilities for enriching technical features with target-aligned statistics."""

from __future__ import annotations

from collections.abc import Sequence
import re

import numpy as np
import pandas as pd
from scipy.stats import skew


TARGET_FEATURE_COLUMNS = (
    "rolling_median_target_1d",
    "rolling_iqr_target_1d",
    "rolling_outlier_ratio_target_1d",
)

DEFAULT_SKEW_BINS = (-np.inf, -0.75, -0.25, 0.25, 0.75, np.inf)
DEFAULT_SKEW_LABELS = ("very_left", "left", "neutral", "right", "very_right")
DEFAULT_RISK_LABELS = ("low_vol", "medium_vol", "high_vol")


def _sanitize_feature_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "feature"


def _outlier_ratio(window_values: pd.Series) -> float:
    if window_values.empty:
        return float("nan")
    q1 = window_values.quantile(0.25)
    q3 = window_values.quantile(0.75)
    iqr = q3 - q1
    if np.isclose(iqr, 0.0):
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (window_values < lower) | (window_values > upper)
    return float(mask.mean())


def _rolling_skew(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        raise ValueError("rolling skew window must be greater than 1")
    return series.rolling(window=window, min_periods=window).apply(
        lambda values: skew(values, bias=False), raw=False
    )


def _qcut_with_fallback(
    series: pd.Series,
    *,
    quantiles: int,
    labels: Sequence[str],
) -> pd.Series:
    if quantiles < 1:
        raise ValueError("quantiles must be a positive integer")
    if len(labels) != quantiles:
        raise ValueError("labels length must equal quantiles")
    if series.empty:
        return pd.Series(pd.Categorical([], categories=labels), index=series.index)
    try:
        categories = pd.qcut(series, q=quantiles, labels=labels, duplicates="drop")
    except (ValueError, IndexError):
        fallback = pd.Categorical([np.nan] * len(series), categories=labels)
        return pd.Series(fallback, index=series.index)
    categories = categories.cat.add_categories([label for label in labels if label not in categories.cat.categories])
    categories = categories.cat.reorder_categories(labels)
    return categories


def compute_target_statistics(
    df: pd.DataFrame,
    *,
    ticker_column: str = "ticker",
    date_column: str = "Date",
    close_column: str = "Close",
    target_shift: int = 1,
    rolling_window: int = 30,
) -> pd.DataFrame:
    """Append 1-day forward target and its rolling statistics per ticker."""

    required_columns = {ticker_column, date_column, close_column}
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        missing_fmt = ", ".join(missing)
        raise KeyError(f"Input frame is missing required columns: {missing_fmt}")

    working = df.sort_values([ticker_column, date_column], kind="mergesort").reset_index(drop=True).copy()

    grouped_close = working.groupby(ticker_column, sort=False)[close_column]
    future_close = grouped_close.shift(-target_shift)
    working["target_1d"] = (future_close - working[close_column]) / working[close_column]

    target_group = working.groupby(ticker_column, sort=False)["target_1d"]
    rolling_target = target_group.rolling(window=rolling_window, min_periods=rolling_window)
    median_series = rolling_target.median().reset_index(level=0, drop=True)
    q75 = rolling_target.quantile(0.75).reset_index(level=0, drop=True)
    q25 = rolling_target.quantile(0.25).reset_index(level=0, drop=True)
    outlier_ratio_series = rolling_target.apply(_outlier_ratio, raw=False).reset_index(level=0, drop=True)

    working["rolling_median_target_1d"] = median_series
    working["rolling_iqr_target_1d"] = q75 - q25
    working["rolling_outlier_ratio_target_1d"] = outlier_ratio_series

    return working


def generate_skew_derivatives(
    df: pd.DataFrame,
    *,
    base_features: Sequence[str],
    ticker_column: str = "ticker",
    date_column: str = "Date",
    skew_window: int = 60,
    skew_threshold: float = 0.25,
    skew_bins: Sequence[float] = DEFAULT_SKEW_BINS,
    skew_labels: Sequence[str] = DEFAULT_SKEW_LABELS,
    delta_lag: int = 7,
    ema_span: int = 20,
    volatility_window: int = 14,
    risk_quantiles: int = 3,
    risk_labels: Sequence[str] = DEFAULT_RISK_LABELS,
) -> pd.DataFrame:
    """Create skew, volatility and interaction derivatives for the supplied base features."""

    if not base_features:
        raise ValueError("base_features must contain at least one feature name")

    missing = [col for col in base_features if col not in df.columns]
    if missing:
        missing_fmt = ", ".join(missing)
        raise KeyError(f"Base feature columns not found in frame: {missing_fmt}")

    if len(skew_bins) - 1 != len(skew_labels):
        raise ValueError("skew_labels length must be exactly one less than the number of skew_bins")
    if volatility_window <= 1:
        raise ValueError("volatility_window must be greater than 1")

    working = df.sort_values([ticker_column, date_column], kind="mergesort").reset_index(drop=True).copy()

    grouped = working.groupby(ticker_column, sort=False)
    derived_columns: dict[str, pd.Series] = {}

    for feature in base_features:
        feature_id = _sanitize_feature_label(feature)
        feature_series = working[feature]

        skew_series = grouped[feature].apply(lambda values: _rolling_skew(values, skew_window))
        skew_series = skew_series.reset_index(level=0, drop=True)

        skew_col = f"{feature_id}_skew_{skew_window}"
        derived_columns[skew_col] = skew_series

        sign_series = skew_series.gt(skew_threshold).astype(int) - skew_series.lt(-skew_threshold).astype(int)
        sign_series = sign_series.where(~skew_series.isna())
        sign_col = f"{feature_id}_skew_sign"
        derived_columns[sign_col] = sign_series

        bucket_series = pd.cut(skew_series, bins=skew_bins, labels=skew_labels)
        dummy_frame = pd.get_dummies(bucket_series, prefix=f"{feature_id}_skew_bucket")
        for column in dummy_frame.columns:
            derived_columns[column] = dummy_frame[column]

        delta_series = skew_series.groupby(working[ticker_column]).diff(delta_lag)
        delta_col = f"{feature_id}_skew_delta_{delta_lag}"
        derived_columns[delta_col] = delta_series

        ema_series = skew_series.groupby(working[ticker_column]).transform(
            lambda values: values.ewm(span=ema_span, adjust=False).mean()
        )
        ema_col = f"{feature_id}_skew_ema_{ema_span}"
        derived_columns[ema_col] = ema_series

        pos_mask = sign_series.eq(1).astype(int)
        neg_mask = sign_series.eq(-1).astype(int)
        derived_columns[f"{feature_id}_x_skew_sign_pos"] = feature_series * pos_mask
        derived_columns[f"{feature_id}_x_skew_sign_neg"] = feature_series * neg_mask

        rolling_mean = grouped[feature].transform(
            lambda values: values.rolling(window=volatility_window, min_periods=volatility_window).mean()
        )
        rolling_std = grouped[feature].transform(
            lambda values: values.rolling(window=volatility_window, min_periods=volatility_window).std()
        )

        volatility_col = f"{feature_id}_volatility_{volatility_window}"
        derived_columns[volatility_col] = rolling_std

        denom = rolling_std.replace(0.0, np.nan)
        zscore_col = f"{feature_id}_zscore_{volatility_window}"
        derived_columns[zscore_col] = (feature_series - rolling_mean) / denom

        risk_series = grouped[feature].apply(
            lambda values: _qcut_with_fallback(values, quantiles=risk_quantiles, labels=risk_labels)
        ).reset_index(level=0, drop=True)
        risk_series.name = f"{feature_id}_risk_level"
        risk_col = f"{feature_id}_risk_level"
        derived_columns[risk_col] = risk_series
        risk_code_col = f"{feature_id}_risk_level_code"
        risk_codes = risk_series.cat.codes.astype("float").replace(-1, np.nan)
        derived_columns[risk_code_col] = risk_codes

    derived_frame = pd.DataFrame(derived_columns, index=working.index)
    enriched = pd.concat([working, derived_frame], axis=1)
    return enriched


def expand_base_features(
    df: pd.DataFrame,
    *,
    base_features: Sequence[str],
    ticker_column: str = "ticker",
    date_column: str = "Date",
    close_column: str = "Close",
    target_shift: int = 1,
    target_window: int = 30,
    skew_window: int = 60,
    skew_threshold: float = 0.25,
    skew_bins: Sequence[float] = DEFAULT_SKEW_BINS,
    skew_labels: Sequence[str] = DEFAULT_SKEW_LABELS,
    delta_lag: int = 7,
    ema_span: int = 20,
    volatility_window: int = 14,
    risk_quantiles: int = 3,
    risk_labels: Sequence[str] = DEFAULT_RISK_LABELS,
) -> pd.DataFrame:
    """Convenience wrapper to compute target statistics and feature-derived enrichments."""

    target_augmented = compute_target_statistics(
        df,
        ticker_column=ticker_column,
        date_column=date_column,
        close_column=close_column,
        target_shift=target_shift,
        rolling_window=target_window,
    )
    enriched = generate_skew_derivatives(
        target_augmented,
        base_features=base_features,
        ticker_column=ticker_column,
        date_column=date_column,
        skew_window=skew_window,
        skew_threshold=skew_threshold,
        skew_bins=skew_bins,
        skew_labels=skew_labels,
        delta_lag=delta_lag,
        ema_span=ema_span,
        volatility_window=volatility_window,
        risk_quantiles=risk_quantiles,
        risk_labels=risk_labels,
    )
    return enriched
