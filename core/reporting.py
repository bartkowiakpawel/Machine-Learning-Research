"""Shared reporting utilities for cross-package analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REPORT_FAMILY = "standard"
_SUMMARY_COLUMNS = ("max_positive_pct", "min_negative_pct", "mean_diff_pct", "samples")
_FLOAT_COLUMNS = ("max_positive_pct", "min_negative_pct", "mean_diff_pct")


def resolve_reports_dir(
    *,
    package: str,
    report_family: str = _DEFAULT_REPORT_FAMILY,
    base_dir: Path | None = None,
    create: bool = True,
) -> Path:
    """Return the directory intended for standardized reports.

    Parameters
    ----------
    package:
        Logical package or component name. A subdirectory with this name is
        created inside the report family folder to keep outputs namespaced.
    report_family:
        High level category for reports (defaults to "standard").
    base_dir:
        Optional root directory to anchor the reports folder. When omitted the
        repository root is used.
    create:
        When True (default) ensure the directory exists.
    """

    root = Path(base_dir) if base_dir is not None else _REPO_ROOT
    target = root / "reports" / report_family / package
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def _empty_summary(columns: Sequence[str]) -> pd.DataFrame:
    data_columns = list(columns) + list(_SUMMARY_COLUMNS)
    frame = pd.DataFrame(columns=data_columns)
    for column in _FLOAT_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype(float)
    if "samples" in frame.columns:
        frame["samples"] = frame["samples"].astype(int)
    return frame


def _round_float_columns(df: pd.DataFrame, columns: Sequence[str], decimals: int) -> pd.DataFrame:
    if df.empty:
        return df
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").round(decimals)
    return df


def _normalise_samples(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "samples" not in df.columns:
        return df
    df["samples"] = pd.to_numeric(df["samples"], errors="coerce").fillna(0).astype(int)
    return df


def _build_pivot_layout(
    by_ticker: pd.DataFrame,
    by_ticker_model: pd.DataFrame,
    overall: pd.DataFrame,
) -> pd.DataFrame:
    if by_ticker.empty and by_ticker_model.empty and overall.empty:
        return _empty_summary(["ticker", "model", "scope"])

    rows: list[dict[str, object]] = []

    ticker_groups = {
        row["ticker"]: row
        for _, row in by_ticker.iterrows()
        if "ticker" in row
    }

    for ticker_value, ticker_row in ticker_groups.items():
        rows.append(
            {
                "ticker": ticker_value,
                "model": "",
                "scope": "ticker",
                "max_positive_pct": ticker_row.get("max_positive_pct"),
                "min_negative_pct": ticker_row.get("min_negative_pct"),
                "mean_diff_pct": ticker_row.get("mean_diff_pct"),
                "samples": ticker_row.get("samples", 0),
            }
        )
        ticker_subset = by_ticker_model[by_ticker_model["ticker"] == ticker_value]
        for _, model_row in ticker_subset.iterrows():
            rows.append(
                {
                    "ticker": ticker_value,
                    "model": model_row.get("model"),
                    "scope": "model",
                    "max_positive_pct": model_row.get("max_positive_pct"),
                    "min_negative_pct": model_row.get("min_negative_pct"),
                    "mean_diff_pct": model_row.get("mean_diff_pct"),
                    "samples": model_row.get("samples", 0),
                }
            )

    if not overall.empty:
        first_row = overall.iloc[0].to_dict()
        rows.append(
            {
                "ticker": "TOTAL",
                "model": "",
                "scope": "overall",
                "max_positive_pct": first_row.get("max_positive_pct"),
                "min_negative_pct": first_row.get("min_negative_pct"),
                "mean_diff_pct": first_row.get("mean_diff_pct"),
                "samples": first_row.get("samples", 0),
            }
        )

    pivot = pd.DataFrame(rows)
    if pivot.empty:
        return _empty_summary(["ticker", "model", "scope"])
    return pivot


def build_last_n_day_prediction_summary(
    predictions: pd.DataFrame,
    *,
    ticker_column: str = "ticker",
    model_column: str = "model",
    diff_column: str = "prediction_diff",
    decimals: int = 4,
) -> Mapping[str, pd.DataFrame]:
    """Prepare standard summary tables for last-N-day validation results.

    The returned mapping contains:
      - ``by_ticker``: metrics aggregated per ticker
      - ``by_ticker_model``: metrics per ticker + model combination
      - ``by_model``: metrics aggregated per model across tickers
      - ``overall``: single-row frame with overall metrics
      - ``pivot_layout``: flattened layout mimicking the Excel pivot shared by users

    Percentages are expressed in percentage points (value * 100) for readability.
    """

    required_columns = {ticker_column, model_column, diff_column}
    missing = required_columns.difference(predictions.columns)
    if missing:
        missing_fmt = ", ".join(sorted(missing))
        raise KeyError(f"Prediction frame missing required columns: {missing_fmt}")

    if predictions.empty:
        empty_by_ticker = _empty_summary([ticker_column])
        empty_by_ticker_model = _empty_summary([ticker_column, model_column])
        empty_by_model = _empty_summary([model_column])
        empty_overall = _empty_summary(["scope"])
        empty_pivot = _empty_summary(["ticker", "model", "scope"])
        return {
            "by_ticker": empty_by_ticker,
            "by_ticker_model": empty_by_ticker_model,
            "by_model": empty_by_model,
            "overall": empty_overall,
            "pivot_layout": empty_pivot,
        }

    working = predictions.copy()
    working[diff_column] = pd.to_numeric(working[diff_column], errors="coerce")
    working = working.dropna(subset=[diff_column])
    if working.empty:
        return build_last_n_day_prediction_summary(
            pd.DataFrame(columns=predictions.columns),
            ticker_column=ticker_column,
            model_column=model_column,
            diff_column=diff_column,
            decimals=decimals,
        )

    working["prediction_diff_pct"] = working[diff_column] * 100.0
    working["positive_diff_pct"] = working["prediction_diff_pct"].where(
        working["prediction_diff_pct"] > 0
    )
    working["negative_diff_pct"] = working["prediction_diff_pct"].where(
        working["prediction_diff_pct"] < 0
    )

    def _aggregate(group_columns: Sequence[str]) -> pd.DataFrame:
        aggregated = (
            working.groupby(list(group_columns), dropna=False)
            .agg(
                max_positive_pct=("positive_diff_pct", "max"),
                min_negative_pct=("negative_diff_pct", "min"),
                mean_diff_pct=("prediction_diff_pct", "mean"),
                samples=(diff_column, "count"),
            )
            .reset_index()
        )
        aggregated = _round_float_columns(aggregated, _FLOAT_COLUMNS, decimals)
        aggregated = _normalise_samples(aggregated)
        return aggregated

    summary_by_ticker_model = _aggregate([ticker_column, model_column])
    summary_by_ticker = _aggregate([ticker_column])
    summary_by_model = _aggregate([model_column])

    overall = pd.DataFrame(
        {
            "scope": ["overall"],
            "max_positive_pct": [working["positive_diff_pct"].max()],
            "min_negative_pct": [working["negative_diff_pct"].min()],
            "mean_diff_pct": [working["prediction_diff_pct"].mean()],
            "samples": [len(working)],
        }
    )
    overall = _round_float_columns(overall, _FLOAT_COLUMNS, decimals)
    overall = _normalise_samples(overall)

    pivot_layout = _build_pivot_layout(summary_by_ticker, summary_by_ticker_model, overall)
    pivot_layout = _round_float_columns(pivot_layout, _FLOAT_COLUMNS, decimals)
    pivot_layout = _normalise_samples(pivot_layout)

    return {
        "by_ticker": summary_by_ticker,
        "by_ticker_model": summary_by_ticker_model,
        "by_model": summary_by_model,
        "overall": overall,
        "pivot_layout": pivot_layout,
    }


__all__ = [
    "resolve_reports_dir",
    "build_last_n_day_prediction_summary",
]
