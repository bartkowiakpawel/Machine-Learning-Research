"""Local helpers for producing standard validation reports."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

from core.reporting import (
    build_last_n_day_prediction_summary,
    resolve_reports_dir,
)
from core.shared_utils import save_dataframe

_DEFAULT_PACKAGE_NAME = "eda_boxplots"
_DEFAULT_REPORT_FAMILY = "standard"


def _safe_sheet_name(label: str) -> str:
    cleaned = label.replace(" ", "_")
    return cleaned[:31] or "summary"


def export_last_n_day_prediction_reports(
    predictions: pd.DataFrame,
    *,
    case_id: str,
    case_output_dir: Path,
    ticker_bundle: str,
    last_n_days: int,
    yeojohnson: bool = False,
    package_name: str = _DEFAULT_PACKAGE_NAME,
    report_family: str = _DEFAULT_REPORT_FAMILY,
    decimals: int = 4,
) -> Mapping[str, Path]:
    """Persist standardized reports for last-N-day validation outputs.

    Parameters
    ----------
    predictions:
        Combined prediction frame emitted by the validation helper.
    case_id:
        Identifier of the running case (e.g. ``case_7``).
    case_output_dir:
        Root output directory for the case. Per-case reports are written under
        ``reports/<report_family>``.
    ticker_bundle:
        Identifier derived from the participating tickers, used in filenames.
    last_n_days:
        Number of look-back days used during validation, included in filenames.
    yeojohnson:
        Flag appended to filenames when the Yeo-Johnson transform is active.
    package_name:
        Logical package namespace for cross-package report collation.
    report_family:
        High-level report family (default: ``standard``).
    decimals:
        Rounding precision applied to percentage columns.
    """

    if predictions.empty:
        return {}

    summary_tables = build_last_n_day_prediction_summary(
        predictions,
        ticker_column="ticker",
        model_column="model",
        diff_column="prediction_diff",
        decimals=decimals,
    )

    suffix = "_yeojohnson" if yeojohnson else ""
    base_filename = (
        f"{ticker_bundle}_{case_id}_last_{last_n_days}_day_predictions{suffix}"
    )

    case_report_dir = Path(case_output_dir) / "reports" / report_family
    case_report_dir.mkdir(parents=True, exist_ok=True)

    standard_dir = resolve_reports_dir(
        package=package_name,
        report_family=report_family,
    ) / case_id
    standard_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}

    for key, df in summary_tables.items():
        if df.empty:
            continue
        filename = f"{base_filename}_{key}.csv"
        case_path = save_dataframe(
            df,
            case_report_dir,
            filename,
            csv_kwargs={"index": False},
        )
        standard_path = save_dataframe(
            df,
            standard_dir,
            filename,
            csv_kwargs={"index": False},
        )
        saved_paths[f"case_{key}"] = case_path
        saved_paths[f"standard_{key}"] = standard_path

    if saved_paths:
        excel_name = f"{base_filename}_summary.xlsx"
        case_excel = case_report_dir / excel_name
        with pd.ExcelWriter(case_excel) as writer:
            for key, df in summary_tables.items():
                if df.empty:
                    continue
                df.to_excel(writer, sheet_name=_safe_sheet_name(key), index=False)
        saved_paths["case_summary_excel"] = case_excel

        standard_excel = standard_dir / excel_name
        with pd.ExcelWriter(standard_excel) as writer:
            for key, df in summary_tables.items():
                if df.empty:
                    continue
                df.to_excel(writer, sheet_name=_safe_sheet_name(key), index=False)
        saved_paths["standard_summary_excel"] = standard_excel

    return saved_paths


__all__ = ["export_last_n_day_prediction_reports"]
