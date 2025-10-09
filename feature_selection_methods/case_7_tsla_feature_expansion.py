"""Case 7: TSLA feature expansion diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import pandas as pd
from pandas.api import types as ptypes

from config import ML_INPUT_DIR
from core.feature_expansion import (
    TARGET_FEATURE_COLUMNS,
    expand_base_features,
)
from core.distribution_plots import compute_distribution_summary, plot_distribution_grid
from core.shared_utils import (
    filter_ticker,
    load_ml_dataset,
    resolve_output_dir,
    save_dataframe,
    write_case_metadata,
)
from feature_selection_methods.settings import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_OUTPUT_ROOT,
    get_case_config,
)

CASE_ID = "case_7"
CASE_CONFIG = get_case_config(CASE_ID)
CASE_NAME = CASE_CONFIG.name
DEFAULT_TICKER = (CASE_CONFIG.tickers or ("TSLA",))[0]
TARGET_COLUMN = "target_1d"


def _normalize_ticker(ticker: str | None) -> str:
    if ticker is None:
        return DEFAULT_TICKER
    value = str(ticker).strip()
    return value.upper() or DEFAULT_TICKER


def _prepare_numeric_series(df: pd.DataFrame, feature: str) -> pd.Series:
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' is missing from the dataframe.")

    series_raw = df[feature]
    if ptypes.is_numeric_dtype(series_raw):
        series = pd.to_numeric(series_raw, errors="coerce")
    else:
        category = series_raw.astype("category")
        series = category.cat.codes.replace(-1, pd.NA).astype("float")
    series = series.dropna()
    if series.empty:
        raise ValueError(f"No numeric values available for feature '{feature}'.")
    return series


def _select_features(enriched_df: pd.DataFrame, base_df_columns: Iterable[str], base_features: Sequence[str]) -> list[str]:
    base_cols = set(base_df_columns)
    newly_created = [
        column for column in enriched_df.columns if column not in base_cols
    ]

    preferred_features: list[str] = []
    target_features = [col for col in TARGET_FEATURE_COLUMNS if col in enriched_df.columns]
    preferred_features.extend(target_features)
    for feature in base_features:
        if feature in enriched_df.columns:
            preferred_features.append(feature)

    for feature in base_features:
        prefix = feature.lower()
        derived = [
            column
            for column in newly_created
            if column.lower().startswith(prefix)
        ]
        preferred_features.extend(derived)

    seen: set[str] = set()
    ordered: list[str] = []
    for col in preferred_features:
        if col not in seen and col in enriched_df.columns:
            ordered.append(col)
            seen.add(col)
    return ordered


def _load_metadata_base_features(dataset_path: Path) -> list[str]:
    metadata_path = dataset_path.with_name(f"{dataset_path.stem}_metadata.json")
    if not metadata_path.exists():
        return []
    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        print(f"[WARN] Failed to parse metadata at {metadata_path}; falling back to inference.")
        return []
    columns = payload.get("technical_feature_columns")
    if not columns:
        return []
    return [str(col) for col in columns]


def _infer_base_features(df: pd.DataFrame) -> list[str]:
    exclude = {TARGET_COLUMN, "Date", "ticker", "target", "target_pct", "target_pct_1d"}
    exclude.update(TARGET_FEATURE_COLUMNS)
    inferred: list[str] = []
    for column in df.columns:
        if column in exclude:
            continue
        if column.startswith("target_"):
            continue
        series = df[column]
        if ptypes.is_numeric_dtype(series):
            inferred.append(column)
    return inferred


def run_case(
    *,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    ticker: str | None = None,
    base_features: Sequence[str] | None = None,
    output_root: Path | str | None = None,
    show_plots: bool = False,
) -> Path:
    """Expand TSLA base features and visualise their distributions."""

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    dataset_path = ML_INPUT_DIR / dataset_filename
    selected_ticker = _normalize_ticker(ticker)
    df = load_ml_dataset(dataset_path)
    subset = filter_ticker(df, selected_ticker)
    if subset.empty:
        raise ValueError(f"No rows available for ticker '{selected_ticker}'.")

    metadata_base = _load_metadata_base_features(dataset_path)
    if base_features:
        selected_base = tuple(base_features)
        base_source = "CLI override"
    elif metadata_base:
        selected_base = tuple(metadata_base)
        base_source = "dataset metadata"
    else:
        selected_base = tuple(_infer_base_features(subset))
        base_source = "inferred from dataset"

    if not selected_base:
        raise ValueError("No base features available for expansion.")

    print(f"=== Running {CASE_ID}: {CASE_NAME} ===")
    print(f"Dataset: {dataset_path}")
    print(f"Ticker: {selected_ticker}")
    print(f"Base features source: {base_source}")
    print(f"Base features ({len(selected_base)}): {', '.join(selected_base)}")

    available_bases = [feature for feature in selected_base if feature in subset.columns]
    missing_bases = [feature for feature in selected_base if feature not in subset.columns]
    if missing_bases:
        print(f"[WARN] Skipping missing base features: {', '.join(missing_bases)}")
    if not available_bases:
        raise ValueError("None of the requested base features are present in the dataset.")
    print(f"Expanding {len(available_bases)} base features.")

    baseline_columns = set(subset.columns)
    enriched = expand_base_features(
        subset,
        base_features=available_bases,
    )

    diagnostics_dir = case_output_dir / "diagnostics"
    plots_dir = diagnostics_dir / "plots"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    features_to_analyse = _select_features(enriched, baseline_columns, available_bases)
    if not features_to_analyse:
        raise ValueError("No enriched features were generated for analysis.")

    summaries: list[dict[str, object]] = []
    figure_paths: dict[str, Path] = {}
    for feature in features_to_analyse:
        try:
            series = _prepare_numeric_series(enriched, feature)
        except ValueError as exc:
            print(f"[WARN] Skipping feature '{feature}': {exc}")
            continue

        summary_stats = compute_distribution_summary(series)
        summaries.append(
            {"ticker": selected_ticker, "feature": feature, **summary_stats}
        )

        figure_path = plot_distribution_grid(
            series,
            ticker=selected_ticker,
            feature=feature,
            case_id=CASE_ID,
            case_name=CASE_NAME,
            summary=summary_stats,
            output_dir=plots_dir,
            show=show_plots,
        )
        figure_paths[feature] = figure_path
        print(f"[INFO] Saved diagnostics for {feature} -> {figure_path}")

    if not summaries:
        raise ValueError("No feature summaries were produced; aborting case.")

    summaries_frame = pd.DataFrame(summaries)
    summary_path = save_dataframe(
        summaries_frame,
        diagnostics_dir,
        f"{selected_ticker.lower()}_{CASE_ID}_feature_stats.csv",
    )
    print(f"Summary statistics saved to: {summary_path}")

    enriched_path = save_dataframe(
        enriched,
        case_output_dir,
        f"{selected_ticker.lower()}_{CASE_ID}_enriched_dataset.csv",
    )
    print(f"Enriched dataset saved to: {enriched_path}")

    relative_plots = {
        feature: path.relative_to(case_output_dir).as_posix()
        for feature, path in figure_paths.items()
    }

    write_case_metadata(
        case_dir=case_output_dir,
        case_id=CASE_ID,
        case_name=CASE_NAME,
        package="feature_selection_methods",
        dataset_path=dataset_path,
        tickers=[selected_ticker],
        features={
            "base_features": list(available_bases),
            "target_features": [col for col in TARGET_FEATURE_COLUMNS if col in enriched.columns],
            "generated_features": list(relative_plots.keys()),
        },
        target="target_1d",
        models=[],
        extras={
            "diagnostics_summary_csv": summary_path.relative_to(case_output_dir).as_posix(),
            "enriched_dataset_csv": enriched_path.relative_to(case_output_dir).as_posix(),
            "base_feature_source": base_source,
            "missing_base_features": missing_bases,
            "plot_paths": relative_plots,
        },
    )

    return case_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CASE_NAME)
    parser.add_argument(
        "--dataset",
        dest="dataset_filename",
        default=DEFAULT_DATASET_FILENAME,
        help="Dataset filename located within the ML input directory.",
    )
    parser.add_argument(
        "--ticker",
        dest="ticker",
        default=None,
        help=f"Ticker symbol to analyse (default: {DEFAULT_TICKER}).",
    )
    parser.add_argument(
        "--base-features",
        dest="base_features",
        nargs="+",
        default=None,
        help="Optional list of base features to pass into feature expansion.",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default=None,
        help="Optional override for the case output directory.",
    )
    parser.add_argument(
        "--show-plots",
        dest="show_plots",
        action="store_true",
        help="If set, display plots interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI convenience
    args = _parse_args()
    run_case(
        dataset_filename=args.dataset_filename,
        ticker=args.ticker,
        base_features=args.base_features,
        output_root=args.output_root,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
