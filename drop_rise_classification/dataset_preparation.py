"""Utilities for preparing classification datasets from ML inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from config import ML_INPUT_DIR
from core.feature_expansion import TARGET_FEATURE_COLUMNS, expand_base_features
from core.shared_utils import filter_ticker, save_dataframe
from drop_rise_classification.settings import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_NEUTRAL_BAND,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TICKER,
)


DEFAULT_CLASS_LABELS = ("drop", "flat", "rise")
CLASSIFICATION_SUBDIR = "classification"


@dataclass(frozen=True)
class ClassificationArtifacts:
    """Paths and metadata describing generated classification assets."""

    dataset_path: Path
    feature_stats_path: Path | None
    distribution_index_path: Path | None
    metadata_path: Path
    feature_columns: tuple[str, ...]
    base_features: tuple[str, ...]
    target_feature_columns: tuple[str, ...]
    derived_feature_columns: tuple[str, ...]
    ticker: str
    neutral_band: float


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    if "ticker" not in df.columns:
        raise KeyError("Dataset must contain a 'ticker' column.")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


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
    exclude = {
        "Date",
        "ticker",
        "target",
        "target_pct",
        "target_pct_1d",
        "target_1d",
    }
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


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if ptypes.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    if ptypes.is_categorical_dtype(series):
        codes = series.cat.codes.replace(-1, np.nan).astype(float)
        return pd.Series(codes, index=series.index, name=series.name)
    try:
        return pd.to_numeric(series, errors="coerce")
    except (TypeError, ValueError):
        return pd.to_numeric(series.astype(str), errors="coerce")


def _derive_target_class(series: pd.Series, *, neutral_band: float) -> pd.Series:
    values = series.to_numpy(dtype=float, copy=False)
    labels = np.select(
        [values < -neutral_band, values > neutral_band],
        [DEFAULT_CLASS_LABELS[0], DEFAULT_CLASS_LABELS[2]],
        default=DEFAULT_CLASS_LABELS[1],
    )
    labelled = pd.Series(labels, index=series.index, dtype="object")
    labelled = labelled.where(series.notna(), other=pd.NA)
    categorical = pd.Categorical(labelled, categories=list(DEFAULT_CLASS_LABELS), ordered=True)
    return pd.Series(categorical, index=series.index, name="target_1d_class")


def prepare_classification_dataset(
    dataset: pd.DataFrame,
    *,
    case_output_dir: Path,
    case_id: str,
    case_name: str,
    dataset_filename: str,
    ticker: str = DEFAULT_TICKER,
    base_features: Sequence[str] | None = None,
    neutral_band: float = DEFAULT_NEUTRAL_BAND,
    extra_columns: Sequence[str] | None = None,
    show_plots: bool = False,
    enable_distribution_diagnostics: bool = True,
) -> tuple[pd.DataFrame, ClassificationArtifacts]:
    """Generate a classification dataset and diagnostics for the requested ticker."""

    ticker = str(ticker).strip().upper() or DEFAULT_TICKER
    subset = filter_ticker(dataset, ticker)
    if subset.empty:
        raise ValueError(f"No rows available for ticker {ticker}.")

    if "Date" in subset.columns:
        subset["Date"] = pd.to_datetime(subset["Date"])
    subset["ticker"] = subset["ticker"].astype(str).str.upper()

    if base_features is None:
        base_features = _infer_base_features(subset)
    else:
        base_features = list(base_features)
    base_features = [feature for feature in base_features if feature in subset.columns]
    if not base_features:
        raise ValueError("No valid base features provided or inferred for classification dataset.")

    enriched = expand_base_features(
        subset,
        base_features=base_features,
        ticker_column="ticker",
        date_column="Date",
        close_column="Close",
        target_shift=1,
        target_window=30,
    )
    enriched = enriched.replace([np.inf, -np.inf], pd.NA)
    enriched = enriched.sort_values("Date").reset_index(drop=True)

    if "target_1d" not in enriched.columns:
        raise KeyError("Expanded dataset is missing 'target_1d'. Unable to build classification labels.")

    target_feature_columns = [col for col in TARGET_FEATURE_COLUMNS if col in enriched.columns]
    original_columns = set(subset.columns)
    derived_feature_columns = [
        col for col in enriched.columns if col not in original_columns and col not in {"target_1d_class"}
    ]

    candidate_columns: list[str] = []
    candidate_columns.extend(base_features)
    candidate_columns.extend(target_feature_columns)
    candidate_columns.extend(derived_feature_columns)
    if extra_columns:
        candidate_columns.extend(extra_columns)

    seen = set()
    candidate_columns = [col for col in candidate_columns if not (col in seen or seen.add(col))]

    data_columns: dict[str, pd.Series] = {}
    data_columns["Date"] = pd.to_datetime(enriched["Date"])
    data_columns["ticker"] = enriched["ticker"].astype(str)

    feature_columns: list[str] = []
    for column in candidate_columns:
        if column not in enriched.columns:
            continue
        numeric_series = _coerce_numeric(enriched[column])
        if numeric_series is None or numeric_series.isna().all():
            continue
        data_columns[column] = numeric_series
        feature_columns.append(column)

    classification_frame = pd.concat(data_columns, axis=1)
    classification_frame["target_1d"] = pd.to_numeric(enriched["target_1d"], errors="coerce")
    classification_frame["target_1d_class"] = _derive_target_class(classification_frame["target_1d"], neutral_band=neutral_band)

    if not feature_columns:
        raise ValueError("No usable numeric features found for classification dataset.")

    max_missing_ratio = 0.35
    usable_features: list[str] = []
    dropped_features: list[str] = []
    for feature in feature_columns:
        ratio = classification_frame[feature].isna().mean()
        if ratio <= max_missing_ratio:
            usable_features.append(feature)
        else:
            dropped_features.append(feature)

    if not usable_features:
        raise ValueError("All candidate features exceeded the missing value threshold.")

    if dropped_features:
        print(
            "[WARN] Dropped features due to missing ratio > "
            f"{max_missing_ratio:.0%}: {', '.join(sorted(dropped_features))}"
        )

    feature_columns = usable_features

    mandatory_columns = list(dict.fromkeys(feature_columns + ["target_1d"]))
    classification_frame = classification_frame.dropna(subset=mandatory_columns)
    if classification_frame.empty:
        raise ValueError("Classification dataset is empty after dropping missing values.")

    classification_frame = classification_frame.dropna(subset=feature_columns, how="any")
    if classification_frame.empty:
        raise ValueError("Insufficient rows with complete feature data after filtering.")

    classification_frame = classification_frame.reset_index(drop=True)

    base_features = [feature for feature in base_features if feature in feature_columns]
    target_feature_columns = [feature for feature in target_feature_columns if feature in feature_columns]
    derived_feature_columns = [
        feature
        for feature in derived_feature_columns
        if feature in feature_columns and feature not in base_features and feature not in target_feature_columns
    ]

    classification_dir = case_output_dir / CLASSIFICATION_SUBDIR
    classification_dir.mkdir(parents=True, exist_ok=True)

    plots_dir: Path | None = None
    if enable_distribution_diagnostics:
        plots_dir = classification_dir / "distribution_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_filename_out = f"{ticker.lower()}_{case_id}_classification_dataset.csv"
    dataset_path = save_dataframe(
        classification_frame,
        classification_dir,
        dataset_filename_out,
    )
    print(f"Classification dataset saved to: {dataset_path}")

    summaries: list[dict[str, object]] = []
    plot_index: list[dict[str, str]] = []
    if enable_distribution_diagnostics and plots_dir is not None:
        try:
            from core.distribution_plots import compute_distribution_summary, plot_distribution_grid  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive guard for optional dependency
            raise RuntimeError(
                "Distribution diagnostics requested but plotting utilities are unavailable."
            ) from exc

        for feature in feature_columns + ["target_1d"]:
            series = pd.to_numeric(classification_frame[feature], errors="coerce").dropna()
            if series.empty:
                continue
            try:
                summary_stats = compute_distribution_summary(series)
            except ValueError:
                continue
            summaries.append({"feature": feature, **summary_stats})
            figure_path = plot_distribution_grid(
                series,
                ticker=ticker,
                feature=feature,
                case_id=f"{case_id}_classification",
                case_name=f"{case_name} classification diagnostics",
                summary=summary_stats,
                output_dir=plots_dir,
                show=show_plots,
            )
            plot_index.append(
                {
                    "feature": feature,
                    "distribution_plot": figure_path.relative_to(case_output_dir).as_posix(),
                }
            )
            print(f"[INFO] Classification diagnostics saved for {feature}: {figure_path}")

    feature_stats_path: Path | None = None
    if summaries:
        feature_stats_path = save_dataframe(
            pd.DataFrame(summaries),
            classification_dir,
            f"{ticker.lower()}_{case_id}_classification_feature_stats.csv",
        )
        print(f"Classification feature summaries saved to: {feature_stats_path}")

    distribution_index_path: Path | None = None
    if plot_index:
        distribution_index_path = save_dataframe(
            pd.DataFrame(plot_index),
            classification_dir,
            f"{ticker.lower()}_{case_id}_classification_distribution_plots.csv",
        )
        print(f"Distribution plot index saved to: {distribution_index_path}")

    metadata_payload = {
        "case_id": case_id,
        "case_name": case_name,
        "ticker": ticker,
        "source_dataset": dataset_filename,
        "neutral_band": neutral_band,
        "row_count": int(classification_frame.shape[0]),
        "feature_columns": feature_columns,
        "base_features": base_features,
        "target_feature_columns": target_feature_columns,
        "derived_feature_columns": derived_feature_columns,
        "class_labels": list(DEFAULT_CLASS_LABELS),
        "classification_dataset": dataset_path.relative_to(case_output_dir).as_posix(),
        "feature_summaries": feature_stats_path.relative_to(case_output_dir).as_posix() if feature_stats_path else None,
        "distribution_index": (
            distribution_index_path.relative_to(case_output_dir).as_posix() if distribution_index_path else None
        ),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }
    metadata_path = classification_dir / f"{ticker.lower()}_{case_id}_classification_meta.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2))
    print(f"Classification metadata saved to: {metadata_path}")

    artifacts = ClassificationArtifacts(
        dataset_path=dataset_path,
        feature_stats_path=feature_stats_path,
        distribution_index_path=distribution_index_path,
        metadata_path=metadata_path,
        feature_columns=tuple(feature_columns),
        base_features=tuple(base_features),
        target_feature_columns=tuple(target_feature_columns),
        derived_feature_columns=tuple(derived_feature_columns),
        ticker=ticker,
        neutral_band=neutral_band,
    )

    return classification_frame, artifacts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate classification dataset and diagnostics for a ticker.")
    parser.add_argument(
        "--dataset",
        default=str((ML_INPUT_DIR / DEFAULT_DATASET_FILENAME).resolve()),
        help="Path to the ML dataset CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help=f"Ticker symbol to analyse (default: {DEFAULT_TICKER}).",
    )
    parser.add_argument(
        "--base-features",
        nargs="+",
        default=None,
        help="Optional list of base feature columns for expansion.",
    )
    parser.add_argument(
        "--extra-columns",
        nargs="+",
        default=None,
        help="Additional columns to include in the classification output.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where outputs should be written (default: package outputs directory).",
    )
    parser.add_argument(
        "--case-id",
        default="classification_preparation_cli",
        help="Identifier used inside generated metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--case-name",
        default="Classification dataset preparation workflow",
        help="Friendly name stored in metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--neutral-band",
        type=float,
        default=DEFAULT_NEUTRAL_BAND,
        help="Absolute threshold around zero treated as 'flat' (default: %(default)s).",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively after saving them.",
    )

    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset).resolve()
    dataset = _load_dataset(dataset_path)

    base_features = args.base_features or _load_metadata_base_features(dataset_path)
    if not base_features:
        base_features = _infer_base_features(dataset)

    if not base_features:
        raise SystemExit("No base features available. Provide them via --base-features.")

    output_root = Path(args.output_dir).resolve() if args.output_dir else (DEFAULT_OUTPUT_ROOT / args.case_id)
    output_root.mkdir(parents=True, exist_ok=True)

    _, artifacts = prepare_classification_dataset(
        dataset=dataset,
        case_output_dir=output_root,
        case_id=args.case_id,
        case_name=args.case_name,
        dataset_filename=str(dataset_path),
        ticker=args.ticker,
        base_features=base_features,
        neutral_band=args.neutral_band,
        extra_columns=args.extra_columns,
        show_plots=args.show_plots,
    )

    print("Classification outputs generated:")
    print(f" - dataset: {artifacts.dataset_path}")
    if artifacts.feature_stats_path is not None:
        print(f" - feature stats: {artifacts.feature_stats_path}")
    if artifacts.distribution_index_path is not None:
        print(f" - distribution index: {artifacts.distribution_index_path}")
    print(f" - metadata: {artifacts.metadata_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "CLASSIFICATION_SUBDIR",
    "ClassificationArtifacts",
    "DEFAULT_CLASS_LABELS",
    "DEFAULT_NEUTRAL_BAND",
    "DEFAULT_TICKER",
    "prepare_classification_dataset",
    "main",
]
