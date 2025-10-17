"""Case 3: TSLA RSI classification across multiple target horizons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

# Ensure imports work when executed as a script
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

try:
    from great_tables import GT
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    GT = None  # type: ignore[assignment]

from config import ML_INPUT_DIR
from core.shared_utils import load_ml_dataset, resolve_output_dir, save_dataframe
from rsi_calibration.case_1_tsla_rsi_baseline import (
    DEFAULT_NEUTRAL_BAND,
    DEFAULT_TICKER,
    DEFAULT_YF_DATASET_PATH,
    RSI_DEFAULT_PERIOD,
    _build_estimators,
    _compute_rsi,
    _derive_class_target,
)
from rsi_calibration.case_1_tsla_rsi_baseline import _save_png_table as _save_metrics_table

CASE_ID = "case_3"
CASE_NAME = "Case 3: TSLA RSI multi-horizon classification"
DEFAULT_OUTPUT_ROOT = Path("rsi_calibration") / "outputs"
DEFAULT_TARGET_HORIZONS = (5, 14, 21)


def _prepare_multihorizon_dataset(
    dataset: pd.DataFrame,
    *,
    ticker: str,
    rsi_period: int,
    neutral_band: float,
    target_horizons: Iterable[int],
) -> dict[int, tuple[pd.DataFrame, pd.Series]]:
    working = dataset.copy()
    working["ticker"] = working["ticker"].astype(str).str.upper()
    ticker_upper = ticker.upper()
    working = working[working["ticker"] == ticker_upper].copy()
    if working.empty:
        raise ValueError(f"No rows available for ticker {ticker_upper}.")

    if "Date" in working.columns:
        working["Date"] = pd.to_datetime(working["Date"], utc=True, errors="coerce").dt.tz_convert(None)
        working = working.sort_values("Date")

    close = pd.to_numeric(working["Close"], errors="coerce")
    rsi = _compute_rsi(close, period=rsi_period)
    working = working.assign(rsi=rsi)

    datasets: dict[int, tuple[pd.DataFrame, pd.Series]] = {}
    for horizon in target_horizons:
        return_column = (close.shift(-horizon) - close) / close
        class_column = _derive_class_target(return_column, neutral_band=neutral_band)

        assembled = working.loc[:, ["Date", "ticker", "Close", "rsi"]].copy()
        assembled[f"target_{horizon}d"] = return_column
        assembled[f"target_{horizon}d_class"] = class_column
        assembled = assembled.dropna().reset_index(drop=True)
        features = assembled.loc[:, ["rsi"]]
        target_class = assembled[f"target_{horizon}d_class"]
        datasets[horizon] = (features, target_class)
    return datasets


def _evaluate_horizon(
    horizon: int,
    features: pd.DataFrame,
    target_class: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty:
        raise ValueError(f"Horizon {horizon}: dataset is empty.")

    split_index = int(len(features) * (1.0 - test_size))
    if split_index <= 0 or split_index >= len(features):
        raise ValueError(f"Horizon {horizon}: chronological split produced invalid train/test division.")

    X_train = features.iloc[:split_index].to_numpy(dtype=float)
    X_test = features.iloc[split_index:].to_numpy(dtype=float)

    target_class = target_class.astype("category")
    label_encoder = target_class.cat.codes
    y_encoded = label_encoder.to_numpy()
    y_train = y_encoded[:split_index]
    y_test = y_encoded[split_index:]

    # Reuse estimator definitions from case 1
    estimators = _build_estimators(random_state=random_state)
    metrics_records: list[dict[str, object]] = []
    for model_name, pipeline in estimators.items():
        pipeline.fit(X_train, y_train)
        if not hasattr(pipeline, "predict_proba"):
            continue
        probas = pipeline.predict_proba(X_test)
        y_pred_labels = pipeline.predict(X_test)

        accuracy = float((y_pred_labels == y_test).mean())
        metrics_records.append(
            {
                "target_horizon": horizon,
                "model": model_name,
                "accuracy": accuracy,
            }
        )

    metrics_df = pd.DataFrame(metrics_records)
    dataset_snapshot = features.copy()
    dataset_snapshot["target_class"] = target_class.astype(str)
    dataset_snapshot["split"] = ["train"] * split_index + ["test"] * (len(features) - split_index)
    dataset_snapshot["target_horizon"] = horizon
    return metrics_df, dataset_snapshot


def run_case(
    *,
    dataset_filename: str | None = None,
    ticker: str = DEFAULT_TICKER,
    neutral_band: float = DEFAULT_NEUTRAL_BAND,
    output_root: Path | None = None,
    test_size: float = 0.2,
    rsi_period: int = RSI_DEFAULT_PERIOD,
    target_horizons: Iterable[int] = DEFAULT_TARGET_HORIZONS,
    random_state: int = 42,
) -> Path:
    if dataset_filename is None:
        dataset_path = DEFAULT_YF_DATASET_PATH
    else:
        dataset_path = Path(dataset_filename)
        if not dataset_path.exists():
            dataset_path = ML_INPUT_DIR / dataset_filename

    dataset = load_ml_dataset(dataset_path)

    prepared = _prepare_multihorizon_dataset(
        dataset,
        ticker=ticker,
        rsi_period=rsi_period,
        neutral_band=neutral_band,
        target_horizons=target_horizons,
    )

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[pd.DataFrame] = []
    all_snapshots: list[pd.DataFrame] = []

    for horizon, (features, target_class) in prepared.items():
        metrics_df, snapshot_df = _evaluate_horizon(
            horizon,
            features,
            target_class,
            test_size=test_size,
            random_state=random_state,
        )
        all_metrics.append(metrics_df)
        all_snapshots.append(snapshot_df)

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_snapshots = pd.concat(all_snapshots, ignore_index=True)

    metrics_path = save_dataframe(
        combined_metrics,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_metrics.csv",
    )
    dataset_snapshot_path = save_dataframe(
        combined_snapshots,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_dataset_snapshot.csv",
    )

    png_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_metrics.png"
    _save_metrics_table(combined_metrics, png_path)

    horizon_summary = combined_metrics.groupby("target_horizon")["accuracy"].agg(["mean", "std"]).reset_index()
    horizon_summary_path = save_dataframe(
        horizon_summary,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_horizon_summary.csv",
    )
    horizon_png_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_horizon_summary.png"
    _save_metrics_table(horizon_summary, horizon_png_path)

    print(f"Combined metrics saved to: {metrics_path}")
    print(f"Dataset snapshot saved to: {dataset_snapshot_path}")
    print(f"Horizon summary saved to: {horizon_summary_path}")

    return case_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CASE_NAME)
    parser.add_argument(
        "--dataset",
        dest="dataset_filename",
        default=None,
        help="Optional path/filename to dataset (default: local tsla_yf_dataset.csv).",
    )
    parser.add_argument(
        "--ticker",
        dest="ticker",
        default=DEFAULT_TICKER,
        help=f"Ticker symbol to analyse (default: {DEFAULT_TICKER}).",
    )
    parser.add_argument(
        "--neutral-band",
        dest="neutral_band",
        type=float,
        default=DEFAULT_NEUTRAL_BAND,
        help="Absolute threshold around zero treated as 'flat'.",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default=None,
        help="Optional override for the case output root directory.",
    )
    parser.add_argument(
        "--test-size",
        dest="test_size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for testing (default: %(default)s).",
    )
    parser.add_argument(
        "--rsi-period",
        dest="rsi_period",
        type=int,
        default=RSI_DEFAULT_PERIOD,
        help="Look-back period for RSI calculation.",
    )
    parser.add_argument(
        "--target-horizons",
        dest="target_horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_TARGET_HORIZONS),
        help="List of horizons (in days) used to derive targets (default: %(default)s).",
    )
    parser.add_argument(
        "--random-state",
        dest="random_state",
        type=int,
        default=42,
        help="Random seed for estimators.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI convenience
    args = _parse_args()
    output_root = Path(args.output_root) if args.output_root else None

    run_case(
        dataset_filename=args.dataset_filename,
        ticker=args.ticker,
        neutral_band=args.neutral_band,
        output_root=output_root,
        test_size=args.test_size,
        rsi_period=args.rsi_period,
        target_horizons=args.target_horizons,
        random_state=args.random_state,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
