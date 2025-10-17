"""Case 7: RSI period model comparison across horizons for TSLA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure imports work when executed as a script
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

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
    _build_estimators,
)
from rsi_calibration.case_3_tsla_rsi_multihorizon import _prepare_multihorizon_dataset

CASE_ID = "case_7"
CASE_NAME = "Case 7: RSI period model comparison"
DEFAULT_OUTPUT_ROOT = Path("rsi_calibration") / "outputs"
DEFAULT_RSI_PERIODS = (5, 14, 21, 50)
DEFAULT_TARGET_HORIZONS = (1, 7, 14, 21)
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def _evaluate_models_for_period(
    *,
    rsi_period: int,
    datasets: Dict[int, Tuple[pd.DataFrame, pd.Series]],
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate baseline estimators for a specific RSI period across horizons."""
    estimators = _build_estimators(random_state=random_state)
    metrics_records: list[dict[str, object]] = []
    snapshot_frames: list[pd.DataFrame] = []

    for horizon, (features, target_class) in datasets.items():
        if features.empty:
            print(f"[WARN] RSI {rsi_period}, horizon {horizon}: empty feature set, skipping.")
            continue

        split_index = int(len(features) * (1.0 - test_size))
        if split_index <= 0 or split_index >= len(features):
            print(f"[WARN] RSI {rsi_period}, horizon {horizon}: invalid train/test split, skipping.")
            continue

        target_cat = target_class.astype("category")
        y_encoded = target_cat.cat.codes.to_numpy()

        if len(np.unique(y_encoded[:split_index])) < 2:
            print(
                f"[WARN] RSI {rsi_period}, horizon {horizon}: training data lacks at least two classes, skipping models."
            )
            continue

        X_train = features.iloc[:split_index].to_numpy(dtype=float)
        X_test = features.iloc[split_index:].to_numpy(dtype=float)
        y_train = y_encoded[:split_index]
        y_test = y_encoded[split_index:]

        for model_name, pipeline in estimators.items():
            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                accuracy = float((y_pred == y_test).mean())
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[WARN] RSI {rsi_period}, horizon {horizon}, model {model_name} failed: {exc}")
                accuracy = np.nan

            metrics_records.append(
                {
                    "rsi_period": int(rsi_period),
                    "target_horizon": int(horizon),
                    "model": model_name,
                    "accuracy": accuracy,
                    "train_samples": split_index,
                    "test_samples": len(features) - split_index,
                }
            )

        snapshot = features.copy()
        snapshot["target_class"] = target_cat.astype(str)
        snapshot["split"] = ["train"] * split_index + ["test"] * (len(features) - split_index)
        snapshot["target_horizon"] = int(horizon)
        snapshot["rsi_period"] = int(rsi_period)
        snapshot_frames.append(snapshot)

    metrics_df = pd.DataFrame(metrics_records)
    snapshot_df = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    return metrics_df, snapshot_df


def _prepare_accuracy_display(pivot_df: pd.DataFrame) -> pd.DataFrame:
    display = pivot_df.copy()
    display.index.name = "RSI period"
    display = display.reset_index()
    column_mapping = {}
    for col in display.columns:
        if isinstance(col, (int, np.integer)):
            column_mapping[col] = f"h{int(col)}d best accuracy"
    display = display.rename(columns=column_mapping)
    float_cols = [col for col in display.columns if col != "RSI period"]
    display[float_cols] = display[float_cols].astype(float).round(4)
    return display


def _prepare_model_display(best_df: pd.DataFrame) -> pd.DataFrame:
    working = best_df.copy()
    working["model_accuracy"] = working.apply(
        lambda row: f"{row['model']} ({row['accuracy']:.4f})", axis=1
    )
    pivot = working.pivot(index="rsi_period", columns="target_horizon", values="model_accuracy")
    pivot.index.name = "RSI period"
    pivot = pivot.reset_index()
    column_mapping = {}
    for col in pivot.columns:
        if isinstance(col, (int, np.integer)):
            column_mapping[col] = f"h{int(col)}d best model"
    return pivot.rename(columns=column_mapping)


def _render_table(df: pd.DataFrame, output_path: Path, *, title: str, subtitle: str) -> None:
    if df.empty:
        return
    if GT is None:
        print("[WARN] great_tables not installed; skipping table rendering.")
        return
    try:
        table = (
            GT(df)
            .tab_header(title=title, subtitle=subtitle)
            .tab_source_note("Accuracy values rounded to 4 decimal places. Random seed fixed at 42.")
        )
        table.save(str(output_path))
        table.write_raw_html(output_path.with_suffix(".html"), inline_css=True, make_page=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to render table with great_tables: {exc}")


def run_case(
    *,
    dataset_filename: str | None = None,
    ticker: str = DEFAULT_TICKER,
    neutral_band: float = DEFAULT_NEUTRAL_BAND,
    rsi_periods: Sequence[int] = DEFAULT_RSI_PERIODS,
    target_horizons: Iterable[int] = DEFAULT_TARGET_HORIZONS,
    test_size: float = DEFAULT_TEST_SIZE,
    output_root: Path | None = None,
    random_state: int = RANDOM_STATE,
) -> Path:
    if dataset_filename is None:
        dataset_path = DEFAULT_YF_DATASET_PATH
    else:
        dataset_path = Path(dataset_filename)
        if not dataset_path.exists():
            dataset_path = ML_INPUT_DIR / dataset_filename

    dataset = load_ml_dataset(dataset_path)

    metrics_frames: list[pd.DataFrame] = []
    snapshot_frames: list[pd.DataFrame] = []

    for rsi_period in rsi_periods:
        prepared = _prepare_multihorizon_dataset(
            dataset,
            ticker=ticker,
            rsi_period=rsi_period,
            neutral_band=neutral_band,
            target_horizons=target_horizons,
        )
        metrics_df, snapshot_df = _evaluate_models_for_period(
            rsi_period=rsi_period,
            datasets=prepared,
            test_size=test_size,
            random_state=random_state,
        )
        if not metrics_df.empty:
            metrics_frames.append(metrics_df)
        if not snapshot_df.empty:
            snapshot_frames.append(snapshot_df)

    if not metrics_frames:
        raise ValueError("No metrics were computed. Check dataset, RSI periods, or horizons.")

    combined_metrics = pd.concat(metrics_frames, ignore_index=True)
    combined_snapshots = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = save_dataframe(
        combined_metrics,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_metrics.csv",
    )
    print(f"Combined metrics saved to: {metrics_path}")

    if not combined_snapshots.empty:
        dataset_snapshot_path = save_dataframe(
            combined_snapshots,
            case_output_dir,
            f"{ticker.lower()}_{CASE_ID}_dataset_snapshot.csv",
        )
        print(f"Dataset snapshot saved to: {dataset_snapshot_path}")

    best_idx = combined_metrics.groupby(["rsi_period", "target_horizon"])["accuracy"].idxmax()
    best_combos = combined_metrics.loc[best_idx].reset_index(drop=True)
    best_combos_path = save_dataframe(
        best_combos,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_best_combos.csv",
    )
    print(f"Best model combinations saved to: {best_combos_path}")

    pivot_accuracy = best_combos.pivot(index="rsi_period", columns="target_horizon", values="accuracy")
    pivot_accuracy = pivot_accuracy.sort_index(axis=0).sort_index(axis=1)
    pivot_accuracy_path = save_dataframe(
        pivot_accuracy.reset_index(),
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_best_accuracy_pivot.csv",
    )
    print(f"Best accuracy pivot saved to: {pivot_accuracy_path}")

    accuracy_display = _prepare_accuracy_display(pivot_accuracy)
    accuracy_display_path = save_dataframe(
        accuracy_display,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_best_accuracy_display.csv",
    )
    print(f"Accuracy display table saved to: {accuracy_display_path}")

    model_display = _prepare_model_display(best_combos)
    model_display_path = save_dataframe(
        model_display,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_best_model_display.csv",
    )
    print(f"Model display table saved to: {model_display_path}")

    horizon_top_idx = combined_metrics.groupby("target_horizon")["accuracy"].idxmax()
    horizon_top = combined_metrics.loc[horizon_top_idx].sort_values("target_horizon").reset_index(drop=True)
    horizon_top_path = save_dataframe(
        horizon_top,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_horizon_top_models.csv",
    )
    print(f"Top models per horizon saved to: {horizon_top_path}")

    accuracy_table_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_best_accuracy_table.png"
    _render_table(
        accuracy_display,
        accuracy_table_path,
        title=f"{CASE_NAME} (accuracy matrix)",
        subtitle="Highest accuracy per RSI period and horizon.",
    )
    model_table_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_best_models_table.png"
    _render_table(
        model_display,
        model_table_path,
        title=f"{CASE_NAME} (best models)",
        subtitle="Best-performing model and accuracy per RSI period and horizon.",
    )
    if accuracy_table_path.exists():
        print(f"Accuracy matrix table saved to: {accuracy_table_path}")
    if model_table_path.exists():
        print(f"Best models table saved to: {model_table_path}")

    print("Top RSI periods per horizon (best model selection):")
    for _, row in horizon_top.iterrows():
        print(
            f"Horizon {int(row['target_horizon']):>2}d -> "
            f"RSI {int(row['rsi_period'])} using {row['model']} "
            f"(accuracy={row['accuracy']:.4f})"
        )

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
        "--rsi-periods",
        dest="rsi_periods",
        type=int,
        nargs="+",
        default=list(DEFAULT_RSI_PERIODS),
        help="List of RSI look-back periods to evaluate (default: %(default)s).",
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
        "--test-size",
        dest="test_size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of samples reserved for testing (default: %(default)s).",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default=None,
        help="Optional override for the case output root directory.",
    )
    parser.add_argument(
        "--random-state",
        dest="random_state",
        type=int,
        default=RANDOM_STATE,
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
        rsi_periods=args.rsi_periods,
        target_horizons=args.target_horizons,
        test_size=args.test_size,
        output_root=output_root,
        random_state=args.random_state,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
