"""Case 1: TSLA RSI calibration baseline with compact evaluation."""

from __future__ import annotations

import argparse
from typing import Mapping, Sequence

# Ensure project root is available on PYTHONPATH when running as script
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_regression

try:
    from great_tables import GT
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    GT = None  # type: ignore[assignment]

from config import ML_INPUT_DIR
from core.shared_utils import load_ml_dataset, resolve_output_dir, save_dataframe
from drop_rise_classification.settings import DEFAULT_NEUTRAL_BAND, DEFAULT_TICKER

CASE_ID = "case_1"
CASE_NAME = "Case 1: TSLA RSI calibration baseline"
DEFAULT_OUTPUT_ROOT = Path("rsi_calibration") / "outputs"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_YF_DATASET_PATH = PACKAGE_DIR / "data" / "tsla_yf_dataset.csv"

RSI_DEFAULT_PERIOD = 14
TARGET_LABELS = ("drop", "flat", "rise")


def _compute_rsi(close: pd.Series, period: int = RSI_DEFAULT_PERIOD) -> pd.Series:
    """Compute the classic Wilder's RSI."""

    close = close.to_numpy(dtype=float)
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    alpha = 1.0 / period
    avg_gain = np.empty_like(gains)
    avg_loss = np.empty_like(losses)

    if gains.size == 0:
        return pd.Series([], dtype=float)

    avg_gain[0] = gains[:period].mean()
    avg_loss[0] = losses[:period].mean()

    for idx in range(1, gains.size):
        avg_gain[idx] = (1 - alpha) * avg_gain[idx - 1] + alpha * gains[idx]
        avg_loss[idx] = (1 - alpha) * avg_loss[idx - 1] + alpha * losses[idx]

    rs = np.divide(
        avg_gain,
        avg_loss,
        out=np.zeros_like(avg_gain),
        where=avg_loss != 0,
    )
    rsi = 100 - 100 / (1 + rs)

    rsi_series = pd.Series(np.concatenate([[np.nan], rsi]), dtype=float)
    return rsi_series


def _derive_class_target(
    series: pd.Series,
    *,
    neutral_band: float,
) -> pd.Series:
    values = series.to_numpy(dtype=float, copy=False)
    labels = np.select(
        [values < -neutral_band, values > neutral_band],
        [TARGET_LABELS[0], TARGET_LABELS[2]],
        default=TARGET_LABELS[1],
    )
    labelled = pd.Series(labels, index=series.index, dtype="object")
    labelled = labelled.where(series.notna(), other=pd.NA)
    categorical = pd.Categorical(labelled, categories=list(TARGET_LABELS), ordered=True)
    return pd.Series(categorical, index=series.index, name="target_1d_class")


def _prepare_rsi_dataset(
    dataset: pd.DataFrame,
    *,
    ticker: str,
    rsi_period: int,
    neutral_band: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    filtered = dataset.copy()
    filtered["ticker"] = filtered["ticker"].astype(str).str.upper()
    ticker_upper = ticker.upper()
    filtered = filtered[filtered["ticker"] == ticker_upper].copy()
    if filtered.empty:
        raise ValueError(f"No rows available for ticker {ticker_upper}.")

    if "Date" in filtered.columns:
        filtered["Date"] = pd.to_datetime(filtered["Date"], utc=True, errors="coerce").dt.tz_convert(None)
        filtered = filtered.sort_values("Date")

    close = pd.to_numeric(filtered["Close"], errors="coerce")
    rsi = _compute_rsi(close, period=rsi_period)
    filtered = filtered.assign(rsi=rsi)

    filtered["target_1d"] = (close.shift(-1) - close) / close
    filtered["target_1d_class"] = _derive_class_target(filtered["target_1d"], neutral_band=neutral_band)

    assembled = filtered.loc[:, ["Date", "ticker", "Close", "rsi", "target_1d", "target_1d_class"]].dropna().reset_index(drop=True)

    features = assembled.loc[:, ["rsi"]]
    target_class = assembled["target_1d_class"]
    target_continuous = assembled["target_1d"]
    return features, target_class, target_continuous


def _build_estimators(random_state: int) -> Mapping[str, Pipeline]:
    estimators: dict[str, Pipeline] = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "SVC": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "KNeighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=15, weights="distance")),
            ]
        ),
        "GaussianNB": Pipeline(
            [
                ("model", GaussianNB()),
            ]
        ),
        "RandomForest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        random_state=random_state,
                        class_weight="balanced_subsample",
                    ),
                )
            ]
        ),
        "ExtraTrees": Pipeline(
            [
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                )
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
        "AdaBoost": Pipeline(
            [
                ("model", AdaBoostClassifier(random_state=random_state, n_estimators=300)),
            ]
        ),
        "DecisionTree": Pipeline(
            [
                ("model", DecisionTreeClassifier(random_state=random_state, class_weight="balanced")),
            ]
        ),
        "MLPClassifier": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        max_iter=2000,
                        random_state=random_state,
                        early_stopping=True,
                        n_iter_no_change=20,
                    ),
                ),
            ]
        ),
    }
    return estimators


def _evaluate_models(
    estimators: Mapping[str, Pipeline],
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    random_state: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    class_names = list(label_encoder.classes_)

    for model_name, pipeline in estimators.items():
        pipeline.fit(X_train, y_train)
        if not hasattr(pipeline, "predict_proba"):
            continue
        probas = pipeline.predict_proba(X_test)
        y_pred_labels = pipeline.predict(X_test)

        overall_accuracy = float(accuracy_score(y_test, y_pred_labels))
        precision_macro = float(precision_score(y_test, y_pred_labels, average="macro", zero_division=0))
        recall_macro = float(recall_score(y_test, y_pred_labels, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred_labels, average="macro", zero_division=0))

        for class_index, class_label in enumerate(class_names):
            y_true = (y_test == class_index).astype(float)
            y_pred = probas[:, class_index].astype(float)

            brier = float(brier_score_loss(y_true, y_pred))
            if np.unique(y_true).size < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_true, y_pred))
            try:
                ll = float(log_loss(y_true, y_pred, labels=[0.0, 1.0]))
            except ValueError:
                ll = float("nan")
            mi = float(
                mutual_info_regression(
                    y_pred.reshape(-1, 1),
                    y_true,
                    random_state=random_state,
                )[0]
            )

            records.append(
                {
                    "model": model_name,
                    "class": class_label,
                    "brier_score": brier,
                    "log_loss": ll,
                    "roc_auc": auc,
                    "mutual_information": mi,
                    "accuracy": overall_accuracy,
                    "precision_macro": precision_macro,
                    "recall_macro": recall_macro,
                    "f1_macro": f1_macro,
                }
            )

    results_df = pd.DataFrame.from_records(records)
    return results_df


def _save_png_table(df: pd.DataFrame, output_path: Path) -> tuple[Path, Path] | None:
    if df.empty:
        return None
    if GT is None:
        print("[WARN] great_tables not installed; skipping PNG/HTML table generation.")
        return None

    display_df = df.copy()
    numeric_columns = [
        "brier_score",
        "log_loss",
        "roc_auc",
        "mutual_information",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]
    for column in numeric_columns:
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce").round(4)

    try:
        table = GT(display_df).tab_header(title=f"{CASE_NAME} metrics")
        table.save(str(output_path))
        html_path = output_path.with_suffix(".html")
        table.write_raw_html(html_path, inline_css=True, make_page=True)
        return output_path, html_path
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to render PNG/HTML table with great_tables: {exc}")
        return None


def run_case(
    *,
    dataset_filename: str | None = None,
    ticker: str = DEFAULT_TICKER,
    neutral_band: float = DEFAULT_NEUTRAL_BAND,
    output_root: Path | None = None,
    test_size: float = 0.2,
    rsi_period: int = RSI_DEFAULT_PERIOD,
    random_state: int = 42,
) -> Path:
    if not 0.0 < test_size < 0.5:
        raise ValueError("test_size should be between 0 and 0.5 for chronological split.")

    if dataset_filename is None:
        dataset_path = DEFAULT_YF_DATASET_PATH
    else:
        dataset_path = Path(dataset_filename)
        if not dataset_path.exists():
            dataset_path = ML_INPUT_DIR / dataset_filename

    dataset = load_ml_dataset(dataset_path)

    features, target_class, target_continuous = _prepare_rsi_dataset(
        dataset,
        ticker=ticker,
        rsi_period=rsi_period,
        neutral_band=neutral_band,
    )
    if features.empty:
        raise ValueError("RSI dataset is empty after preparation.")

    split_index = int(len(features) * (1.0 - test_size))
    if split_index <= 0 or split_index >= len(features):
        raise ValueError("Chronological split produced invalid train/test division.")

    X_train = features.iloc[:split_index].to_numpy(dtype=float)
    X_test = features.iloc[split_index:].to_numpy(dtype=float)

    label_encoder = LabelEncoder()
    label_encoder.fit(target_class)
    y_encoded = label_encoder.transform(target_class)
    y_train = y_encoded[:split_index]
    y_test = y_encoded[split_index:]

    estimators = _build_estimators(random_state=random_state)
    metrics_df = _evaluate_models(
        estimators,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
        random_state=random_state,
    )

    case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_snapshot = features.copy()
    dataset_snapshot["target_1d_class"] = target_class
    dataset_snapshot["target_1d"] = target_continuous
    dataset_snapshot["split"] = ["train"] * split_index + ["test"] * (len(features) - split_index)

    dataset_path_out = save_dataframe(
        dataset_snapshot.reset_index(drop=True),
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_rsi_dataset.csv",
    )

    metrics_path = save_dataframe(
        metrics_df,
        case_output_dir,
        f"{ticker.lower()}_{CASE_ID}_metrics.csv",
    )

    png_path = case_output_dir / f"{ticker.lower()}_{CASE_ID}_metrics.png"
    table_paths = _save_png_table(metrics_df, png_path)

    print(f"Prepared dataset snapshot: {dataset_path_out}")
    print(f"Model metrics saved to: {metrics_path}")
    if table_paths:
        png_generated, html_generated = table_paths
        if png_generated.exists():
            print(f"Metrics table rendered to: {png_generated}")
        if html_generated.exists():
            print(f"Metrics table HTML saved to: {html_generated}")

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
        random_state=args.random_state,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
