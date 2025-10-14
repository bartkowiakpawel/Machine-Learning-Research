"""Case 2: TSLA drop/rise classification without distribution diagnostics."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import ML_INPUT_DIR
from core.shared_utils import load_ml_dataset, resolve_output_dir, save_dataframe, write_case_metadata
from drop_rise_classification.dataset_preparation import (
    ClassificationArtifacts,
    prepare_classification_dataset,
)
from drop_rise_classification.settings import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_NEUTRAL_BAND,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TICKER,
    get_case_config,
)

CASE_ID = "case_2"
CASE_CONFIG = get_case_config(CASE_ID)
CASE_NAME = CASE_CONFIG.name
TARGET_COLUMN = "target_1d_class"
CONTINUOUS_TARGET_COLUMN = "target_1d"
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def _acquire_display_awake() -> Callable[[], None]:
    if os.name != "nt":
        return lambda: None
    try:
        set_state = ctypes.windll.kernel32.SetThreadExecutionState
    except AttributeError:
        return lambda: None

    set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)

    def _restore() -> None:
        set_state(ES_CONTINUOUS)

    return _restore


def _select_numeric_features(df: pd.DataFrame, candidate_features: Sequence[str]) -> list[str]:
    numeric_features: list[str] = []
    for feature in candidate_features:
        if feature not in df.columns:
            continue
        series = df[feature]
        if ptypes.is_numeric_dtype(series):
            numeric_features.append(feature)
    return numeric_features


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
                        multi_class="auto",
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }
    return estimators


def run_case(
    *,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    ticker: str = DEFAULT_TICKER,
    base_features: Sequence[str] | None = None,
    extra_columns: Sequence[str] | None = None,
    neutral_band: float = DEFAULT_NEUTRAL_BAND,
    output_root: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Path:
    """Execute TSLA classification workflow with feature selection and benchmarking."""

    release_display = _acquire_display_awake()
    if not 0.0 < test_size < 0.5:
        raise ValueError("test_size should be between 0 and 0.5 for stratified split.")

    try:
        dataset_path = ML_INPUT_DIR / dataset_filename
        dataset = load_ml_dataset(dataset_path)
        if "Date" in dataset.columns:
            dataset["Date"] = pd.to_datetime(dataset["Date"])

        case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
        diagnostics_dir = case_output_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        classification_df, artifacts = prepare_classification_dataset(
            dataset=dataset,
            case_output_dir=case_output_dir,
            case_id=CASE_ID,
            case_name=CASE_NAME,
            dataset_filename=str(dataset_path.resolve()),
            ticker=ticker,
            base_features=base_features,
            neutral_band=neutral_band,
            extra_columns=extra_columns,
            enable_distribution_diagnostics=False,
        )

        if TARGET_COLUMN not in classification_df.columns:
            raise KeyError(f"Classification dataset missing target column '{TARGET_COLUMN}'.")
        if classification_df[TARGET_COLUMN].nunique(dropna=True) < 2:
            raise ValueError("Classification target contains fewer than two classes after preparation.")

        feature_columns = _select_numeric_features(classification_df, artifacts.feature_columns)
        if len(feature_columns) <= 1:
            raise ValueError("Insufficient numeric features available for RFECV.")

        classification_df = classification_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        label_encoder = LabelEncoder()
        label_encoder.fit(classification_df[TARGET_COLUMN])

        train_df, test_df = train_test_split(
            classification_df,
            test_size=test_size,
            stratify=classification_df[TARGET_COLUMN],
            random_state=random_state,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        X_train = train_df.loc[:, feature_columns].to_numpy(dtype=float)
        y_train = label_encoder.transform(train_df[TARGET_COLUMN])

        if np.unique(y_train).shape[0] < 2:
            raise ValueError("Training split contains fewer than two classes; adjust neutral band or dataset.")

        scaler_for_rfecv = StandardScaler()
        X_train_scaled = scaler_for_rfecv.fit_transform(X_train)

        if len(feature_columns) < 2:
            raise ValueError("Need at least two features for RFECV.")
        min_features = 1 if len(feature_columns) < 5 else min(10, len(feature_columns) - 1)
        min_features = max(1, min_features)

        rfecv = RFECV(
            estimator=LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="auto",
            ),
            step=1,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            scoring="f1_macro",
            min_features_to_select=min_features,
            n_jobs=-1,
        )
        rfecv.fit(X_train_scaled, y_train)

        selected_mask = rfecv.support_
        selected_features = [feature_columns[idx] for idx, keep in enumerate(selected_mask) if keep]
        if not selected_features:
            best_index = int(np.argmin(rfecv.ranking_))
            selected_features = [feature_columns[best_index]]

        ranking_records = [
            {
                "feature": feature,
                "selected": bool(selected_mask[idx]),
                "rank": int(rfecv.ranking_[idx]),
            }
            for idx, feature in enumerate(feature_columns)
        ]
        ranking_df = pd.DataFrame(ranking_records).sort_values(
            ["selected", "rank", "feature"], ascending=[False, True, True]
        )
        ranking_path = save_dataframe(
            ranking_df,
            diagnostics_dir,
            f"{ticker.lower()}_{CASE_ID}_rfecv_ranking.csv",
        )

        try:
            rfecv_scores = rfecv.cv_results_["mean_test_score"]  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fallback for older sklearn
            rfecv_scores = getattr(rfecv, "grid_scores_", [])
        scores_df = pd.DataFrame(
            {
                "n_features": np.arange(1, len(rfecv_scores) + 1),
                "mean_cv_score": np.asarray(rfecv_scores, dtype=float),
            }
        )
        rfecv_scores_path = save_dataframe(
            scores_df,
            diagnostics_dir,
            f"{ticker.lower()}_{CASE_ID}_rfecv_scores.csv",
        )

        X_test = test_df.loc[:, feature_columns].to_numpy(dtype=float)
        y_test = label_encoder.transform(test_df[TARGET_COLUMN])

        estimators = _build_estimators(random_state)

        selected_train = train_df.loc[:, selected_features].to_numpy(dtype=float)
        selected_test = test_df.loc[:, selected_features].to_numpy(dtype=float)

        metrics_records: list[dict[str, object]] = []
        prediction_frame = test_df[["Date", "ticker"]].copy()
        prediction_frame["actual_label"] = label_encoder.inverse_transform(y_test)
        if CONTINUOUS_TARGET_COLUMN in test_df.columns:
            prediction_frame[CONTINUOUS_TARGET_COLUMN] = test_df[CONTINUOUS_TARGET_COLUMN]

        confusion_matrices: dict[str, Path] = {}
        classification_reports: dict[str, Path] = {}

        for name, pipeline in estimators.items():
            pipeline.fit(selected_train, y_train)
            y_pred = pipeline.predict(selected_test)
            metrics_records.append(
                {
                    "estimator": name,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                }
            )
            predicted_labels = label_encoder.inverse_transform(y_pred)
            prediction_frame[f"pred_{name.lower()}"] = predicted_labels

            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(selected_test)
                for class_idx, class_label in enumerate(label_encoder.classes_):
                    column_name = f"prob_{name.lower()}_{class_label.lower()}"
                    prediction_frame[column_name] = proba[:, class_idx]

            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(label_encoder.classes_)))
            cm_df = pd.DataFrame(
                cm,
                index=label_encoder.classes_,
                columns=label_encoder.classes_,
            )
            cm_path = save_dataframe(
                cm_df.reset_index().rename(columns={"index": "actual"}),
                diagnostics_dir,
                f"{ticker.lower()}_{CASE_ID}_{name.lower()}_confusion_matrix.csv",
            )
            confusion_matrices[name] = cm_path

            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=label_encoder.classes_,
                zero_division=0,
                output_dict=True,
            )
            report_df = pd.DataFrame(report_dict).transpose()
            report_path = save_dataframe(
                report_df,
                diagnostics_dir,
                f"{ticker.lower()}_{CASE_ID}_{name.lower()}_classification_report.csv",
            )
            classification_reports[name] = report_path

        metrics_df = pd.DataFrame(metrics_records).sort_values("f1_weighted", ascending=False)
        metrics_path = save_dataframe(
            metrics_df,
            case_output_dir,
            f"{ticker.lower()}_{CASE_ID}_model_metrics.csv",
        )

        selected_features_path = save_dataframe(
            pd.DataFrame({"feature": selected_features}),
            diagnostics_dir,
            f"{ticker.lower()}_{CASE_ID}_selected_features.csv",
        )

        predictions_path = save_dataframe(
            prediction_frame,
            case_output_dir,
            f"{ticker.lower()}_{CASE_ID}_test_predictions.csv",
        )

        metadata_extras: dict[str, object] = {
            "neutral_band": neutral_band,
            "rfecv_optimal_features": int(getattr(rfecv, "n_features_", len(selected_features))),
            "rfecv_ranking_csv": ranking_path.relative_to(case_output_dir).as_posix(),
            "rfecv_scores_csv": rfecv_scores_path.relative_to(case_output_dir).as_posix(),
            "selected_features_csv": selected_features_path.relative_to(case_output_dir).as_posix(),
            "metrics_csv": metrics_path.relative_to(case_output_dir).as_posix(),
            "predictions_csv": predictions_path.relative_to(case_output_dir).as_posix(),
            "label_encoder_classes": label_encoder.classes_.tolist(),
            "confusion_matrices": {
                name: path.relative_to(case_output_dir).as_posix() for name, path in confusion_matrices.items()
            },
            "classification_reports": {
                name: path.relative_to(case_output_dir).as_posix() for name, path in classification_reports.items()
            },
        }

        write_case_metadata(
            case_dir=case_output_dir,
            case_id=CASE_ID,
            case_name=CASE_NAME,
            package="drop_rise_classification",
            dataset_path=artifacts.dataset_path,
            tickers=[ticker],
            features={
                "candidate_features": list(feature_columns),
                "selected_features": selected_features,
            },
            target=TARGET_COLUMN,
            models=list(estimators.keys()),
            extras=metadata_extras,
        )

        return case_output_dir
    finally:
        release_display()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CASE_NAME)
    parser.add_argument(
        "--dataset",
        dest="dataset_filename",
        default=DEFAULT_DATASET_FILENAME,
        help="Dataset filename within the ML input directory.",
    )
    parser.add_argument(
        "--ticker",
        dest="ticker",
        default=DEFAULT_TICKER,
        help=f"Ticker symbol to analyse (default: {DEFAULT_TICKER}).",
    )
    parser.add_argument(
        "--base-features",
        dest="base_features",
        nargs="+",
        default=None,
        help="Optional list of base features to seed feature expansion.",
    )
    parser.add_argument(
        "--extra-columns",
        dest="extra_columns",
        nargs="+",
        default=None,
        help="Additional columns to include in the classification dataset.",
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
        "--random-state",
        dest="random_state",
        type=int,
        default=42,
        help="Random seed used for splitting and estimators.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI convenience
    args = _parse_args()
    output_root = Path(args.output_root) if args.output_root else None

    run_case(
        dataset_filename=args.dataset_filename,
        ticker=args.ticker,
        base_features=args.base_features,
        extra_columns=args.extra_columns,
        neutral_band=args.neutral_band,
        output_root=output_root,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
