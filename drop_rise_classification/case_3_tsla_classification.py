"""Case 3: TSLA drop/rise classification with fixed features and model-derived enrichments."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
from typing import Callable, Sequence

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
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

CASE_ID = "case_3"
CASE_CONFIG = get_case_config(CASE_ID)
CASE_NAME = CASE_CONFIG.name
TARGET_COLUMN = "target_1d_class"
CONTINUOUS_TARGET_COLUMN = "target_1d"
PRIMARY_MODEL_NAME = "GradientBoosting"
AUXILIARY_MODEL_NAME = "LogisticRegression"
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

FIXED_SELECTED_FEATURES: tuple[str, ...] = (
    "rolling_median_target_1d",
    "target_1d",
    "close_skew_bucket_neutral",
    "openclosereturn_x_skew_sign_pos",
    "rsi_14_skew_bucket_neutral",
    "avg_volume_14_zscore_14",
    "atr_14_risk_level_code",
    "dmi_plus_14_skew_bucket_neutral",
    "stoch_k_50_skew_bucket_very_right",
    "dmi_plus_50_x_skew_sign_pos",
    "dmi_minus_50_skew_bucket_left",
    "dmi_minus_50_skew_bucket_neutral",
    "macd_skew_bucket_neutral",
    "bb_width_14_skew_delta_7",
)


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


def _validate_feature_availability(
    df: pd.DataFrame,
    required_features: Sequence[str],
) -> list[str]:
    missing = [feature for feature in required_features if feature not in df.columns]
    if missing:
        missing_fmt = ", ".join(missing)
        raise KeyError(f"Classification dataset is missing required features: {missing_fmt}")
    return list(required_features)


def _compute_log_odds(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


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
    """Execute TSLA classification with fixed features and enriched outputs."""

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

        selected_features = _validate_feature_availability(classification_df, FIXED_SELECTED_FEATURES)

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

        X_train = train_df.loc[:, selected_features].to_numpy(dtype=float)
        X_test = test_df.loc[:, selected_features].to_numpy(dtype=float)
        y_train = label_encoder.transform(train_df[TARGET_COLUMN])
        y_test = label_encoder.transform(test_df[TARGET_COLUMN])

        if np.unique(y_train).shape[0] < 2:
            raise ValueError("Training split contains fewer than two classes; adjust neutral band or dataset.")

        gbm = GradientBoostingClassifier(random_state=random_state)
        gbm.fit(X_train, y_train)
        y_pred_gbm = gbm.predict(X_test)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logreg = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
            random_state=random_state,
        )
        logreg.fit(X_train_scaled, y_train)
        y_pred_logreg = logreg.predict(X_test_scaled)

        classes = label_encoder.classes_
        if "rise" not in classes:
            raise KeyError("Label encoder is missing the 'rise' class required for probability features.")
        rise_index = int(np.where(classes == "rise")[0][0])

        gbm_test_proba = gbm.predict_proba(X_test)[:, rise_index]
        logreg_test_proba = logreg.predict_proba(X_test_scaled)[:, rise_index]
        logreg_test_log_odds = _compute_log_odds(logreg_test_proba)

        metrics_records: list[dict[str, object]] = [
            {
                "model": PRIMARY_MODEL_NAME,
                "accuracy": accuracy_score(y_test, y_pred_gbm),
                "precision_weighted": precision_score(y_test, y_pred_gbm, average="weighted", zero_division=0),
                "recall_weighted": recall_score(y_test, y_pred_gbm, average="weighted", zero_division=0),
                "f1_weighted": f1_score(y_test, y_pred_gbm, average="weighted", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred_gbm, average="macro", zero_division=0),
            },
            {
                "model": AUXILIARY_MODEL_NAME,
                "accuracy": accuracy_score(y_test, y_pred_logreg),
                "precision_weighted": precision_score(y_test, y_pred_logreg, average="weighted", zero_division=0),
                "recall_weighted": recall_score(y_test, y_pred_logreg, average="weighted", zero_division=0),
                "f1_weighted": f1_score(y_test, y_pred_logreg, average="weighted", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred_logreg, average="macro", zero_division=0),
            },
        ]

        prediction_frame = test_df[["Date", "ticker"]].copy()
        prediction_frame["actual_label"] = label_encoder.inverse_transform(y_test)
        if CONTINUOUS_TARGET_COLUMN in test_df.columns:
            prediction_frame[CONTINUOUS_TARGET_COLUMN] = test_df[CONTINUOUS_TARGET_COLUMN]

        prediction_frame[f"pred_{PRIMARY_MODEL_NAME.lower()}"] = label_encoder.inverse_transform(y_pred_gbm)
        prediction_frame[f"pred_{AUXILIARY_MODEL_NAME.lower()}"] = label_encoder.inverse_transform(y_pred_logreg)
        prediction_frame["rise_proba_gbm"] = gbm_test_proba
        prediction_frame["log_odds_logreg"] = logreg_test_log_odds

        metrics_df = pd.DataFrame(metrics_records).sort_values("f1_weighted", ascending=False)
        metrics_path = save_dataframe(
            metrics_df,
            case_output_dir,
            f"{ticker.lower()}_{CASE_ID}_model_metrics.csv",
        )

        predictions_path = save_dataframe(
            prediction_frame,
            case_output_dir,
            f"{ticker.lower()}_{CASE_ID}_test_predictions.csv",
        )

        confusion_matrices: dict[str, Path] = {}
        classification_reports: dict[str, Path] = {}

        model_eval_payload = {
            PRIMARY_MODEL_NAME: (y_pred_gbm, gbm_test_proba),
            AUXILIARY_MODEL_NAME: (y_pred_logreg, logreg_test_proba),
        }
        for model_name, (y_pred, rise_proba) in model_eval_payload.items():
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes)))
            cm_df = pd.DataFrame(
                cm,
                index=classes,
                columns=classes,
            )
            cm_path = save_dataframe(
                cm_df.reset_index().rename(columns={"index": "actual"}),
                diagnostics_dir,
                f"{ticker.lower()}_{CASE_ID}_{model_name.lower()}_confusion_matrix.csv",
            )
            confusion_matrices[model_name] = cm_path

            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=classes,
                zero_division=0,
                output_dict=True,
            )
            report_df = pd.DataFrame(report_dict).transpose()
            report_path = save_dataframe(
                report_df,
                diagnostics_dir,
                f"{ticker.lower()}_{CASE_ID}_{model_name.lower()}_classification_report.csv",
            )
            classification_reports[model_name] = report_path

        feature_catalog_records = [
            {"feature": feature, "category": "case2_selected"} for feature in selected_features
        ]
        feature_catalog_records.extend(
            [
                {"feature": "rise_proba_gbm", "category": "model_generated"},
                {"feature": "log_odds_logreg", "category": "model_generated"},
            ]
        )
        feature_catalog_df = pd.DataFrame(feature_catalog_records)
        feature_catalog_path = save_dataframe(
            feature_catalog_df,
            diagnostics_dir,
            f"{ticker.lower()}_{CASE_ID}_feature_catalog.csv",
        )

        full_features = classification_df.loc[:, selected_features].to_numpy(dtype=float)
        gbm_full_proba = gbm.predict_proba(full_features)[:, rise_index]
        full_scaled_features = scaler.transform(full_features)
        logreg_full_proba = logreg.predict_proba(full_scaled_features)[:, rise_index]
        logreg_full_log_odds = _compute_log_odds(logreg_full_proba)

        feature_enriched_frame = classification_df.copy()
        feature_enriched_frame["rise_proba_gbm"] = gbm_full_proba
        feature_enriched_frame["log_odds_logreg"] = logreg_full_log_odds
        enriched_dataset_path = save_dataframe(
            feature_enriched_frame,
            case_output_dir,
            f"{ticker.lower()}_{CASE_ID}_feature_enriched_dataset.csv",
        )

        metadata_extras: dict[str, object] = {
            "neutral_band": neutral_band,
            "fixed_selected_features": selected_features,
            "primary_model": PRIMARY_MODEL_NAME,
            "auxiliary_model": AUXILIARY_MODEL_NAME,
            "feature_catalog_csv": feature_catalog_path.relative_to(case_output_dir).as_posix(),
            "metrics_csv": metrics_path.relative_to(case_output_dir).as_posix(),
            "predictions_csv": predictions_path.relative_to(case_output_dir).as_posix(),
            "feature_enriched_dataset_csv": enriched_dataset_path.relative_to(case_output_dir).as_posix(),
            "label_encoder_classes": classes.tolist(),
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
                "selected_features": list(selected_features),
                "model_generated_features": ["rise_proba_gbm", "log_odds_logreg"],
            },
            target=TARGET_COLUMN,
            models=[PRIMARY_MODEL_NAME, AUXILIARY_MODEL_NAME],
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

