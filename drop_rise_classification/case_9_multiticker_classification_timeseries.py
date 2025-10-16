"""Case 9: Multi-ticker drop/rise classification with time-series CV baseline."""

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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import ML_INPUT_DIR
from core.feature_expansion import TARGET_FEATURE_COLUMNS, expand_base_features
from core.shared_utils import load_ml_dataset, resolve_output_dir, save_dataframe, write_case_metadata
from drop_rise_classification.dataset_preparation import (
    _coerce_numeric,
    _derive_target_class,
    _infer_base_features,
    _load_metadata_base_features,
)
from drop_rise_classification.settings import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_NEUTRAL_BAND,
    DEFAULT_OUTPUT_ROOT,
    get_case_config,
)

CASE_ID = "case_9"
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


def _screen_features(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    target: pd.Series,
    missing_ratio_threshold: float,
    max_features: int,
) -> tuple[list[str], pd.DataFrame]:
    stats_records: list[dict[str, object]] = []

    feature_data = df.loc[:, features]
    missing_ratios = feature_data.isna().mean(axis=0)
    variance = feature_data.var(axis=0, ddof=1)

    valid_mask = (missing_ratios <= missing_ratio_threshold) & variance.gt(0.0)
    filtered_features = list(feature_data.columns[valid_mask])
    dropped_missing = [col for col in feature_data.columns if missing_ratios[col] > missing_ratio_threshold]
    dropped_zero_var = [col for col in feature_data.columns if variance[col] <= 0.0]

    if not filtered_features:
        raise ValueError("All features removed during missing/variance screening.")

    imputer = SimpleImputer(strategy="median")
    imputed_matrix = imputer.fit_transform(feature_data.loc[:, filtered_features])

    mi_scores = mutual_info_classif(
        imputed_matrix,
        target.to_numpy(dtype=int),
        random_state=0,
    )
    mi_series = pd.Series(mi_scores, index=filtered_features)
    mi_series = mi_series.fillna(0.0)

    nonzero_mask = mi_series.gt(0.0)
    mi_filtered_features = list(mi_series[nonzero_mask].index)
    if not mi_filtered_features:
        mi_filtered_features = [mi_series.idxmax()]
    top_features = mi_series[mi_filtered_features].sort_values(ascending=False).head(max_features)
    selected_features = list(top_features.index)

    stats_records.extend(
        [
            {
                "feature": feature,
                "missing_ratio": float(missing_ratios[feature]),
                "variance": float(variance[feature]),
                "mutual_information": float(mi_series.get(feature, 0.0)),
                "kept_after_screening": feature in selected_features,
                "dropped_missing_ratio": feature in dropped_missing,
                "dropped_zero_variance": feature in dropped_zero_var,
                "dropped_zero_mi": feature in mi_series.index and feature not in mi_filtered_features,
            }
            for feature in feature_data.columns
        ]
    )
    stats_df = pd.DataFrame(stats_records)
    return selected_features, stats_df


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


def _prepare_multiticker_classification_dataset(
    dataset: pd.DataFrame,
    *,
    dataset_path: Path,
    tickers: Sequence[str],
    base_features: Sequence[str] | None,
    extra_columns: Sequence[str] | None,
    neutral_band: float,
    missing_ratio_threshold: float,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    working = dataset.copy()
    working["ticker"] = working["ticker"].astype(str).str.upper()
    if "Date" in working.columns:
        working["Date"] = pd.to_datetime(working["Date"])

    ticker_set = sorted({str(t).upper() for t in tickers})
    working = working[working["ticker"].isin(ticker_set)].copy()
    if working.empty:
        raise ValueError("No rows available for the requested tickers.")

    if base_features is None:
        metadata_features = _load_metadata_base_features(dataset_path)
        if metadata_features:
            base_features = metadata_features
        else:
            base_features = _infer_base_features(working)
    else:
        base_features = list(base_features)
    base_features = [feature for feature in base_features if feature in working.columns]
    if not base_features:
        raise ValueError("No valid base features provided or inferred for multi-ticker classification dataset.")

    enriched = expand_base_features(
        working,
        base_features=base_features,
        ticker_column="ticker",
        date_column="Date",
        close_column="Close",
        target_shift=1,
        target_window=30,
    )
    enriched = enriched.replace([np.inf, -np.inf], pd.NA)
    enriched = enriched.sort_values(["Date", "ticker"]).reset_index(drop=True)

    if "target_1d" not in enriched.columns:
        raise KeyError("Expanded dataset is missing 'target_1d'. Unable to build classification labels.")

    target_feature_columns = [col for col in TARGET_FEATURE_COLUMNS if col in enriched.columns]
    original_columns = set(working.columns)
    derived_feature_columns = [
        col for col in enriched.columns if col not in original_columns and col not in {"target_1d_class"}
    ]

    extra_columns = list(extra_columns or [])
    candidate_columns: list[str] = []
    candidate_columns.extend(base_features)
    candidate_columns.extend(target_feature_columns)
    candidate_columns.extend(derived_feature_columns)
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

    if not feature_columns:
        raise ValueError("No usable numeric features found for multi-ticker classification dataset.")

    protected_targets = {
        TARGET_COLUMN,
        CONTINUOUS_TARGET_COLUMN,
        "target",
        "target_pct",
        "target_pct_1d",
    }
    feature_columns = [column for column in feature_columns if column not in protected_targets]
    if not feature_columns:
        raise ValueError("Protected target columns removed all available features; adjust feature selection inputs.")

    classification_frame = pd.concat(data_columns, axis=1)
    classification_frame[CONTINUOUS_TARGET_COLUMN] = pd.to_numeric(enriched["target_1d"], errors="coerce")
    classification_frame[TARGET_COLUMN] = _derive_target_class(
        classification_frame[CONTINUOUS_TARGET_COLUMN], neutral_band=neutral_band
    )

    filtered_features: list[str] = []
    dropped_features: list[str] = []
    for feature in feature_columns:
        ratio = classification_frame[feature].isna().mean()
        if ratio <= missing_ratio_threshold:
            filtered_features.append(feature)
        else:
            dropped_features.append(feature)

    if not filtered_features:
        raise ValueError("All candidate features exceeded the missing value threshold.")

    if dropped_features:
        print(
            "[WARN] Dropped features due to missing ratio > "
            f"{missing_ratio_threshold:.0%}: {', '.join(sorted(dropped_features))}"
        )

    feature_columns = filtered_features

    mandatory_columns = list(dict.fromkeys(feature_columns + [CONTINUOUS_TARGET_COLUMN]))
    classification_frame = classification_frame.dropna(subset=mandatory_columns)
    if classification_frame.empty:
        raise ValueError("Classification dataset is empty after dropping missing target values.")

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

    return classification_frame, feature_columns, base_features, target_feature_columns, derived_feature_columns


def run_case(
    *,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
    tickers: Sequence[str] | None = None,
    base_features: Sequence[str] | None = None,
    extra_columns: Sequence[str] | None = None,
    neutral_band: float = DEFAULT_NEUTRAL_BAND,
    output_root: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    show_plots: bool = False,
    missing_ratio_threshold: float = 0.35,
    max_screened_features: int = 400,
    enable_distribution_diagnostics: bool = False,
) -> Path:
    """Execute multi-ticker classification with feature screening prior to RFECV."""

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

        available_tickers = sorted(dataset["ticker"].astype(str).str.upper().unique())
        if not available_tickers:
            raise ValueError("Dataset does not contain any tickers.")

        if tickers:
            requested_tickers = [str(t).upper() for t in tickers]
            requested_tickers = list(dict.fromkeys(requested_tickers))
            missing_tickers = [t for t in requested_tickers if t not in available_tickers]
            if missing_tickers:
                missing_fmt = ", ".join(sorted(missing_tickers))
                raise ValueError(f"Requested tickers not present in dataset: {missing_fmt}")
            tickers_to_use = requested_tickers
        elif CASE_CONFIG.tickers:
            tickers_to_use = list(dict.fromkeys(str(t).upper() for t in CASE_CONFIG.tickers))
        else:
            tickers_to_use = available_tickers

        if not tickers_to_use:
            raise ValueError("No tickers selected for multi-ticker classification.")

        classification_df, feature_columns, base_features_used, target_features_used, derived_features_used = (
            _prepare_multiticker_classification_dataset(
                dataset=dataset,
                dataset_path=dataset_path,
                tickers=tickers_to_use,
                base_features=base_features,
                extra_columns=extra_columns,
                neutral_band=neutral_band,
                missing_ratio_threshold=missing_ratio_threshold,
            )
        )

        aggregated_dataset_path = save_dataframe(
            classification_df,
            case_output_dir,
            f"all_tickers_{CASE_ID}_classification_dataset.csv",
        )

        if "Date" in classification_df.columns:
            classification_df = classification_df.sort_values(["Date", "ticker"])
        classification_df = classification_df.reset_index(drop=True)

        label_encoder = LabelEncoder()
        label_encoder.fit(classification_df[TARGET_COLUMN])

        split_index = int(len(classification_df) * (1.0 - test_size))
        if split_index <= 0 or split_index >= len(classification_df):
            raise ValueError("test_size produced an invalid chronological split; adjust the parameter.")

        train_df = classification_df.iloc[:split_index].reset_index(drop=True)
        test_df = classification_df.iloc[split_index:].reset_index(drop=True)

        y_train_encoded = label_encoder.transform(train_df[TARGET_COLUMN])
        y_test_encoded = label_encoder.transform(test_df[TARGET_COLUMN])

        screening_features, screening_stats_df = _screen_features(
            train_df.loc[:, feature_columns],
            features=feature_columns,
            target=pd.Series(y_train_encoded, index=train_df.index, name=TARGET_COLUMN),
            missing_ratio_threshold=missing_ratio_threshold,
            max_features=max_screened_features,
        )

        if len(screening_features) < 2:
            raise ValueError("Too few features remain after screening; adjust thresholds or dataset.")

        screening_stats_path = save_dataframe(
            screening_stats_df,
            diagnostics_dir,
            f"all_tickers_{CASE_ID}_screening_stats.csv",
        )

        X_train = train_df.loc[:, screening_features].to_numpy(dtype=float)
        y_train = y_train_encoded

        if np.unique(y_train).shape[0] < 2:
            raise ValueError("Training split contains fewer than two classes; adjust neutral band or dataset.")

        scaler_for_rfecv = StandardScaler()
        X_train_scaled = scaler_for_rfecv.fit_transform(X_train)

        min_features = 1 if len(screening_features) < 5 else min(10, len(screening_features) - 1)
        class_counts = np.bincount(y_train, minlength=len(label_encoder.classes_))
        nonzero_counts = class_counts[class_counts > 0]
        if nonzero_counts.size == 0:
            raise ValueError("Training split contains no samples after encoding.")
        min_class_count = int(nonzero_counts.min())
        if min_class_count < 2:
            raise ValueError(
                "Training split has fewer than two samples in at least one class; adjust neutral band or dataset."
            )
        cv_splits = min(5, min_class_count, len(X_train) - 1)
        if cv_splits < 2:
            raise ValueError(
                "Not enough samples to perform time-series cross-validation; consider reducing test_size or adjusting the dataset."
            )

        rfecv_estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
        rfecv = RFECV(
            estimator=rfecv_estimator,
            step=1,
            cv=TimeSeriesSplit(n_splits=cv_splits),
            scoring="f1_macro",
            min_features_to_select=min_features,
            n_jobs=None,
        )
        rfecv.fit(X_train_scaled, y_train)

        support_mask = getattr(rfecv, "support_", None)
        if support_mask is None:
            raise AttributeError("RFECV failed to produce a support mask for selected features.")

        selected_features = [feature for feature, keep in zip(screening_features, support_mask) if keep]
        if not selected_features:
            best_index = int(np.argmin(rfecv.ranking_))
            selected_features = [screening_features[best_index]]

        ranking_records = [
            {"feature": feature, "rank": int(rank)}
            for feature, rank in zip(screening_features, getattr(rfecv, "ranking_", np.ones_like(support_mask)))
        ]
        ranking_df = pd.DataFrame(ranking_records)
        ranking_path = save_dataframe(
            ranking_df,
            diagnostics_dir,
            f"all_tickers_{CASE_ID}_rfecv_ranking.csv",
        )

        rfecv_scores = getattr(rfecv, "cv_results_", {}).get("mean_test_score")
        if rfecv_scores is None:
            rfecv_scores = getattr(rfecv, "grid_scores_", [])
        rfecv_scores = np.asarray(rfecv_scores, dtype=float)
        rfecv_scores_df = pd.DataFrame(
            {
                "n_features": np.arange(1, rfecv_scores.shape[0] + 1),
                "mean_cv_score": rfecv_scores,
            }
        )
        rfecv_scores_path = save_dataframe(
            rfecv_scores_df,
            diagnostics_dir,
            f"all_tickers_{CASE_ID}_rfecv_scores.csv",
        )

        selected_features_df = pd.DataFrame({"feature": selected_features})
        selected_features_path = save_dataframe(
            selected_features_df,
            diagnostics_dir,
            f"all_tickers_{CASE_ID}_selected_features.csv",
        )

        estimators = _build_estimators(random_state=random_state)

        X_train_selected = train_df.loc[:, selected_features].to_numpy(dtype=float)
        X_test_selected = test_df.loc[:, selected_features].to_numpy(dtype=float)

        y_train_selected = y_train
        y_test_selected = y_test_encoded

        metrics_records: list[dict[str, object]] = []
        predictions_rows: list[dict[str, object]] = []
        confusion_matrices: dict[str, Path] = {}
        classification_reports: dict[str, Path] = {}

        for model_name, pipeline in estimators.items():
            pipeline.fit(X_train_selected, y_train_selected)
            y_pred = pipeline.predict(X_test_selected)
            y_prob = pipeline.predict_proba(X_test_selected)[:, 1] if hasattr(pipeline, "predict_proba") else None

            metrics_records.append(
                {
                    "model": model_name,
                    "accuracy": float(accuracy_score(y_test_selected, y_pred)),
                    "precision_macro": float(precision_score(y_test_selected, y_pred, average="macro", zero_division=0)),
                    "recall_macro": float(recall_score(y_test_selected, y_pred, average="macro", zero_division=0)),
                    "f1_macro": float(f1_score(y_test_selected, y_pred, average="macro", zero_division=0)),
                }
            )

            confusion = confusion_matrix(y_test_selected, y_pred)
            confusion_df = pd.DataFrame(confusion, index=label_encoder.classes_, columns=label_encoder.classes_)
            confusion_path = save_dataframe(
                confusion_df,
                case_output_dir,
                f"all_tickers_{CASE_ID}_{model_name.lower()}_confusion_matrix.csv",
            )
            confusion_matrices[model_name] = confusion_path

            report = classification_report(
                y_test_selected,
                y_pred,
                target_names=label_encoder.classes_,
                zero_division=0,
                output_dict=True,
            )
            report_df = pd.DataFrame(report).transpose()
            report_path = save_dataframe(
                report_df,
                case_output_dir,
                f"all_tickers_{CASE_ID}_{model_name.lower()}_classification_report.csv",
            )
            classification_reports[model_name] = report_path

            for idx, original_index in enumerate(test_df.index):
                row = {
                    "row_index": int(original_index),
                    "actual": int(y_test_selected[idx]),
                    "predicted": int(y_pred[idx]),
                    "ticker": str(test_df.loc[original_index, "ticker"]),
                }
                if y_prob is not None:
                    row["probability_class_1"] = float(y_prob[idx])
                predictions_rows.append(row)

        metrics_df = pd.DataFrame(metrics_records)
        metrics_path = save_dataframe(
            metrics_df,
            case_output_dir,
            f"all_tickers_{CASE_ID}_metrics.csv",
        )

        prediction_frame = pd.DataFrame(predictions_rows)
        predictions_path = save_dataframe(
            prediction_frame,
            case_output_dir,
            f"all_tickers_{CASE_ID}_test_predictions.csv",
        )

        metadata_extras: dict[str, object] = {
            "neutral_band": neutral_band,
            "screening_missing_ratio_threshold": missing_ratio_threshold,
            "screening_max_features": max_screened_features,
            "rfecv_min_features": min_features,
            "rfecv_cv_splits": cv_splits,
            "rfecv_ranking_path": ranking_path.relative_to(case_output_dir).as_posix(),
            "rfecv_scores_path": rfecv_scores_path.relative_to(case_output_dir).as_posix(),
            "selected_features_path": selected_features_path.relative_to(case_output_dir).as_posix(),
            "screening_stats_path": screening_stats_path.relative_to(case_output_dir).as_posix(),
            "aggregated_dataset_path": aggregated_dataset_path.relative_to(case_output_dir).as_posix(),
            "feature_groups": {
                "base": list(base_features_used),
                "target": list(target_features_used),
                "derived": list(derived_features_used),
            },
            "confusion_matrices": {
                model_name: path.relative_to(case_output_dir).as_posix()
                for model_name, path in confusion_matrices.items()
            },
            "classification_reports": {
                model_name: path.relative_to(case_output_dir).as_posix()
                for model_name, path in classification_reports.items()
            },
            "metrics_path": metrics_path.relative_to(case_output_dir).as_posix(),
            "predictions_path": predictions_path.relative_to(case_output_dir).as_posix(),
        }

        write_case_metadata(
            case_dir=case_output_dir,
            case_id=CASE_ID,
            case_name=CASE_NAME,
            package="drop_rise_classification",
            dataset_path=aggregated_dataset_path,
            tickers=tuple(tickers_to_use),
            features={
                "candidate_features": list(feature_columns),
                "screened_features": screening_features,
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
        "--tickers",
        dest="tickers",
        nargs="+",
        default=None,
        help="Optional list of ticker symbols to include (default: all available).",
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
    parser.add_argument(
        "--show-plots",
        dest="show_plots",
        action="store_true",
        help="Display distribution plots interactively when generating the dataset.",
    )
    parser.add_argument(
        "--missing-ratio-threshold",
        dest="missing_ratio_threshold",
        type=float,
        default=0.35,
        help="Maximum allowed fraction of missing values retained during screening.",
    )
    parser.add_argument(
        "--max-screened-features",
        dest="max_screened_features",
        type=int,
        default=400,
        help="Maximum number of features retained after mutual information screening.",
    )
    parser.add_argument(
        "--enable-distribution-diagnostics",
        dest="enable_distribution_diagnostics",
        action="store_true",
        help="Generate distribution plots and summaries during dataset preparation.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI convenience
    args = _parse_args()
    output_root = Path(args.output_root) if args.output_root else None

    run_case(
        dataset_filename=args.dataset_filename,
        tickers=args.tickers,
        base_features=args.base_features,
        extra_columns=args.extra_columns,
        neutral_band=args.neutral_band,
        output_root=output_root,
        test_size=args.test_size,
        random_state=args.random_state,
        show_plots=args.show_plots,
        missing_ratio_threshold=args.missing_ratio_threshold,
        max_screened_features=args.max_screened_features,
        enable_distribution_diagnostics=args.enable_distribution_diagnostics,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
