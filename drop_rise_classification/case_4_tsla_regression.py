"""Case 4: TSLA target regression with RFECV-driven feature selection."""

from __future__ import annotations

import argparse
import ctypes
import math
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from config import ML_INPUT_DIR
from core.shared_utils import resolve_output_dir, save_dataframe, write_case_metadata
from drop_rise_classification.settings import (
    DEFAULT_NEUTRAL_BAND,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TICKER,
    get_case_config,
)

CASE_ID = "case_4"
CASE_CONFIG = get_case_config(CASE_ID)
CASE_NAME = CASE_CONFIG.name
TARGET_COLUMN = "target_1d"
DEFAULT_DATASET_PATH = Path("drop_rise_classification") / "outputs" / "case_3" / "tsla_case_3_feature_enriched_dataset.csv"
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None  # type: ignore[assignment]


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


def _resolve_dataset_path(dataset_arg: str | None) -> Path:
    if dataset_arg:
        candidate = Path(dataset_arg)
        if candidate.exists():
            return candidate.resolve()
        fallback = ML_INPUT_DIR / dataset_arg
        if fallback.exists():
            return fallback.resolve()
    default_path = DEFAULT_DATASET_PATH
    if default_path.exists():
        return default_path.resolve()
    raise FileNotFoundError(
        "Unable to locate the regression dataset. Provide --dataset pointing to the Case 3 feature enriched CSV."
    )


def _select_numeric_features(df: pd.DataFrame, excludes: Sequence[str]) -> list[str]:
    excludes_set = {col.lower() for col in excludes}
    features: list[str] = []
    for column in df.columns:
        if column.lower() in excludes_set:
            continue
        series = df[column]
        if ptypes.is_numeric_dtype(series):
            features.append(column)
    return features


def _build_estimators(random_state: int) -> Mapping[str, Pipeline]:
    estimators: dict[str, Pipeline] = {
        "GradientBoosting": Pipeline(
            [
                ("model", GradientBoostingRegressor(random_state=random_state)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=600,
                        max_depth=None,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "ElasticNet": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "model",
                    ElasticNet(
                        alpha=0.001,
                        l1_ratio=0.5,
                        max_iter=10000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    if XGBRegressor is not None:
        estimators["XGBoost"] = Pipeline(
            [
                (
                    "model",
                    XGBRegressor(
                        n_estimators=600,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        max_depth=5,
                        random_state=random_state,
                        objective="reg:squarederror",
                        tree_method="hist",
                    ),
                ),
            ]
        )
    return estimators


def run_case(
    *,
    dataset: str | None = None,
    ticker: str = DEFAULT_TICKER,
    neutral_band: float = DEFAULT_NEUTRAL_BAND,
    output_root: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Path:
    """Execute TSLA regression workflow with RFECV and multi-model benchmarking."""

    release_display = _acquire_display_awake()
    if not 0.0 < test_size < 0.5:
        raise ValueError("test_size should be between 0 and 0.5 for stratified-like splits.")

    try:
        dataset_path = _resolve_dataset_path(dataset)
        regression_df = pd.read_csv(dataset_path)
        if regression_df.empty:
            raise ValueError("Regression dataset is empty.")
        if "Date" in regression_df.columns:
            regression_df["Date"] = pd.to_datetime(regression_df["Date"])
        regression_df["ticker"] = regression_df["ticker"].astype(str).str.upper()

        ticker = str(ticker).strip().upper() or DEFAULT_TICKER
        subset = regression_df[regression_df["ticker"] == ticker].copy()
        if subset.empty:
            raise ValueError(f"No rows available for ticker {ticker} in regression dataset.")

        if TARGET_COLUMN not in subset.columns:
            raise KeyError(f"Regression dataset missing target column '{TARGET_COLUMN}'.")

        subset = subset.replace([np.inf, -np.inf], np.nan)
        subset = subset.dropna(subset=[TARGET_COLUMN])

        excludes = ["Date", "ticker", TARGET_COLUMN]
        feature_columns = _select_numeric_features(subset, excludes)
        if not feature_columns:
            raise ValueError("No numeric features found for regression modelling.")

        subset = subset.reset_index(drop=True)
        train_df, test_df = train_test_split(
            subset,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        X_train_raw = train_df.loc[:, feature_columns]
        non_missing_mask = X_train_raw.notna().any(axis=0)
        keep_index = non_missing_mask[non_missing_mask].index
        drop_index = non_missing_mask[~non_missing_mask].index
        X_train_raw = X_train_raw.loc[:, keep_index]
        usable_feature_columns = list(X_train_raw.columns)
        dropped_all_nan_features = list(drop_index)
        if not usable_feature_columns:
            raise ValueError("All candidate features are entirely missing in the training split.")
        X_test_raw = test_df.loc[:, usable_feature_columns]

        imputer = SimpleImputer(strategy="median")

        X_train_array = imputer.fit_transform(X_train_raw)
        X_test_array = imputer.transform(X_test_raw)
        X_train = pd.DataFrame(
            X_train_array,
            columns=usable_feature_columns,
            index=X_train_raw.index,
        )
        X_test = pd.DataFrame(
            X_test_array,
            columns=usable_feature_columns,
            index=X_test_raw.index,
        )

        y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
        y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

        if np.isclose(y_train.std(), 0.0):
            raise ValueError("Training target has near-zero variance; regression is ill-posed.")

        case_output_dir = resolve_output_dir(CASE_ID, DEFAULT_OUTPUT_ROOT, output_root)
        diagnostics_dir = case_output_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        rfecv_estimator = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
        min_features = max(1, min(50, int(len(feature_columns) * 0.25)))
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        rfecv = RFECV(
            estimator=rfecv_estimator,
            step=1,
            cv=cv,
            scoring="neg_mean_absolute_error",
            min_features_to_select=min_features,
            n_jobs=-1,
        )
        X_train_matrix = X_train.to_numpy(dtype=float)
        rfecv.fit(X_train_matrix, y_train)

        selected_mask = rfecv.support_
        selected_features = [usable_feature_columns[idx] for idx, keep in enumerate(selected_mask) if keep]
        if not selected_features:
            best_idx = int(np.argmin(rfecv.ranking_))
            selected_features = [usable_feature_columns[best_idx]]

        ranking_records = [
            {
                "feature": feature,
                "selected": bool(selected_mask[idx]),
                "rank": int(rfecv.ranking_[idx]),
            }
            for idx, feature in enumerate(usable_feature_columns)
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
        except Exception:  # pragma: no cover - fallback guard
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

        selected_train = X_train.loc[:, selected_features].to_numpy(dtype=float)
        selected_test = X_test.loc[:, selected_features].to_numpy(dtype=float)

        estimators = _build_estimators(random_state)
        metrics_records: list[dict[str, object]] = []
        predictions_frame = test_df[["Date", "ticker"]].copy()
        predictions_frame[f"actual_{TARGET_COLUMN}"] = y_test

        for name, pipeline in estimators.items():
            fitted = pipeline.fit(selected_train, y_train)
            y_pred = fitted.predict(selected_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            metrics_records.append(
                {
                    "estimator": name,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
            )
            predictions_frame[f"pred_{name.lower()}"] = y_pred

        metrics_df = pd.DataFrame(metrics_records).sort_values("mae", ascending=True)
        metrics_path = save_dataframe(
            metrics_df,
            case_output_dir,
            f"{ticker.lower()}_{CASE_ID}_model_metrics.csv",
        )

        predictions_path = save_dataframe(
            predictions_frame,
            case_output_dir,
            f"{ticker.lower()}_{CASE_ID}_test_predictions.csv",
        )

        selected_features_path = save_dataframe(
            pd.DataFrame({"feature": selected_features}),
            diagnostics_dir,
            f"{ticker.lower()}_{CASE_ID}_selected_features.csv",
        )

        metadata_extras: dict[str, object] = {
            "neutral_band": neutral_band,
            "rfecv_optimal_features": int(getattr(rfecv, "n_features_", len(selected_features))),
            "rfecv_ranking_csv": ranking_path.relative_to(case_output_dir).as_posix(),
            "rfecv_scores_csv": rfecv_scores_path.relative_to(case_output_dir).as_posix(),
            "selected_features_csv": selected_features_path.relative_to(case_output_dir).as_posix(),
            "metrics_csv": metrics_path.relative_to(case_output_dir).as_posix(),
            "predictions_csv": predictions_path.relative_to(case_output_dir).as_posix(),
            "selected_features": selected_features,
            "imputation_strategy": "median",
            "labelled_target": TARGET_COLUMN,
            "dropped_all_nan_features": dropped_all_nan_features,
        }

        write_case_metadata(
            case_dir=case_output_dir,
            case_id=CASE_ID,
            case_name=CASE_NAME,
            package="drop_rise_classification",
            dataset_path=dataset_path,
            tickers=[ticker],
            features={
                "candidate_features": usable_feature_columns,
                "selected_features": selected_features,
                "dropped_all_nan_features": dropped_all_nan_features,
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
        dest="dataset",
        default=None,
        help="Path to the regression dataset CSV (default: Case 3 enriched dataset).",
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
        help="Retained for metadata continuity with classification cases.",
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
        dataset=args.dataset,
        ticker=args.ticker,
        neutral_band=args.neutral_band,
        output_root=output_root,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
