"""Utilities for validating short-term predictions across tickers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class PredictionValidationResult:
    """Artifacts produced by the last-N-day validation helper."""

    dataset_label: str
    model_metrics: pd.DataFrame
    per_ticker_metrics: pd.DataFrame
    predictions: pd.DataFrame
    plot_paths: dict[str, Path | None]


def _sanitize_datetime(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    return pd.to_datetime(series, errors="coerce")


def _split_last_n_days(
    metadata: pd.DataFrame,
    *,
    ticker_column: str,
    date_column: str,
    n_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    if ticker_column not in metadata.columns:
        raise KeyError(f"Column '{ticker_column}' not found in metadata frame")
    if date_column not in metadata.columns:
        raise KeyError(f"Column '{date_column}' not found in metadata frame")

    working = metadata.copy()
    working.loc[:, date_column] = _sanitize_datetime(working[date_column])
    if working[date_column].isna().any():
        raise ValueError(
            "Date column contains values that could not be parsed to datetime for last-N-day split",
        )

    working = working.sort_values([ticker_column, date_column])

    train_segments: list[np.ndarray] = []
    test_segments: list[np.ndarray] = []

    for _, group in working.groupby(ticker_column, sort=False):
        indices = group.index.to_numpy()
        if len(indices) <= n_days:
            train_segments.append(indices)
            continue
        train_segments.append(indices[:-n_days])
        test_segments.append(indices[-n_days:])

    train_idx = (
        np.concatenate(train_segments)
        if train_segments
        else np.empty(0, dtype=metadata.index.dtype)
    )
    test_idx = (
        np.concatenate(test_segments)
        if test_segments
        else np.empty(0, dtype=metadata.index.dtype)
    )
    return train_idx, test_idx


def _sanitize_identifier(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", label.strip().lower())
    cleaned = cleaned.strip("._")
    return cleaned or "model"


def _apply_preprocessor(preproc: Any | None, X: pd.DataFrame, *, fit: bool) -> Any:
    if preproc is None:
        return X
    if fit:
        transformed = preproc.fit_transform(X)
    else:
        transformed = preproc.transform(X)
    if isinstance(transformed, pd.DataFrame):
        return transformed
    if isinstance(X, pd.DataFrame):
        if hasattr(preproc, "get_feature_names_out"):
            try:
                columns = preproc.get_feature_names_out(input_features=X.columns)
            except TypeError:
                columns = preproc.get_feature_names_out()
        else:
            columns = X.columns
        return pd.DataFrame(transformed, columns=columns, index=X.index)
    return transformed


def _to_numpy(data: Any, feature_order: list[str] | None) -> Any:
    if isinstance(data, pd.DataFrame):
        if feature_order is not None:
            missing = [col for col in feature_order if col not in data.columns]
            if missing:
                raise KeyError(f"Missing columns in feature matrix: {missing}")
            data = data.loc[:, feature_order]
        return data.to_numpy()
    return data


def _absolute_percentage_errors(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    """Return absolute percentage errors (in percent) excluding near-zero actuals."""

    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    denom = np.abs(true_arr)
    mask = denom > epsilon
    if not np.any(mask):
        return np.empty(0, dtype=float)
    return np.abs((true_arr[mask] - pred_arr[mask]) / denom[mask]) * 100.0



def _mean_absolute_percentage_error(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, *, epsilon: float = 1e-8) -> float:
    """Return the mean absolute percentage error in percent, ignoring near-zero actuals."""

    errors = _absolute_percentage_errors(y_true, y_pred, epsilon=epsilon)
    if errors.size == 0:
        return float('nan')
    return float(np.mean(errors))



def _median_absolute_percentage_error(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, *, epsilon: float = 1e-8) -> float:
    """Return the median absolute percentage error in percent, ignoring near-zero actuals."""

    errors = _absolute_percentage_errors(y_true, y_pred, epsilon=epsilon)
    if errors.size == 0:
        return float('nan')
    return float(np.median(errors))


def _compute_per_ticker_metrics(
    frame: pd.DataFrame,
    *,
    ticker_column: str,
    true_col: str,
    pred_col: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for ticker, group in frame.groupby(ticker_column, sort=False):
        mae = mean_absolute_error(group[true_col], group[pred_col])
        rmse = math.sqrt(mean_squared_error(group[true_col], group[pred_col]))
        mape = _mean_absolute_percentage_error(group[true_col], group[pred_col])
        median_mape = _median_absolute_percentage_error(group[true_col], group[pred_col])
        records.append({
            ticker_column: ticker,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Median_MAPE": median_mape,
            "n_samples": len(group),
        })
    return pd.DataFrame(records)


def _plot_predictions_grid(
    pred_frame: pd.DataFrame,
    *,
    ticker_column: str,
    date_column: str,
    true_col: str,
    pred_col: str,
    title: str,
    n_cols: int,
    output_dir: Path | None,
    filename_prefix: str,
    show_plot: bool,
    n_days: int,
) -> Path | None:
    if pred_frame.empty:
        return None

    tickers = pred_frame[ticker_column].unique()
    n_tickers = len(tickers)
    n_cols = max(1, min(n_cols, n_tickers))
    n_rows = math.ceil(n_tickers / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharey=True,
    )
    axes_arr = np.atleast_1d(axes).ravel()

    for idx, ticker in enumerate(tickers):
        ax = axes_arr[idx]
        subset = pred_frame[pred_frame[ticker_column] == ticker].sort_values(date_column)
        ax.plot(subset[date_column], subset[true_col], marker="o", label="Actual")
        ax.plot(subset[date_column], subset[pred_col], marker="x", label="Prediction")
        ax.set_title(str(ticker))
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))

    for extra_ax in axes_arr[n_tickers:]:
        fig.delaxes(extra_ax)

    fig.suptitle(title, fontsize=16)
    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{filename_prefix}_last_{n_days}_days.png"
        plot_path = output_dir / filename
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(fig)
    return plot_path


def evaluate_last_n_day_predictions(
    model_dict: Mapping[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    *,
    metadata: pd.DataFrame,
    dataset_label: str,
    n_days: int = 5,
    ticker_column: str = "ticker",
    date_column: str = "Date",
    output_dir: Path | str | None = None,
    plot_prefix: str | None = None,
    n_plot_columns: int = 3,
    show_plot: bool = False,
    preprocessor: Any | None = None,
    model_store_dir: Path | str | None = None,
) -> PredictionValidationResult:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be provided as a pandas DataFrame to preserve column names")

    target_series = target.copy()
    if not isinstance(target_series, pd.Series):
        target_series = pd.Series(target, index=features.index, name=getattr(target, "name", "target"))

    meta = metadata.loc[features.index, [ticker_column, date_column]].copy()
    valid_idx = meta[ticker_column].notna() & meta[date_column].notna()
    if not valid_idx.all():
        meta = meta.loc[valid_idx]
        features = features.loc[meta.index]
        target_series = target_series.loc[meta.index]

    if not len(features):
        raise ValueError("No rows available after aligning features with metadata for prediction validation")

    train_idx, test_idx = _split_last_n_days(
        meta,
        ticker_column=ticker_column,
        date_column=date_column,
        n_days=n_days,
    )

    if len(test_idx) == 0:
        raise ValueError(
            "Last-N-day validation requires at least one ticker with more than n_days observations",
        )

    X_train = features.loc[train_idx]
    y_train = target_series.loc[train_idx]
    X_test = features.loc[test_idx]
    y_test = target_series.loc[test_idx]

    model_records: list[dict[str, Any]] = []
    ticker_records: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    plot_paths: dict[str, Path | None] = {}

    output_path_obj = Path(output_dir) if output_dir is not None else None
    model_store_path = Path(model_store_dir) if model_store_dir is not None else None
    if model_store_path is not None:
        model_store_path.mkdir(parents=True, exist_ok=True)

    for model_name, estimator in model_dict.items():
        print(f"Running last-{n_days}-day validation for model: {model_name}")

        bundle = None
        model_path = None
        preproc_fitted: Any | None = None
        feature_order: list[str] | None = None

        if model_store_path is not None:
            identifier = f"{_sanitize_identifier(dataset_label)}__{_sanitize_identifier(model_name)}.joblib"
            model_path = model_store_path / identifier
            if model_path.exists():
                bundle = joblib.load(model_path)
                model = bundle.get("model")
                preproc_fitted = bundle.get("preprocessor")
                feature_order = bundle.get("feature_order")
            else:
                model = clone(estimator)
        else:
            model = clone(estimator)

        if bundle is not None:
            if preproc_fitted is not None:
                X_train_proc = _apply_preprocessor(preproc_fitted, X_train, fit=False)
                X_test_proc = _apply_preprocessor(preproc_fitted, X_test, fit=False)
            else:
                X_train_proc = X_train
                X_test_proc = X_test
        else:
            preproc_fitted = clone(preprocessor) if preprocessor is not None else None
            if preproc_fitted is not None:
                X_train_proc = _apply_preprocessor(preproc_fitted, X_train, fit=True)
                X_test_proc = _apply_preprocessor(preproc_fitted, X_test, fit=False)
            else:
                X_train_proc = X_train
                X_test_proc = X_test

        if feature_order is None:
            if isinstance(X_train_proc, pd.DataFrame):
                feature_order = list(X_train_proc.columns)
            elif isinstance(X_train, pd.DataFrame):
                feature_order = list(X_train.columns)

        X_train_input = _to_numpy(X_train_proc, feature_order)
        X_test_input = _to_numpy(X_test_proc, feature_order)

        if bundle is None:
            model.fit(X_train_input, y_train)
            if model_path is not None:
                joblib.dump(
                    {
                        "model": model,
                        "preprocessor": preproc_fitted,
                        "feature_order": feature_order,
                    },
                    model_path,
                )
        predictions = pd.Series(model.predict(X_test_input), index=y_test.index, name="prediction")

        mae = mean_absolute_error(y_test, predictions)
        rmse = math.sqrt(mean_squared_error(y_test, predictions))
        mape = _mean_absolute_percentage_error(y_test, predictions)
        median_mape = _median_absolute_percentage_error(y_test, predictions)
        model_records.append(
            {
                "dataset": dataset_label,
                "model": model_name,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "Median_MAPE": median_mape,
            }
        )

        pred_frame = meta.loc[y_test.index, [ticker_column, date_column]].copy()
        pred_frame["actual"] = y_test.values
        pred_frame["prediction"] = predictions.values
        pred_frame["model"] = model_name
        pred_frame[date_column] = _sanitize_datetime(pred_frame[date_column])
        prediction_frames.append(pred_frame)

        per_ticker = _compute_per_ticker_metrics(
            pred_frame,
            ticker_column=ticker_column,
            true_col="actual",
            pred_col="prediction",
        )
        per_ticker.insert(0, "model", model_name)
        per_ticker.insert(0, "dataset", dataset_label)
        ticker_records.append(per_ticker)

        plot_title = f"{dataset_label} - {model_name} (last {n_days} days)"
        prefix = plot_prefix or dataset_label.replace(" ", "_").lower()
        model_prefix = f"{prefix}_{model_name.replace(' ', '_').lower()}"
        plot_paths[model_name] = _plot_predictions_grid(
            pred_frame,
            ticker_column=ticker_column,
            date_column=date_column,
            true_col="actual",
            pred_col="prediction",
            title=plot_title,
            n_cols=n_plot_columns,
            output_dir=output_path_obj,
            filename_prefix=model_prefix,
            show_plot=show_plot,
            n_days=n_days,
        )

    combined_predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    combined_metrics = pd.DataFrame(model_records)
    combined_ticker_metrics = (
        pd.concat(ticker_records, ignore_index=True)
        if ticker_records
        else pd.DataFrame()
    )

    return PredictionValidationResult(
        dataset_label=dataset_label,
        model_metrics=combined_metrics,
        per_ticker_metrics=combined_ticker_metrics,
        predictions=combined_predictions,
        plot_paths=plot_paths,
    )


__all__ = ["PredictionValidationResult", "evaluate_last_n_day_predictions"]
