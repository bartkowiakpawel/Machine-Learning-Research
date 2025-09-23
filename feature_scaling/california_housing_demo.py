"""Demonstration of feature scaling on the California Housing dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PowerTransformer

from core.visualization import plot_features_distribution_grid

_OUTPUT_DIR = Path("feature_scaling") / "outputs"
_MODEL_FILENAME = "linear_regression_california.joblib"


def _display_dataframe(
    df: pd.DataFrame,
    label: str,
    formatter: Callable[[pd.DataFrame], None] | None = None,
) -> None:
    """Display a DataFrame head with fallback for non-notebook environments."""

    print(f"\n{label}")
    print(f"Shape: {df.shape}")

    if formatter is not None:
        formatter(df)
        return

    try:
        from IPython.display import display  # type: ignore

        display(df.head())
    except ImportError:
        print(df.head())


def _sample_train_test(
    df: pd.DataFrame,
    target_column: str,
    sample_size: int,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split the DataFrame into train/test sets using a simple random sample."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    if sample_size >= len(df):
        raise ValueError("sample_size must be smaller than the dataframe length")

    rng = np.random.default_rng(random_seed)
    test_indices = rng.choice(df.index.to_numpy(), size=sample_size, replace=False)

    df_test = df.loc[test_indices]
    df_train = df.drop(test_indices)

    x_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    x_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]

    return x_train, y_train, x_test, y_test


def _save_prediction_bar_chart(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    title: str,
    filename_suffix: str,
) -> Path:
    """Create a bar chart comparing predictions with targets and save to PNG."""

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_true_array = np.asarray(list(y_true))
    y_pred_array = np.asarray(list(y_pred))

    labels = [f"Sample {i + 1}" for i in range(len(y_true_array))]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, y_true_array, width, label="Real Value")
    rects2 = ax.bar(x + width / 2, y_pred_array, width, label="Model Prediction")

    ax.set_ylabel("MedHouseVal")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    plt.tight_layout()

    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    filename = f"california_housing_predictions_{filename_suffix}.png"
    plot_path = _OUTPUT_DIR / filename
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _train_and_evaluate(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    title: str,
    filename_suffix: str,
    sample_size: int = 10,
    random_seed: int | None = None,
) -> None:
    """Train Linear Regression, persist the model, and report metrics."""

    x_train, y_train, x_test, y_test = _sample_train_test(
        pd.concat([features, target], axis=1),
        target.name,
        sample_size,
        random_seed,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _OUTPUT_DIR / _MODEL_FILENAME
    joblib.dump(model, model_path)

    del model
    model_loaded = joblib.load(model_path)

    y_pred = model_loaded.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"--- Research result: {title} ---\n")
    print(f"Model saved to: {model_path}")
    print(
        f"MAE score for {len(y_test)} random unseen records "
        f"(loaded model - joblib): {mae:.3f}"
    )

    differences = y_test.to_numpy() - y_pred
    abs_percentage_errors = np.abs(differences / y_test.to_numpy()) * 100

    for i, diff in enumerate(differences):
        print(
            f"Sample {i + 1:2}: "
            f"Real value = {y_test.to_numpy()[i]:.3f}, "
            f"Prediction = {y_pred[i]:.3f}, "
            f"Difference = {diff:.3f}, "
            f"MAPE = {abs_percentage_errors[i]:.3f}%"
        )

    mean_abs_diff = float(np.mean(np.abs(differences)))
    mean_mape = float(np.mean(abs_percentage_errors))

    print(f"\nAverage absolute difference (MAE) calculated manually: {mean_abs_diff:.3f}")
    print(f"Average MAPE: {mean_mape:.3f}%\n")

    plot_path = _save_prediction_bar_chart(
        y_test,
        y_pred,
        title="Prediction vs actual values for 10 random unseen records",
        filename_suffix=filename_suffix,
    )
    print(f"Prediction comparison chart saved to: {plot_path}\n")


def run_demo(n_features_per_row: int = 3) -> None:
    """Fetch the California Housing dataset and visualise Yeo-Johnson scaling."""

    data = fetch_california_housing(as_frame=True)
    original_frame = data.frame

    _display_dataframe(original_frame, "Original data (first rows)")

    features = original_frame.drop(columns=["MedHouseVal"])
    target = original_frame["MedHouseVal"]

    transformer = PowerTransformer(method="yeo-johnson")
    transformed_features = transformer.fit_transform(features)
    transformed_frame = pd.DataFrame(transformed_features, columns=features.columns)

    _display_dataframe(
        transformed_frame,
        "Data after Yeo-Johnson transformation (first rows)",
    )

    plot_path = plot_features_distribution_grid(
        transformed_frame,
        features,
        title="Yeo-Johnson applied for California Housing set",
        n_features_per_row=n_features_per_row,
    )
    print(f"Feature distribution chart saved to: {plot_path}")

    _train_and_evaluate(
        features,
        target,
        title="not transformed data",
        filename_suffix="original",
    )

    transformed_with_target = transformed_frame.copy()
    transformed_with_target.loc[:, "MedHouseVal"] = target.values

    _train_and_evaluate(
        transformed_with_target.drop(columns=["MedHouseVal"]),
        transformed_with_target["MedHouseVal"],
        title="transformed data with Yeo-Johnson",
        filename_suffix="yeo_johnson",
    )


if __name__ == "__main__":
    run_demo()
