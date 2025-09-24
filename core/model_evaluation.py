"""Model evaluation utilities (learning curves, diagnostics)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve

from .visualization import plot_features_distribution_grid


def _sanitize_filename(label: str, suffix: str) -> str:
    base = re.sub(r"[^\w.-]+", "_", label.strip().lower()) or "learning_curves"
    base = base.strip("._")
    suffix = suffix.strip("._")
    if suffix:
        return f"{base}_{suffix}.png"
    return f"{base}.png"


def analyze_learning_curve(
    df_curve: pd.DataFrame,
    model_name: str,
    *,
    scoring: str = "MAE",
) -> pd.DataFrame:
    """Provide a simple diagnostic summary for a learning curve."""

    if df_curve.empty:
        return pd.DataFrame(
            {
                "model": [model_name],
                "final_train_mae": [np.nan],
                "final_val_mae": [np.nan],
                "score_gap": [np.nan],
                "pattern": ["insufficient data"],
            }
        )

    train_final = float(df_curve["train_error"].iloc[-1])
    val_final = float(df_curve["val_error"].iloc[-1])
    gap = val_final - train_final

    tolerance = max(0.05 * max(abs(train_final), abs(val_final), 1e-9), 0.01)

    if gap > tolerance:
        pattern = "overfitting"
    elif gap < -tolerance:
        pattern = "unclear"
    else:
        avg_error = (train_final + val_final) / 2.0
        if avg_error > 1.0:
            pattern = "underfitting"
        else:
            pattern = "good generalization"

    return pd.DataFrame(
        {
            "model": [model_name],
            "final_train_mae": [train_final],
            "final_val_mae": [val_final],
            "score_gap": [gap],
            "pattern": [pattern],
            "scoring": [scoring],
        }
    )


def plot_learning_curves_for_models(
    model_dict: Dict[str, object],
    X: pd.DataFrame | np.ndarray,
    y: Iterable,
    *,
    main_title: str = "",
    train_sizes: Iterable[float] | None = None,
    cv: int = 5,
    scoring: str = "neg_mean_absolute_error",
    n_cols: int = 3,
    output_dir: Path | str | None = None,
    show: bool = False,
    feature_distribution: pd.DataFrame | None = None,
    original_feature_distribution: pd.DataFrame | None = None,
    distribution_title: str | None = None,
    distribution_output_dir: Path | str | None = None,
    n_features_per_row: int = 3,
) -> Tuple[pd.DataFrame, Optional[Path], Optional[Path]]:
    """Plot learning curves for several models and optionally feature distributions."""

    if not model_dict:
        raise ValueError("model_dict must contain at least one estimator")

    X = np.asarray(X)
    y = np.asarray(list(y))

    train_sizes = train_sizes or np.linspace(0.1, 1.0, 8)

    n_models = len(model_dict)
    n_cols = max(1, n_cols)
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes_arr = np.atleast_1d(axes).ravel()

    title_text = main_title or "Learning curves"
    fig.suptitle(title_text, fontsize=16)

    results_frames = []

    for idx, (model_name, model_obj) in enumerate(model_dict.items()):
        print(f"--- {model_name} ---")
        train_sizes_arr, train_scores, val_scores = learning_curve(
            model_obj,
            X,
            y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
        )

        df_curve = pd.DataFrame(
            {
                "train_size": train_sizes_arr,
                "train_error": -train_scores.mean(axis=1),
                "val_error": -val_scores.mean(axis=1),
            }
        )

        summary = analyze_learning_curve(df_curve, model_name=model_name)
        results_frames.append(summary)
        pattern = summary["pattern"].iloc[0]

        ax = axes_arr[idx]
        ax.plot(df_curve["train_size"], df_curve["train_error"], label="Train error", marker="o")
        ax.plot(df_curve["train_size"], df_curve["val_error"], label="Validation error", marker="o")
        ax.set_title(f"{model_name} ({pattern})", fontsize=11)
        ax.set_xlabel("Train size")
        ax.set_ylabel("MAE")
        ax.legend(fontsize=8)

    for extra_ax in axes_arr[len(model_dict) :]:
        fig.delaxes(extra_ax)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = _sanitize_filename(title_text or "learning_curves", "grid")
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=300, bbox_inches="tight")


    plt.close(fig)

    distribution_path = None
    if feature_distribution is not None:
        distribution_title = distribution_title or f"Feature distribution: {title_text}"
        distribution_path = plot_features_distribution_grid(
            feature_distribution,
            original_feature_distribution if original_feature_distribution is not None else feature_distribution,
            title=distribution_title,
            n_features_per_row=n_features_per_row,
            output_dir=distribution_output_dir or output_dir,
        )

    results = pd.concat(results_frames, ignore_index=True) if results_frames else pd.DataFrame()
    return results, output_path, distribution_path

