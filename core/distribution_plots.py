"""Reusable plotting helpers for feature distribution diagnostics."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

try:
    import ptitprince as pt
except ImportError:  # pragma: no cover - optional dependency safeguard
    pt = None


def compute_distribution_summary(series: pd.Series) -> dict[str, float]:
    """Return key descriptive statistics for the supplied numeric series."""

    clean = pd.to_numeric(series.dropna(), errors="coerce")
    clean = clean.astype(float).dropna()
    if clean.empty:
        raise ValueError("compute_distribution_summary requires at least one non-null value.")

    quantiles = np.percentile(clean.to_numpy(dtype=float), [25, 50, 75])
    q1, median, q3 = map(float, quantiles)
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)),
        "min": float(clean.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(clean.max()),
        "iqr": float(q3 - q1),
        "skew": float(clean.skew()),
        "kurtosis": float(clean.kurt()),
    }


def plot_distribution_grid(
    series: pd.Series,
    *,
    ticker: str,
    feature: str,
    case_id: str,
    case_name: str,
    output_dir: Path,
    summary: Mapping[str, float] | None = None,
    show: bool = False,
    dpi: int = 300,
) -> Path:
    """Create a 2x3 grid of distribution diagnostics for the given series."""

    clean = series.dropna()
    clean = pd.to_numeric(clean, errors="coerce").astype(float).dropna()
    if clean.empty:
        raise ValueError("plot_distribution_grid requires at least one non-null value.")

    stats_summary = dict(summary) if summary is not None else compute_distribution_summary(clean)
    sns.set_theme(style="whitegrid")

    fig, axes_matrix = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes_matrix.ravel()

    median = stats_summary["median"]
    q1 = stats_summary["q1"]
    q3 = stats_summary["q3"]
    iqr = stats_summary["iqr"]
    skew_value = stats_summary["skew"]

    # Boxplot
    sns.boxplot(y=clean, color="#86c5da", ax=axes[0])
    axes[0].set_title("Boxplot")
    axes[0].set_ylabel("Value")
    axes[0].axhline(median, ls="--", lw=1.1, color="tab:orange", label=f"median={median:.4f}")
    axes[0].legend(loc="upper right")

    # Histogram
    bins = max(10, int(math.sqrt(stats_summary["count"])))
    sns.histplot(clean, bins=bins, kde=False, ax=axes[1], color="#5d9fc7")
    axes[1].set_title("Histogram")
    axes[1].set_xlabel("Value")
    axes[1].axvline(median, ls="--", lw=1.1, color="tab:orange")

    # Violin plot
    sns.violinplot(y=clean, inner="quartile", color="#b3a2c8", ax=axes[2])
    axes[2].set_title("Violin Plot")
    axes[2].set_ylabel("Value")

    # RainCloud plot
    rain_ax = axes[3]
    rain_ax.set_title("RainCloud Plot")
    rain_ax.set_ylabel("Value")
    if pt is not None:
        pt.RainCloud(y=clean, bw=0.2, width_viol=0.6, ax=rain_ax, move=0.0, pointplot=False, alpha=0.85)
    else:
        rain_ax.text(
            0.5,
            0.5,
            "ptitprince not installed\nRainCloud skipped",
            transform=rain_ax.transAxes,
            ha="center",
            va="center",
        )
        rain_ax.set_yticks([])

    # KDE plot
    kde_ax = axes[4]
    if clean.nunique(dropna=True) > 1 and not np.isclose(clean.var(ddof=0), 0.0):
        sns.kdeplot(clean, fill=True, color="#6baed6", alpha=0.6, ax=kde_ax, warn_singular=False)
        kde_ax.set_title("KDE (Density)")
        kde_ax.set_xlabel("Value")
        kde_ax.axvline(median, ls="--", lw=1.1, color="tab:orange")
    else:
        kde_ax.set_title("KDE (Density)")
        kde_ax.set_xlabel("Value")
        kde_ax.text(0.5, 0.5, "Variance ~ 0\nKDE skipped", transform=kde_ax.transAxes, ha="center", va="center")
        kde_ax.set_yticks([])
        kde_ax.set_xlim(clean.min(), clean.max() if clean.max() != clean.min() else clean.min() + 1)

    # QQ plot
    stats.probplot(clean, dist="norm", plot=axes[5])
    axes[5].set_title("QQ Plot")

    suptitle = (
        f"{case_name}\n"
        f"{ticker} | {feature} | Q1={q1:.4f} | Q3={q3:.4f} | IQR={iqr:.4f} | "
        f"median={median:.4f} | skew={skew_value:.3f}"
    )
    fig.suptitle(suptitle, fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    safe_ticker = "".join(ch if ch.isalnum() else "_" for ch in ticker).strip("_") or "ticker"
    safe_feature = "".join(ch if ch.isalnum() else "_" for ch in feature).strip("_") or "feature"

    max_token_length = 64
    def _truncate_token(token: str) -> str:
        if len(token) <= max_token_length:
            return token
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:10]
        trimmed = token[: max_token_length - 12].rstrip("_")
        return f"{trimmed}_{digest}"

    safe_feature = _truncate_token(safe_feature.lower())
    safe_ticker = _truncate_token(safe_ticker.lower())
    filename = f"{safe_ticker}_{safe_feature}_{case_id}_distribution.png"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return output_path
