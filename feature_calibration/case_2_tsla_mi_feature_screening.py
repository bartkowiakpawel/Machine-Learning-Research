"""Mutual information feature screening for TSLA technical indicators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

try:
    from great_tables import GT
except ImportError:  # pragma: no cover - optional dependency
    GT = None


DEFAULT_DATASET = Path("feature_calibration/data/tsla_technical_features.csv")
DEFAULT_OUTPUT_DIR = Path("feature_calibration/outputs/case_2")
DEFAULT_HORIZONS = (1, 5, 14, 21)
NEUTRAL_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mutual information between classic technical indicators and TSLA future returns."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the dataset containing OHLCV data and technical indicators.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where outputs (CSVs, charts, tables) will be saved.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def compute_future_returns(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    enriched = df.copy()
    for horizon in horizons:
        future_price = enriched["Close"].shift(-horizon)
        enriched[f"future_return_{horizon}d"] = future_price / enriched["Close"] - 1.0
    return enriched


def label_returns(series: pd.Series) -> pd.Series:
    def _label(value: float) -> str | pd.NA:
        if pd.isna(value):
            return pd.NA
        if value > NEUTRAL_TOLERANCE:
            return "rise"
        if value < -NEUTRAL_TOLERANCE:
            return "drop"
        return "flat"

    return series.apply(_label)


def prepare_features(df: pd.DataFrame, horizons: Iterable[int]) -> tuple[pd.DataFrame, list[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.to_list()
    exclude_cols = []
    for horizon in horizons:
        exclude_cols.append(f"future_return_{horizon}d")
        exclude_cols.append(f"label_{horizon}d")

    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    return df, feature_cols


def compute_mutual_information(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizons: Iterable[int],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for horizon in horizons:
        label_col = f"label_{horizon}d"
        if label_col not in df.columns:
            continue
        subset = df[feature_cols + [label_col]].dropna()
        if subset.empty:
            continue

        X = subset[feature_cols].to_numpy()
        y = subset[label_col].astype("category").cat.codes.to_numpy()
        mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)

        for feature, score in zip(feature_cols, mi_scores):
            records.append(
                {
                    "feature": feature,
                    "horizon_days": horizon,
                    "mutual_information": float(score),
                    "observations": int(len(subset)),
                }
            )

    return pd.DataFrame(records)


def save_mi_heatmap(mi_long: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    if mi_long.empty:
        return

    pivot = mi_long.pivot(index="feature", columns="horizon_days", values="mutual_information").fillna(0.0)
    pivot["avg_mi"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg_mi", ascending=False).head(top_n).drop(columns="avg_mi")
    pivot = pivot.iloc[::-1]  # flip for better readability in heatmap

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.35)))
    cmap = plt.colormaps["viridis"]
    mesh = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{int(col)}d" for col in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Horizon")
    ax.set_title(f"Top {top_n} features by mutual information (TSLA)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=7)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mutual information")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_top_feature_bars(mi_long: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    if mi_long.empty:
        return

    horizons = sorted(mi_long["horizon_days"].unique())
    n_cols = 2
    n_rows = int(np.ceil(len(horizons) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False)

    for idx, horizon in enumerate(horizons):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        subset = mi_long[mi_long["horizon_days"] == horizon].sort_values("mutual_information", ascending=False).head(top_n)
        ax.barh(subset["feature"][::-1], subset["mutual_information"][::-1], color="#3498db")
        ax.set_title(f"Top {top_n} MI features - {horizon}d horizon")
        ax.set_xlabel("Mutual information")
        ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.3)

    for idx in range(len(horizons), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        fig.delaxes(axes[row][col])

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_with_great_tables(mi_long: pd.DataFrame, output_dir: Path, top_n: int = 10) -> None:
    if GT is None or mi_long.empty:
        return

    top_records = (
        mi_long.sort_values(["horizon_days", "mutual_information"], ascending=[True, False])
        .groupby("horizon_days")
        .head(top_n)
        .reset_index(drop=True)
    )
    pivot_ranked = []
    for horizon, group in top_records.groupby("horizon_days"):
        ranked = group.copy()
        ranked["rank"] = np.arange(1, len(group) + 1)
        pivot_ranked.append(ranked)
    ranked_df = pd.concat(pivot_ranked, ignore_index=True)

    table = (
        GT(ranked_df)
        .tab_header(title=f"Top {top_n} TSLA Technical Features by Mutual Information")
        .fmt_number(columns=["mutual_information"], decimals=4)
        .fmt_number(columns=["observations"], decimals=0)
    )
    try:  # pragma: no cover - IO heavy branch
        table.save_png(str(output_dir / "tsla_top_mi_features.png"), scale=0.85)
        table.save_html(str(output_dir / "tsla_top_mi_features.html"))
    except AttributeError:
        pass


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    enriched = compute_future_returns(df, DEFAULT_HORIZONS)
    enriched = enriched.replace([np.inf, -np.inf], np.nan)
    for horizon in DEFAULT_HORIZONS:
        enriched[f"label_{horizon}d"] = label_returns(enriched[f"future_return_{horizon}d"])

    enriched, feature_cols = prepare_features(enriched, DEFAULT_HORIZONS)
    mi_long = compute_mutual_information(enriched, feature_cols, DEFAULT_HORIZONS)

    pivot_path = output_dir / "tsla_mi_scores_pivot.csv"
    long_path = output_dir / "tsla_mi_scores_long.csv"
    top_path = output_dir / "tsla_top_mi_features.csv"
    heatmap_path = output_dir / "tsla_mi_heatmap.png"
    barplot_path = output_dir / "tsla_mi_top_features_barplot.png"
    meta_path = output_dir / "case_metadata.json"

    if not mi_long.empty:
        mi_long.to_csv(long_path, index=False)
        pivot = mi_long.pivot(index="feature", columns="horizon_days", values="mutual_information")
        pivot.to_csv(pivot_path)

        top_features = (
            mi_long.sort_values(["horizon_days", "mutual_information"], ascending=[True, False])
            .groupby("horizon_days")
            .head(15)
            .reset_index(drop=True)
        )
        top_features.to_csv(top_path, index=False)

        save_mi_heatmap(mi_long, heatmap_path, top_n=20)
        save_top_feature_bars(mi_long, barplot_path, top_n=10)
        save_with_great_tables(mi_long, output_dir, top_n=10)
    else:
        pivot = pd.DataFrame()
        top_features = pd.DataFrame()

    gt_outputs = {}
    for stem in ["tsla_top_mi_features.png", "tsla_top_mi_features.html"]:
        candidate = output_dir / stem
        if candidate.exists():
            gt_outputs[stem] = str(candidate)

    metadata = {
        "dataset": str(args.dataset),
        "horizons": list(DEFAULT_HORIZONS),
        "n_features": len(feature_cols),
        "outputs": {
            "mi_scores_long": str(long_path) if long_path.exists() else None,
            "mi_scores_pivot": str(pivot_path) if pivot_path.exists() else None,
            "top_features": str(top_path) if top_path.exists() else None,
            "heatmap": str(heatmap_path) if heatmap_path.exists() else None,
            "barplot": str(barplot_path) if barplot_path.exists() else None,
            "gt_tables": gt_outputs or None,
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
