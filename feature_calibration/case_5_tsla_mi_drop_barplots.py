"""Mutual information barplots per feature drop for TSLA 21-day horizon."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


DEFAULT_DATASET = Path("feature_calibration/data/tsla_technical_features.csv")
DEFAULT_OUTPUT_DIR = Path("feature_calibration/outputs/case_5")
DEFAULT_HORIZON = 21
TOP_N = 20
NEUTRAL_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate mutual-information barplots for TSLA technical indicators on 21-day horizon, "
            "dropping one feature at a time."
        )
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
        help="Directory where outputs (CSVs, charts, metadata) will be saved.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Forward return horizon (in trading days) to evaluate.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help="Number of top features to show on each barplot.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def compute_future_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    enriched = df.copy()
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


def select_feature_columns(df: pd.DataFrame, horizon: int, label_col: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.to_list()
    exclude = {f"future_return_{horizon}d", label_col}
    return [col for col in numeric_cols if col not in exclude]


def compute_mutual_information(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    label_col: str,
) -> pd.DataFrame:
    feature_cols = list(feature_cols)
    subset = df[feature_cols + [label_col]].dropna()
    if subset.empty:
        return pd.DataFrame(columns=["feature", "mutual_information", "observations"])

    X = subset[feature_cols].to_numpy()
    y = subset[label_col].astype("category").cat.codes.to_numpy()
    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)

    records = []
    for feature, score in zip(feature_cols, mi_scores, strict=False):
        records.append(
            {
                "feature": feature,
                "mutual_information": float(score),
                "observations": int(len(subset)),
            }
        )
    return pd.DataFrame(records)


def sanitize_subdir_name(feature: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature)
    return safe.strip("_") or "all_features"


def save_barplot(
    mi_df: pd.DataFrame,
    output_path: Path,
    dropped_feature: str,
    horizon: int,
    top_n: int,
) -> None:
    if mi_df.empty:
        return
    subset = mi_df.sort_values("mutual_information", ascending=True).tail(top_n)
    fig_height = max(4, 0.35 * len(subset) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(subset["feature"], subset["mutual_information"], color="#3498db")
    title_feature = "none" if dropped_feature == "none" else dropped_feature
    ax.set_title(f"Top {top_n} MI features ({horizon}d) – dropped: {title_feature}")
    ax.set_xlabel("Mutual information")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_image_html(image_path: Path, html_path: Path, title: str) -> None:
    if not image_path.exists():
        return
    html_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ background-color: #f7f7f7; font-family: Arial, sans-serif; margin: 1.5rem; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    h1 {{ font-size: 1.4rem; margin-bottom: 1rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <img src="{image_path.name}" alt="{title}" />
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    enriched = compute_future_return(df, args.horizon)
    enriched = enriched.replace([np.inf, -np.inf], np.nan)

    label_col = f"label_{args.horizon}d"
    enriched[label_col] = label_returns(enriched[f"future_return_{args.horizon}d"])

    feature_cols = select_feature_columns(enriched, args.horizon, label_col)
    drop_candidates = ["none"] + feature_cols

    all_records = []
    meta_outputs: dict[str, dict[str, str | None]] = {}

    for dropped in drop_candidates:
        active_features = feature_cols if dropped == "none" else [f for f in feature_cols if f != dropped]
        if not active_features:
            continue
        mi_df = compute_mutual_information(enriched, active_features, label_col)
        if mi_df.empty:
            continue

        mi_df = mi_df.sort_values("mutual_information", ascending=False).reset_index(drop=True)
        scenario_records = mi_df.copy()
        scenario_records.insert(0, "dropped_feature", dropped)
        all_records.append(scenario_records)

        subdir_name = sanitize_subdir_name(dropped)
        scenario_dir = output_dir / f"dropped_{subdir_name}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        csv_path = scenario_dir / "mi_scores.csv"
        mi_df.to_csv(csv_path, index=False)

        plot_path = scenario_dir / "mi_barplot.png"
        save_barplot(mi_df, plot_path, dropped, args.horizon, args.top_n)

        html_path = scenario_dir / "mi_barplot.html"
        write_image_html(
            plot_path,
            html_path,
            f"Top {args.top_n} MI features ({args.horizon}d) – dropped: {dropped if dropped != 'none' else 'none'}",
        )

        meta_outputs[dropped] = {
            "csv": str(csv_path),
            "plot": str(plot_path) if plot_path.exists() else None,
            "plot_html": str(html_path) if html_path.exists() else None,
        }

    long_df = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    long_path = output_dir / "tsla_mi_drop_long.csv"
    long_df.to_csv(long_path, index=False)

    metadata = {
        "dataset": str(args.dataset),
        "horizon": args.horizon,
        "top_n": args.top_n,
        "n_features": len(feature_cols),
        "drop_scenarios": drop_candidates,
        "outputs": {
            "long_csv": str(long_path),
            "per_drop": meta_outputs,
        },
    }
    (output_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

