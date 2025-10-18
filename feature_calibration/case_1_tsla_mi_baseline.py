"""Baseline MI-focused calibration overview for TSLA."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from great_tables import GT
except ImportError:  # pragma: no cover - optional dependency
    GT = None


DEFAULT_HORIZONS = (1, 5, 14, 21)
NEUTRAL_TOLERANCE = 1e-9


@dataclass
class CaseConfig:
    dataset_path: Path
    output_dir: Path
    horizons: tuple[int, ...] = DEFAULT_HORIZONS


def parse_args() -> CaseConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Generate baseline probabilities for TSLA future returns to benchmark "
            "mutual-information based feature calibration experiments."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("feature_calibration/data/tsla_yf_dataset.csv"),
        help="Path to the input TSLA dataset (CSV with OHLCV data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("feature_calibration/outputs/case_1"),
        help="Directory where outputs (CSVs, charts, tables) will be written.",
    )
    args = parser.parse_args()

    return CaseConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    if "Close" not in df.columns:
        raise ValueError("Expected column 'Close' in the dataset.")
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


def compute_probability_summary(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for horizon in horizons:
        return_col = f"future_return_{horizon}d"
        label_col = f"label_{horizon}d"
        labels = label_returns(df[return_col])
        counts = labels.value_counts(dropna=True)
        total = counts.sum()
        probs = counts.div(total) if total else pd.Series(dtype=float)
        records.append(
            {
                "horizon_days": horizon,
                "observations": int(total),
                "drop_prob": float(probs.get("drop", 0.0)),
                "flat_prob": float(probs.get("flat", 0.0)),
                "rise_prob": float(probs.get("rise", 0.0)),
            }
        )
        df[label_col] = labels
    summary = pd.DataFrame(records)
    summary["baseline_accuracy"] = summary[["drop_prob", "flat_prob", "rise_prob"]].max(axis=1)
    return summary


def compute_overall_probabilities(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    label_series = []
    for horizon in horizons:
        label_col = f"label_{horizon}d"
        if label_col in df.columns:
            label_series.append(df[label_col].dropna())
    if not label_series:
        return pd.DataFrame(columns=["movement", "probability", "observations"])
    stacked = pd.concat(label_series, ignore_index=True)
    counts = stacked.value_counts()
    total = counts.sum()
    probs = counts / total
    data = [
        {"movement": label, "probability": float(prob), "observations": int(counts[label])}
        for label, prob in probs.items()
    ]
    overall = pd.DataFrame(data)
    overall["baseline_accuracy"] = overall["probability"].max() if not overall.empty else np.nan
    return overall


def save_probability_chart(summary: pd.DataFrame, output_path: Path) -> None:
    label_order = ["drop_prob", "flat_prob", "rise_prob"]
    colors = {
        "drop_prob": "#e74c3c",
        "flat_prob": "#95a5a6",
        "rise_prob": "#27ae60",
    }
    x = np.arange(len(summary))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for idx, label in enumerate(label_order):
        ax.bar(
            x + idx * bar_width,
            summary[label].values,
            width=bar_width,
            color=colors[label],
            label=label.replace("_prob", "").title(),
        )

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"{int(h)}d" for h in summary["horizon_days"]])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Movement probability per horizon (TSLA)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_probability_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    label_order = ["drop_prob", "flat_prob", "rise_prob"]
    label_names = ["Drop", "Flat", "Rise"]
    data = summary[label_order].to_numpy().T

    fig, ax = plt.subplots(figsize=(8, 3.5))
    cmap = plt.colormaps["RdYlGn"]
    mesh = ax.imshow(data, vmin=0, vmax=1, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(summary)))
    ax.set_xticklabels([f"{int(h)}d" for h in summary["horizon_days"]])
    ax.set_yticks(np.arange(len(label_order)))
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Horizon")
    ax.set_title("Movement probability heatmap (TSLA)")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(
                j,
                i,
                f"{value * 100:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Probability")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_with_great_tables(summary: pd.DataFrame, overall: pd.DataFrame, output_dir: Path) -> None:
    if GT is None:
        return

    horizon_table = (
        GT(summary)
        .tab_header(title="TSLA Movement Probabilities by Horizon")
        .fmt_percent(columns=["drop_prob", "flat_prob", "rise_prob", "baseline_accuracy"], decimals=1)
        .fmt_number(columns=["observations"], decimals=0)
    )
    try:  # pragma: no cover - IO heavy branch
        horizon_table.save_png(str(output_dir / "tsla_movement_probabilities.png"), scale=0.85)
        horizon_table.save_html(str(output_dir / "tsla_movement_probabilities.html"))
    except AttributeError:
        pass

    if overall.empty:
        return

    overall_table = (
        GT(overall)
        .tab_header(title="Overall Movement Probabilities (All Horizons)")
        .fmt_percent(columns=["probability", "baseline_accuracy"], decimals=1)
        .fmt_number(columns=["observations"], decimals=0)
    )
    try:  # pragma: no cover - IO heavy branch
        overall_table.save_png(str(output_dir / "tsla_overall_movement_probabilities.png"), scale=0.85)
        overall_table.save_html(str(output_dir / "tsla_overall_movement_probabilities.html"))
    except AttributeError:
        pass


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(config.dataset_path)
    enriched = compute_future_returns(df, config.horizons)
    summary = compute_probability_summary(enriched, config.horizons)
    overall = compute_overall_probabilities(enriched, config.horizons)

    enriched_path = config.output_dir / "tsla_future_returns_enriched.csv"
    summary_path = config.output_dir / "tsla_horizon_probability_summary.csv"
    overall_path = config.output_dir / "tsla_overall_probability_summary.csv"
    plot_path = config.output_dir / "tsla_movement_probability_chart.png"
    heatmap_path = config.output_dir / "tsla_movement_probability_heatmap.png"
    meta_path = config.output_dir / "case_metadata.json"

    enriched.to_csv(enriched_path, index=False)
    summary.to_csv(summary_path, index=False)
    overall.to_csv(overall_path, index=False)
    save_probability_chart(summary, plot_path)
    save_probability_heatmap(summary, heatmap_path)
    save_with_great_tables(summary, overall, config.output_dir)

    gt_outputs = {}
    for stem in [
        "tsla_movement_probabilities.png",
        "tsla_movement_probabilities.html",
        "tsla_overall_movement_probabilities.png",
        "tsla_overall_movement_probabilities.html",
    ]:
        candidate = config.output_dir / stem
        if candidate.exists():
            gt_outputs[stem] = str(candidate)

    metadata = {
        "dataset": str(config.dataset_path),
        "horizons": list(config.horizons),
        "outputs": {
            "enriched_dataset": str(enriched_path),
            "horizon_summary": str(summary_path),
            "overall_summary": str(overall_path),
            "probability_chart": str(plot_path),
            "probability_heatmap": str(heatmap_path),
            "gt_tables": gt_outputs or None,
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
