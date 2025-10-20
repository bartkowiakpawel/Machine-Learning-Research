"""Run a SMA-200 regime analysis for TSLA daily data. The script loads the
technical feature dataset, computes forward returns for the selected horizons,
labels each day as above or below the 200-day average, and aggregates regime
probabilities together with average and median forward returns. It also derives
lift metrics, produces static plots/HTML wrappers, and optionally exports Great
Tables views when the dependency is available.

CSV outputs:
- tsla_sma200_regime_enriched.csv — per-day features with forward returns and the above/below SMA-200 regime label.
- tsla_sma200_regime_summary.csv — per-horizon sample counts plus rise/drop/flat probabilities and return statistics split by regime.
- tsla_sma200_regime_lift.csv — above-minus-below SMA-200 differences for each probability and return metric across horizons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from great_tables import GT
except ImportError:  # pragma: no cover - optional dependency
    GT = None


DEFAULT_DATASET = Path("feature_calibration/data/tsla_technical_features.csv")
DEFAULT_OUTPUT_DIR = Path("feature_calibration/outputs/case_3")
DEFAULT_HORIZONS = (7, 14, 21)
NEUTRAL_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SMA-200 regime filter impact on TSLA future return probabilities."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the TSLA technical feature dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where case outputs will be stored.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Forward return horizons (in trading days) to evaluate.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def resolve_sma_column(df: pd.DataFrame) -> str:
    candidates = ["sma_200", "SMA200", "SMA_200"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not locate a 200-day SMA column among {candidates}.")


def compute_future_returns(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    enriched = df.copy()
    for horizon in horizons:
        future_price = enriched["Close"].shift(-horizon)
        enriched[f"future_return_{horizon}d"] = future_price / enriched["Close"] - 1.0
    return enriched


def build_regime_label(df: pd.DataFrame, sma_column: str) -> pd.Series:
    return (df["Close"] > df[sma_column]).astype("Int64")


def compute_regime_summary(
    df: pd.DataFrame,
    horizons: Iterable[int],
    regime_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        return_col = f"future_return_{horizon}d"
        if return_col not in df.columns:
            continue

        for regime_value in (0, 1):
            subset = df[df[regime_col] == regime_value][["Date", return_col]].dropna()
            if subset.empty:
                continue
            returns = subset[return_col]
            rise_prob = float((returns > NEUTRAL_TOLERANCE).mean())
            drop_prob = float((returns < -NEUTRAL_TOLERANCE).mean())
            flat_prob = max(0.0, 1.0 - rise_prob - drop_prob)

            rows.append(
                {
                    "horizon_days": horizon,
                    "above_sma200": regime_value,
                    "observations": int(len(subset)),
                    "rise_prob": rise_prob,
                    "drop_prob": drop_prob,
                    "flat_prob": flat_prob,
                    "avg_return": float(returns.mean()),
                    "median_return": float(returns.median()),
                }
            )
    return pd.DataFrame(rows)


def compute_lift(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    pivot = summary.pivot(
        index="horizon_days",
        columns="above_sma200",
        values=["rise_prob", "drop_prob", "flat_prob", "avg_return", "median_return"],
    )
    # Ensure both regimes exist
    pivot = pivot.dropna(axis=0, how="any")
    diff_records = []
    for horizon, row in pivot.iterrows():
        diff_records.append(
            {
                "horizon_days": int(horizon),
                "rise_prob_lift": float(row[("rise_prob", 1)] - row[("rise_prob", 0)]),
                "drop_prob_lift": float(row[("drop_prob", 1)] - row[("drop_prob", 0)]),
                "flat_prob_lift": float(row[("flat_prob", 1)] - row[("flat_prob", 0)]),
                "avg_return_lift": float(row[("avg_return", 1)] - row[("avg_return", 0)]),
                "median_return_lift": float(row[("median_return", 1)] - row[("median_return", 0)]),
            }
        )
    return pd.DataFrame(diff_records)


def save_probability_bars(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    horizons = sorted(summary["horizon_days"].unique())
    bar_width = 0.35
    x = np.arange(len(horizons))

    prob_above = summary[summary["above_sma200"] == 1].set_index("horizon_days")["rise_prob"]
    prob_below = summary[summary["above_sma200"] == 0].set_index("horizon_days")["rise_prob"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - bar_width / 2, prob_below.reindex(horizons).values, width=bar_width, label="Below SMA-200", color="#e74c3c")
    ax.bar(x + bar_width / 2, prob_above.reindex(horizons).values, width=bar_width, label="Above SMA-200", color="#27ae60")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_ylabel("Probability of positive return")
    ax.set_xlabel("Horizon")
    ax.set_ylim(0, 1)
    ax.set_title("TSLA rise probability vs. SMA-200 regime")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_image_html(image_path: Path, html_path: Path, title: str) -> None:
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 1.5rem;
      background: #f9fafb;
      color: #111827;
    }}
    h1 {{
      font-size: 1.5rem;
      margin-bottom: 1rem;
    }}
    img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      box-shadow: 0 4px 10px rgba(15, 23, 42, 0.12);
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <img src="{image_path.name}" alt="{title}">
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")


def save_probability_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    pivot = summary.pivot(index="above_sma200", columns="horizon_days", values="rise_prob")
    pivot = pivot.reindex([0, 1]).fillna(0.0)

    fig, ax = plt.subplots(figsize=(7, 3))
    cmap = plt.colormaps["RdYlGn"]
    mesh = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap=cmap)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{int(col)}d" for col in pivot.columns])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Below SMA-200", "Above SMA-200"])
    ax.set_xlabel("Horizon")
    ax.set_title("Rise probability heatmap (SMA-200 regimes)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value * 100:.1f}%", ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Probability")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_lift_plot(lift_df: pd.DataFrame, output_path: Path) -> None:
    if lift_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lift_df["horizon_days"], lift_df["rise_prob_lift"], marker="o", color="#1abc9c", label="Rise probability lift")
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)

    ax.set_xlabel("Horizon")
    ax.set_ylabel("Probability lift (Above - Below)")
    ax.set_title("SMA-200 regime lift on rise probability")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_with_great_tables(summary: pd.DataFrame, lift_df: pd.DataFrame, output_dir: Path) -> None:
    if GT is None or summary.empty:
        return

    summary_table = (
        GT(summary)
        .tab_header(title="TSLA SMA-200 Regime Summary")
        .fmt_percent(columns=["rise_prob", "drop_prob", "flat_prob"], decimals=1)
        .fmt_number(columns=["observations"], decimals=0)
        .fmt_number(columns=["avg_return", "median_return"], decimals=4)
    )
    try:  # pragma: no cover - IO heavy branch
        summary_table.save_png(str(output_dir / "tsla_sma200_regime_summary.png"), scale=0.85)
        summary_table.save_html(str(output_dir / "tsla_sma200_regime_summary.html"))
    except AttributeError:
        pass

    if not lift_df.empty:
        lift_table = (
            GT(lift_df)
            .tab_header(title="Probability and Return Lift (Above vs Below SMA-200)")
            .fmt_percent(columns=["rise_prob_lift", "drop_prob_lift", "flat_prob_lift"], decimals=1)
            .fmt_number(columns=["avg_return_lift", "median_return_lift"], decimals=4)
        )
        try:  # pragma: no cover
            lift_table.save_png(str(output_dir / "tsla_sma200_regime_lift.png"), scale=0.85)
            lift_table.save_html(str(output_dir / "tsla_sma200_regime_lift.html"))
        except AttributeError:
            pass


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    sma_col = resolve_sma_column(df)
    enriched = compute_future_returns(df, args.horizons)
    enriched = enriched.replace([np.inf, -np.inf], np.nan)
    enriched["above_sma200"] = build_regime_label(enriched, sma_col)

    summary = compute_regime_summary(enriched, args.horizons, "above_sma200")
    lift_df = compute_lift(summary)

    summary_path = output_dir / "tsla_sma200_regime_summary.csv"
    lift_path = output_dir / "tsla_sma200_regime_lift.csv"
    enriched_path = output_dir / "tsla_sma200_regime_enriched.csv"

    barplot_path = output_dir / "tsla_sma200_regime_barplot.png"
    heatmap_path = output_dir / "tsla_sma200_regime_heatmap.png"
    lift_plot_path = output_dir / "tsla_sma200_regime_lift_plot.png"
    barplot_html = output_dir / "tsla_sma200_regime_barplot.html"
    heatmap_html = output_dir / "tsla_sma200_regime_heatmap.html"
    lift_plot_html = output_dir / "tsla_sma200_regime_lift_plot.html"
    meta_path = output_dir / "case_metadata.json"

    enriched.to_csv(enriched_path, index=False)
    summary.to_csv(summary_path, index=False)
    lift_df.to_csv(lift_path, index=False)

    save_probability_bars(summary, barplot_path)
    save_probability_heatmap(summary, heatmap_path)
    save_lift_plot(lift_df, lift_plot_path)

    if barplot_path.exists():
        write_image_html(barplot_path, barplot_html, "TSLA rise probability vs. SMA-200 regime")
    if heatmap_path.exists():
        write_image_html(heatmap_path, heatmap_html, "Rise probability heatmap (SMA-200 regimes)")
    if lift_plot_path.exists():
        write_image_html(lift_plot_path, lift_plot_html, "SMA-200 regime lift on rise probability")

    save_with_great_tables(summary, lift_df, output_dir)

    gt_outputs = {}
    for stem in [
        "tsla_sma200_regime_summary.png",
        "tsla_sma200_regime_summary.html",
        "tsla_sma200_regime_lift.png",
        "tsla_sma200_regime_lift.html",
    ]:
        candidate = output_dir / stem
        if candidate.exists():
            gt_outputs[stem] = str(candidate)

    metadata = {
        "dataset": str(args.dataset),
        "sma_column": sma_col,
        "horizons": list(args.horizons),
        "outputs": {
            "enriched_dataset": str(enriched_path),
            "regime_summary": str(summary_path),
            "regime_lift": str(lift_path),
            "barplot": str(barplot_path) if barplot_path.exists() else None,
            "barplot_html": str(barplot_html) if barplot_html.exists() else None,
            "heatmap": str(heatmap_path) if heatmap_path.exists() else None,
            "heatmap_html": str(heatmap_html) if heatmap_html.exists() else None,
            "lift_plot": str(lift_plot_path) if lift_plot_path.exists() else None,
            "lift_plot_html": str(lift_plot_html) if lift_plot_html.exists() else None,
            "gt_tables": gt_outputs or None,
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
