"""SMA-200 mutual information isolation study for TSLA 21-day horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif

try:
    from great_tables import GT
except ImportError:  # pragma: no cover - optional dependency
    GT = None


DEFAULT_DATASET = Path("feature_calibration/data/tsla_technical_features.csv")
DEFAULT_OUTPUT_DIR = Path("feature_calibration/outputs/case_4")
DEFAULT_HORIZON = 21
NEUTRAL_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect how SMA-200 mutual information behaves when other technical features are removed."
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
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Forward return horizon (in trading days) to evaluate.",
    )
    parser.add_argument(
        "--quantile-bins",
        type=int,
        default=12,
        help="Number of quantile bins for SMA-200 curve summaries.",
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


def resolve_sma_column(df: pd.DataFrame) -> str:
    candidates = ["sma_200", "SMA200", "SMA_200"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not locate a 200-day SMA column among {candidates}.")


def select_feature_columns(
    df: pd.DataFrame,
    sma_column: str,
    horizon: int,
    label_column: str,
) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.to_list()
    exclude = {
        f"future_return_{horizon}d",
        label_column,
    }
    features = [col for col in numeric_cols if col not in exclude]
    if sma_column not in features:
        features.append(sma_column)
    return features


def compute_mutual_information(
    df: pd.DataFrame,
    features: Iterable[str],
    label_column: str,
) -> pd.DataFrame:
    features = list(features)
    subset = df[features + [label_column]].dropna()
    if subset.empty:
        return pd.DataFrame(columns=["feature", "mutual_information", "observations"])

    X = subset[features].to_numpy()
    y = subset[label_column].astype("category").cat.codes.to_numpy()
    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)

    records = []
    for feature, score in zip(features, mi_scores, strict=False):
        records.append(
            {
                "feature": feature,
                "mutual_information": float(score),
                "observations": int(len(subset)),
            }
        )
    return pd.DataFrame(records)


def compute_anova_scores(
    df: pd.DataFrame,
    features: Iterable[str],
    label_column: str,
) -> pd.DataFrame:
    features = list(features)
    subset = df[features + [label_column]].dropna()
    if subset.empty:
        return pd.DataFrame(columns=["feature", "f_statistic", "p_value"])

    X = subset[features].to_numpy()
    y = subset[label_column].astype("category").cat.codes.to_numpy()
    try:
        f_scores, p_values = f_classif(X, y)
    except Exception:  # pragma: no cover - defensive
        f_scores = np.full(len(features), np.nan)
        p_values = np.full(len(features), np.nan)

    records = []
    for feature, f_stat, p_val in zip(features, f_scores, p_values, strict=False):
        records.append(
            {
                "feature": feature,
                "f_statistic": float(f_stat) if np.isfinite(f_stat) else np.nan,
                "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
            }
        )
    return pd.DataFrame(records)


def run_drop_analysis(
    df: pd.DataFrame,
    features: list[str],
    sma_column: str,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    other_features = [f for f in features if f != sma_column]

    baseline_mi = compute_mutual_information(df, features, label_column)
    baseline_row = baseline_mi[baseline_mi["feature"] == sma_column]
    baseline_mi_value = float(baseline_row["mutual_information"].iloc[0]) if not baseline_row.empty else np.nan
    baseline_obs = int(baseline_row["observations"].iloc[0]) if not baseline_row.empty else 0

    summary_records: list[dict[str, object]] = []
    long_records: list[pd.DataFrame] = []

    baseline_summary = {
        "dropped_feature": "none (all features kept)",
        "mi_sma200": baseline_mi_value,
        "delta_from_baseline": 0.0,
        "observations": baseline_obs,
        "features_included": len(features),
    }
    summary_records.append(baseline_summary)
    baseline_mi = baseline_mi.assign(dropped_feature="none (all features kept)")
    long_records.append(baseline_mi)

    for feature_to_drop in other_features:
        active_features = [sma_column] + [f for f in other_features if f != feature_to_drop]
        mi_df = compute_mutual_information(df, active_features, label_column)
        if mi_df.empty:
            continue
        long_records.append(mi_df.assign(dropped_feature=feature_to_drop))

        sma_row = mi_df[mi_df["feature"] == sma_column]
        if sma_row.empty:
            continue
        mi_value = float(sma_row["mutual_information"].iloc[0])
        observations = int(sma_row["observations"].iloc[0])
        summary_records.append(
            {
                "dropped_feature": feature_to_drop,
                "mi_sma200": mi_value,
                "delta_from_baseline": mi_value - baseline_mi_value,
                "observations": observations,
                "features_included": len(active_features),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    long_df = pd.concat(long_records, ignore_index=True) if long_records else pd.DataFrame()
    return summary_df, long_df


def compute_quantile_summary(
    df: pd.DataFrame,
    sma_column: str,
    return_column: str,
    label_column: str,
    quantile_bins: int,
) -> pd.DataFrame:
    subset = df[[sma_column, return_column, label_column]].dropna()
    if subset.empty:
        return pd.DataFrame()

    subset = subset.copy()
    try:
        subset["quantile"] = pd.qcut(subset[sma_column], q=quantile_bins, duplicates="drop")
    except ValueError:
        subset["quantile"] = pd.qcut(subset[sma_column], q=max(2, quantile_bins // 2), duplicates="drop")

    grouped = subset.groupby("quantile", observed=False)
    records = []
    for idx, (interval, group) in enumerate(grouped, start=1):
        if isinstance(interval, pd.Interval):
            midpoint = float(interval.mid)
            label = f"Q{idx}"
        else:
            midpoint = np.nan
            label = str(interval)

        rise_prob = float((group[label_column] == "rise").mean())
        drop_prob = float((group[label_column] == "drop").mean())
        flat_prob = max(0.0, 1.0 - rise_prob - drop_prob)

        records.append(
            {
                "quantile": label,
                "sma200_midpoint": midpoint,
                "sma200_min": float(group[sma_column].min()),
                "sma200_max": float(group[sma_column].max()),
                "avg_return": float(group[return_column].mean()),
                "median_return": float(group[return_column].median()),
                "rise_prob": rise_prob,
                "drop_prob": drop_prob,
                "flat_prob": flat_prob,
                "observations": int(len(group)),
            }
        )
    return pd.DataFrame(records)


def compute_linear_vs_nonlinear_metrics(
    df: pd.DataFrame,
    sma_column: str,
    return_column: str,
    label_column: str,
) -> pd.DataFrame:
    subset = df[[sma_column, return_column, label_column]].dropna()
    if subset.empty:
        return pd.DataFrame(columns=["metric", "value", "notes"])

    subset = subset.copy()
    subset[label_column] = subset[label_column].astype("category")
    y_codes = subset[label_column].cat.codes.to_numpy()
    X = subset[[sma_column]].to_numpy()

    mi_score = mutual_info_classif(X, y_codes, discrete_features=False, random_state=42)[0]
    try:
        f_stat, p_value = f_classif(X, y_codes)
        f_stat_value = float(f_stat[0])
        p_value_value = float(p_value[0])
    except Exception:
        f_stat_value = np.nan
        p_value_value = np.nan

    pearson = subset[sma_column].corr(subset[return_column], method="pearson")
    spearman = subset[sma_column].corr(subset[return_column], method="spearman")
    kendall = subset[sma_column].corr(subset[return_column], method="kendall")

    records = [
        {"metric": "mutual_information", "value": float(mi_score), "notes": "Non-linear dependency capture (kNN estimator)."},
        {"metric": "anova_f_statistic", "value": f_stat_value, "notes": "Linear separability (ANOVA F test)."},
        {"metric": "anova_p_value", "value": p_value_value, "notes": "p-value from ANOVA F test."},
        {"metric": "pearson_correlation", "value": float(pearson), "notes": "Linear correlation with 21d returns."},
        {"metric": "spearman_correlation", "value": float(spearman), "notes": "Rank correlation (monotonic signal)."},
        {"metric": "kendall_tau", "value": float(kendall), "notes": "Ordinal association strength."},
        {"metric": "sample_size", "value": int(len(subset)), "notes": "Rows used after dropping NA values."},
    ]
    return pd.DataFrame(records)


def write_image_html(image_path: Path, html_path: Path, title: str) -> None:
    if not image_path.exists():
        return
    content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 1.5rem; background-color: #fafafa; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    h1 {{ font-size: 1.5rem; margin-bottom: 1rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <img src="{image_path.name}" alt="{title}" />
</body>
</html>
"""
    html_path.write_text(content, encoding="utf-8")


def save_mi_drop_barplot(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    plot_df = summary_df.sort_values("mi_sma200", ascending=True)
    fig_height = max(4, 0.35 * len(plot_df) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = ["#2c3e50" if row == "none (all features kept)" else "#1abc9c" for row in plot_df["dropped_feature"]]
    ax.barh(plot_df["dropped_feature"], plot_df["mi_sma200"], color=colors)
    ax.set_xlabel("Mutual information (SMA-200 vs 21d label)")
    ax.set_ylabel("Dropped feature")
    ax.set_title("SMA-200 mutual information when dropping one feature at a time")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_mi_delta_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    deltas = summary_df[summary_df["dropped_feature"] != "none (all features kept)"]
    if deltas.empty:
        return

    deltas = deltas.sort_values("delta_from_baseline")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(deltas) + 2)))
    bars = ax.barh(deltas["dropped_feature"], deltas["delta_from_baseline"], color="#e67e22")
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Δ mutual information vs. baseline (SMA-200 only)")
    ax.set_ylabel("Dropped feature")
    ax.set_title("Change in SMA-200 mutual information when removing a single feature")
    ax.grid(axis="x", alpha=0.3)
    ax.bar_label(bars, fmt="%.4f", padding=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_quantile_return_plot(quantile_df: pd.DataFrame, output_path: Path) -> None:
    if quantile_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(quantile_df["sma200_midpoint"], quantile_df["avg_return"], marker="o", color="#2980b9", label="Average return")
    ax.plot(
        quantile_df["sma200_midpoint"],
        quantile_df["median_return"],
        marker="s",
        linestyle="--",
        color="#16a085",
        label="Median return",
    )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("SMA-200 midpoint (per quantile)")
    ax.set_ylabel("Forward 21d return")
    ax.set_title("21d return vs. SMA-200 quantiles")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_quantile_probability_plot(quantile_df: pd.DataFrame, output_path: Path) -> None:
    if quantile_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(quantile_df["sma200_midpoint"], quantile_df["rise_prob"], marker="o", color="#27ae60", label="Rise")
    ax.plot(quantile_df["sma200_midpoint"], quantile_df["drop_prob"], marker="s", color="#c0392b", label="Drop")
    ax.plot(quantile_df["sma200_midpoint"], quantile_df["flat_prob"], marker="^", color="#7f8c8d", label="Flat")
    ax.set_xlabel("SMA-200 midpoint (per quantile)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Return regime probabilities vs. SMA-200 quantiles")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_scatter_with_trend(
    df: pd.DataFrame,
    sma_column: str,
    return_column: str,
    output_path: Path,
    quantile_df: pd.DataFrame | None = None,
) -> None:
    subset = df[[sma_column, return_column]].dropna()
    if subset.empty:
        return

    plot_sample = subset
    if len(subset) > 5000:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(subset), size=5000, replace=False)
        plot_sample = subset.iloc[sample_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(plot_sample[sma_column], plot_sample[return_column], alpha=0.25, color="#34495e", s=10, label="Daily samples")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.6)

    if quantile_df is not None and not quantile_df.empty:
        ax.plot(
            quantile_df["sma200_midpoint"],
            quantile_df["avg_return"],
            color="#e74c3c",
            linewidth=2,
            label="Quantile average",
        )

    ax.set_xlabel("SMA-200")
    ax.set_ylabel("Forward 21d return")
    ax.set_title("Forward return vs. SMA-200 (scatter with quantile trend)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_with_great_tables(
    mi_summary: pd.DataFrame,
    quantile_summary: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if GT is None:
        return

    if not mi_summary.empty:
        table = (
            GT(mi_summary)
            .tab_header(title="SMA-200 Mutual Information Drop Test (21d Horizon)")
            .fmt_number(columns=["mi_sma200", "delta_from_baseline"], decimals=4)
            .fmt_number(columns=["observations", "features_included"], decimals=0)
        )
        try:  # pragma: no cover - IO heavy branch
            table.save_png(str(output_dir / "tsla_sma200_mi_drop_table.png"), scale=0.85)
            table.save_html(str(output_dir / "tsla_sma200_mi_drop_table.html"))
        except AttributeError:
            pass

    if not quantile_summary.empty:
        table = (
            GT(quantile_summary)
            .tab_header(title="SMA-200 Quantile Summary (21d Horizon)")
            .fmt_number(columns=["avg_return", "median_return"], decimals=4)
            .fmt_percent(columns=["rise_prob", "drop_prob", "flat_prob"], decimals=1)
            .fmt_number(columns=["observations"], decimals=0)
        )
        try:  # pragma: no cover
            table.save_png(str(output_dir / "tsla_sma200_quantile_table.png"), scale=0.85)
            table.save_html(str(output_dir / "tsla_sma200_quantile_table.html"))
        except AttributeError:
            pass

    if not metrics_df.empty:
        table = (
            GT(metrics_df)
            .tab_header(title="Linear vs. Non-linear Diagnostics (SMA-200, 21d Horizon)")
            .fmt_number(columns=["value"], decimals=6)
        )
        try:  # pragma: no cover
            table.save_png(str(output_dir / "tsla_sma200_metrics_table.png"), scale=0.85)
            table.save_html(str(output_dir / "tsla_sma200_metrics_table.html"))
        except AttributeError:
            pass


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    sma_column = resolve_sma_column(df)
    enriched = compute_future_return(df, args.horizon)
    enriched = enriched.replace([np.inf, -np.inf], np.nan)

    return_column = f"future_return_{args.horizon}d"
    label_column = f"label_{args.horizon}d"
    enriched[label_column] = label_returns(enriched[return_column])

    features = select_feature_columns(enriched, sma_column, args.horizon, label_column)
    mi_summary, mi_long = run_drop_analysis(enriched, features, sma_column, label_column)
    quantile_summary = compute_quantile_summary(
        enriched, sma_column, return_column, label_column, quantile_bins=args.quantile_bins
    )
    metrics_df = compute_linear_vs_nonlinear_metrics(enriched, sma_column, return_column, label_column)
    anova_scores = compute_anova_scores(enriched, [sma_column], label_column)

    mi_summary_path = output_dir / "tsla_sma200_mi_drop_summary.csv"
    mi_long_path = output_dir / "tsla_sma200_mi_drop_long.csv"
    quantile_path = output_dir / "tsla_sma200_quantile_summary.csv"
    metrics_path = output_dir / "tsla_sma200_linear_vs_nonlinear_metrics.csv"
    anova_path = output_dir / "tsla_sma200_anova_scores.csv"

    mi_summary.to_csv(mi_summary_path, index=False)
    mi_long.to_csv(mi_long_path, index=False)
    quantile_summary.to_csv(quantile_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    anova_scores.to_csv(anova_path, index=False)

    mi_barplot_path = output_dir / "tsla_sma200_mi_drop_barplot.png"
    mi_delta_path = output_dir / "tsla_sma200_mi_delta.png"
    quantile_return_path = output_dir / "tsla_sma200_quantile_returns.png"
    quantile_prob_path = output_dir / "tsla_sma200_quantile_probabilities.png"
    scatter_path = output_dir / "tsla_sma200_return_scatter.png"

    save_mi_drop_barplot(mi_summary, mi_barplot_path)
    save_mi_delta_plot(mi_summary, mi_delta_path)
    save_quantile_return_plot(quantile_summary, quantile_return_path)
    save_quantile_probability_plot(quantile_summary, quantile_prob_path)
    save_scatter_with_trend(enriched, sma_column, return_column, scatter_path, quantile_df=quantile_summary)

    write_image_html(mi_barplot_path, output_dir / "tsla_sma200_mi_drop_barplot.html", "SMA-200 MI drop barplot")
    write_image_html(mi_delta_path, output_dir / "tsla_sma200_mi_delta.html", "SMA-200 MI delta plot")
    write_image_html(quantile_return_path, output_dir / "tsla_sma200_quantile_returns.html", "SMA-200 quantile returns")
    write_image_html(quantile_prob_path, output_dir / "tsla_sma200_quantile_probabilities.html", "SMA-200 quantile regime probabilities")
    write_image_html(scatter_path, output_dir / "tsla_sma200_return_scatter.html", "Forward return vs. SMA-200 scatter")

    save_with_great_tables(mi_summary, quantile_summary, metrics_df, output_dir)

    metadata = {
        "dataset": str(args.dataset),
        "horizon": args.horizon,
        "sma_column": sma_column,
        "features_considered": features,
        "outputs": {
            "mi_summary": str(mi_summary_path),
            "mi_long": str(mi_long_path),
            "quantile_summary": str(quantile_path),
            "linear_vs_nonlinear_metrics": str(metrics_path),
            "anova_scores": str(anova_path),
            "mi_drop_barplot": str(mi_barplot_path) if mi_barplot_path.exists() else None,
            "mi_delta_plot": str(mi_delta_path) if mi_delta_path.exists() else None,
            "quantile_return_plot": str(quantile_return_path) if quantile_return_path.exists() else None,
            "quantile_probability_plot": str(quantile_prob_path) if quantile_prob_path.exists() else None,
            "scatter_plot": str(scatter_path) if scatter_path.exists() else None,
        },
    }

    (output_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

