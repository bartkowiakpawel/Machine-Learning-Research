"""Experimental renderer for Case 1 metrics with annotated great_tables output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from great_tables import GT
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    GT = None  # type: ignore[assignment]

COLUMN_METADATA = (
    ("model", "Classifier name", "string", "-", "e.g. RandomForest"),
    ("class", "Target class (drop/flat/rise)", "string", "-", "Evaluated prediction class"),
    ("brier_score", "Probability calibration error", "float", "0.0–1.0", "Lower is better"),
    ("log_loss", "Logarithmic loss penalty", "float", "0.0–∞", "Lower is better"),
    ("roc_auc", "Area Under ROC Curve", "float", "0.0–1.0", "Higher is better"),
    ("mutual_information", "Mutual Information", "float", "0.0–1.0", "Higher is better"),
    ("accuracy", "Overall accuracy", "float", "0.0–1.0", "Higher is better"),
    ("precision_macro", "Macro-averaged precision", "float", "0.0–1.0", "Higher is better"),
    ("recall_macro", "Macro-averaged recall", "float", "0.0–1.0", "Higher is better"),
    ("f1_macro", "Macro-averaged F1-score", "float", "0.0–1.0", "Higher is better"),
)


def _format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in df.columns:
        if column in {"model", "class"}:
            continue
        formatted[column] = pd.to_numeric(formatted[column], errors="coerce").round(4)
    return formatted


def render_metrics_table(metrics_path: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    if GT is None:
        raise RuntimeError(
            "great_tables is not installed in the current environment. "
            "Install it together with selenium, webdriver-manager and css-inline."
        )

    metrics_df = pd.read_csv(metrics_path)
    formatted_df = _format_metrics(metrics_df)

    column_explainer = pd.DataFrame(
        COLUMN_METADATA,
        columns=["Column", "Description", "Data type", "Range", "Interpretation"],
    )

    if output_dir is None:
        output_dir = metrics_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_title = "Case 1 – TSLA RSI baseline metrics"
    png_path = output_dir / (metrics_path.stem + "_annotated.png")
    html_path = png_path.with_suffix(".html")
    glossary_path = output_dir / (metrics_path.stem + "_glossary.csv")

    formatted_df.to_csv(glossary_path, index=False)

    base_table = GT(formatted_df).tab_header(
        title=table_title,
        subtitle="Model-level evaluation on TSLA RSI dataset (train/test split aligned chronologically).",
    )

    note_lines = [
        "Columns rounded to four decimal places.",
        "Log loss / Brier score: lower values indicate better calibration.",
        "ROC AUC / Mutual Information / Accuracy / Precision / Recall / F1: higher values indicate better performance.",
    ]
    annotated_table = base_table.tab_source_note("Notes: " + " ".join(note_lines))

    annotated_table.save(str(png_path))
    annotated_table.write_raw_html(html_path, inline_css=True, make_page=True)

    glossary_table = GT(column_explainer).tab_header(
        title="Column glossary",
        subtitle="Interpretation of metrics used in Case 1 RSI baseline.",
    )
    glossary_html_path = output_dir / (metrics_path.stem + "_glossary.html")
    glossary_table.write_raw_html(glossary_html_path, inline_css=True, make_page=True)

    return png_path, html_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render annotated tables for Case 1 metrics.")
    parser.add_argument(
        "--metrics",
        dest="metrics_path",
        default=Path("rsi_calibration") / "outputs" / "case_1" / "tsla_case_1_metrics.csv",
        help="Path to the metrics CSV produced by case_1_tsla_rsi_baseline.py",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Optional directory where the annotated tables should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics_path = Path(args.metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}. Run the case_1 pipeline first.")
    output_dir = Path(args.output_dir) if args.output_dir else None
    png_path, html_path = render_metrics_table(metrics_path, output_dir=output_dir)
    print(f"Annotated PNG table saved to: {png_path}")
    print(f"Annotated HTML table saved to: {html_path}")


if __name__ == "__main__":
    main()
