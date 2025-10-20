"""Utility script to inspect available Gemini models for the configured API key."""

from __future__ import annotations

import sys
from pathlib import Path

import google.generativeai as genai

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llms_science_debate.config.settings import GEMINI_API_KEY


def main() -> None:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Listing Gemini models available for this API key:\n")

    models = list(genai.list_models())
    output_lines = ["# Google Gemini Models", ""]
    if not models:
        print("No models returned. Check API key permissions in Google AI Studio.")
        output_lines.append("_No models returned for this API key._")
        _write_output(output_lines)
        return

    for model in models:
        methods = getattr(model, "supported_generation_methods", None)
        methods_display = ", ".join(sorted(methods)) if methods else "n/a"
        print(f"- {model.name} :: methods={methods_display}")
        output_lines.append(f"- {model.name} (methods={methods_display})")

    _write_output(output_lines)


def _write_output(lines: list[str]) -> None:
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gemini_models.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nModel list saved to {output_path}")


if __name__ == "__main__":
    main()

