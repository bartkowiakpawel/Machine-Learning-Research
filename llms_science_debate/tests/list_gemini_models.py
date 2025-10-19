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
    if not models:
        print("No models returned. Check API key permissions in Google AI Studio.")
        return

    for model in models:
        methods = getattr(model, "supported_generation_methods", None)
        methods_display = ", ".join(sorted(methods)) if methods else "n/a"
        print(f"- {model.name} :: methods={methods_display}")


if __name__ == "__main__":
    main()

