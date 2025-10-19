"""Utility script to inspect available Anthropic (Claude) models for the configured API key."""

from __future__ import annotations

import sys
from pathlib import Path

from anthropic import Anthropic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llms_science_debate.config.settings import ANTHROPIC_API_KEY


def main() -> None:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured. Provide it in llms_science_debate/.env or environment.")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    print("Listing Anthropic models available for this API key:\n")

    models_page = client.models.list()
    models = getattr(models_page, "data", None) or []
    if not models:
        print("No models returned. Check API key permissions in Anthropic Console.")
        return

    for model in models:
        max_output = getattr(model, "default_output_token_limit", "n/a")
        print(f"- {model.id} :: default_output_tokens={max_output}")


if __name__ == "__main__":
    main()

