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
    output_lines = ["# Anthropic (Claude) Models", ""]
    if not models:
        print("No models returned. Check API key permissions in Anthropic Console.")
        output_lines.append("_No models returned for this API key._")
        _write_output(output_lines)
        return

    for model in models:
        max_output = getattr(model, "default_output_token_limit", "n/a")
        print(f"- {model.id} :: default_output_tokens={max_output}")
        output_lines.append(f"- {model.id} (default_output_tokens={max_output})")

    _write_output(output_lines)


def _write_output(lines: list[str]) -> None:
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "claude_models.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nModel list saved to {output_path}")


if __name__ == "__main__":
    main()

