"""Utility script to inspect available OpenAI models for the configured API key."""

from __future__ import annotations

import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llms_science_debate.config.settings import OPENAI_API_KEY

OPENAI_MODELS_ENDPOINT = "https://api.openai.com/v1/models"


def main() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it in llms_science_debate/.env or the environment to list models."
        )

    print("Listing OpenAI models available for this API key:\n")
    output_lines = ["# OpenAI Models", ""]

    response = requests.get(
        OPENAI_MODELS_ENDPOINT,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Accept": "application/json",
        },
        timeout=30,
    )
    if response.status_code != 200:
        print(f"OpenAI API returned {response.status_code}: {response.text}")
        output_lines.append(f"_OpenAI API returned {response.status_code}: {response.text}_")
        _write_output(output_lines, filename="openai_models.md")
        return

    payload = response.json()
    models = payload.get("data") or []
    if not models:
        print("No models returned. Check API key permissions at platform.openai.com.")
        output_lines.append("_No models returned for this API key._")
        _write_output(output_lines, filename="openai_models.md")
        return

    for model in models:
        model_id = model.get("id", "<unknown>")
        created = model.get("created", "n/a")
        owned_by = model.get("owned_by", "n/a")
        print(f"- {model_id} :: owner={owned_by}, created={created}")
        output_lines.append(f"- {model_id} (owner={owned_by}, created={created})")

    _write_output(output_lines, filename="openai_models.md")


def _write_output(lines: list[str], *, filename: str) -> None:
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nModel list saved to {output_path}")


if __name__ == "__main__":
    main()
