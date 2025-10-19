"""Smoke test for Anthropic Claude 3 Haiku connectivity."""

from __future__ import annotations

import sys
from pathlib import Path

from anthropic import Anthropic, APIError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llms_science_debate.config.models_config import MODELS
from llms_science_debate.config.settings import ANTHROPIC_API_KEY


def test_claude_haiku() -> None:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured. Provide it in llms_science_debate/.env or environment.")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    model_name = MODELS["claude_haiku"]["name"]
    print(f">> Using model: {model_name}")

    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            temperature=0.2,
            messages=[
                {"role": "user", "content": "Provide a short greeting as a connectivity test."},
            ],
        )
        print("Response:")
        for block in response.content:
            if block.type == "text":
                print(block.text)
    except APIError as exc:
        print(f"API call failed: {exc}")
        print("Tip: verify the model name is accessible for your Anthropic API key.")


if __name__ == "__main__":
    test_claude_haiku()

