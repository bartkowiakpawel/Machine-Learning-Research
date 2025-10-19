"""Smoke test for the OpenAI mini configuration."""

from __future__ import annotations

import sys
from pathlib import Path

from openai import APIError, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llms_science_debate.config.models_config import MODELS
from llms_science_debate.config.settings import OPENAI_API_KEY


def test_openai_mini() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured. Provide it in llms_science_debate/.env or environment.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    model_name = MODELS["openai_mini"]["name"]
    print(f">> Using model: {model_name}")

    try:
        response = client.responses.create(
            model=model_name,
            input="Say hello in one short sentence for a connectivity test.",
            max_output_tokens=50,
        )
        print("Response:")
        print(response.output_text)
    except APIError as exc:
        print(f"API call failed: {exc}")
        print("Tip: verify the model name is accessible for your OpenAI API key.")


if __name__ == "__main__":
    test_openai_mini()
