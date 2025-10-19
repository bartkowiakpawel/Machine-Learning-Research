"""Smoke test for the Gemini Flash client configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llms_science_debate.config.models_config import MODELS
from llms_science_debate.config.settings import GEMINI_API_KEY


def test_gemini_flash() -> None:
    genai.configure(api_key=GEMINI_API_KEY)

    model_name = MODELS["gemini_flash"]["name"]
    print(f">> Using model: {model_name}")

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say hello.")
        print("Response:")
        print(response.text)
    except GoogleAPICallError as exc:
        print(f"API call failed: {exc}")
        print("Tip: verify the model name is available for your API key (run genai.list_models()).")


if __name__ == "__main__":
    test_gemini_flash()
