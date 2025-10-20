"""Configuration package for llms_science_debate."""

from .models_config import MODELS  # noqa: F401
from .settings import ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY  # noqa: F401

__all__ = ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "MODELS"]
