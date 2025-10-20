"""Summarize round-two debate transcript using Groq-hosted Llama 3.3 70B model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

try:
    from pyautogen import AssistantAgent, GroupChat, GroupChatManager  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    from llms_science_debate.autogen_compat import AssistantAgent, GroupChat, GroupChatManager

from llms_science_debate.config import GROQ_API_KEY, MODELS  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "output"
ROUND_ONE_SUMMARY = OUTPUT_DIR / "round_one_summary.md"
ROUND_TWO_TRANSCRIPT = OUTPUT_DIR / "round_two_transcript.md"
ROUND_TWO_SUMMARY = OUTPUT_DIR / "round_two_summary.md"


def _require_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path.read_text(encoding="utf-8")


def _build_llm_config(model_key: str, api_key: str | None, api_type: str) -> dict:
    if not api_key:
        raise RuntimeError(f"Missing API key for {model_key}. Update llms_science_debate/.env or environment variables.")
    model_name = MODELS[model_key]["name"]
    return {
        "config_list": [
            {
                "model": model_name,
                "api_key": api_key,
                "api_type": api_type,
            }
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    round_two_text = _require_file(ROUND_TWO_TRANSCRIPT)
    round_one_summary = ROUND_ONE_SUMMARY.read_text(encoding="utf-8") if ROUND_ONE_SUMMARY.exists() else ""

    groq_config = _build_llm_config("groq_llama_70b", GROQ_API_KEY, "groq")

    moderator_prompt = f"""
You are an impartial academic moderator producing a concise synthesis of Round 2.
The transcript below captures a theoretical debate about SMA-200 regime feature engineering.
Reference the prior Round 1 consensus summary where helpful, but focus on Round 2 insights, critiques, and next steps.
Avoid financial advice; keep the tone educational and research-oriented.

Round 1 consensus summary (optional reference):
{round_one_summary}

Round 2 transcript:
{round_two_text}
""".strip()

    moderator = AssistantAgent(
        name="GroqModeratorRound2",
        llm_config=groq_config,
        system_message="Deliver a neutral, academically styled synthesis of the provided debate material.",
    )

    groupchat = GroupChat(agents=[moderator], messages=[])
    manager = GroupChatManager(groupchat=groupchat, llm_config=groq_config)

    groupchat.initiate_chat(manager, message=moderator_prompt)

    summary_content = groupchat.messages[-1]["content"]
    ROUND_TWO_SUMMARY.write_text(summary_content, encoding="utf-8")
    print(f"Round 2 summary saved to {ROUND_TWO_SUMMARY}")


if __name__ == "__main__":
    main()
