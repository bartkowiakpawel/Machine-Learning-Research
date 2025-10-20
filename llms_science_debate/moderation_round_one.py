"""Generate a research-style summary of the round-one debate using Groq."""

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
TRANSCRIPT_PATH = OUTPUT_DIR / "round_one_transcript.md"
SUMMARY_PATH = OUTPUT_DIR / "round_one_summary.md"


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

    transcript_text = _require_file(TRANSCRIPT_PATH)

    groq_config = _build_llm_config("groq_llama_70b", GROQ_API_KEY, "groq")

    user_prompt = f"""
You are an impartial academic moderator tasked with synthesizing a two-part theoretical debate.
The transcript below contains research-only commentary from Claude and GPT about machine-learning feature calibration.
Provide a concise, educational summary that highlights consensus, disagreements, methodological critiques, and proposed next steps.
Do not introduce financial advice or operational recommendations—stay purely in the realm of theoretical analysis.

Transcript:
{transcript_text}
""".strip()

    moderator = AssistantAgent(
        name="GroqModerator",
        llm_config=groq_config,
        system_message="Deliver a neutral, academically styled synthesis of the provided debate transcript.",
    )

    groupchat = GroupChat(agents=[moderator], messages=[])
    manager = GroupChatManager(groupchat=groupchat, llm_config=groq_config)

    groupchat.initiate_chat(manager, message=user_prompt)

    summary_content = groupchat.messages[-1]["content"]
    SUMMARY_PATH.write_text(summary_content, encoding="utf-8")
    print(f"Round one summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
