"""Entry point for running the first-round multi-LLM debate."""

from __future__ import annotations

import sys
from pathlib import Path

from pyautogen import AssistantAgent, GroupChat, GroupChatManager

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from llms_science_debate.config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    MODELS,
    OPENAI_API_KEY,
)

DATA_DIR = PROJECT_ROOT / "input_data"
OUTPUT_DIR = PROJECT_ROOT / "output"


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

    code_summary = _require_file(DATA_DIR / "code_explanation.md")
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    csv_sections = []
    for csv_file in csv_files:
        csv_sections.append(f"{csv_file.name}:\n{csv_file.read_text(encoding='utf-8')[:1500]}")
    csv_block = "\n\n".join(csv_sections)

    user_prompt = f"""
Analyze the ML experiment results and code.
- Code summary:
{code_summary[:1500]}
- CSV results snapshot:
{csv_block}

Focus on overfitting and suggest improvements.
""".strip()

    gemini_config = _build_llm_config("gemini_flash", GEMINI_API_KEY, "google")
    claude_config = _build_llm_config("claude_haiku", ANTHROPIC_API_KEY, "anthropic")
    openai_config = _build_llm_config("openai_full", OPENAI_API_KEY, "openai")

    gemini = AssistantAgent(name="Gemini", llm_config=gemini_config)
    claude = AssistantAgent(name="Claude", llm_config=claude_config)
    gpt = AssistantAgent(name="GPT", llm_config=openai_config, system_message="Moderator")

    groupchat = GroupChat(agents=[gemini, claude, gpt], messages=[])
    manager = GroupChatManager(groupchat=groupchat, llm_config=openai_config)

    groupchat.initiate_chat(manager, message=user_prompt)

    summary_path = OUTPUT_DIR / "round_one_summary.md"
    summary_path.write_text(groupchat.messages[-1]["content"], encoding="utf-8")
    print(f"Debate summary saved to {summary_path}")

    transcript_lines = ["# LLM Debate Transcript", ""]
    for message in groupchat.messages:
        speaker = message.get("name") or message.get("role", "unknown")
        content = message.get("content", "")
        transcript_lines.append(f"## {speaker}")
        transcript_lines.append("")
        transcript_lines.append(content)
        transcript_lines.append("")

    transcript_path = OUTPUT_DIR / "round_one_transcript.md"
    transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
    print(f"Full transcript saved to {transcript_path}")


if __name__ == "__main__":
    main()
