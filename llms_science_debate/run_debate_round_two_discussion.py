"""Round-two theoretical debate between GPT and Claude focusing on critique and refinement."""

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

from llms_science_debate.config import ANTHROPIC_API_KEY, MODELS, OPENAI_API_KEY  # noqa: E402

INPUT_DIR = PROJECT_ROOT / "input_data"
OUTPUT_DIR = PROJECT_ROOT / "reasoning_outputs"


def _require_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path.read_text(encoding="utf-8")


def _build_llm_config(model_key: str, api_key: str | None, api_type: str) -> dict:
    if not api_key:
        raise RuntimeError(f"Missing API key for {model_key}. Set it in llms_science_debate/.env or the environment.")
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

    code_summary = _require_file(INPUT_DIR / "code_explanation.md")

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")
    csv_sections = []
    for csv_file in csv_files:
        csv_sections.append(f"{csv_file.name}:\n{csv_file.read_text(encoding='utf-8')[:1500]}")
    csv_block = "\n\n".join(csv_sections)

    round_one_summary = _require_file(OUTPUT_DIR / "round_one_summary.md")

    prompt = f"""
You are engaging in a second-round academic debate about machine-learning feature engineering.
All commentary must remain theoretical, research-oriented, and explicitly non-financial.

Resources:
- Round 1 consensus summary:
{round_one_summary}

- Code summary (truncated to 1500 chars):
{code_summary[:1500]}

- CSV snapshots (each truncated to 1500 chars):
{csv_block}

Round 2 objectives:
1. Critically assess the reasoning in the Round 1 summary. Identify any logical gaps, overlooked risks, or methodological issues.
2. Incorporate the updated context (code and CSV excerpts) to refine or challenge prior recommendations.
3. Propose academically grounded next steps for validating or enhancing the SMA-200 regime feature within ML workflows.

Keep the tone collegial and focused on research methods, not financial guidance.
""".strip()

    claude_config = _build_llm_config("claude_haiku", ANTHROPIC_API_KEY, "anthropic")
    openai_config = _build_llm_config("openai_full", OPENAI_API_KEY, "openai")

    gpt = AssistantAgent(
        name="GPT",
        llm_config=openai_config,
        system_message="Provide an academically rigorous critique rooted in ML best practices.",
    )
    claude = AssistantAgent(
        name="Claude",
        llm_config=claude_config,
        system_message="Offer a theoretical perspective on SMA-200 regime analysis with emphasis on methodological rigor.",
    )

    groupchat = GroupChat(agents=[gpt, claude], messages=[])
    manager = GroupChatManager(groupchat=groupchat, llm_config=openai_config)

    groupchat.initiate_chat(manager, message=prompt)

    transcript_lines = ["# LLM Debate Transcript (Round 2)", ""]
    for message in groupchat.messages:
        speaker = message.get("name") or message.get("role", "unknown")
        content = message.get("content", "")
        transcript_lines.append(f"## {speaker}")
        transcript_lines.append("")
        transcript_lines.append(content)
        transcript_lines.append("")

    transcript_path = OUTPUT_DIR / "round_two_transcript.md"
    transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
    print(f"Round 2 transcript saved to {transcript_path}")


if __name__ == "__main__":
    main()
