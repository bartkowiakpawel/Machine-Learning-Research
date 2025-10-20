"""Minimal compatibility layer emulating the legacy ``pyautogen`` interface.

The latest AutoGen releases split the project into ``autogen-agentchat`` and
``autogen-core`` packages and removed the top-level re-exports that older
code used. This module provides a very small subset of the former
``pyautogen`` API that is sufficient for the round-one and round-two debate
scripts in this project:

* ``AssistantAgent`` – wraps an LLM provider (OpenAI, Anthropic, Gemini) and
  produces a reply given the accumulated conversation.
* ``GroupChat`` and ``GroupChatManager`` – orchestrate a single pass where each
  specialist agent responds once and the final agent acts as moderator.

The implementation deliberately avoids depending on the new AutoGen runtime so
that the scripts keep working with just the official SDKs for each provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Dict, Iterable, List, Mapping, MutableSequence, Sequence

try:
    # OpenAI Python SDK >= 1.0
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

try:
    # Anthropic SDK
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - optional dependency
    Anthropic = None  # type: ignore[assignment]

try:
    # Gemini SDK
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]

import requests


ConversationMessage = Dict[str, str]


def _extract_single_config(llm_config: Mapping[str, Sequence[Mapping[str, str]]]) -> Mapping[str, str]:
    config_list = llm_config.get("config_list")
    if not config_list:
        raise ValueError("llm_config must provide a non-empty 'config_list'.")
    if len(config_list) != 1:
        raise ValueError("Only single-model configs are supported in this compatibility layer.")
    return config_list[0]


def _format_conversation(messages: Iterable[ConversationMessage]) -> str:
    lines: List[str] = []
    for message in messages:
        name = message.get("name") or message.get("role", "unknown")
        content = message.get("content", "")
        lines.append(f"{name}: {content}")
    return "\n".join(lines)


@dataclass
class AssistantAgent:
    """Light-weight replacement for ``pyautogen.AssistantAgent``."""

    name: str
    llm_config: Mapping[str, Sequence[Mapping[str, str]]]
    system_message: str | None = None
    max_output_tokens: int = 800
    temperature: float = 0.7
    top_p: float = 0.95

    def __post_init__(self) -> None:
        entry = _extract_single_config(self.llm_config)
        self._model = entry.get("model")
        self._api_key = entry.get("api_key")
        self._api_type = (entry.get("api_type") or "").lower()
        if not self._model:
            raise ValueError(f"Missing 'model' in llm_config for agent '{self.name}'.")
        if not self._api_key:
            raise ValueError(f"Missing API key in llm_config for agent '{self.name}'.")
        if self._api_type not in {"openai", "anthropic", "google", "groq"}:
            raise ValueError(
                f"Unsupported api_type '{entry.get('api_type')}'. Expected one of: openai, anthropic, google, groq."
            )

    # Public API -----------------------------------------------------------------
    def generate_reply(
        self,
        *,
        task_prompt: str,
        conversation: MutableSequence[ConversationMessage],
        summary_mode: bool = False,
    ) -> str:
        """Produce a single assistant reply given the accumulated conversation."""
        prompt = self._build_prompt(task_prompt=task_prompt, conversation=conversation, summary_mode=summary_mode)
        if self._api_type == "openai":
            return self._call_openai(prompt)
        if self._api_type == "anthropic":
            return self._call_anthropic(prompt)
        if self._api_type == "groq":
            return self._call_groq(prompt)
        return self._call_gemini(prompt)

    # Prompt preparation ---------------------------------------------------------
    def _build_prompt(
        self,
        *,
        task_prompt: str,
        conversation: Sequence[ConversationMessage],
        summary_mode: bool,
    ) -> str:
        base_instruction = self.system_message or dedent(
            f"""
            You are {self.name}, an expert machine learning researcher taking part in a debate.
            Engage constructively, highlight insights, and reference evidence from the data when helpful.
            """
        ).strip()

        conversation_text = _format_conversation(conversation)
        if conversation_text:
            history_block = f"Conversation so far:\n{conversation_text}"
        else:
            history_block = "Conversation so far: (no prior assistant messages)"

        if summary_mode:
            role_guidance = (
                "Synthesize the discussion above into a concise, actionable consensus summary. "
                "Highlight agreements, key risks, and next steps."
            )
        else:
            role_guidance = (
                "Add your perspective. Build on earlier comments, introduce complementary evidence, "
                "and keep the discussion focused on the user's task."
            )

        return dedent(
            f"""
            {base_instruction}

            Original task:
            {task_prompt}

            {history_block}

            Your objective:
            {role_guidance}
            """
        ).strip()

    # Backend calls --------------------------------------------------------------
    def _call_openai(self, prompt: str) -> str:
        if OpenAI is None:  # pragma: no cover - hard dependency branch
            raise RuntimeError("The openai package is not installed in this environment.")

        client = OpenAI(api_key=self._api_key)
        response = client.responses.create(
            model=self._model,
            input=prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return getattr(response, "output_text", "").strip()

    def _call_anthropic(self, prompt: str) -> str:
        if Anthropic is None:  # pragma: no cover - hard dependency branch
            raise RuntimeError("The anthropic package is not installed in this environment.")

        client = Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parts: List[str] = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                parts.append(getattr(block, "text", ""))
        return "\n".join(parts).strip()

    def _call_gemini(self, prompt: str) -> str:
        if genai is None:  # pragma: no cover - hard dependency branch
            raise RuntimeError("The google-generativeai package is not installed in this environment.")

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_output_tokens": self.max_output_tokens,
            },
            safety_settings=safety_settings,
        )
        candidates = list(getattr(response, "candidates", []) or [])
        chunks: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if parts:
                for part in parts:
                    text_part = getattr(part, "text", None)
                    if text_part:
                        chunks.append(text_part)
        if chunks:
            return "\n".join(chunks).strip()

        finish_reason = None
        safety_ratings = None
        if candidates:
            candidate0 = candidates[0]
            finish_reason = getattr(candidate0, "finish_reason", None)
            safety_ratings = getattr(candidate0, "safety_ratings", None)
        prompt_feedback = getattr(response, "prompt_feedback", None)

        reason_bits: list[str] = []
        if finish_reason is not None:
            reason_bits.append(f"finish_reason={finish_reason}")
        if safety_ratings:
            reason_bits.append(f"safety_ratings={safety_ratings}")
        if prompt_feedback:
            reason_bits.append(f"prompt_feedback={prompt_feedback}")
        detail = "; ".join(reason_bits) if reason_bits else "unknown blocking reason"

        return (
            "Gemini could not provide content because the response was blocked by safety or policy filters "
            f"({detail}). Please tweak the prompt or adjust the Gemini configuration."
        )

    def _call_groq(self, prompt: str) -> str:
        messages: List[dict[str, str]] = []
        system_message = self.system_message or (
            f"You are {self.name}, an academic moderator analyzing machine learning discussions."
        )
        messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": messages,
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Groq API error {response.status_code}: {response.text}"
            )
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("Groq API returned no choices for the completion request.")
        content = choices[0].get("message", {}).get("content", "")
        return content.strip()


class GroupChat:
    """Container mirroring the legacy AutoGen ``GroupChat`` object."""

    def __init__(self, *, agents: Sequence[AssistantAgent], messages: Sequence[ConversationMessage] | None = None):
        if not agents:
            raise ValueError("GroupChat requires at least one agent.")
        self.agents: List[AssistantAgent] = list(agents)
        self.messages: List[ConversationMessage] = list(messages or [])

    def initiate_chat(self, manager: "GroupChatManager", message: str) -> None:
        manager.run(initial_prompt=message)


class GroupChatManager:
    """Single-round debate orchestrator.

    The manager asks each specialist agent to respond once in order and then
    lets the final agent (typically the moderator) provide a closing summary.
    """

    def __init__(
        self,
        *,
        groupchat: GroupChat,
        llm_config: Mapping[str, Sequence[Mapping[str, str]]] | None = None,
        summary_agent_index: int | None = None,
    ) -> None:
        self.groupchat = groupchat
        self.llm_config = llm_config  # kept for signature compatibility
        if summary_agent_index is None:
            summary_agent_index = len(groupchat.agents) - 1
        if not (0 <= summary_agent_index < len(groupchat.agents)):
            raise ValueError("summary_agent_index must reference a valid agent index.")
        self.summary_agent_index = summary_agent_index

    def run(self, *, initial_prompt: str) -> None:
        conversation: List[ConversationMessage] = self.groupchat.messages or []
        conversation.append({"role": "user", "name": "User", "content": initial_prompt})

        for index, agent in enumerate(self.groupchat.agents):
            summary_mode = index == self.summary_agent_index
            role = "moderator" if summary_mode else "assistant"
            reply = agent.generate_reply(
                task_prompt=initial_prompt,
                conversation=conversation,
                summary_mode=summary_mode,
            ).strip()
            if not reply:
                raise RuntimeError(f"{agent.name} did not return any content for the debate.")
            conversation.append({"role": role, "name": agent.name, "content": reply})

        self.groupchat.messages = conversation
