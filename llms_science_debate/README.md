# llms_science_debate

This package hosts a lightweight “science debate” workflow that critiques feature-engineering experiments with multiple LLM providers. The code is intentionally compact, so a newcomer can understand the flow in one pass.

## Repository layout

- `input_data/` – source material shared with the models. Each run expects `code_explanation.md` plus one or more CSV files with experiment metrics. Prompts trim each file to ~1 500 characters to control token usage.
- `reasoning_outputs/` – generated artefacts. Round 1 and Round 2 each produce a transcript (`round_one_transcript.md`, `round_two_transcript.md`) and a moderated summary (`round_one_summary.md`, `round_two_summary.md`). Feel free to inspect or clear these files between runs.
- `config/models_config.py` – mapping of short keys (e.g. `openai_full`, `claude_haiku`, `groq_llama_70b`) to provider model identifiers. Update this file to swap models or add new aliases.
- `config/settings.py` – loads API credentials from environment variables and a local `.env`. **The `.env` file is not committed**; you must create it yourself (or export the variables) with entries such as `OPENAI_API_KEY=…`, `ANTHROPIC_API_KEY=…`, and `GROQ_API_KEY=…`.
- `run_debate_round_one_discussion.py` / `run_debate_round_two_discussion.py` – stage the GPT↔Claude conversations. Each script reads the inputs, crafts an academic prompt, and writes the transcript into `reasoning_outputs/`.
- `moderation_round_one.py` / `moderation_round_two.py` – call the Groq-hosted Llama 3.3 70B model to generate neutral, research-style summaries.
- `tests/list_*_models.py` – helper scripts that list available models for Anthropic, Gemini, Groq, and OpenAI. They print to stdout and also drop Markdown summaries under `tests/output/`.

## Running the debates

1. Populate `input_data/code_explanation.md` with a concise overview of the experiment.
2. Place the relevant CSVs in `input_data/`.
3. Ensure the environment exposes valid API keys. For the current flow you need at least `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GROQ_API_KEY` (set via `.env` or shell variables).
4. Run `python llms_science_debate/run_debate_round_one_discussion.py` to generate the Round 1 transcript.
5. Run `python llms_science_debate/moderation_round_one.py` to produce the Round 1 summary.
6. Repeat the previous two steps with the Round 2 scripts. The second-round discussion automatically reads the Round 1 summary so it can critique or extend it.

All outputs are plain Markdown; re-running a step simply overwrites the existing file.

## Customising models

The orchestration helpers refer to models through the keys defined in `config/models_config.py`. To switch models, edit or add an entry there—no other files need to change. Remember that each key still requires the matching provider API key (e.g. adding a new Groq alias still depends on `GROQ_API_KEY`).

## Notes on credentials and safety

- Secrets live outside the repository. The `.env` file mentioned above is ignored by git; create your own copy locally.
- If you deploy the workflow elsewhere (CI, containers, cloud functions), you must provide equivalent environment variables at runtime.
- Some providers enforce strict safety filters. The current pipeline uses Groq for the moderation steps to avoid Gemini’s policy blocks encountered earlier.

With this structure in mind, a new contributor should be able to run both debate rounds, adjust prompts, or extend the flow for additional feature experiments.
