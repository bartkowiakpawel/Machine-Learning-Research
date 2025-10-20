# llms_science_debate

This module hosts the multi‑round “science debate” workflow that I use to critique feature-engineering experiments with several LLM APIs. The code is intentionally small, so a new contributor can read through it in one sitting.

## Repository layout

- `input_data/` – source material shared with the models. Every run expects `code_explanation.md` plus one or more CSV files with experiment results. The debate prompts trim each file to ~1500 characters to stay under token limits.
- `output/` – generated artefacts. Round 1 and Round 2 each emit a transcript (`round_one_transcript.md`, `round_two_transcript.md`) and a moderated summary (`round_one_summary.md`, `round_two_summary.md`). Feel free to inspect or delete these files between runs.
- `config/models_config.py` – mapping of short keys to provider/model names. Update this file if you want to switch to a different model family.
- `config/settings.py` – reads credentials from environment variables and a local `.env`. **The `.env` file is not tracked in git.** You must create your own file (or export variables manually) with entries such as `OPENAI_API_KEY=...`, `ANTHROPIC_API_KEY=...`, and `GROQ_API_KEY=...`.
- `run_debate_round_one_discussion.py` / `run_debate_round_two_discussion.py` – stage the GPT↔Claude conversations. Both scripts load the input data, craft a research-only prompt, and write the transcript to `output/`.
- `moderation_round_one.py` / `moderation_round_two.py` – call the Groq Llama 3.3 70B model to summarise each transcript.
- `tests/list_*_models.py` – helper scripts that list available models for Anthropic, Gemini, Groq, and OpenAI. Each command prints to stdout and also writes a Markdown table under `tests/output/`.

## Running the debates

1. Populate `input_data/code_explanation.md` with a high-level summary of the experiment.
2. Drop the relevant CSV files into `input_data/`.
3. Ensure your `.env` contains valid API keys. At minimum you need `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GROQ_API_KEY` for the current flow.
4. Run `python llms_science_debate/run_debate_round_one_discussion.py` to create the Round 1 transcript.
5. Run `python llms_science_debate/moderation_round_one.py` to generate the Round 1 summary.
6. Repeat the same two steps with the Round 2 scripts. Round 2 will automatically read the Round 1 summary to critique it.

You can re-run any step at will; new outputs overwrite the previous Markdown files.

## Customising models

The execution helpers rely on short keys (e.g. `openai_full`, `claude_haiku`, `groq_llama_70b`). To swap a model, edit `config/models_config.py` and adjust the entry’s `name` or add a new key. The debate scripts call `_build_llm_config` with these keys, so you only need to change the dictionary in one place. Remember that each entry also depends on the matching API key (for example, adding a new Groq alias still requires a valid `GROQ_API_KEY`).

## Notes on credentials

- The project never commits secrets. The `.env` mentioned above lives in your working copy only.
- If you deploy this code elsewhere (CI, containers, etc.) you must provide equivalent environment variables.
- Some providers enforce safety filters that may refuse to answer even harmless prompts. The workflow now uses Groq for moderation to avoid Gemini’s stricter policies.

This should be enough information for a new collaborator to reproduce the debate pipeline or modify it for other features. Ping me if anything in the flow feels unclear.***
