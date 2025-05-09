# Contributing

This repo is a multi-service playground. Keep changes small, explicit, and local to the service you are touching unless the work is intentionally repo-wide.

## Repo Layout

- Root docs and shared tooling live at the repo root.
- Services live under `src/<service>`.
- Each service should have its own `README.md`.
- Python service tests live under `src/<service>/tests`.
- Large generated artifacts should stay out of git unless there is a clear reason to track them.

## Dependencies

- Root `requirements.txt` is for shared dependencies only.
- Service-specific dependencies belong in `src/<service>/requirements.txt`.
- If you add a new service requirements file, also add it to [environment.yml](/Users/toxa/git/playground/environment.yml).

## Environment And Config

- Prefer environment variables for runtime configuration.
- Use clear uppercase names like `PORT`, `SERVICE_NAME`, `GMAPS_API_KEY`, or `<SERVICE>_MODEL_PATH`.
- Document every new required env var in the relevant service README.
- If a service supports Docker or CI smoke checks, keep its env var behavior compatible there too.

## Running Services

- Use `make run <service> [port]` when possible.
- Browser-only apps should stay easy to run locally with a simple `python main.py`.
- FastAPI or Gradio services should expose `/health` if they are intended to be smoke-tested or containerized.

## Tests

- Run the narrowest relevant test suite for the service you changed.
- Current root `pytest.ini` includes:
  - `tests/test_static_app_smokes.py`
  - `src/test/tests`
  - `src/disasters/tests`
  - `src/ctscan/tests`
  - `src/voiceforge/tests`
- Other services may still have local tests even if they are not yet included in root pytest collection.
- Update or add tests when behavior changes, not only when bugs appear.

## Docs

- Update docs in the same change when commands, env vars, endpoints, or workflows change.
- Keep the root [README.md](/Users/toxa/git/playground/README.md) aligned with the actual service inventory.
- Add a short note to [RECENT_CHANGES.md](/Users/toxa/git/playground/RECENT_CHANGES.md) when a repo-wide command, workflow, or user-visible service contract changes.
- Keep README code blocks pointed at real repo-managed paths; CI validates those references now.
- Keep README heading structure simple and consistent; CI now lints the markdown shape too.
- Prefer correct prose spellings in docs even when legacy file or folder names are misspelled; keep the typo isolated to code paths, not user-facing text.
- Keep tracked workflow and config files valid JSON/YAML; CI now checks syntax and duplicate keys for that small config surface.
- Keep service READMEs focused on how to run, test, and troubleshoot that service.
- Prefer short, accurate docs over long aspirational docs.

## Service Conventions

- Static browser apps should keep most app logic in the browser and keep the Python entrypoint thin.
- Python services should keep startup code in `main.py` and move reusable logic into service modules.
- If a service has Docker support, its Dockerfile should work from the repo root as documented.
- If a service writes generated data, logs, checkpoints, or previews, keep those paths predictable and git-ignored.

## Hygiene

- Do not commit `.DS_Store`, `__pycache__`, local logs, or throwaway experiment outputs.
- Let your editor follow the root `.editorconfig` so whitespace, final newlines, and indentation stay predictable.
- Keep filenames, env vars, and user-facing service names consistent.
- Prefer incremental changes that are easy to review and easy to revert.

## Before You Finish A Change

- Run the relevant tests or smoke checks if available.
- Re-read the changed README or docs once after editing.
- Check `git diff` for accidental generated-file churn.
- If you changed behavior, make sure the docs and tests both reflect it.
