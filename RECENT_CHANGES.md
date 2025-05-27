# Recent Changes

This is a short, human-maintained summary of notable repo-level changes. It is not a full replacement for `git log`.

Use it to answer questions like:

- what changed recently at the repo level
- which workflows or commands are new
- which service behaviors changed in visible ways

## Repo-Wide Docs And Structure

- Expanded the root [README.md](/Users/toxa/git/playground/README.md) so it reflects the full service inventory under `src/`.
- Added a root service inventory table, repo map, browser app checklist, troubleshooting section, first-run internet note, and fastest-local-command summary.
- Added [CONTRIBUTING.md](/Users/toxa/git/playground/CONTRIBUTING.md) and [ARCHITECTURE.md](/Users/toxa/git/playground/ARCHITECTURE.md) to make repo conventions and service shapes easier to understand quickly.
- Added a root [.editorconfig](/Users/toxa/git/playground/.editorconfig) for whitespace and indentation consistency.

## New Root Tooling

- Added `make smoke <service> [port]` for lightweight local smoke checks where the service can be started and probed automatically.
- Added `make test <service>` so service test suites can be run without remembering raw pytest paths.
- Added `make lint` and `make format` with a deliberately narrow Ruff-backed scope that is already green.
- Extended CI and root pytest collection to include `src/voiceforge/tests`.
- Added a root static-app smoke suite so the browser-only apps are exercised in pytest and CI instead of relying only on manual local runs.
- Added a CI-backed README code-block path checker so repo-managed file references in docs drift less easily.
- Added a README markdown consistency linter and wired it into both CI and `make lint`.
- Added a lightweight docs spellcheck pass and cleaned up repeated user-facing `hurricanes` typos in prose.
- Added tracked JSON/YAML config linting for workflow and repo config files, including duplicate-key detection.
- Added changed-path filtering in CI so docs-only or unrelated pushes can skip the heavier service-test and container-smoke jobs.
- Added a workflow summary artifact and job summary page that report suite results, smoke results, and skipped jobs for each CI run.
- Expanded CI caching to cover the full shared Python dependency set and Docker Buildx layer cache for image-based smoke jobs.
- Added a shared HTTP health-polling script that is now used by both local smoke checks and CI web-smoke jobs.
- Added a dedicated tracked-file guard in CI and `make lint` for `.DS_Store` and `__pycache__` regressions.
- Added helper scripts under `scripts/` for:
  - listing services
  - checking required service files
  - checking Dockerized HTTP services for `/health`
  - checking tracked and unignored file hygiene

## Service Contract Cleanup

- Standardized displayed service names for `CT Scan`, `Disasters`, and `VoiceForge`.
- Added consistent `service` fields to health responses where that contract was missing.
- Added or tightened focused health/app smoke tests for the relevant services.

## VoiceForge

- Added device logging so local runs record requested and resolved device information.
- Tuned the local Apple Silicon training workflow and documented the practical MPS run shape.
- Worked around a Gradio API-schema crash for file-backed components.
- Switched the reference upload flow away from the `ffprobe`-dependent Gradio path and added a more resilient audio-loading fallback.

## How To Keep This Useful

- Add a short note here when a repo-wide command, workflow, or documented service contract changes.
- Prefer updating this file for notable user-visible changes, not every tiny internal refactor.
- Keep entries short and high-signal so the file stays worth reading.
