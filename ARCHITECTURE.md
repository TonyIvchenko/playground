# Repo Shape

This repo is a multi-service playground. Keep each service in the smallest
shape that still does the job, and keep repo-wide rules short.

## Where Things Live

- `src/<service>`: each app or service
- `shared/`: browser-app shared assets
- `scripts/`: repo tooling
- `tests/`: root smoke and shared behavior tests

## Service Shapes

- Static browser apps: keep product logic in the browser, keep `main.py` thin,
  and use the shared launcher behavior for `HOST`, `PORT`, and `/health`.
- Python web services: use these only when the app needs server-side inference,
  uploads, larger local assets, or Gradio/FastAPI runtime behavior. Keep
  `main.py` thin and move reusable logic into service modules.
- Worker service: `src/test` is a narrow Redis-backed runtime check. Keep it
  explicit, one-shot friendly, and separate from the user-facing apps.

Some heavier services also keep training or data-prep code next to the app.
That is acceptable here when the app and training flow share assets and
assumptions.

## Working Rules

- Keep changes local to one service unless the work is intentionally repo-wide.
- Put service-specific dependencies in `src/<service>/requirements.txt`; keep
  root dependencies shared-only, and add new service requirements files to
  `environment.yml`.
- Document service-specific env vars, data setup, and caveats in that service's
  README instead of repeating them at the root.
- Run the narrowest relevant test or smoke check for the service you changed.
- When behavior changes, update the service README and tests in the same change.
- Keep generated outputs, logs, caches, and checkpoints predictable and out of
  git unless there is a clear reason to track them.
