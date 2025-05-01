# Playground

Minimal multi-service playground.

For a rolling summary of notable repo-level changes, see [RECENT_CHANGES.md](/Users/toxa/git/playground/RECENT_CHANGES.md).

## Services

- `src/bert`: browser-side toxicity analyzer
- `src/counterpoint`: browser-based argument ghostwriter
- `src/ctscan`: chest CT semantic segmentation viewer with dataset builders, training scripts, and model search tooling
- `src/debate`: browser-based argument sparring simulator
- `src/disasters`: Google Maps + Gradio service for wildfires and hurricanes overlays, plus training/notebook tooling
- `src/facemesh`: browser-based MediaPipe face mesh demo
- `src/manipulation`: browser-based manipulation-pattern analyzer
- `src/memorypalace`: browser-based 3D memory palace builder
- `src/realitycheck`: browser-based AI-likelihood analyzer for text, images, video, and URLs
- `src/realitymix`: browser-based live neural style transfer
- `src/test`: minimal Redis write-loop service used for basic runtime/container checks
- `src/vibedj`: browser-based creative direction generator for writing
- `src/voiceforge`: SpeechT5-based voice-cloning text-to-speech service

### Service Inventory

Use this as the quick repo-level reference. Service-specific env vars and caveats still live in each service README.

For a filesystem-backed view of the same repo surface, run `python scripts/list_services.py`.
To verify the minimum service docs and entrypoints exist, run `python scripts/check_service_files.py`.
To verify each service directory has the minimum expected files for its service type, run `python scripts/check_service_type_files.py`.
To verify Dockerized web services expose `/health`, run `python scripts/check_docker_health.py`.
To scan tracked and unignored files for `.DS_Store`, `__pycache__`, logs, and large files, run `python scripts/check_repo_hygiene.py`.
To validate repo-managed paths referenced inside README code blocks, run `python scripts/check_readme_code_paths.py`.
To lint README markdown structure, run `python scripts/check_markdown_readmes.py`.
To run the lightweight docs spellcheck, run `python scripts/check_docs_spelling.py`.
To verify Dockerized service READMEs still document the expected smoke commands, run `python scripts/check_docker_smoke_docs.py`.
To lint tracked JSON and YAML workflow/config files, run `python scripts/check_json_yaml_configs.py`.
To reuse the shared HTTP health poller used by both local smoke checks and CI, run `python scripts/poll_http_health.py --url http://127.0.0.1:8080/health`.
To fail fast on tracked `.DS_Store` or `__pycache__` paths, run `python scripts/check_tracked_junk.py`.
To verify the browser apps still use the shared top-of-page header contract, run `python scripts/check_browser_app_headers.py`.
To verify the browser apps still use the shared empty-state wording pattern, run `python scripts/check_browser_app_empty_states.py`.
To verify the browser apps still use the shared error-state wording pattern, run `python scripts/check_browser_app_error_states.py`.
To verify each service README's `Local Run` command still starts that service, run `python scripts/check_service_local_run.py --service <name>`.

| Service | Type | Run Command | Tests | Docker | Health Endpoint |
| --- | --- | --- | --- | --- | --- |
| `bert` | Static browser app | `make run bert 8080` | — | — | — |
| `counterpoint` | Static browser app | `make run counterpoint 8080` | — | — | — |
| `ctscan` | FastAPI + Gradio + ML service | `make run ctscan 8080` | `src/ctscan/tests` | `src/ctscan/Dockerfile` | `/health` |
| `debate` | Static browser app | `make run debate 8080` | — | — | — |
| `disasters` | FastAPI + Gradio + ML service | `make run disasters 8080` | `src/disasters/tests` | `src/disasters/Dockerfile` | `/health` |
| `facemesh` | Static browser app | `make run facemesh 8080` | — | — | — |
| `manipulation` | Static browser app | `make run manipulation 8080` | — | — | — |
| `memorypalace` | Static browser app | `make run memorypalace 8080` | — | — | — |
| `realitycheck` | Static browser app + local proxy endpoint | `make run realitycheck 8080` | — | — | — |
| `realitymix` | Static browser app | `make run realitymix 8080` | — | — | — |
| `test` | Redis worker service | `make run test` | `src/test/tests` | `src/test/Dockerfile` | — |
| `vibedj` | Static browser app | `make run vibedj 8080` | — | — | — |
| `voiceforge` | FastAPI + Gradio + TTS service | `make run voiceforge 8080` | `src/voiceforge/tests` | `src/voiceforge/Dockerfile` | `/health` |

### Repo Map

The repo currently falls into three broad shapes:

For the reasoning behind those shapes, see [ARCHITECTURE.md](/Users/toxa/git/playground/ARCHITECTURE.md).

- Static browser apps:
  - `bert`
  - `counterpoint`
  - `debate`
  - `facemesh`
  - `manipulation`
  - `memorypalace`
  - `realitycheck`
  - `realitymix`
  - `vibedj`
- Python web apps:
  - `ctscan`
  - `disasters`
  - `voiceforge`
- Runtime and worker tooling:
  - `test`

Training-heavy or data-heavy areas currently live in:

- `ctscan`
- `disasters`
- `voiceforge`

### Shippable Service Checklist

Before a service should be treated as shippable in this repo, it should have:

- a real `README.md` with local run instructions
- a working `main.py` entrypoint
- a clear place in the root service inventory
- documented env vars for anything required at runtime
- at least one practical smoke path:
  - `GET /health` for web services, or
  - a narrow local run path that is easy to verify
- tests if the service has meaningful Python behavior
- predictable output paths for generated data, logs, checkpoints, or previews
- Docker support if the service is meant to be containerized or CI-smoked

For browser-only demos, “shippable” can still mean lightweight. The bar is clarity and repeatability, not enterprise ceremony.

### Service Maturity

Use this as a rough guide, not a hard promise. In this repo, “maturity” means local repeatability and documentation quality, not production readiness.

- More operational:
  - `ctscan`
  - `disasters`
  - `test`
  - `voiceforge`
- Usable demos and experiments:
  - `bert`
  - `counterpoint`
  - `debate`
  - `facemesh`
  - `manipulation`
  - `memorypalace`
  - `realitycheck`
  - `realitymix`
  - `vibedj`

The first group is where we currently have the strongest combination of tests, Docker or health-check support, and fuller run documentation. The second group is still useful, but it is better to think of those services as polished local demos rather than operational tools.

## Dependency Layout

1. Root `requirements.txt` contains only shared environment dependencies.
2. Each service defines its own dependencies in `src/<service>/requirements.txt`.
3. Root `environment.yml` installs root requirements plus all service requirements files.
4. New service checklist: create `src/<service>/main.py`, `Dockerfile`, `requirements.txt`, `README.md`, add service tests under `src/<service>/tests`, then add the service requirements file to `environment.yml`.

### New Browser App Checklist

For browser-first apps that do not need FastAPI, Gradio, or Docker on day one:

1. Create `src/<service>/main.py` as a thin static file server.
2. Add `src/<service>/index.html`.
3. Add `src/<service>/README.md` with local run instructions and app notes.
4. Keep the Python entrypoint thin and put most product logic in the browser.
5. Load the shared browser base tokens from `/shared/browser-tokens.css` before inventing a new spacing, type, or color foundation.
6. Use the shared `app-header`, `app-kicker`, `app-title`, and `app-subtitle` classes for the top-of-page header so browser apps still feel related.
7. Use shared error-state wording and styling too: prefer `Couldn't <action>.` copy plus shared `is-error` status or pill states.
8. Use `PORT` for local run consistency.
9. Add the service to the root README service inventory once it exists.
10. Add tests only when the app has meaningful Python behavior or a smoke path worth automating.

## Environment

```bash
make setup
make update
```

## First-Run Internet

Some services need internet on first run because they download browser model assets or server-side model/data dependencies.

- Confirmed browser-model first run downloads:
  - `bert`
  - `counterpoint`
  - `debate`
  - `realitymix`
  - `vibedj`
- Browser-model apps that also appear to load model assets at runtime based on current `index.html` imports:
  - `manipulation`
  - `realitycheck`
- Heavier server-side setup paths that may download datasets or pretrained assets if they are not already cached locally:
  - `voiceforge`
  - `ctscan`
  - `disasters`

If you need a fully offline demo path, prefer services that are already cached locally or the simpler browser apps that do not depend on model downloads.

## Local Run

Use:

```bash
make run <service> [port]
```

### Fastest Local Commands

Use this when you just want the shortest practical command to get a service up locally.

| Service | Fastest Local Command | Notes |
| --- | --- | --- |
| `bert` | `make run bert 8080` | Static browser app. |
| `counterpoint` | `make run counterpoint 8080` | Static browser app. |
| `ctscan` | `make run ctscan 8080` | FastAPI + Gradio app; `/health` is available. |
| `debate` | `make run debate 8080` | Static browser app. |
| `disasters` | `make run disasters 8080` | FastAPI + Gradio app; set `GMAPS_API_KEY` first. |
| `facemesh` | `make run facemesh 8080` | Static browser app. |
| `manipulation` | `make run manipulation 8080` | Static browser app. |
| `memorypalace` | `make run memorypalace 8080` | Static browser app. |
| `realitycheck` | `make run realitycheck 8080` | Static browser app with local proxy endpoints. |
| `realitymix` | `make run realitymix 8080` | Static browser app. |
| `test` | `make run test` | Redis-backed worker; set `REDIS_*` vars first. |
| `vibedj` | `make run vibedj 8080` | Static browser app. |
| `voiceforge` | `make run voiceforge 8080` | FastAPI + Gradio app; local model files may be needed. |

For a quick local smoke check that starts a service, probes one endpoint, and shuts it back down, use:

```bash
make smoke <service> [port]
```

Examples:

```bash
make run disasters 8080
make run ctscan 8080
make run test
make smoke bert 8090
make smoke voiceforge 8091
```

If you omit the port for `disasters` or `ctscan`, `PORT` defaults to `8080`. The `test` service only needs `make run test`.
`make smoke` currently supports the browser apps plus the web services with `/health`. The `test` service is still checked through the dedicated README-local-run path instead of the HTTP smoke helper because it is a Redis-backed worker, not a web app.
Browser apps now also serve the shared token stylesheet at `/shared/browser-tokens.css`, so base browser UI tokens can stay aligned across the repo.

Set service-specific environment variables as needed:

- `GMAPS_API_KEY` for `disasters`
- `CTSCAN_MODEL_PATH`, `CTSCAN_SAMPLES_MANIFEST_PATH`, `CTSCAN_DEMO_CT_ZIPS_ROOT`, `CTSCAN_LIDC_ROOT` for `ctscan` (`CTSCAN_DEMO_CT_ZIPS_ROOT` expects study zip bundles)
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_KEY`, `REDIS_VALUE`, `SLEEP_SECONDS`, `REDIS_SOCKET_CONNECT_TIMEOUT`, `REDIS_SOCKET_TIMEOUT`, `REDIS_BACKOFF_INITIAL_SECONDS`, `REDIS_BACKOFF_MAX_SECONDS`, `REDIS_BACKOFF_MULTIPLIER` for `test` (`REDIS_HOST=localhost` for a local Redis)

For `test`, `REDIS_HOST`, `REDIS_KEY`, and `REDIS_VALUE` are trimmed at startup, and blank values are rejected.
When you run `test` in Docker, use `REDIS_HOST=host.docker.internal` on Docker Desktop or `REDIS_HOST=127.0.0.1` with `--network host` on Linux.

## Troubleshooting

### Port Already In Use

- Pass a different port to `make run <service> <port>` or `make smoke <service> <port>`.
- If you are not sure what is listening, run `lsof -iTCP:<port> -sTCP:LISTEN`.

### Missing Environment Variables

- Start with the root README and the service README for the app you are running.
- Common examples:
  - `GMAPS_API_KEY` for `disasters`
  - CT sample/model paths for `ctscan`
  - `REDIS_*` settings for `test`
- If a service boots but key features are missing, check the `/health` endpoint when it exists.

### First-Run Model Or Asset Downloads

- Some services need local models or generated assets before the full experience works.
- `voiceforge` may need model files under `src/voiceforge/models`.
- `ctscan` and `disasters` both have heavier data or tile assets than the browser-only apps.
- If first run is slow, check the service README before assuming the app is hung.

### Apple Silicon Notes

- MPS can work for local ML tasks, but it is often much slower than CUDA for heavier training.
- For `voiceforge`, expect training and model prep to be the slowest parts on Apple Silicon.
- If a process is alive but feels stalled, check actual step progress before restarting it.
- Browser-only apps are usually unaffected by these hardware differences.

## Docker

Build Docker images from the repo root so each Dockerfile can copy the shared `src/` tree:

```bash
docker build --pull -t test -f src/test/Dockerfile .
docker build --pull -t disasters -f src/disasters/Dockerfile .
docker build --pull -t ctscan -f src/ctscan/Dockerfile .
```

The shared Docker build context excludes checked-out service data, notebooks, tests, and CT experiment artifacts.

## Tests

Run `make update` first so the active `playground` env has `pytest` plus all service dependencies.

Run the same service test targets the CI workflow uses:

```bash
python -m pytest -q tests/test_static_app_smokes.py
make test test
make test disasters
make test ctscan
make test voiceforge
```

After `make update`, a plain root run also targets the same suites through `pytest.ini`:

```bash
python -m pytest -q
```

The root suite also includes browser-app smoke coverage in [tests/test_static_app_smokes.py](/Users/toxa/git/playground/tests/test_static_app_smokes.py), which starts each static app locally and verifies both `GET /` HTML and the shared `/shared/browser-tokens.css` asset.

If a service does not have a `tests/` directory yet, `make test <service>` fails fast and tells you which service test targets currently exist.

## Lint

Run the lightweight repo lint entrypoint with:

```bash
make lint
```

The first pass intentionally stays small and green: root helper scripts, README markdown consistency, docs spelling, Docker smoke doc checks for containerized services, tracked JSON/YAML config linting, tracked junk-file checks, browser-app header, empty-state, and error-state contract checks, type-specific service file checks, the shared browser token server/helper paths, service `main.py` entrypoints, the static browser-app smoke suite, and the health/app smoke tests we already maintain for `ctscan`, `disasters`, and `voiceforge`.

## Format

Run the matching formatter entrypoint with:

```bash
make format
```

The initial formatter scope matches `make lint`, so we keep formatting predictable while the older ML scripts and tests still have some style debt to burn down.

## CI

The GitHub workflow currently checks thirteen things on every push and pull request:

1. test collection from the repo root
2. README code-block path validation for repo-managed paths like scripts, tests, notebooks, and service entrypoints
3. README markdown consistency linting across the root and service READMEs
4. lightweight docs spellchecking against a curated typo list
5. Docker smoke documentation checks for the Dockerized service READMEs
6. tracked JSON/YAML workflow and config linting with syntax plus duplicate-key checks
7. a type-aware service file check that verifies each service directory still has the minimum expected files for its role
8. service test suites split across `tests/test_static_app_smokes.py`, `src/test/tests`, `src/disasters/tests`, `src/ctscan/tests`, and `src/voiceforge/tests`
9. a service-by-service Local Run check that verifies each service README still documents the expected startup command and that the app process actually boots
10. container health smokes for `disasters` and `ctscan` via `GET /health`, including fresh image builds
11. a Redis-backed runtime smoke for the `test` container, including a fresh image build
12. an uploaded `workflow-summary` artifact plus job summary page that lists suite results, smoke outcomes, and skipped jobs
13. a tracked-file guard that fails if `.DS_Store` or `__pycache__` paths ever re-enter git

It now also uses changed-path filters so docs-only or config-only pushes can skip the heavier service-test and container-smoke jobs.
The CI setup action now also caches the full shared Python dependency set, and Docker smoke builds use cached Buildx layers across runs.
The local `make smoke` flow and the CI web-smoke jobs now share the same HTTP health polling script instead of maintaining separate retry loops.

There is also a separate scheduled `ML Smoke` workflow for the heavier services. It runs capped synthetic smoke jobs for `ctscan`, `disasters`, and `voiceforge` on a weekly schedule and on manual dispatch, and uploads each job's tiny outputs as artifacts for debugging.

## Service Docs

- `src/bert/README.md`: local run, browser model notes
- `src/counterpoint/README.md`: local run, counterargument workflow, fallback notes
- `src/disasters/README.md`: local run, training, tiles, notebooks, Docker, tests
- `src/debate/README.md`: local run, sparring workflow, model notes
- `src/facemesh/README.md`: local run, webcam demo notes
- `src/manipulation/README.md`: local run, scoring caveats
- `src/memorypalace/README.md`: local run, controls, notes
- `src/realitycheck/README.md`: local run, input modes, caveats
- `src/realitymix/README.md`: local run, live style-transfer notes
- `src/vibedj/README.md`: local run, audio/model notes
- `src/voiceforge/README.md`: local run, data prep, training, Docker, tests
- `src/ctscan/README.md`: data ingest, legacy dataset build, model training/search, app run, tests
- `src/test/README.md`: local run, Docker, tests
