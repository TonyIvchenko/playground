# Playground

Minimal multi-service playground.

## Services

- `src/bert`: browser-side toxicity analyzer
- `src/counterpoint`: browser-based argument ghostwriter
- `src/ctscan`: chest CT semantic segmentation viewer with dataset builders, training scripts, and model search tooling
- `src/debate`: browser-based argument sparring simulator
- `src/disasters`: Google Maps + Gradio service for wildfires and huricaines overlays, plus training/notebook tooling
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
To verify Dockerized web services expose `/health`, run `python scripts/check_docker_health.py`.

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
5. Use `PORT` for local run consistency.
6. Add the service to the root README service inventory once it exists.
7. Add tests only when the app has meaningful Python behavior or a smoke path worth automating.

## Environment

```bash
make setup
make update
```

## Local Run

Use:

```bash
make run <service> [port]
```

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
`make smoke` currently supports the browser apps plus the web services with `/health`; the Redis-backed `test` service still uses its README-specific smoke flow.

Set service-specific environment variables as needed:

- `GMAPS_API_KEY` for `disasters`
- `CTSCAN_MODEL_PATH`, `CTSCAN_SAMPLES_MANIFEST_PATH`, `CTSCAN_DEMO_CT_ZIPS_ROOT`, `CTSCAN_LIDC_ROOT` for `ctscan` (`CTSCAN_DEMO_CT_ZIPS_ROOT` expects study zip bundles)
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_KEY`, `REDIS_VALUE`, `SLEEP_SECONDS`, `REDIS_SOCKET_CONNECT_TIMEOUT`, `REDIS_SOCKET_TIMEOUT`, `REDIS_BACKOFF_INITIAL_SECONDS`, `REDIS_BACKOFF_MAX_SECONDS`, `REDIS_BACKOFF_MULTIPLIER` for `test` (`REDIS_HOST=localhost` for a local Redis)

For `test`, `REDIS_HOST`, `REDIS_KEY`, and `REDIS_VALUE` are trimmed at startup, and blank values are rejected.
When you run `test` in Docker, use `REDIS_HOST=host.docker.internal` on Docker Desktop or `REDIS_HOST=127.0.0.1` with `--network host` on Linux.

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
make test test
make test disasters
make test ctscan
make test voiceforge
```

After `make update`, a plain root run also targets the same suites through `pytest.ini`:

```bash
python -m pytest -q
```

If a service does not have a `tests/` directory yet, `make test <service>` fails fast and tells you which service test targets currently exist.

## Lint

Run the lightweight repo lint entrypoint with:

```bash
make lint
```

The first pass intentionally stays small and green: root helper scripts, service `main.py` entrypoints, and the health/app smoke tests we already maintain for `ctscan`, `disasters`, and `voiceforge`.

## Format

Run the matching formatter entrypoint with:

```bash
make format
```

The initial formatter scope matches `make lint`, so we keep formatting predictable while the older ML scripts and tests still have some style debt to burn down.

## CI

The GitHub workflow currently checks four things on every push and pull request:

1. test collection from the repo root
2. service test suites split across `src/test/tests`, `src/disasters/tests`, and `src/ctscan/tests`
3. container health smokes for `disasters` and `ctscan` via `GET /health`, including fresh image builds
4. a Redis-backed runtime smoke for the `test` container, including a fresh image build

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
