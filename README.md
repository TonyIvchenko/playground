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

## Dependency Layout

1. Root `requirements.txt` contains only shared environment dependencies.
2. Each service defines its own dependencies in `src/<service>/requirements.txt`.
3. Root `environment.yml` installs root requirements plus all service requirements files.
4. New service checklist: create `src/<service>/main.py`, `Dockerfile`, `requirements.txt`, `README.md`, add service tests under `src/<service>/tests`, then add the service requirements file to `environment.yml`.

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

Examples:

```bash
make run disasters 8080
make run ctscan 8080
make run test
```

If you omit the port for `disasters` or `ctscan`, `PORT` defaults to `8080`. The `test` service only needs `make run test`.

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
python -m pytest -q src/test/tests
python -m pytest -q src/disasters/tests
python -m pytest -q src/ctscan/tests
```

After `make update`, a plain root run also targets the same suites through `pytest.ini`:

```bash
python -m pytest -q
```

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
