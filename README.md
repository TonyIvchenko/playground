# Playground

Minimal multi-service playground.

## Services

- `src/disasters`: Google Maps + Gradio service for wildfires and huricaines overlays, plus training/notebook tooling
- `src/ctscan`: chest CT semantic segmentation viewer with dataset builders, training scripts, and model search tooling
- `src/test`: minimal Redis write-loop service used for basic runtime/container checks

## Dependency Layout

1. Root `requirements.txt` contains only shared environment dependencies.
2. Each service defines its own dependencies in `src/<service>/requirements.txt`.
3. Root `environment.yml` installs root requirements plus all service requirements files.
4. New service checklist: create `src/<service>/main.py`, `Dockerfile`, `requirements.txt`, then add its requirements file to `environment.yml`.

## Setup

```bash
make setup
```

## Update Environment

```bash
make update
```

## Local Run

```bash
make run disasters 8080
make run ctscan 8080
make run test
```

Set service-specific environment variables as needed:

- `GMAPS_API_KEY` for `disasters`
- `CTSCAN_MODEL_PATH`, `CTSCAN_SAMPLES_MANIFEST_PATH`, `CTSCAN_DEMO_CT_ZIPS_ROOT`, `CTSCAN_LIDC_ROOT` for `ctscan`
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_KEY`, `REDIS_VALUE`, `SLEEP_SECONDS` for `test`

## Make Commands

```bash
make setup
make update
make run disasters 8080
make run ctscan 8080
make run test
```

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
3. Docker image builds for `test`, `disasters`, and `ctscan`
4. container health smokes for `disasters` and `ctscan` via `GET /health`

## Service Docs

- `src/disasters/README.md`: local run, training, tiles, notebooks, Docker, tests
- `src/ctscan/README.md`: data ingest, legacy dataset build, model training/search, app run, tests
- `src/test/README.md`: local run, Docker, tests
