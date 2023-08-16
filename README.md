# Playground

Minimal multi-service playground.

## Services

- `src/test_service`: redis test service
- `src/hurricane_service`: hurricane intensity-risk API + Gradio UI
- `src/wildfire_service`: wildfire ignition-risk API + Gradio UI

## Dependency Layout

1. Root `requirements.txt` contains only shared environment dependencies.
2. Each service defines its own dependencies in `src/<service>/requirements.txt`.
3. Root `environment.yml` installs root requirements plus each service requirements file.
4. When adding a service, create `src/<service>/main.py`, `Dockerfile`, and `requirements.txt`, then reference it in `environment.yml`.

## Setup

```bash
make setup
```

## Test

```bash
make test
```

## Run Locally (No Docker)

```bash
PYTHONPATH=src API_PORT=8000 python3 -m hurricane_service.main
PYTHONPATH=src API_PORT=8010 python3 -m wildfire_service.main
```

## Docker

```bash
make build-hurricane-service
make start-hurricane-service
make smoke-hurricane-service

make build-wildfire-service
make start-wildfire-service
make smoke-wildfire-service
```

## Service Docs

- `src/hurricane_service/README.md`
- `src/wildfire_service/README.md`
