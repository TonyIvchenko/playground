# Playground

Minimal multi-service playground.

## Services

- `src/test_service`: redis test service
- `src/hurricane`: hurricane intensity-risk API + Gradio UI
- `src/wildfire`: wildfire ignition-risk API + Gradio UI

## Dependency Layout

1. Root `requirements.txt` contains only shared environment dependencies.
2. Each service defines its own dependencies in `src/<service>/requirements.txt`.
3. Root `environment.yml` installs root requirements plus each service requirements file.
4. When adding a service, create `src/<service>/main.py`, `Dockerfile`, and `requirements.txt`, then reference it in `environment.yml`.

## Setup

```bash
make setup
```

## Update Environment

```bash
make update
```

## Run Locally (No Docker)

```bash
PYTHONPATH=src API_PORT=8000 python3 -m hurricane.main
PYTHONPATH=src API_PORT=8010 python3 -m wildfire.main
```

## Docker

```bash
make build app=hurricane
make start app=hurricane port=8000
make smoke port=8000

make build app=wildfire
make start app=wildfire port=8010
make smoke port=8010
```

## Service Docs

- `src/hurricane/README.md`
- `src/wildfire/README.md`
