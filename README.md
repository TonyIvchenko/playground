# Playground

Minimal multi-service playground.

## Services

- `src/test`: redis test service
- `src/hurricane`: standalone Gradio app for hurricane intensity risk (IBTrACS + HURDAT2 training data)
- `src/wildfire`: standalone Gradio app for wildfire ignition risk (multi-source tabular training data)

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
make build hurricane
make start hurricane 8000
make smoke 8000

make build wildfire
make start wildfire 8010
make smoke 8010
```

## Service Docs

- `src/hurricane/README.md`
- `src/wildfire/README.md`
