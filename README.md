# Playground

Minimal multi-service playground.

## Services

- `src/test`: Redis write-loop service used for basic container/runtime checks
- `src/hurricane`: Gradio hurricane intensity-risk service with PyTorch inference
- `src/wildfire`: Gradio wildfire ignition-risk service with PyTorch inference

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

## Clean Caches

```bash
make clean
```

## Local Run (No Docker)

```bash
PYTHONPATH=src GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8000 python3 -m hurricane.main
PYTHONPATH=src GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8010 python3 -m wildfire.main
REDIS_HOST=localhost REDIS_PORT=6379 python3 src/test/main.py
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

## Tests

```bash
conda run -n playground pytest -q
```

## Service Docs

- `src/hurricane/README.md`
- `src/wildfire/README.md`
- `src/test/README.md`
