# Wildfire Service

Minimal self-contained service with:
- FastAPI endpoints: `/health`, `/service-metadata`, `/predict`
- Embedded Gradio UI: `/ui`

## Local Run

From repo root:

```bash
PYTHONPATH=src API_PORT=8010 python3 -m wildfire.main
```

## Docker Run

From repo root:

```bash
docker build -t wildfire -f src/wildfire/Dockerfile .
docker run --rm --name wildfire -p 8010:8010 -e API_PORT=8010 wildfire
```

## Tests

```bash
pytest -q src/wildfire/tests/test_main.py
```
