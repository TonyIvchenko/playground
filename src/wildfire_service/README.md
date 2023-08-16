# Wildfire Service

Minimal self-contained service with:
- FastAPI endpoints: `/health`, `/service-metadata`, `/predict`
- Embedded Gradio UI: `/ui`

## Local Run

From repo root:

```bash
PYTHONPATH=src API_PORT=8010 python3 -m wildfire_service.main
```

## Docker Run

From repo root:

```bash
docker build -t wildfire-service-docker -f src/wildfire_service/Dockerfile .
docker run --rm -p 8010:8010 wildfire-service-docker
```

## Tests

```bash
pytest -q src/wildfire_service/tests/test_main.py
```
