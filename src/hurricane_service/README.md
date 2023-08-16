# Hurricane Service

Minimal self-contained service with:
- FastAPI endpoints: `/health`, `/service-metadata`, `/predict`
- Embedded Gradio UI: `/ui`

## Local Run

From repo root:

```bash
PYTHONPATH=src API_PORT=8000 python3 -m hurricane_service.main
```

## Docker Run

From repo root:

```bash
docker build -t hurricane-service-docker -f src/hurricane_service/Dockerfile .
docker run --rm -p 8000:8000 hurricane-service-docker
```

## Tests

```bash
pytest -q src/hurricane_service/tests/test_main.py
```
