# Hurricane Service

Minimal self-contained service with:
- FastAPI endpoints: `/health`, `/service-metadata`, `/predict`
- Embedded Gradio UI: `/ui`

## Local Run

From repo root:

```bash
PYTHONPATH=src API_PORT=8000 python3 -m hurricane.main
```

## Docker Run

From repo root:

```bash
docker build -t hurricane -f src/hurricane/Dockerfile .
docker run --rm --name hurricane -p 8000:8000 -e API_PORT=8000 hurricane
```

## Tests

```bash
pytest -q src/hurricane/tests/test_main.py
```
