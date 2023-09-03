# Hurricane Service

Minimal standalone Gradio app.

## Local Run

From repo root:

```bash
PYTHONPATH=src API_PORT=8000 python3 -m hurricane.main
```

Open `http://localhost:8000/`.

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
