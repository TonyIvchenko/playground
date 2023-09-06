# Wildfire Service

Minimal standalone Gradio app backed by a trained PyTorch model.

## Local Run

From repo root:

```bash
PYTHONPATH=src python3 src/wildfire/scripts/train_model.py
PYTHONPATH=src API_PORT=8010 python3 -m wildfire.main
```

Open `http://localhost:8010/`.

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
