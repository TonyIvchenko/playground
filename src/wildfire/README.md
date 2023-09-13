# Wildfire Service

Standalone Gradio app backed by a PyTorch model trained on real wildfire records.

## Data + Modeling Workflow

From repo root:

```bash
PYTHONPATH=src python3 src/wildfire/scripts/download_data.py
```

Explore and evaluate in notebook:

- `src/wildfire/notebooks/wildfire_modeling.ipynb`

Train from script:

```bash
PYTHONPATH=src python3 src/wildfire/scripts/train_model.py --model-version 0.2.0
```

The trained artifact is loaded by the app from:

- `src/wildfire/model/wildfire_model.pt`

## Local Run

```bash
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
