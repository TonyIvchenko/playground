# Hurricane Service

Standalone Gradio app backed by a PyTorch model trained on real hurricane tracks.

## Data + Modeling Workflow

From repo root:

```bash
PYTHONPATH=src python3 src/hurricane/scripts/download_data.py
```

This pulls and harmonizes multiple public sources:

- IBTrACS North Atlantic CSV
- Latest HURDAT2 Atlantic text file

and writes merged canonical tracks to:

- `src/hurricane/data/raw/hurricane_tracks_merged.csv`

Explore and evaluate in notebook:

- `src/hurricane/notebooks/hurricane_modeling.ipynb`

Train from script:

```bash
PYTHONPATH=src python3 src/hurricane/scripts/train_model.py --model-version 0.5.1
```

The trained artifact is loaded by the app from:

- `src/hurricane/model/hurricane_model.pt`

The model includes lag-aware dynamics:

- baseline state: `vmax_kt`, `min_pressure_mb`, `lat`, `lon`, `month`
- derived season/location terms: `month_sin`, `month_cos`, `abs_lat`, `pressure_deficit`
- recent trend terms: `dvmax_6h`, `dpres_6h`

## Local Run

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
pytest -q src/hurricane/tests
```
