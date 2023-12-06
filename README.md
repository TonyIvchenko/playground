# Playground

Minimal multi-service playground.

## Services

- `src/test`: Redis write-loop service used for basic container/runtime checks
- `src/riskmap`: unified Gradio app service combining wildfire + hurricane overlays/inference
- `src/hurricane`: hurricane data/model pipeline (download, notebook EDA/eval, training, overlay generation)
- `src/wildfire`: wildfire data/model pipeline (download, notebook EDA/eval, training, overlay generation)

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

## Local Run (No Docker)

```bash
GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8080 python -m src.riskmap.main
REDIS_HOST=localhost REDIS_PORT=6379 python -m src.test.main
```

## Rebuild Data + Models

```bash
python -m src.hurricane.scripts.download_data
python -m src.hurricane.scripts.train_model --model-version 0.5.2
python -m src.hurricane.scripts.generate_overlay_tiles

python -m src.wildfire.scripts.download_data
python -m src.wildfire.scripts.train_model --model-version 0.5.3
python -m src.wildfire.scripts.generate_overlay_tiles
```

Open notebooks for EDA/eval:

```bash
conda run -n playground jupyter lab src/hurricane/notebooks/hurricane_modeling.ipynb
conda run -n playground jupyter lab src/wildfire/notebooks/wildfire_modeling.ipynb
```

## Make Commands

```bash
make setup
make update
make run riskmap 8080
```

## Tests

```bash
pytest -q
```

## Service Docs

- `src/riskmap/README.md`
- `src/hurricane/README.md`
- `src/wildfire/README.md`
- `src/test/README.md`
