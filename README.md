# Playground

Minimal multi-service playground.

## Services

- `src/test`: Redis write-loop service used for basic container/runtime checks
- `src/disasters`: unified app + modeling workspace with this layout:
- `src/disasters/models`: `wildfires.py`, `huricaines.py`, `wildfires.pt`, `huricaines.pt`
- `src/disasters/scripts/wildfires`: `download_data.py`, `train_model.py`, `generate_tiles.py`
- `src/disasters/scripts/huricaines`: `download_data.py`, `train_model.py`, `generate_tiles.py`
- `src/disasters/notebooks`: wildfires/huricaines EDA + evaluation notebooks
- `src/disasters/tests/wildfires`: download/train/tile tests
- `src/disasters/tests/huricaines`: download/train/tile tests
- `src/disasters/tests/test_disasters_main.py`: unified app inference tests

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
GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8080 python -m src.disasters.main
REDIS_HOST=localhost REDIS_PORT=6379 python -m src.test.main
```

## Rebuild Data + Models

```bash
python -m src.disasters.scripts.huricaines.download_data
python -m src.disasters.scripts.huricaines.train_model --model-version 0.5.4
python -m src.disasters.scripts.huricaines.generate_tiles

python -m src.disasters.scripts.wildfires.download_data
python -m src.disasters.scripts.wildfires.train_model --model-version 0.5.3
python -m src.disasters.scripts.wildfires.generate_tiles
```

Open notebooks for EDA/eval:

```bash
conda run -n playground jupyter lab src/disasters/notebooks/huricaines.ipynb
conda run -n playground jupyter lab src/disasters/notebooks/wildfires.ipynb
```

## Make Commands

```bash
make setup
make update
make run disasters 8080
```

## Tests

```bash
pytest -q
```

## Service Docs

- `src/disasters/README.md`
- `src/test/README.md`
