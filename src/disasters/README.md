# Disasters Service

Unified Gradio app service that overlays wildfire and hurricane predictions on one Google Map.

## Layout

- `models/wildfire_model.py`
- `models/hurricane_model.py`
- `models/wildfire_artifact.py`
- `models/hurricane_artifact.py`
- `scripts/wildfire_download_data.py`
- `scripts/wildfire_train_model.py`
- `scripts/wildfire_generate_overlay_tiles.py`
- `scripts/hurricane_download_data.py`
- `scripts/hurricane_train_model.py`
- `scripts/hurricane_generate_overlay_tiles.py`
- `notebooks/wildfire_modeling.ipynb`
- `notebooks/hurricane_modeling.ipynb`
- `tests/test_disasters_main.py`
- `tests/test_wildfire_download_data.py`
- `tests/test_hurricane_download_data.py`

## Local Run

From repo root:

```bash
GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8080 python -m src.disasters.main
```

Open `http://localhost:8080/`.

## Data + Training

```bash
python -m src.disasters.scripts.hurricane_download_data
python -m src.disasters.scripts.hurricane_train_model --model-version 0.5.2
python -m src.disasters.scripts.hurricane_generate_overlay_tiles

python -m src.disasters.scripts.wildfire_download_data
python -m src.disasters.scripts.wildfire_train_model --model-version 0.5.3
python -m src.disasters.scripts.wildfire_generate_overlay_tiles
```

Notebooks:

```bash
conda run -n playground jupyter lab src/disasters/notebooks/hurricane_modeling.ipynb
conda run -n playground jupyter lab src/disasters/notebooks/wildfire_modeling.ipynb
```

## Endpoints

- `GET /health`
- `GET /tiles/{hazard}/{layer}/{frame_idx}/{z}/{x}/{y}.png`

Allowed values:

- `hazard`: `wildfire` or `hurricane`
- `layer`: `risk`, `activity`, `confidence`

## Docker

```bash
docker build -t disasters -f src/disasters/Dockerfile .
docker run --rm --name disasters -p 8080:8080 -e API_PORT=8080 disasters
```

## Tests

```bash
pytest -q src/disasters/tests
```
