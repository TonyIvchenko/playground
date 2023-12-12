# Disasters Service

Unified Gradio app service that overlays wildfires and hiricaines predictions on one Google Map.

## Layout

- `models/wildfires.py`
- `models/hiricaines.py`
- `models/wildfires.pt`
- `models/hiricaines.pt`
- `scripts/wildfires_download_data.py`
- `scripts/wildfires_train_model.py`
- `scripts/wildfires_generate_overlay_tiles.py`
- `scripts/hiricaines_download_data.py`
- `scripts/hiricaines_train_model.py`
- `scripts/hiricaines_generate_overlay_tiles.py`
- `notebooks/wildfires_modeling.ipynb`
- `notebooks/hiricaines_modeling.ipynb`
- `tests/test_disasters_main.py`
- `tests/test_wildfires_download_data.py`
- `tests/test_hiricaines_download_data.py`

## Local Run

From repo root:

```bash
GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8080 python -m src.disasters.main
```

Open `http://localhost:8080/`.

## Data + Training

```bash
python -m src.disasters.scripts.hiricaines_download_data
python -m src.disasters.scripts.hiricaines_train_model --model-version 0.5.2
python -m src.disasters.scripts.hiricaines_generate_overlay_tiles

python -m src.disasters.scripts.wildfires_download_data
python -m src.disasters.scripts.wildfires_train_model --model-version 0.5.3
python -m src.disasters.scripts.wildfires_generate_overlay_tiles
```

Notebooks:

```bash
conda run -n playground jupyter lab src/disasters/notebooks/hiricaines_modeling.ipynb
conda run -n playground jupyter lab src/disasters/notebooks/wildfires_modeling.ipynb
```

## Endpoints

- `GET /health`
- `GET /tiles/{hazard}/{layer}/{frame_idx}/{z}/{x}/{y}.png`

Allowed values:

- `hazard`: `wildfires` or `hiricaines`
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
