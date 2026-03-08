# Disasters Service

Unified Gradio app service that overlays wildfires and huricaines predictions on one Google Map.

## Layout

- `models/wildfires.py`
- `models/huricaines.py`
- `models/wildfires.pt`
- `models/huricaines.pt`
- `scripts/wildfires/download_data.py`
- `scripts/wildfires/train_model.py`
- `scripts/wildfires/generate_tiles.py`
- `scripts/huricaines/download_data.py`
- `scripts/huricaines/train_model.py`
- `scripts/huricaines/generate_tiles.py`
- `notebooks/wildfires.ipynb`
- `notebooks/huricaines.ipynb`
- `tests/test_disasters_main.py`
- `tests/wildfires/test_download_data.py`
- `tests/wildfires/test_train_model.py`
- `tests/wildfires/test_generate_tiles.py`
- `tests/huricaines/test_download_data.py`
- `tests/huricaines/test_train_model.py`
- `tests/huricaines/test_generate_tiles.py`

## Local Run

From `src/disasters`:

```bash
GMAPS_API_KEY=<google_maps_js_api_key> PORT=8080 python main.py
```

Open `http://localhost:8080/`.

## Data + Training

```bash
python scripts/huricaines/download_data.py
python scripts/huricaines/train_model.py --model-version 0.5.4
python scripts/huricaines/generate_tiles.py

python scripts/wildfires/download_data.py
python scripts/wildfires/train_model.py --model-version 0.5.3
python scripts/wildfires/generate_tiles.py
```

Notebooks:

```bash
conda run -n playground jupyter lab src/disasters/notebooks/huricaines.ipynb
conda run -n playground jupyter lab src/disasters/notebooks/wildfires.ipynb
```

## Endpoints

- `GET /health`
- `GET /tiles/{hazard}/{layer}/{frame_idx}/{z}/{x}/{y}.png`

Allowed values:

- `hazard`: `wildfires` or `huricaines`
- `layer`: `risk`, `activity`, `confidence`

## Docker

```bash
docker build -t disasters -f src/disasters/Dockerfile .
docker run --rm --name disasters -p 8080:8080 -e PORT=8080 disasters
```

## Tests

```bash
pytest -q src/disasters/tests
```
