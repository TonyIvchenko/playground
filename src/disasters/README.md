# Disasters

Unified Gradio app service that overlays wildfires and hurricanes predictions on one Google Map.

## What It Is

- unified wildfire and hurricane overlay app on one Google Map
- Gradio app with server-side tile serving
- UI says `hurricanes`, but legacy internal paths and API hazard keys still use `huricaines`

## Local Run

```bash
GMAPS_API_KEY=<google_maps_js_api_key> PORT=8080 python main.py
```

Open `http://localhost:8080/`.
`PORT` defaults to `8080`.

## Data And Training

```bash
python scripts/huricaines/download_data.py
python scripts/huricaines/train_model.py --model-version 0.5.4
python scripts/huricaines/generate_tiles.py

python scripts/wildfires/download_data.py
python scripts/wildfires/train_model.py --model-version 0.5.3
python scripts/wildfires/generate_tiles.py
```

- wildfire training data prep also writes `data/wildfires/raw/wildfires_us_overlay.csv`
- wildfire map overlays are generated over CONUS bounds

Notebooks:

```bash
conda run -n playground jupyter lab src/disasters/notebooks/huricaines.ipynb
conda run -n playground jupyter lab src/disasters/notebooks/wildfires.ipynb
```

## API

- `GET /health`
- `GET /tiles/{hazard}/{frame_idx}/{z}/{x}/{y}.png`
- `GET /tiles/{hazard}/{layer}/{frame_idx}/{z}/{x}/{y}.png`

Allowed values:

- `hazard`: `wildfires` or `huricaines`
- legacy `layer`: `risk`, `activity`, `confidence`

## Docker

```bash
docker build --pull -t disasters -f src/disasters/Dockerfile .
docker run --rm --name disasters -p 8080:8080 -e PORT=8080 disasters
curl --fail --silent --show-error http://127.0.0.1:8080/health
```

The CI workflow uses the same `/health` smoke pattern for this service.

## Tests

```bash
python -m pytest -q src/disasters/tests
```

## Key Caveats

- `GMAPS_API_KEY` is required for the map UI
- the API still accepts the legacy `huricaines` hazard key for compatibility
