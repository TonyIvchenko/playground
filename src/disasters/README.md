# Disasters Service

Unified Gradio app service that overlays wildfire and hurricane predictions on one Google Map.

## Local Run

From repo root:

```bash
GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8080 python -m src.disasters.main
```

Open `http://localhost:8080/`.

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
