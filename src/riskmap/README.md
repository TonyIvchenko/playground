# Riskmap Service

Unified Gradio app service that overlays wildfire and hurricane predictions on one Google Map.

## Local Run

From repo root:

```bash
GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8080 python -m src.riskmap.main
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
docker build -t riskmap -f src/riskmap/Dockerfile .
docker run --rm --name riskmap -p 8080:8080 -e API_PORT=8080 riskmap
```

## Tests

```bash
pytest -q src/riskmap/tests
```
