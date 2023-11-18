# Hurricane Service

Gradio app + PyTorch model for 24h rapid-intensification risk inference, with monthly Google Maps overlays (2000-2030).

## End-to-end Local Workflow (No Docker)

Run from repo root.

1. Download and merge hurricane datasets:

```bash
PYTHONPATH=src python src/hurricane/scripts/download_data.py
```

Sources used by the downloader:

- IBTrACS North Atlantic CSV
- latest HURDAT2 Atlantic file from NHC index

Merged output:

- `src/hurricane/data/raw/hurricane_tracks_merged.csv`

2. Open notebook for EDA and evaluation:

```bash
conda run -n playground jupyter lab src/hurricane/notebooks/hurricane_modeling.ipynb
```

Notebook responsibilities:

- data sanity checks and feature distributions
- training/eval run with accuracy, balanced accuracy, and AUROC
- optional artifact save back to `src/hurricane/model/hurricane_model.pt`

3. Train model from script:

```bash
PYTHONPATH=src python src/hurricane/scripts/train_model.py --model-version 0.5.2
```

Script outputs:

- processed training rows: `src/hurricane/data/processed/hurricane_training.csv`
- model artifact: `src/hurricane/model/hurricane_model.pt`

4. Regenerate monthly overlay cube:

```bash
PYTHONPATH=src python src/hurricane/scripts/generate_overlay_tiles.py
```

Overlay outputs:

- `src/hurricane/tiles/overlay_cube.npz`
- `src/hurricane/tiles/overlay_config.json`

5. Launch app locally:

```bash
PYTHONPATH=src GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8000 python -m hurricane.main
```

If `GMAPS_API_KEY` is omitted, inference still works and map area shows a clear setup message.

6. Smoke test:

```bash
curl http://localhost:8000/health
```

Open UI at `http://localhost:8000/`.

## API Endpoints

- `GET /health`
- `GET /tiles/{layer}/{frame_idx}/{z}/{x}/{y}.png`

## Docker

```bash
docker build -t hurricane -f src/hurricane/Dockerfile .
docker run --rm --name hurricane -p 8000:8000 -e API_PORT=8000 hurricane
```

## Tests

```bash
PYTHONPATH=src pytest -q src/hurricane/tests
```
