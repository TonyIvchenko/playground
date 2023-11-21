# Wildfire Service

Gradio app + PyTorch model for 24h wildfire ignition-risk inference, with monthly Google Maps overlays (2000-2030).

## End-to-end Local Workflow (No Docker)

Run from repo root.

1. Download and merge wildfire datasets:

```bash
PYTHONPATH=src python src/wildfire/scripts/download_data.py
```

Sources used by the downloader:

- UCI Forest Fires
- UCI Algerian Forest Fires

Merged output:

- `src/wildfire/data/raw/wildfire_training_merged.csv`

Current merged row mix:

- `uci_forestfires`: 517 rows
- `uci_algerian`: 243 rows

2. Open notebook for EDA and evaluation:

```bash
conda run -n playground jupyter lab src/wildfire/notebooks/wildfire_modeling.ipynb
```

Notebook responsibilities:

- data sanity checks and feature distributions
- training/eval run with accuracy, balanced accuracy, and AUROC
- optional artifact save back to `src/wildfire/model/wildfire_model.pt`

3. Train model from script:

```bash
PYTHONPATH=src python src/wildfire/scripts/train_model.py --model-version 0.5.3
```

Script outputs:

- processed training rows: `src/wildfire/data/processed/wildfire_training.csv`
- model artifact: `src/wildfire/model/wildfire_model.pt`

Current artifact snapshot (trained on March 7, 2026):

- `model_version`: `0.5.3`
- `dataset_rows`: `760`
- `val_accuracy`: `0.7697`
- `val_balanced_accuracy`: `0.7693`
- `val_auc`: `0.8119`

4. Regenerate monthly overlay cube:

```bash
PYTHONPATH=src python src/wildfire/scripts/generate_overlay_tiles.py
```

Overlay outputs:

- `src/wildfire/tiles/overlay_cube.npz`
- `src/wildfire/tiles/overlay_config.json`

5. Launch app locally:

```bash
PYTHONPATH=src GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8010 python -m wildfire.main
```

If `GMAPS_API_KEY` is omitted, inference still works and map area shows a clear setup message.

6. Smoke test:

```bash
curl http://localhost:8010/health
```

Open UI at `http://localhost:8010/`.

## API Endpoints

- `GET /health`
- `GET /tiles/{layer}/{frame_idx}/{z}/{x}/{y}.png`

## Docker

```bash
docker build -t wildfire -f src/wildfire/Dockerfile .
docker run --rm --name wildfire -p 8010:8010 -e API_PORT=8010 wildfire
```

## Tests

```bash
PYTHONPATH=src pytest -q src/wildfire/tests
```
