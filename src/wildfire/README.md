# Wildfire Service

Wildfire data + modeling pipeline for 24h ignition-risk training and overlay generation.

Scope note:

- current model is trained on public UCI wildfire tables (Portugal + Algeria), not a full U.S. nationwide feature stack yet
- this is a lightweight end-to-end reference implementation of the workflow

## End-to-end Local Workflow (No Docker)

Run from repo root.

1. Download and merge wildfire datasets:

```bash
python -m src.wildfire.scripts.download_data
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
python -m src.wildfire.scripts.train_model --model-version 0.5.3
```

Script outputs:

- processed training rows: `src/wildfire/data/processed/wildfire_training.csv`
- model artifact: `src/wildfire/model/wildfire_model.pt`

Current artifact snapshot:

- `model_version`: `0.5.3`
- `dataset_rows`: `760`
- `val_accuracy`: `0.7697`
- `val_balanced_accuracy`: `0.7693`
- `val_auc`: `0.8119`

4. Regenerate monthly overlay cube:

```bash
python -m src.wildfire.scripts.generate_overlay_tiles
```

Overlay outputs:

- `src/wildfire/tiles/overlay_cube.npz`
- `src/wildfire/tiles/overlay_config.json`

5. Launch combined app locally:

```bash
GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8080 python -m src.disasters.main
```

The wildfire layers are served through the unified `disasters` service.

6. Smoke test:

```bash
curl http://localhost:8080/health
```

Open UI at `http://localhost:8080/`.

## API Endpoints

- `GET /tiles/wildfire/{layer}/{frame_idx}/{z}/{x}/{y}.png` (served by `disasters`)

## Docker

```bash
docker build -t disasters -f src/disasters/Dockerfile .
docker run --rm --name disasters -p 8080:8080 -e API_PORT=8080 disasters
```

## Tests

```bash
pytest -q src/wildfire/tests
```
