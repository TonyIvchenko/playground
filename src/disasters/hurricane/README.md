# Hurricane Service

Hurricane data + modeling pipeline for 24h rapid-intensification risk training and overlay generation.

## End-to-end Local Workflow (No Docker)

Run from repo root.

1. Download and merge hurricane datasets:

```bash
python -m src.disasters.hurricane.scripts.download_data
```

Sources used by the downloader:

- IBTrACS North Atlantic CSV
- latest HURDAT2 Atlantic file from NHC index

Merged output:

- `src/disasters/hurricane/data/raw/hurricane_tracks_merged.csv`

Current merged row mix:

- `merged` (matched records from both sources): 55,549 rows
- `ibtracs` only: 52,885 rows
- `hurdat2` only: 56 rows

2. Open notebook for EDA and evaluation:

```bash
conda run -n playground jupyter lab src/disasters/hurricane/notebooks/hurricane_modeling.ipynb
```

Notebook responsibilities:

- data sanity checks and feature distributions
- training/eval run with accuracy, balanced accuracy, and AUROC
- optional artifact save back to `src/disasters/hurricane/model/hurricane_model.pt`

3. Train model from script:

```bash
python -m src.disasters.hurricane.scripts.train_model --model-version 0.5.2
```

Script outputs:

- processed training rows: `src/disasters/hurricane/data/processed/hurricane_training.csv`
- model artifact: `src/disasters/hurricane/model/hurricane_model.pt`

Current artifact snapshot:

- `model_version`: `0.5.2`
- `dataset_rows`: `94,123`
- `val_accuracy`: `0.8376`
- `val_balanced_accuracy`: `0.7969`
- `val_auc`: `0.8772`

4. Regenerate monthly overlay cube:

```bash
python -m src.disasters.hurricane.scripts.generate_overlay_tiles
```

Overlay outputs:

- `src/disasters/hurricane/tiles/overlay_cube.npz`
- `src/disasters/hurricane/tiles/overlay_config.json`

5. Launch combined app locally:

```bash
GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8080 python -m src.disasters.main
```

The hurricane layers are served through the unified `disasters` service.

6. Smoke test:

```bash
curl http://localhost:8080/health
```

Open UI at `http://localhost:8080/`.

## API Endpoints

- `GET /tiles/hurricane/{layer}/{frame_idx}/{z}/{x}/{y}.png` (served by `disasters`)

## Docker

```bash
docker build -t disasters -f src/disasters/Dockerfile .
docker run --rm --name disasters -p 8080:8080 -e API_PORT=8080 disasters
```

## Tests

```bash
pytest -q src/disasters/hurricane/tests
```
