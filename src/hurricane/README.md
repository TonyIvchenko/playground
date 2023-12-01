# Hurricane Service

Hurricane data + modeling pipeline for 24h rapid-intensification risk training and overlay generation.

## End-to-end Local Workflow (No Docker)

Run from repo root.

1. Download and merge hurricane datasets:

```bash
python -m src.hurricane.scripts.download_data
```

Sources used by the downloader:

- IBTrACS North Atlantic CSV
- latest HURDAT2 Atlantic file from NHC index

Merged output:

- `src/hurricane/data/raw/hurricane_tracks_merged.csv`

Current merged row mix:

- `merged` (matched records from both sources): 55,549 rows
- `ibtracs` only: 52,885 rows
- `hurdat2` only: 56 rows

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
python -m src.hurricane.scripts.train_model --model-version 0.5.2
```

Script outputs:

- processed training rows: `src/hurricane/data/processed/hurricane_training.csv`
- model artifact: `src/hurricane/model/hurricane_model.pt`

Current artifact snapshot:

- `model_version`: `0.5.2`
- `dataset_rows`: `94,123`
- `val_accuracy`: `0.8376`
- `val_balanced_accuracy`: `0.7969`
- `val_auc`: `0.8772`

4. Regenerate monthly overlay cube:

```bash
python -m src.hurricane.scripts.generate_overlay_tiles
```

Overlay outputs:

- `src/hurricane/tiles/overlay_cube.npz`
- `src/hurricane/tiles/overlay_config.json`

5. Launch combined app locally:

```bash
GMAPS_API_KEY=<google_maps_js_api_key> API_PORT=8080 python -m src.riskmap.main
```

The hurricane layers are served through the unified `riskmap` service.

6. Smoke test:

```bash
curl http://localhost:8080/health
```

Open UI at `http://localhost:8080/`.

## API Endpoints

- `GET /tiles/hurricane/{layer}/{frame_idx}/{z}/{x}/{y}.png` (served by `riskmap`)

## Docker

```bash
docker build -t riskmap -f src/riskmap/Dockerfile .
docker run --rm --name riskmap -p 8080:8080 -e API_PORT=8080 riskmap
```

## Tests

```bash
pytest -q src/hurricane/tests
```
