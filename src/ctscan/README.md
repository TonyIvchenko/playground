# CT Scan

Chest CT semantic-segmentation service with an axial slice viewer and uploadable
study bundles.

## What It Is

- chest CT only
- axial viewer with lung and finding overlays
- default rule-based issue labels:
  - emphysema
  - fibrotic pattern
  - ground-glass opacity
  - consolidation
- research/demo workflow only, not diagnosis

## Local Run

```bash
python main.py
```

Open `http://localhost:8080/`.
`PORT` defaults to `8080`.

Optional runtime overrides:

- `CTSCAN_MODEL_PATH`
- `CTSCAN_SAMPLES_MANIFEST_PATH`
- `CTSCAN_DEMO_CT_ZIPS_ROOT`
- `CTSCAN_LIDC_ROOT`

Upload caveat:

- the UI says `Upload DICOM file`
- the current service contract expects one zip bundle containing one DICOM study

To prebuild cached sample viewer assets before opening the app:

```bash
python scripts/warm_sample_viewer_cache.py
```

Subset or JSON examples:

```bash
python scripts/warm_sample_viewer_cache.py demo_lung1-001 --json
python scripts/warm_sample_viewer_cache.py --limit 3
```

## Data And Training

The short local path is:

```bash
python scripts/segmentation/download_data.py
python scripts/segmentation/build_dataset.py --overwrite
python scripts/segmentation/train_unet.py --model-version 0.1.0
```

Heavier real-data ingest and alternative training flows still live under
`scripts/segmentation/`, including:

- `download_lidc.py`
- `build_lidc_manifest.py`
- `build_luna_manifest.py`
- `build_nlstseg_manifest.py`
- `build_lndb_manifest.py`
- `train_unet_backbone.py`
- `search_unet_backbone.py`
- `train_legacy_vgg11_unet.py`

Use the relevant script `--help` for exact knobs instead of keeping long
runbooks in this README.

## Docker

```bash
docker build --pull -t ctscan -f src/ctscan/Dockerfile .
docker run --rm --name ctscan -p 8080:8080 -e PORT=8080 ctscan
curl --fail --silent --show-error http://127.0.0.1:8080/health
```

## API

- `GET /health`
- `POST /predict`

`POST /predict` accepts multipart form data:

- `study_zip` required unless `sample_id` is provided
- optional `sample_id`
- optional `age`
- optional `sex`
- optional `smoking_history`

Response includes:

- issue types
- per-issue lung damage percentage
- study summary and QC

## Tests

```bash
python -m pytest -q src/ctscan/tests
```

## Key Caveats

- full public-dataset ingest and training are large and long-running
- public dataset licenses vary; do not redistribute raw or derived data unless
  the source license allows it
- if you want the optional lungmask backend, install `SimpleITK` and `lungmask`
  manually in your environment
