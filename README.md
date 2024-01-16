# Playground

Minimal multi-service playground.

## Services

- `src/test`: Redis write-loop service used for basic container/runtime checks
- `src/disasters`: unified app + modeling workspace with this layout:
- `src/disasters/models`: `wildfires.py`, `huricaines.py`, `wildfires.pt`, `huricaines.pt`
- `src/disasters/scripts/wildfires`: `download_data.py`, `train_model.py`, `generate_tiles.py`
- `src/disasters/scripts/huricaines`: `download_data.py`, `train_model.py`, `generate_tiles.py`
- `src/disasters/notebooks`: wildfires/huricaines EDA + evaluation notebooks
- `src/disasters/tests/wildfires`: download/train/tile tests
- `src/disasters/tests/huricaines`: download/train/tile tests
- `src/disasters/tests/test_disasters_main.py`: unified app inference tests
- `src/ctscan`: chest CT semantic segmentation service:
- `src/ctscan/study.py`: DICOM loading, lung segmentation, issue segmentation, overlay rendering
- `src/ctscan/model/unet.py`: U-Net architecture for semantic segmentation
- `src/ctscan/scripts/segmentation`: `download_data.py`, `download_lidc.py`, `build_lidc_manifest.py`, `build_luna_manifest.py`, `build_nlstseg_manifest.py`, `build_lndb_manifest.py`, `build_dataset.py`, `train_unet.py`
- `src/ctscan/tests`: study and API tests

## Dependency Layout

1. Root `requirements.txt` contains only shared environment dependencies.
2. Each service defines its own dependencies in `src/<service>/requirements.txt`.
3. Root `environment.yml` installs root requirements plus all service requirements files.
4. New service checklist: create `src/<service>/main.py`, `Dockerfile`, `requirements.txt`, then add its requirements file to `environment.yml`.

## Setup

```bash
make setup
```

## Update Environment

```bash
make update
```

## Local Run (No Docker)

```bash
cd src/disasters
GMAPS_API_KEY=<google_maps_js_api_key> PORT=8080 python main.py

cd ../ctscan
PORT=8080 python main.py

cd ../test
REDIS_HOST=localhost REDIS_PORT=6379 python main.py
```

## Rebuild Data + Models

```bash
cd src/disasters
python scripts/huricaines/download_data.py
python scripts/huricaines/train_model.py --model-version 0.5.4
python scripts/huricaines/generate_tiles.py

python scripts/wildfires/download_data.py
python scripts/wildfires/train_model.py --model-version 0.5.3
python scripts/wildfires/generate_tiles.py

cd ../ctscan
python scripts/segmentation/download_data.py
python scripts/segmentation/download_lidc.py --max-series 0
python scripts/segmentation/build_lidc_manifest.py --replace-lidc-rows --overwrite
python scripts/segmentation/build_luna_manifest.py --replace-luna-rows --overwrite
python scripts/segmentation/build_nlstseg_manifest.py --replace-nlstseg-rows --overwrite
python scripts/segmentation/build_lndb_manifest.py --replace-lndb-rows --overwrite
python scripts/segmentation/build_dataset.py --overwrite
python scripts/segmentation/train_unet.py --model-version 0.2.0
```

Open notebooks for EDA/eval:

```bash
conda run -n playground jupyter lab src/disasters/notebooks/huricaines.ipynb
conda run -n playground jupyter lab src/disasters/notebooks/wildfires.ipynb
```

## Make Commands

```bash
make setup
make update
make run disasters 8080
make run ctscan 8080
```

## Tests

```bash
pytest -q
```

## Service Docs

- `src/disasters/README.md`
- `src/ctscan/README.md`
- `src/test/README.md`
