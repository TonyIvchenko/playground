# CT Scan Service

Chest CT semantic segmentation service.

## Scope

- chest CT only
- semantic issue overlays on axial slices
- issue types in this version:
  - emphysema
  - fibrotic pattern
  - ground-glass opacity
  - consolidation
- per-issue damage as `% of lung volume`
- research/demo workflow only, not diagnosis

## Layout

- `main.py`
- `study.py`
- `model/unet.py`
- `scripts/segmentation/download_data.py`
- `scripts/segmentation/download_lidc.py`
- `scripts/segmentation/build_lidc_manifest.py`
- `scripts/segmentation/build_luna_manifest.py`
- `scripts/segmentation/build_nlstseg_manifest.py`
- `scripts/segmentation/build_lndb_manifest.py`
- `scripts/segmentation/build_dataset.py`
- `scripts/segmentation/train_unet.py`
- `tests/test_study.py`
- `tests/test_ctscan_main.py`

## Data Setup

From `src/ctscan`:

```bash
python scripts/segmentation/download_data.py
python scripts/segmentation/build_dataset.py --overwrite
python scripts/segmentation/train_unet.py --model-version 0.1.0
```

## Dataset Licenses

- LIDC-IDRI: CC BY 3.0 (`https://creativecommons.org/licenses/by/3.0/`) + TCIA attribution policy.
- LUNA16: CC BY 4.0 (`https://creativecommons.org/licenses/by/4.0/`).
- NLSTseg (Zenodo `14838349`): CC BY 4.0 (`https://creativecommons.org/licenses/by/4.0/`).
- LNDb (Grand Challenge rules): CC BY-NC-ND 4.0 (`https://creativecommons.org/licenses/by-nc-nd/4.0/`).

Do not redistribute raw or derived data unless the source license allows it.

## Full LIDC Ingest (Real Data)

From `src/ctscan`:

```bash
python scripts/segmentation/download_lidc.py --max-series 0
python scripts/segmentation/build_lidc_manifest.py --replace-lidc-rows --overwrite
python scripts/segmentation/build_luna_manifest.py --replace-luna-rows --overwrite
python scripts/segmentation/build_nlstseg_manifest.py --replace-nlstseg-rows --overwrite
python scripts/segmentation/build_lndb_manifest.py --replace-lndb-rows --overwrite
python scripts/segmentation/build_dataset.py --overwrite
python scripts/segmentation/train_unet.py --model-version 0.2.0
```

Notes:
- `--max-series 0` means all available LIDC CT series in TCIA NBIA metadata (currently 1,018 series).
- Full download is large and long-running.
- `build_lidc_manifest.py` converts LIDC nodules to class `5` voxel masks.
- `build_luna_manifest.py` converts LUNA16 world-coordinate nodules into class `5` voxel masks.
- `build_nlstseg_manifest.py` converts NLSTseg image/mask pairs into composite rows.
- `build_lndb_manifest.py` converts LNDb image/mask pairs into composite rows.
- For transient TCIA SSL/network errors, downloader now retries with both `requests` and `urllib` backends.
- Resume from a failed UID directly:
  `python scripts/segmentation/download_lidc.py --resume-series-uid <series_uid>`

## NLSTseg + LNDb Downloads

From `src/ctscan`:

1. Download NLSTseg (Zenodo record `14838349`) and extract:

```bash
mkdir -p data/ctscan/raw/nlstseg
python - <<'PY'
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile

root = Path("data/ctscan/raw/nlstseg")
with urlopen("https://zenodo.org/api/records/14838349", timeout=60) as response:
    record = json.load(response)
for file_obj in record["files"]:
    out_path = root / file_obj["key"]
    if out_path.exists():
        print("skip", out_path.name)
        continue
    print("download", out_path.name)
    with urlopen(file_obj["links"]["self"], timeout=60) as src, out_path.open("wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
for zip_path in sorted(root.glob("*.zip")):
    print("extract", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(root)
PY
```

2. Download LNDb (Zenodo record `8309612`) and extract:

```bash
mkdir -p data/ctscan/raw/lndb
python - <<'PY'
import json
from pathlib import Path
from urllib.request import urlopen

root = Path("data/ctscan/raw/lndb")
wanted = {
    "masks.rar",
    "data0.rar",
    "data1.rar",
    "data2.rar",
    "data3.rar",
    "data4.rar",
    "data5.rar",
    "allNods.csv",
    "LNDbAcqParams.csv",
    "trainset_csv.zip",
}
with urlopen("https://zenodo.org/api/records/8309612", timeout=60) as response:
    record = json.load(response)
for file_obj in record["files"]:
    key = file_obj["key"]
    if key not in wanted:
        continue
    out_path = root / key
    if out_path.exists():
        print("skip", out_path.name)
        continue
    print("download", out_path.name)
    with urlopen(file_obj["links"]["self"], timeout=60) as src, out_path.open("wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
PY
```

3. Extract LNDb archives (requires `7z`):

```bash
# macOS: brew install p7zip
for archive in data/ctscan/raw/lndb/*.rar; do
  7z x -y "$archive" -odata/ctscan/raw/lndb
done
python - <<'PY'
from pathlib import Path
import zipfile

root = Path("data/ctscan/raw/lndb")
for zip_path in sorted(root.glob("*.zip")):
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(root)
PY
```

4. Build manifest rows for all datasets:

```bash
python scripts/segmentation/build_lidc_manifest.py --replace-lidc-rows --overwrite
python scripts/segmentation/build_luna_manifest.py --replace-luna-rows --overwrite
python scripts/segmentation/build_nlstseg_manifest.py --replace-nlstseg-rows --overwrite
python scripts/segmentation/build_lndb_manifest.py --replace-lndb-rows --overwrite
python scripts/segmentation/build_dataset.py --overwrite
```

This writes:
- `data/ctscan/raw/public_datasets.json`
- `data/ctscan/samples/samples.json` (plus demo DICOM zip files)
- `data/ctscan/processed/unet_composite/manifest.json`
- `data/ctscan/processed/unet_composite/train.csv`
- `data/ctscan/processed/unet_composite/val.csv`
- `data/ctscan/processed/unet_composite/cases/*.npz`
- `model/unet.pt`
- `model/unet.metrics.json`

The composite builder merges:
- pre-labeled mask pairs listed in `data/ctscan/raw/composite_manifest.csv` (if provided)
- pseudo-labeled sample CT studies from `data/ctscan/samples/samples.json`

`composite_manifest.csv` columns:
- `case_id`
- `source`
- `image_path` (`.npy` or `.npz`)
- `mask_path` (`.npy` or `.npz`)
- optional `label_map` (JSON map from source ids to service ids, e.g. `{"1":3,"2":4}`)
- optional `spacing_z`, `spacing_y`, `spacing_x`

Case `.npz` format is U-Net ready:
- `image`: normalized float32 tensor `[z, y, x]`
- `mask`: uint8 semantic labels `[z, y, x]`
- `spacing`: float32 `[z, y, x]`

Class ids:
- `0`: background
- `1`: emphysema
- `2`: fibrotic pattern
- `3`: ground-glass opacity
- `4`: consolidation
- `5`: nodule

## U-Net Training

From `src/ctscan`:

```bash
python scripts/segmentation/train_unet.py \
  --dataset-dir data/ctscan/processed/unet_composite \
  --output-path model/unet.pt \
  --model-version 0.1.0
```

Useful knobs:
- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--image-size` (resizes every slice to a fixed square for batching)
- `--negative-stride` (keeps fewer empty/background-only slices)
- `--device auto|cpu|cuda|mps`

## Pretrained-Backbone Baseline

This path exports PNG image/mask slice pairs and trains a 2D U-Net with a pretrained encoder backbone.

From `src/ctscan`:

```bash
python scripts/segmentation/build_slice_dataset.py \
  --processed-dir /Volumes/Extreme\ Pro/data/ctscan/processed/unet_composite_full \
  --output-dir /Volumes/Extreme\ Pro/data/ctscan/processed/slice_dataset_backbone_smoke \
  --max-cases 2 \
  --max-slices-per-case 24 \
  --negative-stride 4 \
  --overwrite

python scripts/segmentation/train_unet_backbone.py \
  --slice-dir /Volumes/Extreme\ Pro/data/ctscan/processed/slice_dataset_backbone_smoke \
  --output-path model/unet_backbone_smoke.pt \
  --metrics-path model/unet_backbone_smoke.metrics.json \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --epochs 1 \
  --batch-size 4 \
  --max-train-batches 4 \
  --max-val-batches 2 \
  --max-test-batches 2 \
  --device cpu
```

Notes:
- This is a smoke baseline only; run with larger `--max-cases`, more epochs, and remove `--max-*-batches` for real training.
- Split files are written to `.../slice_dataset_backbone_smoke/splits/*.csv`.
- PNG pairs are written under `.../slice_dataset_backbone_smoke/images` and `.../slice_dataset_backbone_smoke/masks`.

## Optional Lungmask Backend

Default backend is an internal threshold method.

If you want lungmask backend, install manually in your env:

```bash
pip install SimpleITK lungmask
```

Then `/health` should report `segmentation_backend: "lungmask"`.

## Local Run

From `src/ctscan`:

```bash
python main.py
```

Open `http://localhost:8080/`.

## Docker

```bash
docker build -t ctscan -f src/ctscan/Dockerfile .
docker run --rm --name ctscan -p 8080:8080 -e PORT=8080 ctscan
```

## API

- `GET /health`
- `POST /predict`

`POST /predict` accepts multipart form data:
- `study_zip` (required unless `sample_id` is provided)
- optional `sample_id`
- optional `age`
- optional `sex`
- optional `smoking_history`

Response includes:
- issue types
- per-issue lung damage `%`
- study summary and QC

## Tests

```bash
pytest -q src/ctscan/tests
```
