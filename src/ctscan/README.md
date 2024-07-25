# CT Scan Service

Chest CT semantic segmentation service.

## Scope

- chest CT only
- axial slice viewer with lung and finding overlays
- default rule-based issue labels:
  - emphysema
  - fibrotic pattern
  - ground-glass opacity
  - consolidation
- legacy VGG11 U-Net checkpoint path supports:
  - ground-glass
  - consolidation
  - pleural effusion
- per-issue damage as `% of lung volume`
- research/demo workflow only, not diagnosis

## Layout

- `main.py`
- `study.py`
- `model/unet.py`
- `model/legacy_vgg11_unet.py`
- `scripts/segmentation/download_data.py`
- `scripts/segmentation/download_legacy_sources.py`
- `scripts/segmentation/download_lidc.py`
- `scripts/segmentation/build_legacy_dataset.py`
- `scripts/segmentation/build_lidc_manifest.py`
- `scripts/segmentation/build_luna_manifest.py`
- `scripts/segmentation/build_nlstseg_manifest.py`
- `scripts/segmentation/build_lndb_manifest.py`
- `scripts/segmentation/build_dataset.py`
- `scripts/segmentation/build_slice_dataset.py`
- `scripts/segmentation/export_vgg11_unet_dataset.py`
- `scripts/segmentation/train_legacy_vgg11_unet.py`
- `scripts/segmentation/train_unet.py`
- `scripts/segmentation/train_unet_backbone.py`
- `scripts/segmentation/search_unet_backbone.py`
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

3D patch mode (directly from processed 3D `.npz` dataset, no PNG export):

```bash
python scripts/segmentation/train_unet.py \
  --dataset-dir /Volumes/Extreme\ Pro/data/ctscan/processed/unet_composite_full \
  --output-path /Volumes/Extreme\ Pro/data/ctscan/experiments/unet3d_full.pt \
  --metrics-path /Volumes/Extreme\ Pro/data/ctscan/experiments/unet3d_full.metrics.json \
  --model-version 0.7.0-unet3d \
  --train-mode 3d \
  --patch-size 64,128,128 \
  --train-patches-per-case 4 \
  --val-patches-per-case 2 \
  --batch-size 1 \
  --epochs 20 \
  --device mps
```

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
  --preset legacy_png_best \
  --epochs 1 \
  --batch-size 4 \
  --max-train-batches 4 \
  --max-val-batches 2 \
  --max-test-batches 2 \
  --device cpu
```

Notes:
- Add `--list-presets` to print the available trainer presets.
- Add `--list-architectures` to print the supported trainer architectures.
- Add `--list-encoders` to print the available SMP encoder backbones for the trainer.
- Add `--list-encoder-weights` to print the supported trainer encoder-weight selectors.
- Add `--list-losses` to print the supported trainer losses.
- Add `--list-optimizers` to print the supported trainer optimizers.
- Add `--list-class-weight-modes` to print the supported trainer class-weight modes.
- Add `--list-devices` to print the supported trainer device selectors.
- Add `--list-schedulers` to print the supported trainer schedulers.
- Add `--list-samplers` to print the supported trainer samplers.
- Add `--list-augmentations` to print the supported trainer augmentations.
- Add `--list-metrics` to print the supported trainer metrics.
- Metric names are validated up front, so a typo in `--selection-metric` fails fast with the supported names.
- Core trainer choices like `--architecture`, `--loss`, and `--optimizer` are also validated up front.
- Trainer class-weight modes are also validated up front.
- Trainer device selectors are also validated up front.
- Trainer encoder-weight selectors are also validated up front.
- Scheduler, sampler, and augmentation names are also validated before the trainer starts.
- Add `--inspect-splits` to print train/val/test row counts before starting a run.
- Split inspection also reports whether rows came from `splits/*.csv` or `splits.json`, plus the exact source path for each split.
- Add `--show-output-paths` to print checkpoint/metrics/config/report locations before starting a run.
- Add `--dry-run` to print the fully resolved trainer config without starting training.
- Each training run writes a resolved config snapshot next to the metrics file as `*.config.json`.
- Each training run also writes a short Markdown report next to the metrics file as `*.md`.
- This is a smoke baseline only; run with larger `--max-cases`, more epochs, and remove `--max-*-batches` for real training.
- Split files are written to `.../slice_dataset_backbone_smoke/splits/*.csv`.
- PNG pairs are written under `.../slice_dataset_backbone_smoke/images` and `.../slice_dataset_backbone_smoke/masks`.
- `--preset legacy_png_best` applies the strongest measured legacy-PNG setting so far:
  `fpn + efficientnet-b1 + lovasz_ce + adamw + image_size=320 + batch_size=6 + lr=2e-4 + weight_decay=1e-4 + sampler=rare_fg`.
  On the current `data/legacy_compatible_png` cache, the latest 1-epoch check reached
  `val_mean_dice_fg=0.6945`, `val_mean_iou_fg=0.5321`, `val_loss=0.2900`,
  `test_mean_dice_fg=0.6730`.
- To run short ranked sweeps and reuse finished trials:

```bash
python scripts/segmentation/search_unet_backbone.py \
  --slice-dir data/legacy_compatible_png \
  --output-dir model/backbone_search \
  --architectures fpn \
  --encoders efficientnet-b0,efficientnet-b1 \
  --losses lovasz_ce \
  --image-sizes 320 \
  --batch-sizes 6 \
  --learning-rates 0.0002 \
  --weight-decays 0.0001 \
  --sampler rare_fg \
  --top-k 3 \
  --skip-existing
```

- Sweep outputs are written to `model/backbone_search/leaderboard.json` and
  `model/backbone_search/leaderboard.csv`.
- A Markdown summary is also written to `model/backbone_search/leaderboard.md`.
- The current winner is also written to `model/backbone_search/best_trial.json`.
- A human-readable winner card is also written to `model/backbone_search/best_trial.md`.
- Leaderboard rows and winner cards now include the resolved encoder-weight selector too.
- Each sweep also writes `model/backbone_search/run_summary.json` with overall trial counts.
- A Markdown version of the run summary is also written to `model/backbone_search/run_summary.md`.
- `--show-output-paths` now includes both run-summary paths, not just the JSON one.
- The run summary also records both the leaderboard sort metric and the trainer selection metric.
- Each sweep also writes `model/backbone_search/trial_plan.json` with the resolved planned trials.
- A Markdown version of the plan is also written to `model/backbone_search/trial_plan.md`.
- Trial plans now include the original 1-based trial index, so chunked sweeps keep the same numbering as the full grid.
- Trial plans and per-trial `*.config.json` snapshots now also record the resolved encoder-weight selector.
- Add `--show-output-paths` to print all sweep artifact paths before running anything.
- Add `--list-architectures` to print the supported sweep architectures.
- Add `--list-encoders` to print the available SMP encoder backbones for the sweep runner.
- Add `--list-encoder-weights` to print the supported sweep encoder-weight selectors.
- Add `--list-losses` to print the supported sweep losses.
- Add `--list-optimizers` to print the supported sweep optimizers.
- Add `--list-class-weight-modes` to print the supported sweep class-weight modes.
- Add `--list-devices` to print the supported sweep device selectors.
- Add `--list-schedulers` to print the supported sweep schedulers.
- Add `--list-samplers` to print the supported sweep samplers.
- Add `--list-augmentations` to print the supported sweep augmentations.
- Add `--list-metrics` to print the supported sweep metrics.
- Sweep metric names are validated up front, so typos in `--sort-metric` or `--selection-metric` fail fast.
- Sweep architecture, loss, and optimizer families are also validated up front before trial planning starts.
- Sweep class-weight modes are also validated up front.
- Sweep device selectors are also validated up front.
- Sweep encoder-weight selectors are also validated up front.
- Sweep scheduler, sampler, and augmentation names are also validated before trial planning starts.
- Each trial also writes `model/backbone_search/<trial>.config.json` with the resolved knob values.
- Add `--dry-run` to print the planned trial slugs without starting training.
- Repeated values in comma-separated sweep knobs are deduplicated in order before planning trials.
- Empty comma-separated sweep knobs now fail fast instead of silently producing an empty sweep.
- Add `--fail-fast` to stop immediately after the first failed trial.
- Use `--start-index` and `--end-index` to split a large sweep into smaller chunks.
- Invalid sweep windows now fail fast instead of silently clamping `--start-index` or accepting an inverted range.


## Legacy End-To-End Runbook

Use this path if you want the old VGG11 U-Net workflow end to end.

Output layout:
- `data/legacy_compatible/dataset/*.nii.gz`
- `data/legacy_compatible/mask/*mask.nii`
- `data/legacy_compatible_png/images/*.png`
- `data/legacy_compatible_png/masks/*.png`
- `model/legacy_vgg11_unet.pt`

Label ids:
- `0`: background
- `1`: ground-glass opacity
- `2`: consolidation
- `3`: pleural effusion

From `src/ctscan`:

1. Download raw source datasets.

```bash
python scripts/segmentation/download_legacy_sources.py \
  --raw-dir data/ctscan/raw/legacy_sources \
  --longciu-archive /path/to/longciu.zip
```

2. Build the legacy-compatible NIfTI dataset.

```bash
python scripts/segmentation/build_legacy_dataset.py \
  --raw-dir data/ctscan/raw/legacy_sources \
  --output-dir data/legacy_compatible \
  --longciu-mask-source staple \
  --plethora-vote-mode union \
  --overwrite
```

3. First training run: generate PNG cache from the NIfTI volumes and train.

```bash
python scripts/segmentation/train_legacy_vgg11_unet.py \
  --data-root data/legacy_compatible \
  --work-dir data/legacy_compatible_png \
  --output-path model/legacy_vgg11_unet.pt \
  --metrics-path model/legacy_vgg11_unet.metrics.json \
  --log-path model/legacy_vgg11_unet.train.log \
  --model-version legacy-vgg11-unet-0.1.0
```

4. Check the PNG cache before reuse.

```bash
python scripts/segmentation/check_png_sizes.py --root data/legacy_compatible_png
```

Healthy output should show:
- `invalid_images=0`
- `invalid_masks=0`
- one size only, usually `512x512`
- `pair_mismatches=0`

5. If the PNG cache is mixed-size, normalize it once.

```bash
python scripts/segmentation/resize_png_dataset.py --root data/legacy_compatible_png --size 512
```

6. Later training runs: reuse the cached PNG slices instead of converting NIfTI again.

```bash
python scripts/segmentation/train_legacy_vgg11_unet.py \
  --data-root data/legacy_compatible \
  --work-dir data/legacy_compatible_png \
  --output-path model/legacy_vgg11_unet.pt \
  --metrics-path model/legacy_vgg11_unet.metrics.json \
  --log-path model/legacy_vgg11_unet.train.log \
  --model-version legacy-vgg11-unet-0.1.0 \
  --skip-existing-png
```

7. Resume from the last best checkpoint if needed.

```bash
python scripts/segmentation/train_legacy_vgg11_unet.py \
  --data-root data/legacy_compatible \
  --work-dir data/legacy_compatible_png \
  --output-path model/legacy_vgg11_unet.pt \
  --metrics-path model/legacy_vgg11_unet.metrics.json \
  --log-path model/legacy_vgg11_unet.train.log \
  --model-version legacy-vgg11-unet-0.1.0 \
  --skip-existing-png \
  --resume-path model/legacy_vgg11_unet.pt
```

8. Launch the app with the trained checkpoint.

```bash
CTSCAN_MODEL_PATH=model/legacy_vgg11_unet.pt python main.py
```

Notes:
- `download_legacy_sources.py` downloads `MedSeg/SIRM` automatically.
- `PleThora` masks and paired CT series are also downloaded automatically. By default the script downloads CTs only for the `78` effusion-positive PleThora patients.
- `LongCIU` still requires a manual `longciu.zip` handoff because the DOI landing page does not expose a stable direct archive URL.
- The builder uses `MedSeg/SIRM` for exact `1/2/3` labels, `LongCIU` for `1/2`, and `PleThora` for `3`.
- `train_legacy_vgg11_unet.py` follows the notebook path closely: nibabel slice conversion, sklearn split, no resize by default, raw `state_dict` saves, and best-model save on validation improvement.
- `train_legacy_vgg11_unet.py` also saves `model/legacy_vgg11_unet.epochNNN.pt` after every epoch.
- If you already have older folders named `data/jemys_compatible` and `data/jemys_compatible_png`, use those same paths consistently in every command instead of mixing naming schemes.

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
