# CT Scan Service

Chest CT pulmonary nodule triage service.

## Scope

- chest CT only
- pulmonary nodule review only
- research/demo workflow only, not diagnosis

## Layout

- `models/nodules.py`
- `models/nodules.pt`
- `scripts/nodules/download_data.py`
- `scripts/nodules/train_model.py`
- `notebooks/nodules.ipynb`
- `tests/nodules/test_download_data.py`
- `tests/nodules/test_train_model.py`
- `tests/nodules/test_nodules.py`
- `tests/test_study.py`
- `tests/test_ctscan_main.py`

## Prepare Data

From `src/ctscan`:

```bash
python scripts/nodules/download_data.py
```

This does three things:
- writes a public dataset manifest for `LIDC-IDRI`, `LUNA16`, and `LNDb`
- downloads two de-identified TCIA chest CT demo studies for the UI sample dropdown
- downloads a small real `LIDC-IDRI` training subset and converts XML annotations into canonical 3D nodule patches
- generates a deterministic smoke fallback dataset for tests/offline fallback

By default the real training subset uses `7` LIDC studies, which is the subset size used for the currently shipped checkpoint. Increase it when you want more real patches:

```bash
python scripts/nodules/download_data.py --lidc-study-limit 10
```

If you only want the real training subset and not the demo studies:

```bash
python scripts/nodules/download_data.py --skip-samples
```

## Train

```bash
python scripts/nodules/train_model.py --model-version 0.3.0
```

The shipped checkpoint is now trained on a real `LIDC-IDRI` subset built from `7` annotated studies and `150` extracted patches. It is still a lightweight prototype model, not a clinically meaningful detector, but it is no longer trained on synthetic-only patches. The smoke dataset remains in the repo only as an offline fallback for tests and local recovery.

## Notebook

Open the notebook from the repo root:

```bash
conda run -n playground jupyter lab src/ctscan/notebooks/nodules.ipynb
```

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
- `study_zip`
- optional `prior_study_zip`
- optional `sample_id`
- optional `age`
- optional `sex`
- optional `smoking_history`

## Tests

```bash
pytest -q src/ctscan/tests
```
