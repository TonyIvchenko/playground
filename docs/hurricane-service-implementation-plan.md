# Hurricane Intensity-Risk Service Implementation Plan

## Goal
Implement a self-contained hurricane ML service in `playground` that:
- trains a hurricane intensity-risk model,
- serves predictions through a Dockerized HTTP API,
- includes a built-in Gradio UI in the same service.

## Deliverables
1. `notebooks/hurricane_eda_and_baseline.ipynb` for data exploration and baseline workflow.
2. `scripts/train_hurricane_intensity_model.py` for reproducible model training.
3. `src/hurricane_service/` API + Gradio service with `/health`, `/predict`, `/service-metadata`, `/ui`.
4. `src/hurricane_service/Dockerfile` and Make targets for local Docker run.
5. Unit and API tests under `tests/hurricane/`.

## Service Contract
### Endpoints
- `GET /health`
- `POST /predict`
- `GET /service-metadata`
- `GET /ui` (Gradio UI)
- `GET /openapi.json`

### Predict Input
- `storm_id`, `issue_time`
- `storm_state`: current position and intensity
- `history_24h`: 6h/12h/24h snapshots
- `environment`: SST, OHC, shear, humidity, vorticity

### Predict Output
- `ri_probability_24h`
- `vmax_quantiles_kt` at 24h and 48h (`p10/p50/p90`)
- `model_version`
- `warnings`

## Training + Artifact Flow
1. Prepare training CSV with required feature and target columns.
2. Run training script to output `model_bundle.joblib`.
3. Service loads model from local `MODEL_BUNDLE_PATH`.
4. Docker image build generates a local model bundle for self-contained runtime.

## Gradio UI Behavior
1. Render JSON request editor pre-filled from metadata example payload.
2. Submit request to in-process predictor (same model/service runtime).
3. Show prediction JSON and validation/runtime error payloads.
4. Show `/health` result and `/service-metadata` in UI.

## Local Runbook
1. `make train-hurricane-model`
2. `make build-hurricane-service`
3. `make start-hurricane-service`
4. `make smoke-hurricane-service`
5. Open `http://localhost:8000/ui`

## Testing Requirements
- Feature engineering tests.
- Model bundle behavior tests (quantile monotonic handling).
- Settings/env parsing tests.
- API contract tests (`/health`, `/service-metadata`, `/predict`).
