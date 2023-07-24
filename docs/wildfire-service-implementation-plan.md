# Wildfire Ignition-Risk Service Implementation Plan

## Goal
Implement a self-contained wildfire ML service in `playground` that:
- trains a 24h wildfire ignition-risk model,
- serves predictions through a Dockerized HTTP API,
- includes a built-in Gradio UI in the same service.

## Deliverables
1. `notebooks/wildfire_eda_and_baseline.ipynb` for data exploration and baseline workflow.
2. `scripts/train_wildfire_ignition_model.py` for reproducible model training.
3. `src/wildfire_service/` API + Gradio service with `/health`, `/predict`, `/service-metadata`, `/ui`.
4. `src/wildfire_service/Dockerfile` and Make targets for local Docker run.
5. Unit and API tests under `tests/wildfire/`.

## Service Contract
### Endpoints
- `GET /health`
- `POST /predict`
- `GET /service-metadata`
- `GET /ui` (Gradio UI)
- `GET /openapi.json`

### Predict Input
- `region_id`
- `location`: `lat`, `lon`
- `forecast_date`
- optional `conditions`: temperature, humidity, wind, precipitation, drought, fuels, terrain

### Predict Output
- `ignition_probability_24h`
- `risk_level` (`low`, `moderate`, `high`, `extreme`)
- `top_drivers` (ranked risk factors)
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
1. `make train-wildfire-model`
2. `make build-wildfire-service`
3. `make start-wildfire-service`
4. `make smoke-wildfire-service`
5. Open `http://localhost:8010/ui`

## Testing Requirements
- Feature engineering tests.
- Model bundle behavior tests.
- Settings/env parsing tests.
- API contract tests (`/health`, `/service-metadata`, `/predict`).
