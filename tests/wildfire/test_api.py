from fastapi.testclient import TestClient

from wildfire_service.api import create_app
from wildfire_service.constants import FEATURE_COLUMNS
from wildfire_service.model_bundle import ModelBundle
from wildfire_service.settings import ServiceSettings


class FakeClassifier:
    def predict_proba(self, rows):
        assert len(rows) == 1
        return [[0.74, 0.26]]


def fake_loader(_settings):
    metadata = {
        "model_version": "test-wildfire-v1",
        "feature_importances": {feature: 0.1 for feature in FEATURE_COLUMNS},
        "feature_mean": {feature: 0.0 for feature in FEATURE_COLUMNS},
        "feature_std": {feature: 1.0 for feature in FEATURE_COLUMNS},
    }
    bundle = ModelBundle(
        ignition_classifier=FakeClassifier(),
        metadata=metadata,
        feature_columns=list(FEATURE_COLUMNS),
    )
    return bundle, "/tmp/model_bundle.joblib"


def failing_loader(_settings):
    raise RuntimeError("no model available")


def sample_payload():
    return {
        "region_id": "norcal-foothills",
        "location": {"lat": 38.72, "lon": -121.2},
        "forecast_date": "2026-08-25",
        "conditions": {
            "temp_c": 34.0,
            "relative_humidity_pct": 21.0,
            "wind_speed_kph": 28.0,
            "precip_7d_mm": 0.5,
            "drought_index": 0.82,
            "fuel_moisture_pct": 12.0,
            "vegetation_dryness": 0.88,
            "human_activity_index": 0.67,
            "elevation_m": 480.0,
            "slope_deg": 18.0,
        },
    }


def test_api_predict_and_metadata_contract():
    app = create_app(settings=ServiceSettings(), loader=fake_loader)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["model_version"] == "test-wildfire-v1"

        metadata = client.get("/service-metadata")
        assert metadata.status_code == 200
        payload = metadata.json()
        assert payload["service_id"] == "wildfire-ignition-risk"
        assert payload["predict_endpoint"] == "/predict"
        assert "request_schema" in payload
        assert "response_schema" in payload

        prediction = client.post("/predict", json=sample_payload())
        assert prediction.status_code == 200
        body = prediction.json()
        assert body["region_id"] == "norcal-foothills"
        assert body["model_version"] == "test-wildfire-v1"
        assert body["ignition_probability_24h"] == 0.26
        assert body["risk_level"] == "moderate"


def test_api_reports_degraded_when_model_loading_fails():
    app = create_app(settings=ServiceSettings(), loader=failing_loader)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert health.json()["model_loaded"] is False

        prediction = client.post("/predict", json=sample_payload())
        assert prediction.status_code == 503
        assert prediction.json()["detail"]["error"] == "MODEL_UNAVAILABLE"
