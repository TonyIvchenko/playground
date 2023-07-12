from fastapi.testclient import TestClient

from hurricane_service.api import create_app
from hurricane_service.constants import FEATURE_COLUMNS
from hurricane_service.model_bundle import ModelBundle
from hurricane_service.settings import ServiceSettings


class FakeClassifier:
    def predict_proba(self, rows):
        assert len(rows) == 1
        return [[0.65, 0.35]]


class FakeRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, rows):
        assert len(rows) == 1
        return [self.value]


def fake_loader(_settings):
    bundle = ModelBundle(
        ri_classifier=FakeClassifier(),
        intensity_models={
            "24h": {"0.1": FakeRegressor(60), "0.5": FakeRegressor(70), "0.9": FakeRegressor(80)},
            "48h": {"0.1": FakeRegressor(65), "0.5": FakeRegressor(75), "0.9": FakeRegressor(85)},
        },
        metadata={"model_version": "test-v1"},
        feature_columns=list(FEATURE_COLUMNS),
    )
    return bundle, "hurricane-intensity/test-v1/model_bundle.joblib"


def failing_loader(_settings):
    raise RuntimeError("no model available")


def sample_payload():
    return {
        "storm_id": "AL09",
        "issue_time": "2026-08-25T12:00:00Z",
        "storm_state": {
            "lat": 22.4,
            "lon": -71.8,
            "vmax_kt": 70,
            "mslp_mb": 987,
            "motion_dir_deg": 300,
            "motion_speed_kt": 12,
        },
        "history_24h": [
            {"hours_ago": 6, "lat": 22.0, "lon": -70.7, "vmax_kt": 65, "mslp_mb": 992},
            {"hours_ago": 12, "lat": 21.6, "lon": -69.8, "vmax_kt": 60, "mslp_mb": 996},
            {"hours_ago": 24, "lat": 20.9, "lon": -68.4, "vmax_kt": 55, "mslp_mb": 1000},
        ],
        "environment": {
            "sst_c": 29.1,
            "ohc_kj_cm2": 68.0,
            "shear_200_850_kt": 9.0,
            "midlevel_rh_pct": 66.0,
            "vorticity_850_s-1": 0.00018,
        },
    }


def test_api_predict_and_metadata_contract():
    app = create_app(settings=ServiceSettings(), loader=fake_loader)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["model_version"] == "test-v1"

        metadata = client.get("/service-metadata")
        assert metadata.status_code == 200
        payload = metadata.json()
        assert payload["service_id"] == "hurricane-intensity-risk"
        assert payload["predict_endpoint"] == "/predict"
        assert "request_schema" in payload
        assert "response_schema" in payload

        prediction = client.post("/predict", json=sample_payload())
        assert prediction.status_code == 200
        body = prediction.json()
        assert body["storm_id"] == "AL09"
        assert body["model_version"] == "test-v1"
        assert body["ri_probability_24h"] == 0.35
        assert set(body["vmax_quantiles_kt"].keys()) == {"24h", "48h"}


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
