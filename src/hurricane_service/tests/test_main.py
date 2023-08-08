from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

# Make package imports work when running pytest from repo root.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hurricane_service.main import create_app, load_settings


def test_load_settings_defaults_and_port_fallback():
    defaults = load_settings({})
    assert defaults.api_port == 8000

    from_port = load_settings({"PORT": "7860"})
    assert from_port.api_port == 7860


def test_load_settings_invalid_port_raises():
    with pytest.raises(ValueError):
        load_settings({"API_PORT": "bad"})


def test_hurricane_service_endpoints():
    settings = load_settings({"SERVICE_VERSION": "test-v1", "UI_PATH": "/ui"})
    app = create_app(settings)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["model_version"] == "test-v1"

        metadata = client.get("/service-metadata")
        assert metadata.status_code == 200
        meta = metadata.json()
        assert meta["service_id"] == "hurricane-intensity-risk"
        assert meta["predict_endpoint"] == "/predict"

        prediction = client.post(
            "/predict",
            json={"storm_id": "AL09", "vmax_kt": 70, "shear_kt": 10, "sst_c": 29.2},
        )
        assert prediction.status_code == 200
        body = prediction.json()
        assert body["storm_id"] == "AL09"
        assert 0.0 <= body["ri_probability_24h"] <= 1.0
        assert body["risk_level"] in {"low", "moderate", "high", "extreme"}
