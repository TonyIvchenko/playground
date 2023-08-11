from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

# Make package imports work when running pytest from repo root.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wildfire_service.main import create_app, load_settings


def test_load_settings_defaults_and_port_fallback():
    defaults = load_settings({})
    assert defaults.api_port == 8010

    from_port = load_settings({"PORT": "7860"})
    assert from_port.api_port == 7860


def test_load_settings_invalid_port_raises():
    with pytest.raises(ValueError):
        load_settings({"API_PORT": "bad"})


def test_wildfire_service_endpoints():
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
        assert meta["service_id"] == "wildfire-ignition-risk"
        assert meta["predict_endpoint"] == "/predict"

        prediction = client.post(
            "/predict",
            json={
                "region_id": "norcal",
                "location": {"lat": 38.7, "lon": -121.2},
                "temp_c": 34,
                "humidity_pct": 22,
                "wind_kph": 28,
                "drought_index": 0.82,
            },
        )
        assert prediction.status_code == 200
        body = prediction.json()
        assert body["region_id"] == "norcal"
        assert 0.0 <= body["ignition_probability_24h"] <= 1.0
        assert body["risk_level"] in {"low", "moderate", "high", "extreme"}
