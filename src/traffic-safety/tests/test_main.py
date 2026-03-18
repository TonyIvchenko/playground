from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def test_health_endpoint_contract():
    client = TestClient(MODULE.api)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "Traffic Safety"
    assert payload["model_ready"] is True
    assert payload["overlay_ready"] is True
    assert payload["frames"] == 168


def test_predict_traffic_safety_shape_and_values():
    result = MODULE.predict_traffic_safety(
        lat=34.0522,
        lon=-118.2437,
        day_of_week=5,
        hour=17,
        month=9,
    )

    assert result["model_version"] == MODULE.MODEL_VERSION
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "high", "extreme"}


def test_map_html_uses_local_static_assets():
    html = MODULE._map_html()

    assert '<div id="risk-map-shell"' in html
    assert 'data-config="' in html
    assert "Traffic Safety" in html


def test_bootstrap_loader_looks_for_local_map_script():
    assert "bootstrapTrafficSafetyMap" in MODULE._map_bootstrap_js()


def test_tiles_are_served():
    client = TestClient(MODULE.api)

    response = client.get("/tiles/0/4/0/0.png")

    assert response.status_code == 200
    assert "image/png" in response.headers["content-type"]
    assert len(response.content) > 0
