from fastapi.testclient import TestClient

from src.voiceforge.main import app, build_app


def test_root_page_renders():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "VoiceForge" in response.text


def test_build_app_api_info_does_not_crash():
    demo = build_app()

    info = demo.get_api_info()

    assert info == {"named_endpoints": {}, "unnamed_endpoints": {}}


def test_health_endpoint_contract():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "VoiceForge"
