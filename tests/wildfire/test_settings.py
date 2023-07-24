import pytest

from wildfire_service.settings import load_settings


def test_load_settings_defaults():
    loaded = load_settings({})
    assert loaded.model_bundle_path == "/app/model/model_bundle.joblib"
    assert loaded.api_host == "0.0.0.0"
    assert loaded.api_port == 8010
    assert loaded.ui_path == "/ui"


def test_load_settings_from_env():
    loaded = load_settings(
        {
            "MODEL_BUNDLE_PATH": "/tmp/model.joblib",
            "API_HOST": "127.0.0.1",
            "API_PORT": "9009",
            "UI_PATH": "/gradio",
            "SERVICE_VERSION": "9.9.9",
        }
    )
    assert loaded.model_bundle_path == "/tmp/model.joblib"
    assert loaded.api_host == "127.0.0.1"
    assert loaded.api_port == 9009
    assert loaded.ui_path == "/gradio"
    assert loaded.service_version == "9.9.9"


def test_load_settings_invalid_port_raises():
    with pytest.raises(ValueError):
        load_settings({"API_PORT": "bad"})
