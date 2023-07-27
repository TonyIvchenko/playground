import pytest

from hurricane_service.settings import load_settings


def test_load_settings_defaults():
    loaded = load_settings({})
    assert loaded.model_bundle_path == "/app/model/model_bundle.joblib"
    assert loaded.api_host == "0.0.0.0"
    assert loaded.api_port == 8000
    assert loaded.ui_path == "/ui"


def test_load_settings_from_env():
    loaded = load_settings(
        {
            "MODEL_BUNDLE_PATH": "/tmp/model.joblib",
            "API_HOST": "127.0.0.1",
            "API_PORT": "8080",
            "UI_PATH": "/gradio",
            "SERVICE_VERSION": "1.2.3",
        }
    )
    assert loaded.model_bundle_path == "/tmp/model.joblib"
    assert loaded.api_host == "127.0.0.1"
    assert loaded.api_port == 8080
    assert loaded.ui_path == "/gradio"
    assert loaded.service_version == "1.2.3"


def test_load_settings_reads_port_fallback():
    loaded = load_settings({"PORT": "7860"})
    assert loaded.api_port == 7860


def test_load_settings_invalid_port_raises():
    with pytest.raises(ValueError):
        load_settings({"API_PORT": "bad"})


def test_service_settings_dataclass_is_immutable():
    loaded = load_settings({})
    with pytest.raises(Exception):
        loaded.api_host = "127.0.0.1"
