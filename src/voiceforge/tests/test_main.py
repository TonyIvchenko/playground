from fastapi.testclient import TestClient

from src.voiceforge.inference import (
    format_inference_failure,
    format_initial_status,
    format_missing_reference_status,
    format_missing_text_status,
)
from src.voiceforge.main import app, build_app
from src.voiceforge.ui import (
    DEFAULT_TEXT,
    REFERENCE_FILE_TYPES,
    VOICEFORGE_INTRO_MARKDOWN,
)


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


def test_voiceforge_ui_constants():
    assert "VoiceForge" in VOICEFORGE_INTRO_MARKDOWN
    assert "ffprobe" in VOICEFORGE_INTRO_MARKDOWN
    assert DEFAULT_TEXT.startswith("I am ready for the fine-tuned voice cloning demo.")
    assert REFERENCE_FILE_TYPES == [".wav", ".flac", ".mp3", ".m4a", ".ogg"]


def test_inference_status_formatters(tmp_path):
    model_dir = tmp_path / "model-dir"

    assert format_initial_status(model_dir) == f"Looking for model in {model_dir}"
    assert format_missing_reference_status() == "Upload a reference clip first."
    assert format_missing_text_status() == "Type text to synthesize first."
    assert (
        format_inference_failure(RuntimeError("boom")) == "Voice synthesis failed: boom"
    )
