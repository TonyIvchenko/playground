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
    REFERENCE_AUDIO_HINT_MARKDOWN,
    REFERENCE_FILE_TYPES,
    VOICEFORGE_INTRO_MARKDOWN,
)


def test_root_page_renders():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "VoiceForge" in response.text
    assert "Resolved device:" in response.text
    assert "Active checkpoint:" in response.text


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
    assert ".wav" in REFERENCE_AUDIO_HINT_MARKDOWN
    assert ".ogg" in REFERENCE_AUDIO_HINT_MARKDOWN
    assert "WAV" in REFERENCE_AUDIO_HINT_MARKDOWN
    assert "clear speaker" in REFERENCE_AUDIO_HINT_MARKDOWN


def test_inference_status_formatters(tmp_path):
    model_dir = tmp_path / "model-dir"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    initial_status = format_initial_status(model_dir)
    missing_reference_status = format_missing_reference_status(model_dir)
    missing_text_status = format_missing_text_status(model_dir)
    failure_status = format_inference_failure(RuntimeError("boom"), model_dir)

    assert "Ready to synthesize" in initial_status
    assert "Resolved device:" in initial_status
    assert f"Active checkpoint: {model_dir.resolve()}" in initial_status
    assert "Upload a reference clip first." in missing_reference_status
    assert "Resolved device:" in missing_reference_status
    assert f"Active checkpoint: {model_dir.resolve()}" in missing_reference_status
    assert "Type text to synthesize first." in missing_text_status
    assert "Resolved device:" in missing_text_status
    assert f"Active checkpoint: {model_dir.resolve()}" in missing_text_status
    assert "Voice synthesis failed: boom" in failure_status
    assert "Resolved device:" in failure_status
    assert f"Active checkpoint: {model_dir.resolve()}" in failure_status
