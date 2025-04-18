import numpy as np
from pathlib import Path

from src.voiceforge.model.speecht5 import load_audio_mono, read_artifact_metadata, resolve_model_source


def test_resolve_model_source_prefers_local_checkpoint(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    source, is_finetuned = resolve_model_source(tmp_path, base_model="base/model")
    assert source == str(tmp_path.resolve())
    assert is_finetuned is True


def test_read_artifact_metadata_returns_empty_when_missing(tmp_path: Path):
    assert read_artifact_metadata(tmp_path) == {}


def test_load_audio_mono_falls_back_to_librosa(monkeypatch):
    import librosa
    import soundfile as sf

    monkeypatch.setattr(sf, "read", lambda _: (_ for _ in ()).throw(RuntimeError("decode failed")))
    monkeypatch.setattr(
        librosa,
        "load",
        lambda path, sr, mono: (np.array([0.25, -0.5, 0.5], dtype=np.float32), sr),
    )

    waveform = load_audio_mono("/tmp/reference.mp3", target_sample_rate=16000)

    np.testing.assert_allclose(waveform, np.array([0.25, -0.5, 0.5], dtype=np.float32))
