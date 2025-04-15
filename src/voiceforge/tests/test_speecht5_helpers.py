from pathlib import Path

from src.voiceforge.model.speecht5 import read_artifact_metadata, resolve_model_source


def test_resolve_model_source_prefers_local_checkpoint(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    source, is_finetuned = resolve_model_source(tmp_path, base_model="base/model")
    assert source == str(tmp_path.resolve())
    assert is_finetuned is True


def test_read_artifact_metadata_returns_empty_when_missing(tmp_path: Path):
    assert read_artifact_metadata(tmp_path) == {}
