from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.ctscan.scripts.nodules.download_data import (
    build_smoke_training_dataset,
    fetch_series_uid,
    write_dataset_manifest,
)


def test_write_dataset_manifest_contains_expected_sources(tmp_path: Path):
    manifest_path = write_dataset_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = {item["name"] for item in payload}
    assert {"LIDC-IDRI", "LUNA16", "LNDb"} <= names


def test_build_smoke_training_dataset_shapes(tmp_path: Path):
    dataset_path = build_smoke_training_dataset(tmp_path, rows=12)
    bundle = np.load(dataset_path)
    assert bundle["patches"].shape == (12, 1, 16, 16, 16)
    assert bundle["nodule_target"].shape == (12,)
    assert bundle["malignancy_target"].shape == (12,)


def test_fetch_series_uid_uses_largest_series(monkeypatch):
    monkeypatch.setattr(
        "src.ctscan.scripts.nodules.download_data._fetch_json",
        lambda *_args, **_kwargs: [
            {"SeriesInstanceUID": "small", "ImageCount": 24},
            {"SeriesInstanceUID": "large", "ImageCount": 133},
        ],
    )
    assert fetch_series_uid("LIDC-IDRI-0001") == "large"
