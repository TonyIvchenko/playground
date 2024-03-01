from __future__ import annotations

import numpy as np

import src.ctscan.study as study_module
from src.ctscan.study import (
    issue_slice_stats,
    issue_volume_stats,
    load_study_from_zip_bytes,
    render_segmentation_slice,
    segment_issues,
    segment_lungs,
)


def test_load_study_contract(make_ct_zip):
    study_path = make_ct_zip()
    loaded = load_study_from_zip_bytes(study_path.read_bytes())
    assert loaded.metadata["body_part_examined"] == "CHEST"
    assert loaded.metadata["slice_count"] == 24
    assert len(loaded.spacing) == 3


def test_load_study_wrong_body_part_is_flagged(make_ct_zip):
    study_path = make_ct_zip(body_part="HEAD", patient_id="head-study")
    loaded = load_study_from_zip_bytes(study_path.read_bytes())
    assert loaded.qc_reasons


def test_segmentation_and_stats(make_ct_zip):
    study_path = make_ct_zip()
    loaded = load_study_from_zip_bytes(study_path.read_bytes())
    lung_mask, backend = segment_lungs(loaded.volume_hu)
    labels = segment_issues(loaded.volume_hu, lung_mask)

    assert backend in {"threshold", "lungmask"}
    assert int(lung_mask.sum()) > 0
    assert int((labels > 0).sum()) > 0

    volume_rows = issue_volume_stats(labels, lung_mask, loaded.spacing)
    assert any(float(row["lung_percent"]) > 0.0 for row in volume_rows)

    slice_rows = issue_slice_stats(labels, lung_mask, slice_index=loaded.volume_hu.shape[0] // 2)
    assert any(float(row["slice_percent"]) > 0.0 for row in slice_rows)


def test_render_segmentation_slice(make_ct_zip):
    study_path = make_ct_zip()
    loaded = load_study_from_zip_bytes(study_path.read_bytes())
    lung_mask, _ = segment_lungs(loaded.volume_hu)
    labels = segment_issues(loaded.volume_hu, lung_mask)
    image = render_segmentation_slice(
        volume_hu=loaded.volume_hu,
        labels=labels,
        lung_mask=lung_mask,
        slice_index=loaded.volume_hu.shape[0] // 2,
        preset="lung",
        selected_issues=[],
    )
    assert image.size == (loaded.volume_hu.shape[2], loaded.volume_hu.shape[1])


def test_supported_issues_legacy_schema(monkeypatch):
    monkeypatch.setattr(study_module, "_inspect_model_checkpoint", lambda: {"model_type": "legacy_vgg11_unet"})
    issues = study_module.supported_issues()
    assert [issue["key"] for issue in issues] == ["ground_glass", "consolidation", "pleural_effusion"]


def test_segment_issues_explicit_model_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(study_module, "_predict_issue_labels_model", lambda volume_hu: None)
    monkeypatch.setattr(study_module, "MODEL_PATH_EXPLICIT", True)
    volume_hu = np.full((2, 8, 8), -700.0, dtype=np.float32)
    lung_mask = np.ones_like(volume_hu, dtype=bool)
    labels = study_module.segment_issues(volume_hu, lung_mask)
    assert not labels.any()
