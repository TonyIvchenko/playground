from __future__ import annotations

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
        focus_issue="all",
    )
    assert image.size == (loaded.volume_hu.shape[2], loaded.volume_hu.shape[1])
