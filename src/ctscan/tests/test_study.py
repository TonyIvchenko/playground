from __future__ import annotations

from src.ctscan.study import estimate_lung_mask, generate_candidates, load_study_from_zip_bytes, match_prior_findings, render_slice_image


def test_load_study_and_resample(make_ct_zip):
    study_path = make_ct_zip()
    loaded = load_study_from_zip_bytes(study_path.read_bytes())
    assert loaded.metadata["body_part_examined"] == "CHEST"
    assert loaded.spacing == (1.0, 1.0, 1.0)
    assert loaded.volume_hu.shape[0] > 24
    assert loaded.qc_reasons == []


def test_load_study_rejects_wrong_body_part(make_ct_zip):
    study_path = make_ct_zip(body_part="HEAD", patient_id="head-study")
    loaded = load_study_from_zip_bytes(study_path.read_bytes())
    assert loaded.qc_reasons


def test_candidate_generation_and_prior_matching(make_ct_zip):
    current_path = make_ct_zip(patient_id="current", nodule_center=(12, 32, 38), nodule_radius=4)
    prior_path = make_ct_zip(patient_id="prior", nodule_center=(12, 32, 38), nodule_radius=2, malignant_boost=80.0)
    current = load_study_from_zip_bytes(current_path.read_bytes())
    prior = load_study_from_zip_bytes(prior_path.read_bytes())

    current_candidates = generate_candidates(current.volume_hu, estimate_lung_mask(current.volume_hu))
    prior_candidates = generate_candidates(prior.volume_hu, estimate_lung_mask(prior.volume_hu))
    matched = match_prior_findings(current_candidates, prior_candidates)

    assert matched
    assert any(item.get("growth") is not None for item in matched)


def test_render_slice_image(make_ct_zip):
    study_path = make_ct_zip()
    loaded = load_study_from_zip_bytes(study_path.read_bytes())
    findings = generate_candidates(loaded.volume_hu, estimate_lung_mask(loaded.volume_hu))
    for index, finding in enumerate(findings):
        finding["lesion_id"] = f"lesion-{index + 1}"
        finding["nodule_probability"] = 0.9
        finding["malignancy_risk"] = 0.6
    image = render_slice_image(loaded.volume_hu, findings, findings[0]["slice_index"], "lung", findings[0]["lesion_id"])
    assert image.size == (loaded.volume_hu.shape[2], loaded.volume_hu.shape[1])
