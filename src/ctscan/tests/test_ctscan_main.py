from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import src.ctscan.main as ctscan_main


def test_analyze_study_bytes_contract(make_ct_zip):
    study_path = make_ct_zip()
    payload = ctscan_main.analyze_study_bytes(study_path.read_bytes(), age=63, sex="male")
    assert payload["version"] == "segmentation-v1"
    assert payload["backend"] in {"threshold", "lungmask"}
    assert payload["qc"]["status"] in {"ok", "rejected"}
    assert "issues" in payload
    assert "summary" in payload
    assert "_viewer" in payload


def test_health_and_predict_endpoint(make_ct_zip):
    client = TestClient(ctscan_main.api)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    study_path = make_ct_zip()
    with study_path.open("rb") as handle:
        response = client.post(
            "/predict",
            files={"study_zip": ("study.zip", handle.read(), "application/zip")},
            data={"age": "67", "sex": "female", "smoking_history": "former smoker"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "segmentation-v1"
    assert "issues" in payload
    assert "_viewer" not in payload


def test_update_viewer_handles_empty_state():
    image_path, slice_df = ctscan_main.update_viewer({}, 0, "lung", "all", True, True, 0.32, 0.45)
    assert Path(image_path).exists()
    assert list(slice_df.columns) == ctscan_main.SLICE_TABLE_COLUMNS
