from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import src.ctscan.main as ctscan_main
from src.ctscan.models.nodules import create_model


def test_analyze_study_bytes_contract(make_ct_zip, monkeypatch):
    model = create_model()
    monkeypatch.setattr(
        ctscan_main,
        "get_model_bundle",
        lambda: (model, -700.0, 250.0, "0.1.0", {"nodule_accuracy": 0.8, "malignancy_auc": 0.82}),
    )
    study_path = make_ct_zip()
    prior_path = make_ct_zip(patient_id="prior", nodule_radius=2, malignant_boost=70.0)
    payload = ctscan_main.analyze_study_bytes(study_path.read_bytes(), prior_study_bytes=prior_path.read_bytes(), age=63, sex="male")
    assert payload["model_version"] == "0.1.0"
    assert payload["qc"]["status"] == "ok"
    assert "study_metadata" in payload
    assert "summary" in payload
    assert isinstance(payload["findings"], list)


def test_health_and_predict_endpoint(make_ct_zip, monkeypatch):
    model = create_model()
    monkeypatch.setattr(
        ctscan_main,
        "get_model_bundle",
        lambda: (model, -700.0, 250.0, "0.1.0", {"nodule_accuracy": 0.8, "malignancy_auc": 0.82}),
    )
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
    assert payload["model_version"] == "0.1.0"
    assert "findings" in payload


def test_update_viewer_handles_empty_state():
    image_path = ctscan_main.update_viewer({}, 0, "lung", None)
    assert Path(image_path).exists()
