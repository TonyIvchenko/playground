from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import json

import src.ctscan.main as ctscan_main


def test_analyze_study_bytes_contract(make_ct_zip):
    study_path = make_ct_zip()
    payload = ctscan_main.analyze_study_bytes(
        study_path.read_bytes(), age=63, sex="male"
    )
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
    assert health.json()["service"] == "CT Scan"

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


def test_blank_viewer_html():
    html = ctscan_main.render_upload_html(None)
    assert "Upload DICOM file." in html
    assert "<img" in html


def test_demo_injects_viewer_head():
    demo = ctscan_main.build_demo()
    assert demo.head == ctscan_main.VIEWER_HEAD
    assert "/ctscan-static/viewer.css" in demo.head
    assert "/ctscan-static/viewer.js" in demo.head


def test_viewer_page_references_external_assets():
    html = ctscan_main._viewer_html(
        {
            "default_slice": 0,
            "default_opacity": 0.2,
            "slice_count": 1,
            "asset_root": "/viewer-cache/demo",
            "rows": [],
        }
    )
    assert '<link rel="stylesheet" href="/ctscan-static/viewer.css">' in html
    assert '<script src="/ctscan-static/viewer.js" defer></script>' in html
    assert '<script type="application/json" class="ctscan-state">' in html
    assert "<style>" not in html


def test_ctscan_static_viewer_assets_are_served():
    client = TestClient(ctscan_main.api)

    css_response = client.get("/ctscan-static/viewer.css")
    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert ".ctscan-viewer-root" in css_response.text

    js_response = client.get("/ctscan-static/viewer.js")
    assert js_response.status_code == 200
    assert "text/javascript" in js_response.headers["content-type"]
    assert "function initViewer" in js_response.text


def test_auto_demo_manifest_from_legacy_ct_zips(
    tmp_path: Path, monkeypatch, make_ct_zip
):
    samples_manifest = tmp_path / "samples" / "samples.json"
    ct_zips_dir = tmp_path / "ct_zips"
    ct_zips_dir.mkdir(parents=True, exist_ok=True)
    sample_zip = make_ct_zip(patient_id="LUNG1-001")
    target_zip = ct_zips_dir / "LUNG1-001.zip"
    target_zip.write_bytes(sample_zip.read_bytes())

    monkeypatch.setattr(ctscan_main, "SAMPLES_MANIFEST_PATH", samples_manifest)
    monkeypatch.setenv("CTSCAN_DEMO_CT_ZIPS_ROOT", str(ct_zips_dir))
    ctscan_main.load_samples_manifest.cache_clear()

    manifest = ctscan_main.load_samples_manifest()
    assert "demo_lung1-001" in manifest
    assert manifest["demo_lung1-001"]["study_zip"] == str(target_zip.resolve())
    assert json.loads(samples_manifest.read_text(encoding="utf-8"))["demo_lung1-001"][
        "study_zip"
    ] == str(target_zip.resolve())
