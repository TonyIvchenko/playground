from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import SimpleITK as sitk

from src.ctscan.scripts.segmentation.build_legacy_dataset import BuildConfig, build_dataset, parse_args as parse_build_legacy_args
from src.ctscan.scripts.segmentation.download_legacy_sources import (
    extract_google_drive_download_form,
    extract_google_drive_file_id,
    list_zip_patient_ids,
)


def write_nifti(path: Path, array: np.ndarray, spacing_zyx: tuple[float, float, float] = (1.5, 1.0, 1.0)) -> None:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]))
    sitk.WriteImage(image, str(path), useCompression=path.name.endswith('.gz'))


def test_build_legacy_dataset_smoke(tmp_path: Path, make_ct_zip):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "out"

    # MedSeg fixture: exact legacy-compatible labels already present.
    medseg_dir = raw_dir / "medseg_sirm"
    medseg_dir.mkdir(parents=True, exist_ok=True)
    medseg_image = medseg_dir / "medseg_train_images.nii.gz"
    medseg_mask = medseg_dir / "medseg_train_masks.nii.gz"
    medseg_img_arr = np.zeros((3, 8, 8), dtype=np.float32)
    medseg_mask_arr = np.zeros((3, 8, 8), dtype=np.uint8)
    medseg_mask_arr[0, 1:3, 1:3] = 1
    medseg_mask_arr[1, 2:4, 2:5] = 2
    medseg_mask_arr[2, 5:7, 5:7] = 3
    write_nifti(medseg_image, medseg_img_arr)
    write_nifti(medseg_mask, medseg_mask_arr)

    # LongCIU fixture: STAPLE mask with GGO + consolidation only.
    longciu_dir = raw_dir / "longciu" / "extracted"
    longciu_dir.mkdir(parents=True, exist_ok=True)
    longciu_image = longciu_dir / "longciu_img.nii.gz"
    longciu_mask = longciu_dir / "longciu_STAPLE_tgt.nii.gz"
    longciu_img_arr = np.zeros((4, 10, 10), dtype=np.float32)
    longciu_mask_arr = np.zeros((4, 10, 10), dtype=np.uint8)
    longciu_mask_arr[1, 1:4, 1:4] = 1
    longciu_mask_arr[2, 5:8, 5:8] = 2
    write_nifti(longciu_image, longciu_img_arr)
    write_nifti(longciu_mask, longciu_mask_arr)

    # PleThora fixture: one CT zip + two reviewer masks.
    plethora_dir = raw_dir / "plethora"
    effusion_dir = plethora_dir / "masks" / "effusions" / "Effusions" / "LUNG1-001"
    effusion_dir.mkdir(parents=True, exist_ok=True)
    ct_zip = make_ct_zip(patient_id="LUNG1-001")
    reviewer_a = np.zeros((24, 64, 64), dtype=np.uint8)
    reviewer_b = np.zeros((24, 64, 64), dtype=np.uint8)
    reviewer_a[10:14, 20:30, 45:55] = 1
    reviewer_b[11:15, 22:32, 46:56] = 1
    write_nifti(effusion_dir / "LUNG1-001_effusion_first_reviewer.nii.gz", reviewer_a)
    write_nifti(effusion_dir / "LUNG1-001_effusion_second_reviewer.nii.gz", reviewer_b)

    manifest = {
        "schema_version": 1,
        "sources": {
            "medseg_sirm": {
                "train_images": str(medseg_image),
                "train_masks": str(medseg_mask),
            },
            "longciu": {
                "extracted_dir": str(longciu_dir),
            },
            "plethora": {
                "effusion_masks_dir": str(plethora_dir / "masks" / "effusions"),
                "ct_series": [
                    {
                        "patient_id": "LUNG1-001",
                        "zip_path": str(ct_zip),
                    }
                ],
            },
        },
    }
    manifest_path = raw_dir / "legacy_sources_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = build_dataset(
        BuildConfig(
            raw_dir=raw_dir,
            output_dir=output_dir,
            manifest_path=manifest_path,
            overwrite=True,
            longciu_mask_source="staple",
            plethora_vote_mode="union",
        )
    )

    assert summary["total_cases"] == 3
    image_files = sorted((output_dir / "dataset").glob("*.nii.gz"))
    mask_files = sorted((output_dir / "mask").glob("*mask.nii"))
    assert len(image_files) == 3
    assert len(mask_files) == 3

    medseg_loaded = sitk.GetArrayFromImage(sitk.ReadImage(str(output_dir / "mask" / "medseg_trainmask.nii")))
    longciu_loaded = sitk.GetArrayFromImage(sitk.ReadImage(str(output_dir / "mask" / "longciu_staplemask.nii")))
    plethora_loaded = sitk.GetArrayFromImage(sitk.ReadImage(str(output_dir / "mask" / "plethora_lung1-001mask.nii")))
    assert set(np.unique(medseg_loaded).tolist()) == {0, 1, 2, 3}
    assert set(np.unique(longciu_loaded).tolist()) == {0, 1, 2}
    assert set(np.unique(plethora_loaded).tolist()) == {0, 3}


def test_download_helpers_parse_ids(tmp_path: Path):
    assert extract_google_drive_file_id("https://drive.google.com/file/d/abc123/view?usp=sharing") == "abc123"
    assert extract_google_drive_file_id("https://drive.google.com/open?id=xyz789") == "xyz789"

    zip_path = tmp_path / "effusions.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("Effusions/LUNG1-001/a.nii.gz", b"x")
        archive.writestr("Effusions/LUNG1-002/b.nii.gz", b"x")
    assert list_zip_patient_ids(zip_path, r"(LUNG1-\d+)") == ["LUNG1-001", "LUNG1-002"]


def test_extract_google_drive_download_form():
    html = """
    <html><body>
      <form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
        <input type="hidden" name="id" value="abc123">
        <input type="hidden" name="export" value="download">
        <input type="hidden" name="confirm" value="t">
        <input type="hidden" name="uuid" value="uuid-456">
      </form>
    </body></html>
    """
    action, params = extract_google_drive_download_form(html)
    assert action == "https://drive.usercontent.google.com/download"
    assert params == {
        "id": "abc123",
        "export": "download",
        "confirm": "t",
        "uuid": "uuid-456",
    }


def test_build_legacy_parse_args_uses_raw_dir_manifest(monkeypatch, tmp_path: Path):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(sys, "argv", ["build_legacy_dataset.py", "--raw-dir", str(raw_dir)])
    config = parse_build_legacy_args()
    assert config.raw_dir == raw_dir.resolve()
    assert config.manifest_path == raw_dir.resolve() / "legacy_sources_manifest.json"
