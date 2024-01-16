from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ctscan.scripts.segmentation.build_lidc_manifest import merge_rows, to_zyx
from src.ctscan.scripts.segmentation.download_lidc import DownloadConfig, _series_dir


def test_to_zyx_reorders_last_axis_to_first():
    volume = np.zeros((4, 5, 6), dtype=np.float32)
    volume[1, 2, 3] = 9.0
    reordered = to_zyx(volume)
    assert reordered.shape == (6, 4, 5)
    assert float(reordered[3, 1, 2]) == 9.0


def test_merge_rows_replaces_old_lidc_rows_only():
    existing = [
        {"case_id": "a", "source": "external", "image_path": "x", "mask_path": "x"},
        {"case_id": "lidc_old", "source": "lidc_idri", "image_path": "y", "mask_path": "y"},
    ]
    lidc_rows = [
        {"case_id": "lidc_new", "source": "lidc_idri", "image_path": "z", "mask_path": "z"},
    ]
    merged = merge_rows(existing, lidc_rows, replace_lidc_rows=True)
    ids = {row["case_id"] for row in merged}
    assert "a" in ids
    assert "lidc_old" not in ids
    assert "lidc_new" in ids


def test_series_dir_layout(tmp_path: Path):
    config = DownloadConfig(
        raw_dir=tmp_path,
        dicom_root=tmp_path / "LIDC-IDRI",
        series_csv=tmp_path / "series.csv",
        max_series=0,
        start_index=0,
        timeout_sec=60,
        retries=1,
        retry_backoff_sec=1,
        resume_series_uid="",
        keep_zip=False,
        overwrite=False,
        dry_run=True,
        stop_on_error=False,
    )
    row = {
        "PatientID": "LIDC-IDRI-0001",
        "StudyInstanceUID": "1.2.3",
        "SeriesInstanceUID": "4.5.6",
    }
    path = _series_dir(config, row)
    assert str(path).endswith("LIDC-IDRI/LIDC-IDRI-0001/1.2.3/4.5.6")
