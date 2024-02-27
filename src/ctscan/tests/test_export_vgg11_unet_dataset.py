from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.ctscan.scripts.segmentation.export_vgg11_unet_dataset import export_dataset


def test_export_vgg11_unet_dataset_smoke(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    cases_dir = processed_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((4, 8, 8), dtype=np.float32)
    image[1:3, 2:6, 2:6] = 1.0
    mask = np.zeros((4, 8, 8), dtype=np.uint8)
    mask[1, 2:4, 2:4] = 5  # nodule -> class 1
    mask[2, 4:6, 4:6] = 3  # ground_glass -> class 2
    mask[3, 1:3, 5:7] = 7  # pleural_effusion -> class 3

    case_path = cases_dir / "case_a.npz"
    np.savez_compressed(
        case_path,
        image=image,
        mask=mask,
        spacing=np.asarray([1.5, 1.0, 1.0], dtype=np.float32),
    )

    train_csv = processed_dir / "train.csv"
    with train_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "source", "path"])
        writer.writeheader()
        writer.writerow({"case_id": "case_a", "source": "fixture", "path": str(case_path)})
    with (processed_dir / "val.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "source", "path"])
        writer.writeheader()

    output_dir = tmp_path / "legacy_out"
    summary = export_dataset(
        processed_dir=processed_dir,
        output_dir=output_dir,
        class_map={0: 0, 5: 1, 3: 2, 7: 3},
        default_class=0,
        max_cases=0,
        overwrite=True,
        skip_existing=False,
    )

    assert summary["exported_cases"] == 1
    image_nii = output_dir / "dataset" / "case_a.nii.gz"
    mask_nii = output_dir / "mask" / "case_amask.nii"
    assert image_nii.exists()
    assert mask_nii.exists()

    loaded_img = sitk.GetArrayFromImage(sitk.ReadImage(str(image_nii)))
    loaded_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_nii)))
    assert loaded_img.shape == image.shape
    assert loaded_mask.shape == mask.shape
    assert set(np.unique(loaded_mask).astype(int).tolist()) == {0, 1, 2, 3}

