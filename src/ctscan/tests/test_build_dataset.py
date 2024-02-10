from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.ctscan.scripts.segmentation.build_dataset import BuildConfig, build_dataset


def _write_samples_manifest(samples_dir: Path, sample_id: str, study_zip: Path) -> None:
    payload = {
        sample_id: {
            "patient_id": "demo",
            "series_instance_uid": "series-demo",
            "study_zip": str(study_zip),
        }
    }
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / "samples.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_dataset_from_sample_zip(tmp_path: Path, make_ct_zip):
    sample_zip = make_ct_zip()
    samples_dir = tmp_path / "samples"
    _write_samples_manifest(samples_dir, "demo_case", sample_zip)

    config = BuildConfig(
        raw_dir=tmp_path / "raw",
        samples_dir=samples_dir,
        output_dir=tmp_path / "processed",
        labeled_manifest=tmp_path / "raw" / "missing.csv",
        include_samples=True,
        max_samples=0,
        target_spacing=(1.5, 1.0, 1.0),
        val_fraction=0.2,
        seed=7,
        min_positive_voxels=1,
        disable_resample=False,
        overwrite=True,
    )
    manifest_path = build_dataset(config)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["total_cases"] == 1
    assert payload["train_cases"] == 1
    assert payload["val_cases"] == 0
    assert any(int(value) > 0 for key, value in payload["class_voxels"].items() if key != "0")

    case_path = Path(payload["cases"][0]["path"])
    case = np.load(case_path)
    assert case["image"].dtype == np.float32
    assert case["mask"].dtype == np.uint8
    assert case["mask_multi"].dtype == np.uint8
    assert case["image"].shape == case["mask"].shape
    assert case["mask_multi"].shape[1:] == case["image"].shape


def test_build_dataset_from_labeled_manifest(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_path = raw_dir / "case_a_image.npy"
    mask_path = raw_dir / "case_a_mask.npy"

    image = np.full((6, 16, 16), -700.0, dtype=np.float32)
    mask = np.zeros((6, 16, 16), dtype=np.uint8)
    mask[2:4, 4:12, 5:11] = 9

    np.save(image_path, image)
    np.save(mask_path, mask)

    manifest_path = raw_dir / "composite_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "source",
                "image_path",
                "mask_path",
                "label_map",
                "spacing_z",
                "spacing_y",
                "spacing_x",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case_a",
                "source": "fixture_labeled",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "label_map": json.dumps({"9": 5}),
                "spacing_z": "1.5",
                "spacing_y": "1.0",
                "spacing_x": "1.0",
            }
        )

    config = BuildConfig(
        raw_dir=raw_dir,
        samples_dir=tmp_path / "samples",
        output_dir=tmp_path / "processed",
        labeled_manifest=manifest_path,
        include_samples=False,
        max_samples=0,
        target_spacing=(1.5, 1.0, 1.0),
        val_fraction=0.2,
        seed=7,
        min_positive_voxels=1,
        disable_resample=False,
        overwrite=True,
    )
    output_manifest = build_dataset(config)
    payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert payload["total_cases"] == 1
    assert payload["class_voxels"]["5"] > 0
    assert payload["task_type"] == "multilabel_segmentation"

    train_csv = tmp_path / "processed" / "train.csv"
    assert train_csv.exists()


def test_build_dataset_skips_corrupted_npz(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    good_image = raw_dir / "good_image.npy"
    good_mask = raw_dir / "good_mask.npy"
    bad_image = raw_dir / "bad_image.npz"

    image = np.full((6, 16, 16), -700.0, dtype=np.float32)
    mask = np.zeros((6, 16, 16), dtype=np.uint8)
    mask[2:4, 3:10, 4:11] = 5
    np.save(good_image, image)
    np.save(good_mask, mask)
    bad_image.write_text("not-a-zip", encoding="utf-8")

    manifest_path = raw_dir / "composite_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "source",
                "image_path",
                "mask_path",
                "label_map",
                "spacing_z",
                "spacing_y",
                "spacing_x",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "good_case",
                "source": "fixture_labeled",
                "image_path": str(good_image),
                "mask_path": str(good_mask),
                "label_map": "",
                "spacing_z": "1.5",
                "spacing_y": "1.0",
                "spacing_x": "1.0",
            }
        )
        writer.writerow(
            {
                "case_id": "bad_case",
                "source": "fixture_labeled",
                "image_path": str(bad_image),
                "mask_path": "",
                "label_map": "",
                "spacing_z": "1.5",
                "spacing_y": "1.0",
                "spacing_x": "1.0",
            }
        )

    config = BuildConfig(
        raw_dir=raw_dir,
        samples_dir=tmp_path / "samples",
        output_dir=tmp_path / "processed",
        labeled_manifest=manifest_path,
        include_samples=False,
        max_samples=0,
        target_spacing=(1.5, 1.0, 1.0),
        val_fraction=0.2,
        seed=7,
        min_positive_voxels=1,
        disable_resample=False,
        overwrite=True,
    )
    output_manifest = build_dataset(config)
    payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert payload["total_cases"] == 1
    assert payload["cases"][0]["case_id"] == "good_case"
