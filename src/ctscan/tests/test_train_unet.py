from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ctscan.scripts.segmentation.train_unet import TrainConfig, train


CHANNEL_CLASS_IDS = [5, 6, 3, 4, 1, 2, 7]


def _write_case(path: Path, channel_index: int) -> None:
    image = np.full((6, 64, 64), 0.2, dtype=np.float32)
    mask = np.zeros((6, 64, 64), dtype=np.uint8)
    mask_multi = np.zeros((len(CHANNEL_CLASS_IDS), 6, 64, 64), dtype=np.uint8)
    yy, xx = np.ogrid[:64, :64]
    blob = (yy - 32) ** 2 + (xx - 32) ** 2 <= 8**2
    class_id = CHANNEL_CLASS_IDS[channel_index]
    mask[2:5, blob] = np.uint8(class_id)
    mask_multi[channel_index, 2:5, blob] = np.uint8(1)
    image[2:5, blob] = 0.8
    np.savez_compressed(
        path,
        image=image,
        mask=mask,
        mask_multi=mask_multi,
        spacing=np.asarray([1.5, 1.0, 1.0], dtype=np.float32),
    )


def test_train_unet_smoke(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    cases_dir = dataset_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    train_case = cases_dir / "train_case.npz"
    val_case = cases_dir / "val_case.npz"
    _write_case(train_case, channel_index=2)
    _write_case(val_case, channel_index=3)

    with (dataset_dir / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "source", "path"])
        writer.writeheader()
        writer.writerow({"case_id": "train_case", "source": "fixture", "path": str(train_case)})

    with (dataset_dir / "val.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "source", "path"])
        writer.writeheader()
        writer.writerow({"case_id": "val_case", "source": "fixture", "path": str(val_case)})

    manifest = {
        "dataset_name": "fixture",
        "version": "0.0.1",
        "task_type": "multilabel_segmentation",
        "total_spatial_voxels": 6 * 64 * 64 * 2,
        "classes": {
            "0": "background",
            "1": "emphysema",
            "2": "fibrotic_pattern",
            "3": "ground_glass",
            "4": "consolidation",
            "5": "nodule",
            "6": "mass_or_tumor",
            "7": "pleural_effusion",
        },
        "class_channels": [
            {"channel_index": 0, "class_id": 5, "name": "nodule"},
            {"channel_index": 1, "class_id": 6, "name": "mass_or_tumor"},
            {"channel_index": 2, "class_id": 3, "name": "ground_glass"},
            {"channel_index": 3, "class_id": 4, "name": "consolidation"},
            {"channel_index": 4, "class_id": 1, "name": "emphysema"},
            {"channel_index": 5, "class_id": 2, "name": "fibrotic_pattern"},
            {"channel_index": 6, "class_id": 7, "name": "pleural_effusion"},
        ],
        "class_voxels": {"0": 10000, "1": 100, "2": 100, "3": 500, "4": 500, "5": 100, "6": 50, "7": 50},
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    output_path = tmp_path / "model" / "unet.pt"
    metrics_path = tmp_path / "model" / "unet.metrics.json"
    config = TrainConfig(
        dataset_dir=dataset_dir,
        output_path=output_path,
        metrics_path=metrics_path,
        model_version="test",
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_workers=0,
        seed=17,
        negative_stride=2,
        base_channels=8,
        image_size=64,
        device="cpu",
        max_train_steps=1,
        max_val_steps=1,
    )

    checkpoint, metrics = train(config)

    assert output_path.exists()
    assert metrics_path.exists()
    assert checkpoint["model_type"] == "unet2d"
    assert checkpoint["task_type"] == "multilabel_segmentation"
    assert checkpoint["model_config"]["num_classes"] == 7
    assert checkpoint["best_epoch"] == 1
    assert metrics["train_slices"] > 0
    assert metrics["val_slices"] > 0

    loaded = torch.load(output_path, map_location="cpu")
    assert "state_dict" in loaded
