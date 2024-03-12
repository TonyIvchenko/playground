from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.ctscan.scripts.segmentation.train_unet_backbone import SlicePairDataset, load_split_rows


def _write_pair(images_dir: Path, masks_dir: Path, name: str, size: int = 64) -> None:
    Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L").save(images_dir / f"{name}.png")
    Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L").save(masks_dir / f"{name}.png")


def test_load_split_rows_falls_back_to_legacy_split_json(tmp_path: Path):
    root = tmp_path / "legacy_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "case001")
    _write_pair(images_dir, masks_dir, "case002")
    (root / "splits.json").write_text(
        json.dumps({"train": ["case001"], "val": ["case002"], "test": []}),
        encoding="utf-8",
    )

    train_rows = load_split_rows(root, "train")
    val_rows = load_split_rows(root, "val")

    assert train_rows == [{"image": "images/case001.png", "mask": "masks/case001.png"}]
    assert val_rows == [{"image": "images/case002.png", "mask": "masks/case002.png"}]


def test_slice_pair_dataset_reads_legacy_rows(tmp_path: Path):
    root = tmp_path / "legacy_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "case001", size=80)
    rows = [{"image": "images/case001.png", "mask": "masks/case001.png"}]

    dataset = SlicePairDataset(root, rows, image_size=64)
    image, mask = dataset[0]

    assert tuple(image.shape) == (1, 64, 64)
    assert tuple(mask.shape) == (64, 64)
