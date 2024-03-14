from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from src.ctscan.scripts.segmentation.train_unet_backbone import (
    MulticlassFocalLoss,
    PRESET_DEFAULTS,
    SlicePairDataset,
    load_split_rows,
    metric_direction,
    parse_args,
)


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


def test_metric_direction_prefers_higher_for_dice_and_lower_for_loss():
    assert metric_direction("val_mean_dice_fg") == 1
    assert metric_direction("val_loss") == -1


def test_multiclass_focal_loss_runs_on_logits_and_integer_targets():
    criterion = MulticlassFocalLoss()
    logits = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))

    loss = criterion(logits, target)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_legacy_png_best_preset_sets_measured_winner(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--preset", "legacy_png_best"])

    config = parse_args()

    assert config.architecture == PRESET_DEFAULTS["legacy_png_best"]["architecture"]
    assert config.encoder_name == PRESET_DEFAULTS["legacy_png_best"]["encoder_name"]
    assert config.classes == PRESET_DEFAULTS["legacy_png_best"]["classes"]
    assert config.image_size == PRESET_DEFAULTS["legacy_png_best"]["image_size"]
    assert config.batch_size == PRESET_DEFAULTS["legacy_png_best"]["batch_size"]
