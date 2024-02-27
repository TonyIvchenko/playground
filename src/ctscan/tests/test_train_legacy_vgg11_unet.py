from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.ctscan.scripts.segmentation.train_legacy_vgg11_unet import LegacyLungDataset


def _write_pair(images_dir: Path, masks_dir: Path, name: str, size: int) -> None:
    image = np.random.randint(0, 255, size=(size, size), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 4: size // 2, size // 4: size // 2] = 2
    Image.fromarray(image, mode="L").save(images_dir / f"{name}.png")
    Image.fromarray(mask, mode="L").save(masks_dir / f"{name}.png")


def test_legacy_dataset_resizes_for_batching(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "a", 334)
    _write_pair(images_dir, masks_dir, "b", 340)

    ds = LegacyLungDataset(["a", "b"], images_dir=images_dir, masks_dir=masks_dir, image_size=320)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    images, masks = next(iter(loader))
    assert isinstance(images, torch.Tensor)
    assert isinstance(masks, torch.Tensor)
    assert tuple(images.shape) == (2, 1, 320, 320)
    assert tuple(masks.shape) == (2, 320, 320)

