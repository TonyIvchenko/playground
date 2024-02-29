from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.ctscan.scripts.segmentation.resize_png_dataset import resize_folder


def _write_png(path: Path, shape: tuple[int, int], value: int) -> None:
    image = np.full(shape, value, dtype=np.uint8)
    Image.fromarray(image, mode="L").save(path)


def test_resize_png_dataset_in_place(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_png(images_dir / "a.png", (630, 630), 100)
    _write_png(images_dir / "b.png", (512, 512), 150)
    _write_png(masks_dir / "a.png", (630, 630), 3)
    _write_png(masks_dir / "b.png", (512, 512), 1)

    image_summary = resize_folder(images_dir, size=512, dry_run=False)
    mask_summary = resize_folder(masks_dir, size=512, dry_run=False)

    assert image_summary["resized"] == 1
    assert image_summary["unchanged"] == 1
    assert mask_summary["resized"] == 1
    assert mask_summary["unchanged"] == 1

    with Image.open(images_dir / "a.png") as image_a:
        assert image_a.size == (512, 512)
    with Image.open(images_dir / "b.png") as image_b:
        assert image_b.size == (512, 512)
    with Image.open(masks_dir / "a.png") as mask_a:
        assert mask_a.size == (512, 512)
        assert set(np.unique(np.asarray(mask_a)).tolist()) == {3}
    with Image.open(masks_dir / "b.png") as mask_b:
        assert mask_b.size == (512, 512)
