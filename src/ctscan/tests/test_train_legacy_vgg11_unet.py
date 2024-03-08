from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import nibabel
import torch
from torch.utils.data import DataLoader
import torchvision

from src.ctscan.scripts.segmentation.train_legacy_vgg11_unet import LegacyLungDataset, TrainConfig, existing_png_names, train


def _write_pair(images_dir: Path, masks_dir: Path, name: str, size: int) -> None:
    image = np.random.randint(0, 255, size=(size, size), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 4: size // 2, size // 4: size // 2] = 2
    Image.fromarray(image, mode="L").save(images_dir / f"{name}.png")
    Image.fromarray(mask, mode="L").save(masks_dir / f"{name}.png")


def test_legacy_dataset_keeps_original_size_by_default(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "a", 334)

    ds = LegacyLungDataset(["a"], images_dir=images_dir, masks_dir=masks_dir, image_size=0)
    image, mask = ds[0]
    assert tuple(image.shape) == (1, 334, 334)
    assert tuple(mask.shape) == (334, 334)


def test_legacy_dataset_optionally_resizes_for_batching(tmp_path: Path):
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


def test_existing_png_names_returns_only_paired_pngs(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "a", 64)
    _write_pair(images_dir, masks_dir, "b", 64)
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L").save(masks_dir / "orphan.png")

    assert existing_png_names(images_dir, masks_dir) == ["a", "b"]


def test_legacy_trainer_writes_epoch_checkpoints(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data_root"
    dataset_dir = data_root / "dataset"
    mask_dir = data_root / "mask"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    volume = np.zeros((4, 128, 128), dtype=np.float32)
    mask = np.zeros((4, 128, 128), dtype=np.uint8)
    mask[:, 16:32, 16:32] = 2

    nibabel.save(nibabel.Nifti1Image(np.transpose(volume, (1, 2, 0)), affine=np.eye(4)), str(dataset_dir / "case001.nii.gz"))
    nibabel.save(nibabel.Nifti1Image(np.transpose(mask, (1, 2, 0)), affine=np.eye(4)), str(mask_dir / "case001mask.nii"))

    original_vgg11 = torchvision.models.vgg11

    def _vgg11_no_weights(*args, **kwargs):
        kwargs["weights"] = None
        return original_vgg11(*args, **kwargs)

    monkeypatch.setattr(torchvision.models, "vgg11", _vgg11_no_weights)

    config = TrainConfig(
        data_root=data_root,
        work_dir=tmp_path / "work",
        output_path=tmp_path / "model" / "legacy_vgg11_unet.pt",
        metrics_path=tmp_path / "model" / "legacy_vgg11_unet.metrics.json",
        log_path=tmp_path / "model" / "legacy_vgg11_unet.train.log",
        resume_path=None,
        model_version="test-legacy-vgg11-unet-0.1.0",
        epochs=2,
        batch_size=1,
        learning_rate=1e-3,
        image_size=0,
        seed=42,
        num_workers=0,
        device="cpu",
        overwrite_workdir=True,
        skip_existing_png=False,
        max_volumes=1,
    )

    metrics = train(config)

    assert (tmp_path / "model" / "legacy_vgg11_unet.epoch001.pt").exists()
    assert (tmp_path / "model" / "legacy_vgg11_unet.epoch002.pt").exists()
    assert config.output_path.exists()
    assert metrics["epoch_checkpoint_pattern"].endswith("legacy_vgg11_unet.epochNNN.pt")
    loaded = torch.load(config.output_path, map_location="cpu")
    assert isinstance(loaded, dict)
    assert all(isinstance(value, torch.Tensor) for value in loaded.values())
