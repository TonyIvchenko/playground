from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn

from src.ctscan.scripts.segmentation.train_unet_backbone import (
    build_scheduler,
    compute_class_weights,
    compute_sample_weights,
    MulticlassFocalLoss,
    PRESET_DEFAULTS,
    SlicePairDataset,
    TrainConfig,
    load_split_rows,
    metric_direction,
    parse_args,
    train,
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


def test_slice_pair_dataset_light_augmentation_preserves_shape(tmp_path: Path):
    root = tmp_path / "legacy_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "case001", size=80)
    rows = [{"image": "images/case001.png", "mask": "masks/case001.png"}]

    dataset = SlicePairDataset(root, rows, image_size=64, augmentation_name="light")
    image, mask = dataset[0]

    assert tuple(image.shape) == (1, 64, 64)
    assert tuple(mask.shape) == (64, 64)
    assert image.min().item() >= 0.0
    assert image.max().item() <= 1.0


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


def test_build_scheduler_returns_onecycle_when_requested():
    model = nn.Conv2d(1, 2, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    scheduler = build_scheduler("onecycle", optimizer, learning_rate=3e-4, steps_per_epoch=4, epochs=2)

    assert scheduler is not None


def test_compute_class_weights_scales_down_background(tmp_path: Path):
    root = tmp_path / "legacy_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "case001")
    sparse = np.zeros((64, 64), dtype=np.uint8)
    sparse[0:4, 0:4] = 1
    sparse[4:6, 4:6] = 2
    sparse[6:10, 6:10] = 3
    Image.fromarray(sparse, mode="L").save(masks_dir / "case001.png")

    weights = compute_class_weights(root, [{"image": "images/case001.png", "mask": "masks/case001.png"}], 4, "inverse_sqrt")

    assert weights is not None
    assert tuple(weights.shape) == (4,)
    assert weights[0].item() < weights[1].item()
    assert weights[0].item() < weights[2].item()
    assert weights[0].item() < weights[3].item()
    assert abs(weights.mean().item() - 1.0) < 1e-6


def test_compute_sample_weights_boosts_rare_foreground_slices(tmp_path: Path):
    root = tmp_path / "legacy_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "bg")
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L").save(masks_dir / "bg.png")

    _write_pair(images_dir, masks_dir, "common")
    common = np.zeros((64, 64), dtype=np.uint8)
    common[0:8, 0:8] = 3
    Image.fromarray(common, mode="L").save(masks_dir / "common.png")

    _write_pair(images_dir, masks_dir, "rare")
    rare = np.zeros((64, 64), dtype=np.uint8)
    rare[0:8, 0:8] = 1
    Image.fromarray(rare, mode="L").save(masks_dir / "rare.png")

    rows = [
        {"image": "images/bg.png", "mask": "masks/bg.png"},
        {"image": "images/common.png", "mask": "masks/common.png"},
        {"image": "images/rare.png", "mask": "masks/rare.png"},
    ]
    weights = compute_sample_weights(root, rows, 4, "rare_fg")

    assert weights is not None
    assert weights[0] == 1.0
    assert weights[1] > weights[0]
    assert weights[2] > weights[0]


def test_legacy_png_best_preset_sets_measured_winner(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--preset", "legacy_png_best"])

    config = parse_args()

    assert config.architecture == PRESET_DEFAULTS["legacy_png_best"]["architecture"]
    assert config.encoder_name == PRESET_DEFAULTS["legacy_png_best"]["encoder_name"]
    assert config.classes == PRESET_DEFAULTS["legacy_png_best"]["classes"]
    assert config.image_size == PRESET_DEFAULTS["legacy_png_best"]["image_size"]
    assert config.batch_size == PRESET_DEFAULTS["legacy_png_best"]["batch_size"]
    assert config.class_weight_mode == PRESET_DEFAULTS["legacy_png_best"]["class_weight_mode"]
    assert config.scheduler_name == PRESET_DEFAULTS["legacy_png_best"]["scheduler"]
    assert config.sampler_name == PRESET_DEFAULTS["legacy_png_best"]["sampler"]
    assert config.augmentation_name == PRESET_DEFAULTS["legacy_png_best"]["augmentation"]


def test_train_evaluates_test_metrics_with_best_checkpoint_state(tmp_path: Path, monkeypatch):
    root = tmp_path / "slice_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    _write_pair(images_dir, masks_dir, "case001")
    rows = [{"image": "images/case001.png", "mask": "masks/case001.png"}]

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.0]))

        def forward(self, image):  # pragma: no cover - not used by patched run_epoch
            batch, _, height, width = image.shape
            return torch.zeros((batch, 2, height, width), dtype=image.dtype, device=image.device)

    call_state = {"train_epoch": 0, "val_epoch": 0}

    def fake_run_epoch(model, loader, optimizer, scheduler, criterion, device, classes, max_batches, progress_desc):
        if optimizer is not None:
            call_state["train_epoch"] += 1
            model.weight.data.fill_(float(call_state["train_epoch"]))
            return {"loss": 1.0, "mean_iou": 0.0, "mean_dice": 0.0, "mean_iou_fg": 0.0, "mean_dice_fg": 0.0}
        if progress_desc.startswith("Epoch"):
            call_state["val_epoch"] += 1
            if call_state["val_epoch"] == 1:
                return {"loss": 0.5, "mean_iou": 0.0, "mean_dice": 0.0, "mean_iou_fg": 0.8, "mean_dice_fg": 0.8}
            return {"loss": 0.6, "mean_iou": 0.0, "mean_dice": 0.0, "mean_iou_fg": 0.1, "mean_dice_fg": 0.1}
        assert float(model.weight.item()) == 1.0
        return {"loss": 0.4, "mean_iou": 0.0, "mean_dice": 0.0, "mean_iou_fg": 0.7, "mean_dice_fg": 0.7}

    monkeypatch.setattr("src.ctscan.scripts.segmentation.train_unet_backbone.build_model", lambda config: DummyModel())
    monkeypatch.setattr("src.ctscan.scripts.segmentation.train_unet_backbone.build_optimizer", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.ctscan.scripts.segmentation.train_unet_backbone.build_loss", lambda *args, **kwargs: nn.CrossEntropyLoss())
    monkeypatch.setattr("src.ctscan.scripts.segmentation.train_unet_backbone.load_split_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr("src.ctscan.scripts.segmentation.train_unet_backbone.run_epoch", fake_run_epoch)

    config = TrainConfig(
        slice_dir=root,
        output_path=tmp_path / "model.pt",
        metrics_path=tmp_path / "metrics.json",
        model_version="test",
        architecture="fpn",
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        classes=2,
        in_channels=1,
        image_size=64,
        batch_size=1,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_name="adamw",
        loss_name="dice_ce",
        class_weight_mode="none",
        scheduler_name="none",
        sampler_name="none",
        augmentation_name="none",
        selection_metric="val_mean_dice_fg",
        num_workers=0,
        seed=17,
        device="cpu",
        max_train_batches=0,
        max_val_batches=0,
        max_test_batches=0,
    )

    metrics = train(config)

    assert metrics["best_epoch"] == 1
