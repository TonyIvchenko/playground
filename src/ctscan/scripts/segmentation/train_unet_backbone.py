"""Train a 2D U-Net with a pretrained encoder backbone on PNG slice pairs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional
    _tqdm = None

import segmentation_models_pytorch as smp


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SLICE_DIR = CTSCAN_ROOT / "data" / "ctscan" / "processed" / "slice_dataset"
DEFAULT_OUTPUT_PATH = CTSCAN_ROOT / "model" / "unet_backbone.pt"
DEFAULT_METRICS_PATH = CTSCAN_ROOT / "model" / "unet_backbone.metrics.json"


@dataclass
class TrainConfig:
    slice_dir: Path
    output_path: Path
    metrics_path: Path
    model_version: str
    encoder_name: str
    encoder_weights: str | None
    classes: int
    in_channels: int
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    num_workers: int
    seed: int
    device: str
    max_train_batches: int
    max_val_batches: int
    max_test_batches: int


class SlicePairDataset(Dataset):
    def __init__(self, root: Path, split_csv: Path, image_size: int):
        self.root = root
        self.image_size = int(image_size)
        self.rows: list[dict[str, str]] = []
        if split_csv.exists():
            with split_csv.open("r", encoding="utf-8", newline="") as handle:
                self.rows = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(self.root / row["image"]).convert("L")
        mask = Image.open(self.root / row["mask"]).convert("L")

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = np.asarray(mask, dtype=np.int64)

        image_t = torch.from_numpy(image_arr).unsqueeze(0)
        mask_t = torch.from_numpy(mask_arr)

        if image_t.shape[-2:] != (self.image_size, self.image_size):
            image_t = F.interpolate(
                image_t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_t = F.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0).float(),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()

        return image_t, mask_t


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train pretrained-backbone U-Net on slice PNG pairs.")
    parser.add_argument("--slice-dir", type=Path, default=DEFAULT_SLICE_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--model-version", type=str, default="0.1.0-backbone")
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    args = parser.parse_args()

    encoder_weights = None if str(args.encoder_weights).strip().lower() in {"", "none"} else str(args.encoder_weights)
    return TrainConfig(
        slice_dir=args.slice_dir.resolve(),
        output_path=args.output_path.resolve(),
        metrics_path=args.metrics_path.resolve(),
        model_version=str(args.model_version),
        encoder_name=str(args.encoder_name),
        encoder_weights=encoder_weights,
        classes=max(int(args.classes), 2),
        in_channels=max(int(args.in_channels), 1),
        image_size=max(int(args.image_size), 64),
        batch_size=max(int(args.batch_size), 1),
        epochs=max(int(args.epochs), 1),
        learning_rate=float(args.learning_rate),
        num_workers=max(int(args.num_workers), 0),
        seed=int(args.seed),
        device=str(args.device).strip().lower(),
        max_train_batches=max(int(args.max_train_batches), 0),
        max_val_batches=max(int(args.max_val_batches), 0),
        max_test_batches=max(int(args.max_test_batches), 0),
    )


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    classes: int,
) -> dict[str, Any]:
    eps = 1e-6
    ious = []
    dices = []
    ious_fg = []
    dices_fg = []
    for c in range(classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        denom_iou = tp + fp + fn
        denom_dice = (2 * tp) + fp + fn
        iou = float(tp / (denom_iou + eps))
        dice = float((2 * tp) / (denom_dice + eps))
        ious.append(iou)
        dices.append(dice)
        if c > 0:
            ious_fg.append(iou)
            dices_fg.append(dice)
    return {
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "mean_iou_fg": float(np.mean(ious_fg)) if ious_fg else 0.0,
        "mean_dice_fg": float(np.mean(dices_fg)) if dices_fg else 0.0,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    classes: int,
    max_batches: int,
    progress_desc: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses = []
    all_pred: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []

    total_batches = len(loader)
    if max_batches > 0:
        total_batches = min(total_batches, max_batches)
    if _tqdm is not None:
        progress_iter = _tqdm(loader, total=total_batches, desc=progress_desc, unit="batch", leave=False)
    else:
        progress_iter = loader

    for batch_idx, (image, mask) in enumerate(progress_iter, start=1):
        image = image.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.long)

        with torch.set_grad_enabled(training):
            logits = model(image)
            loss = criterion(logits, mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        pred = torch.argmax(logits, dim=1).detach().cpu()
        all_pred.append(pred)
        all_target.append(mask.detach().cpu())
        if max_batches > 0 and batch_idx >= max_batches:
            break

    if _tqdm is not None and hasattr(progress_iter, "close"):
        progress_iter.close()

    if not losses:
        return {"loss": 0.0, "mean_iou": 0.0, "mean_dice": 0.0, "mean_iou_fg": 0.0, "mean_dice_fg": 0.0}

    pred_cat = torch.cat(all_pred, dim=0)
    target_cat = torch.cat(all_target, dim=0)
    metrics = class_metrics(pred=pred_cat, target=target_cat, classes=classes)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    print(f"device={device}")

    splits_dir = config.slice_dir / "splits"
    train_ds = SlicePairDataset(config.slice_dir, splits_dir / "train.csv", config.image_size)
    val_ds = SlicePairDataset(config.slice_dir, splits_dir / "val.csv", config.image_size)
    test_ds = SlicePairDataset(config.slice_dir, splits_dir / "test.csv", config.image_size)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training rows found in {splits_dir / 'train.csv'}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        classes=config.classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        train_m = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            classes=config.classes,
            max_batches=config.max_train_batches,
            progress_desc=f"Epoch {epoch}/{config.epochs} train",
        )
        with torch.no_grad():
            val_m = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                criterion=criterion,
                device=device,
                classes=config.classes,
                max_batches=config.max_val_batches,
                progress_desc=f"Epoch {epoch}/{config.epochs} val",
            ) if len(val_ds) > 0 else {"loss": 0.0, "mean_iou": 0.0, "mean_dice": 0.0, "mean_iou_fg": 0.0, "mean_dice_fg": 0.0}

        row = {
            "epoch": float(epoch),
            "train_loss": train_m["loss"],
            "train_mean_iou_fg": train_m["mean_iou_fg"],
            "train_mean_dice_fg": train_m["mean_dice_fg"],
            "val_loss": val_m["loss"],
            "val_mean_iou_fg": val_m["mean_iou_fg"],
            "val_mean_dice_fg": val_m["mean_dice_fg"],
        }
        history.append(row)
        print(
            f"epoch={epoch} "
            f"train_loss={train_m['loss']:.4f} train_iou_fg={train_m['mean_iou_fg']:.4f} "
            f"val_loss={val_m['loss']:.4f} val_iou_fg={val_m['mean_iou_fg']:.4f}"
        )

        if val_m["loss"] <= best_val:
            best_val = val_m["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    checkpoint = {
        "model_version": config.model_version,
        "model_type": "unet_pretrained_backbone",
        "encoder_name": config.encoder_name,
        "encoder_weights": config.encoder_weights,
        "in_channels": config.in_channels,
        "classes": config.classes,
        "image_size": config.image_size,
        "best_epoch": best_epoch,
        "history": history,
        "state_dict": best_state,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, config.output_path)

    with torch.no_grad():
        test_m = run_epoch(
            model=model,
            loader=test_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            classes=config.classes,
            max_batches=config.max_test_batches,
            progress_desc="Test",
        ) if len(test_ds) > 0 else {"loss": 0.0, "mean_iou": 0.0, "mean_dice": 0.0, "mean_iou_fg": 0.0, "mean_dice_fg": 0.0}

    metrics = {
        "model_version": config.model_version,
        "encoder_name": config.encoder_name,
        "encoder_weights": config.encoder_weights,
        "device": str(device),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "test_rows": len(test_ds),
        "best_epoch": best_epoch,
        "history": history,
        "test": test_m,
    }
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved_checkpoint={config.output_path}")
    print(f"saved_metrics={config.metrics_path}")
    return metrics


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
