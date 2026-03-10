"""Train a 2D U-Net on the composite CT semantic segmentation dataset."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
if str(CTSCAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CTSCAN_ROOT))

from model.unet import UNet2D


DEFAULT_DATASET_DIR = CTSCAN_ROOT / "data" / "ctscan" / "processed" / "unet_composite"
DEFAULT_OUTPUT_PATH = CTSCAN_ROOT / "model" / "unet.pt"
DEFAULT_MODEL_VERSION = "0.1.0"


@dataclass
class TrainConfig:
    dataset_dir: Path
    output_path: Path
    metrics_path: Path
    model_version: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    negative_stride: int
    base_channels: int
    image_size: int
    device: str
    max_train_steps: int
    max_val_steps: int


class SliceDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        dataset_dir: Path,
        negative_stride: int,
        image_size: int,
        augment: bool,
        seed: int,
    ):
        self.dataset_dir = dataset_dir
        self.samples: list[tuple[Path, int]] = []
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)
        self._cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

        with split_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                case_path = self._resolve_case_path(str(row["path"]))
                if not case_path.exists():
                    continue
                payload = np.load(case_path)
                mask = payload["mask"].astype(np.uint8)
                z_dim = int(mask.shape[0])
                flattened = mask.reshape(z_dim, -1)
                positive = np.where(flattened.max(axis=1) > 0)[0].tolist()
                negative = np.where(flattened.max(axis=1) == 0)[0].tolist()

                if negative_stride <= 1:
                    selected_negative = negative
                else:
                    selected_negative = negative[::negative_stride]

                for index in positive + selected_negative:
                    self.samples.append((case_path, int(index)))

        if self.samples:
            self.rng.shuffle(self.samples)

    def _resolve_case_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        candidate = (CTSCAN_ROOT / path).resolve()
        if candidate.exists():
            return candidate
        return (self.dataset_dir / path).resolve()

    def _load_case(self, case_path: Path) -> tuple[np.ndarray, np.ndarray]:
        if case_path not in self._cache:
            payload = np.load(case_path)
            image = payload["image"].astype(np.float32)
            mask = payload["mask"].astype(np.uint8)
            self._cache[case_path] = (image, mask)
        return self._cache[case_path]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        case_path, slice_index = self.samples[index]
        image_volume, mask_volume = self._load_case(case_path)
        image = image_volume[slice_index].copy()
        mask = mask_volume[slice_index].copy()

        if self.augment:
            if bool(self.rng.integers(0, 2)):
                image = np.flip(image, axis=0)
                mask = np.flip(mask, axis=0)
            if bool(self.rng.integers(0, 2)):
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=1)
            image = image.copy()
            mask = mask.copy()

        image_tensor = torch.from_numpy(image[None, :, :]).float()
        mask_tensor = torch.from_numpy(mask).long()

        if image_tensor.shape[-2:] != (self.image_size, self.image_size):
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_tensor = (
                F.interpolate(
                    mask_tensor[None, None, :, :].float(),
                    size=(self.image_size, self.image_size),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )
        return image_tensor, mask_tensor


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train CT semantic segmentation U-Net.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--model-version", type=str, default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--negative-stride", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    args = parser.parse_args()

    metrics_path = args.metrics_path
    if metrics_path is None:
        metrics_path = args.output_path.with_suffix(".metrics.json")

    return TrainConfig(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        metrics_path=metrics_path,
        model_version=str(args.model_version),
        epochs=max(int(args.epochs), 1),
        batch_size=max(int(args.batch_size), 1),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        num_workers=max(int(args.num_workers), 0),
        seed=int(args.seed),
        negative_stride=max(int(args.negative_stride), 1),
        base_channels=max(int(args.base_channels), 8),
        image_size=max(int(args.image_size), 64),
        device=str(args.device).strip().lower(),
        max_train_steps=max(int(args.max_train_steps), 0),
        max_val_steps=max(int(args.max_val_steps), 0),
    )


def resolve_device(name: str) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def class_weights_from_manifest(manifest: dict[str, Any], num_classes: int) -> torch.Tensor:
    class_voxels = manifest.get("class_voxels", {})
    counts = np.array(
        [float(class_voxels.get(str(index), 1.0)) for index in range(num_classes)],
        dtype=np.float32,
    )
    counts = np.clip(counts, 1.0, None)
    weights = 1.0 / np.sqrt(counts)
    if len(weights) > 0:
        weights[0] *= 0.35
    weights = weights / float(np.mean(weights))
    return torch.from_numpy(weights.astype(np.float32))


def compute_epoch_metrics(
    intersections: np.ndarray,
    unions: np.ndarray,
    correct_pixels: int,
    total_pixels: int,
) -> tuple[float, float]:
    accuracy = float(correct_pixels / max(total_pixels, 1))

    ious: list[float] = []
    for class_id in range(1, len(intersections)):
        union = float(unions[class_id])
        if union <= 0.0:
            continue
        iou = float(intersections[class_id] / union)
        ious.append(iou)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return accuracy, mean_iou


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int,
    max_steps: int,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_batches = 0
    intersections = np.zeros(num_classes, dtype=np.float64)
    unions = np.zeros(num_classes, dtype=np.float64)
    correct_pixels = 0
    total_pixels = 0

    for step_index, (images, masks) in enumerate(loader, start=1):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device=device, dtype=torch.long, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, masks)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += float(loss.item())
        total_batches += 1
        correct_pixels += int((predictions == masks).sum().item())
        total_pixels += int(masks.numel())

        for class_id in range(num_classes):
            pred_class = predictions == class_id
            mask_class = masks == class_id
            intersections[class_id] += float((pred_class & mask_class).sum().item())
            unions[class_id] += float((pred_class | mask_class).sum().item())

        if max_steps > 0 and step_index >= max_steps:
            break

    accuracy, mean_iou = compute_epoch_metrics(intersections, unions, correct_pixels, total_pixels)
    return {
        "loss": float(total_loss / max(total_batches, 1)),
        "accuracy": accuracy,
        "mean_iou": mean_iou,
    }


def train(config: TrainConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    print(f"device={device}")

    manifest = read_manifest(config.dataset_dir)
    classes = manifest.get("classes", {})
    num_classes = max(int(key) for key in classes.keys()) + 1 if classes else 6

    train_csv = config.dataset_dir / "train.csv"
    val_csv = config.dataset_dir / "val.csv"

    train_dataset = SliceDataset(
        split_csv=train_csv,
        dataset_dir=config.dataset_dir,
        negative_stride=config.negative_stride,
        image_size=config.image_size,
        augment=True,
        seed=config.seed,
    )
    val_dataset = SliceDataset(
        split_csv=val_csv,
        dataset_dir=config.dataset_dir,
        negative_stride=1,
        image_size=config.image_size,
        augment=False,
        seed=config.seed + 1,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No training samples found in dataset split.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    ) if len(val_dataset) > 0 else None

    model = UNet2D(in_channels=1, num_classes=num_classes, base_channels=config.base_channels).to(device)
    class_weights = class_weights_from_manifest(manifest, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            max_steps=config.max_train_steps,
        )
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    optimizer=None,
                    device=device,
                    num_classes=num_classes,
                    max_steps=config.max_val_steps,
                )
            monitor_score = float(val_metrics["mean_iou"])
        else:
            val_metrics = None
            monitor_score = float(train_metrics["mean_iou"])

        if monitor_score >= best_score:
            best_score = monitor_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        row = {
            "epoch": epoch,
            "train_loss": round(float(train_metrics["loss"]), 6),
            "train_accuracy": round(float(train_metrics["accuracy"]), 6),
            "train_mean_iou": round(float(train_metrics["mean_iou"]), 6),
        }
        if val_metrics is not None:
            row.update(
                {
                    "val_loss": round(float(val_metrics["loss"]), 6),
                    "val_accuracy": round(float(val_metrics["accuracy"]), 6),
                    "val_mean_iou": round(float(val_metrics["mean_iou"]), 6),
                }
            )
        history.append(row)

        if val_metrics is None:
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} train_miou={train_metrics['mean_iou']:.4f}"
            )
        else:
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} train_miou={train_metrics['mean_iou']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_miou={val_metrics['mean_iou']:.4f}"
            )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_version": config.model_version,
        "model_type": "unet2d",
        "model_config": {
            "in_channels": 1,
            "num_classes": num_classes,
            "base_channels": config.base_channels,
        },
        "class_names": classes,
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "history": history,
        "dataset_manifest": str((config.dataset_dir / "manifest.json").resolve()),
        "state_dict": best_state,
    }
    torch.save(checkpoint, config.output_path)

    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "model_version": config.model_version,
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "history": history,
        "device": str(device),
        "train_slices": len(train_dataset),
        "val_slices": len(val_dataset),
        "image_size": config.image_size,
    }
    config.metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return checkpoint, metrics_payload


def main() -> None:
    config = parse_args()
    checkpoint, metrics = train(config)
    print(f"saved_checkpoint={config.output_path}")
    print(f"saved_metrics={config.metrics_path}")
    print(
        f"best_epoch={checkpoint['best_epoch']} "
        f"best_miou={checkpoint['best_score']:.4f} "
        f"train_slices={metrics['train_slices']} val_slices={metrics['val_slices']}"
    )


if __name__ == "__main__":
    main()
