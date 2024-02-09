"""Train U-Net models on the composite CT multi-label segmentation dataset."""

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
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional
    _tqdm = None


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
if str(CTSCAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CTSCAN_ROOT))

from model.unet import UNet2D


DEFAULT_DATASET_DIR = CTSCAN_ROOT / "data" / "ctscan" / "processed" / "unet_composite"
DEFAULT_OUTPUT_PATH = CTSCAN_ROOT / "model" / "unet.pt"
DEFAULT_MODEL_VERSION = "0.1.0"
DEFAULT_CLASS_CHANNELS = [5, 6, 3, 4, 1, 2, 7]


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
    train_mode: str = "2d"
    negative_stride: int = 4
    base_channels: int = 32
    image_size: int = 256
    patch_size: tuple[int, int, int] = (64, 128, 128)
    train_patches_per_case: int = 4
    val_patches_per_case: int = 2
    device: str = "auto"
    max_train_steps: int = 0
    max_val_steps: int = 0


class SliceDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        dataset_dir: Path,
        class_ids: list[int],
        negative_stride: int,
        image_size: int,
        augment: bool,
        seed: int,
    ):
        self.dataset_dir = dataset_dir
        self.class_ids = list(class_ids)
        self.samples: list[tuple[Path, int]] = []
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

        with split_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                case_path = self._resolve_case_path(str(row["path"]))
                if not case_path.exists():
                    continue
                image, mask_multi, roi_mask = self._load_case(case_path)
                if image.ndim != 3 or mask_multi.ndim != 4:
                    continue
                if image.shape != tuple(mask_multi.shape[1:]):
                    continue
                if roi_mask.shape != image.shape:
                    continue

                z_dim = int(image.shape[0])
                positive_map = np.any(mask_multi > 0, axis=0) & (roi_mask > 0)
                flattened = positive_map.reshape(z_dim, -1)
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

    def _scalar_to_multilabel(self, mask: np.ndarray) -> np.ndarray:
        class_id_to_channel = {class_id: index for index, class_id in enumerate(self.class_ids)}
        output = np.zeros((len(self.class_ids),) + tuple(mask.shape), dtype=np.uint8)
        for class_id, channel_index in class_id_to_channel.items():
            output[channel_index, mask == int(class_id)] = np.uint8(1)
        return output

    def _load_case(self, case_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        payload = np.load(case_path)
        image = payload["image"].astype(np.float32)
        if "roi_mask" in payload:
            roi_mask = (payload["roi_mask"].astype(np.uint8) > 0).astype(np.uint8)
        else:
            roi_mask = np.ones(image.shape, dtype=np.uint8)
        if "mask_multi" in payload:
            mask_multi = payload["mask_multi"].astype(np.uint8)
            if mask_multi.ndim == 4 and mask_multi.shape[0] == len(self.class_ids):
                return image, (mask_multi > 0).astype(np.uint8), roi_mask

        mask = payload["mask"].astype(np.uint8)
        mask_multi = self._scalar_to_multilabel(mask)
        return image, mask_multi, roi_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        case_path, slice_index = self.samples[index]
        image_volume, mask_volume, roi_volume = self._load_case(case_path)
        image = image_volume[slice_index].copy()
        mask = mask_volume[:, slice_index].copy()
        roi = roi_volume[slice_index].copy()

        if self.augment:
            if bool(self.rng.integers(0, 2)):
                image = np.flip(image, axis=0)
                mask = np.flip(mask, axis=1)
                roi = np.flip(roi, axis=0)
            if bool(self.rng.integers(0, 2)):
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=2)
                roi = np.flip(roi, axis=1)
            image = image.copy()
            mask = mask.copy()
            roi = roi.copy()

        image_tensor = torch.from_numpy(image[None, :, :]).float()
        mask_tensor = torch.from_numpy(mask).float()
        roi_tensor = torch.from_numpy((roi > 0).astype(np.float32)[None, :, :]).float()

        if image_tensor.shape[-2:] != (self.image_size, self.image_size):
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0)
            roi_tensor = F.interpolate(
                roi_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0)
        return image_tensor, mask_tensor, roi_tensor


class VolumePatchDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        dataset_dir: Path,
        class_ids: list[int],
        patch_size: tuple[int, int, int],
        patches_per_case: int,
        augment: bool,
        seed: int,
    ):
        self.dataset_dir = dataset_dir
        self.class_ids = list(class_ids)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.patches_per_case = max(int(patches_per_case), 1)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)
        self.case_paths: list[Path] = []

        with split_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                case_path = self._resolve_case_path(str(row["path"]))
                if case_path.exists():
                    self.case_paths.append(case_path)

    def _resolve_case_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        candidate = (CTSCAN_ROOT / path).resolve()
        if candidate.exists():
            return candidate
        return (self.dataset_dir / path).resolve()

    def _scalar_to_multilabel(self, mask: np.ndarray) -> np.ndarray:
        class_id_to_channel = {class_id: index for index, class_id in enumerate(self.class_ids)}
        output = np.zeros((len(self.class_ids),) + tuple(mask.shape), dtype=np.uint8)
        for class_id, channel_index in class_id_to_channel.items():
            output[channel_index, mask == int(class_id)] = np.uint8(1)
        return output

    def _load_case(self, case_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        payload = np.load(case_path)
        image = payload["image"].astype(np.float32)
        if "roi_mask" in payload:
            roi_mask = (payload["roi_mask"].astype(np.uint8) > 0).astype(np.uint8)
        else:
            roi_mask = np.ones(image.shape, dtype=np.uint8)
        if "mask_multi" in payload:
            mask_multi = payload["mask_multi"].astype(np.uint8)
            if mask_multi.ndim == 4 and mask_multi.shape[0] == len(self.class_ids):
                return image, (mask_multi > 0).astype(np.uint8), roi_mask
        mask = payload["mask"].astype(np.uint8)
        return image, self._scalar_to_multilabel(mask), roi_mask

    def _choose_center(self, mask_multi: np.ndarray, roi_mask: np.ndarray) -> tuple[int, int, int]:
        positive = np.argwhere((np.any(mask_multi > 0, axis=0)) & (roi_mask > 0))
        if len(positive) > 0 and (self.augment or bool(self.rng.random() < 0.7)):
            center = positive[int(self.rng.integers(0, len(positive)))]
            return int(center[0]), int(center[1]), int(center[2])
        roi_points = np.argwhere(roi_mask > 0)
        if len(roi_points) > 0:
            center = roi_points[int(self.rng.integers(0, len(roi_points)))]
            return int(center[0]), int(center[1]), int(center[2])
        shape = roi_mask.shape
        return shape[0] // 2, shape[1] // 2, shape[2] // 2

    def _extract_patch(self, volume: np.ndarray, center: tuple[int, int, int]) -> np.ndarray:
        target_z, target_y, target_x = self.patch_size
        c_z, c_y, c_x = center
        shape_z, shape_y, shape_x = volume.shape[-3], volume.shape[-2], volume.shape[-1]

        z0 = c_z - (target_z // 2)
        y0 = c_y - (target_y // 2)
        x0 = c_x - (target_x // 2)
        z1 = z0 + target_z
        y1 = y0 + target_y
        x1 = x0 + target_x

        pad_before = [0, 0, 0]
        pad_after = [0, 0, 0]
        if z0 < 0:
            pad_before[0] = -z0
            z0 = 0
        if y0 < 0:
            pad_before[1] = -y0
            y0 = 0
        if x0 < 0:
            pad_before[2] = -x0
            x0 = 0
        if z1 > shape_z:
            pad_after[0] = z1 - shape_z
            z1 = shape_z
        if y1 > shape_y:
            pad_after[1] = y1 - shape_y
            y1 = shape_y
        if x1 > shape_x:
            pad_after[2] = x1 - shape_x
            x1 = shape_x

        patch = volume[..., z0:z1, y0:y1, x0:x1]
        if any(value > 0 for value in pad_before + pad_after):
            pad_spec = [(0, 0)] * (patch.ndim - 3) + [
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            ]
            patch = np.pad(patch, pad_spec, mode="constant")
        return patch

    def __len__(self) -> int:
        return len(self.case_paths) * self.patches_per_case

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.case_paths:
            raise IndexError("No cases in dataset.")
        case_index = int(index // self.patches_per_case) % len(self.case_paths)
        case_path = self.case_paths[case_index]
        image, mask_multi, roi_mask = self._load_case(case_path)

        center = self._choose_center(mask_multi, roi_mask)
        image_patch = self._extract_patch(image, center).astype(np.float32)
        mask_patch = self._extract_patch(mask_multi, center).astype(np.float32)
        roi_patch = self._extract_patch(roi_mask, center).astype(np.float32)

        if self.augment:
            if bool(self.rng.integers(0, 2)):
                image_patch = np.flip(image_patch, axis=1)
                mask_patch = np.flip(mask_patch, axis=2)
                roi_patch = np.flip(roi_patch, axis=1)
            if bool(self.rng.integers(0, 2)):
                image_patch = np.flip(image_patch, axis=2)
                mask_patch = np.flip(mask_patch, axis=3)
                roi_patch = np.flip(roi_patch, axis=2)
            if bool(self.rng.integers(0, 2)):
                image_patch = np.flip(image_patch, axis=0)
                mask_patch = np.flip(mask_patch, axis=1)
                roi_patch = np.flip(roi_patch, axis=0)
            image_patch = image_patch.copy()
            mask_patch = mask_patch.copy()
            roi_patch = roi_patch.copy()

        image_tensor = torch.from_numpy(image_patch[None, :, :, :]).float()
        mask_tensor = torch.from_numpy(mask_patch).float()
        roi_tensor = torch.from_numpy((roi_patch > 0).astype(np.float32)[None, :, :, :]).float()
        return image_tensor, mask_tensor, roi_tensor


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock3D((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            diff_z = skip.size(2) - x.size(2)
            diff_y = skip.size(3) - x.size(3)
            diff_x = skip.size(4) - x.size(4)
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                    diff_z // 2,
                    diff_z - diff_z // 2,
                ],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 7, base_channels: int = 16):
        super().__init__()
        width = int(base_channels)
        self.inc = ConvBlock3D(in_channels, width)
        self.down1 = DownBlock3D(width, width * 2)
        self.down2 = DownBlock3D(width * 2, width * 4)
        self.down3 = DownBlock3D(width * 4, width * 8)
        self.down4 = DownBlock3D(width * 8, width * 16)
        self.up1 = UpBlock3D(width * 16, width * 8, width * 8)
        self.up2 = UpBlock3D(width * 8, width * 4, width * 4)
        self.up3 = UpBlock3D(width * 4, width * 2, width * 2)
        self.up4 = UpBlock3D(width * 2, width, width)
        self.outc = nn.Conv3d(width, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


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
    parser.add_argument("--train-mode", type=str, default="2d", choices=["2d", "3d"])
    parser.add_argument("--negative-stride", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=str, default="64,128,128", help="Patch size z,y,x for 3d mode.")
    parser.add_argument("--train-patches-per-case", type=int, default=4)
    parser.add_argument("--val-patches-per-case", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    args = parser.parse_args()

    metrics_path = args.metrics_path
    if metrics_path is None:
        metrics_path = args.output_path.with_suffix(".metrics.json")

    patch_parts = [part.strip() for part in str(args.patch_size).split(",") if part.strip()]
    if len(patch_parts) != 3:
        raise ValueError("--patch-size must be z,y,x")
    patch_size = tuple(max(int(part), 16) for part in patch_parts)

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
        train_mode=str(args.train_mode).strip().lower(),
        negative_stride=max(int(args.negative_stride), 1),
        base_channels=max(int(args.base_channels), 8),
        image_size=max(int(args.image_size), 64),
        patch_size=patch_size,
        train_patches_per_case=max(int(args.train_patches_per_case), 1),
        val_patches_per_case=max(int(args.val_patches_per_case), 1),
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


def resolve_class_channels(manifest: dict[str, Any]) -> tuple[list[int], list[str]]:
    class_channels = manifest.get("class_channels")
    if isinstance(class_channels, list) and class_channels:
        rows = sorted(class_channels, key=lambda row: int(row.get("channel_index", 0)))
        class_ids = [int(row["class_id"]) for row in rows]
        class_names = [str(row["name"]) for row in rows]
        return class_ids, class_names

    classes = manifest.get("classes", {})
    if isinstance(classes, dict) and classes:
        ids = sorted(int(key) for key in classes.keys() if int(key) != 0)
        if ids:
            return ids, [str(classes[str(class_id)]) for class_id in ids]

    return list(DEFAULT_CLASS_CHANNELS), [str(class_id) for class_id in DEFAULT_CLASS_CHANNELS]


def class_pos_weights_from_manifest(manifest: dict[str, Any], class_ids: list[int]) -> torch.Tensor:
    class_voxels = manifest.get("class_voxels", {})
    total_spatial_voxels = float(manifest.get("total_spatial_voxels", 0) or 0)
    if total_spatial_voxels <= 0.0:
        total_spatial_voxels = float(class_voxels.get("0", 1.0))
    total_spatial_voxels = max(total_spatial_voxels, 1.0)

    weights = []
    for class_id in class_ids:
        positive = max(float(class_voxels.get(str(class_id), 1.0)), 1.0)
        negative = max(total_spatial_voxels - positive, 1.0)
        weights.append(float(np.clip(negative / positive, 1.0, 200.0)))
    return torch.tensor(weights, dtype=torch.float32)


def multilabel_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    roi_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities * roi_mask
    targets = targets * roi_mask
    dims = (0,) + tuple(range(2, int(logits.ndim)))
    intersection = (probabilities * targets).sum(dim=dims)
    cardinality = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def compute_epoch_metrics(
    intersections: np.ndarray,
    unions: np.ndarray,
    predicted_pixels: np.ndarray,
    target_pixels: np.ndarray,
    correct_pixels: int,
    total_pixels: int,
) -> tuple[float, float, float]:
    accuracy = float(correct_pixels / max(total_pixels, 1))

    ious: list[float] = []
    dices: list[float] = []
    for class_index in range(len(intersections)):
        union = float(unions[class_index])
        pred = float(predicted_pixels[class_index])
        target = float(target_pixels[class_index])
        if union > 0.0:
            ious.append(float(intersections[class_index] / union))
        if pred + target > 0.0:
            dices.append(float((2.0 * intersections[class_index]) / (pred + target)))

    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_dice = float(np.mean(dices)) if dices else 0.0
    return accuracy, mean_iou, mean_dice


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int,
    max_steps: int,
    progress_desc: str,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    total_batches = 0

    intersections = np.zeros(num_classes, dtype=np.float64)
    unions = np.zeros(num_classes, dtype=np.float64)
    predicted_pixels = np.zeros(num_classes, dtype=np.float64)
    target_pixels = np.zeros(num_classes, dtype=np.float64)
    correct_pixels = 0
    total_pixels = 0

    total_steps = len(loader)
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)

    if _tqdm is not None:
        progress_iter = _tqdm(loader, total=total_steps, desc=progress_desc, unit="batch", leave=False)
    else:
        progress_iter = loader

    for step_index, (images, masks, rois) in enumerate(progress_iter, start=1):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
        rois = rois.to(device=device, dtype=torch.float32, non_blocking=True)
        if float(rois.sum().item()) <= 0.0:
            continue

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            bce_map = criterion(logits, masks)
            bce_weighted = bce_map * rois
            denom = max(float(rois.sum().item()) * float(num_classes), 1.0)
            bce_loss = bce_weighted.sum() / denom
            dice_loss = multilabel_dice_loss(logits, masks, rois)
            loss = bce_loss + (0.5 * dice_loss)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5) & (rois > 0.5)
        targets = (masks >= 0.5) & (rois > 0.5)
        repeats = [1, num_classes] + [1] * max(int(rois.ndim) - 2, 0)
        roi_targets = rois.repeat(*repeats) > 0.5

        total_loss += float(loss.item())
        total_bce_loss += float(bce_loss.item())
        total_dice_loss += float(dice_loss.item())
        total_batches += 1
        correct_pixels += int((predictions == targets)[roi_targets].sum().item())
        total_pixels += int(roi_targets.sum().item())

        for class_index in range(num_classes):
            pred_class = predictions[:, class_index]
            target_class = targets[:, class_index]
            intersections[class_index] += float((pred_class & target_class).sum().item())
            unions[class_index] += float((pred_class | target_class).sum().item())
            predicted_pixels[class_index] += float(pred_class.sum().item())
            target_pixels[class_index] += float(target_class.sum().item())

        if max_steps > 0 and step_index >= max_steps:
            break

    if _tqdm is not None and hasattr(progress_iter, "close"):
        progress_iter.close()

    accuracy, mean_iou, mean_dice = compute_epoch_metrics(
        intersections=intersections,
        unions=unions,
        predicted_pixels=predicted_pixels,
        target_pixels=target_pixels,
        correct_pixels=correct_pixels,
        total_pixels=total_pixels,
    )
    return {
        "loss": float(total_loss / max(total_batches, 1)),
        "bce_loss": float(total_bce_loss / max(total_batches, 1)),
        "dice_loss": float(total_dice_loss / max(total_batches, 1)),
        "accuracy": accuracy,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
    }


def train(config: TrainConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    print(f"device={device}")

    manifest = read_manifest(config.dataset_dir)
    class_ids, class_names = resolve_class_channels(manifest)
    num_classes = len(class_ids)

    train_csv = config.dataset_dir / "train.csv"
    val_csv = config.dataset_dir / "val.csv"

    if config.train_mode == "3d":
        train_dataset = VolumePatchDataset(
            split_csv=train_csv,
            dataset_dir=config.dataset_dir,
            class_ids=class_ids,
            patch_size=config.patch_size,
            patches_per_case=config.train_patches_per_case,
            augment=True,
            seed=config.seed,
        )
        val_dataset = VolumePatchDataset(
            split_csv=val_csv,
            dataset_dir=config.dataset_dir,
            class_ids=class_ids,
            patch_size=config.patch_size,
            patches_per_case=config.val_patches_per_case,
            augment=False,
            seed=config.seed + 1,
        )
    else:
        train_dataset = SliceDataset(
            split_csv=train_csv,
            dataset_dir=config.dataset_dir,
            class_ids=class_ids,
            negative_stride=config.negative_stride,
            image_size=config.image_size,
            augment=True,
            seed=config.seed,
        )
        val_dataset = SliceDataset(
            split_csv=val_csv,
            dataset_dir=config.dataset_dir,
            class_ids=class_ids,
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

    if config.train_mode == "3d":
        model = UNet3D(in_channels=1, num_classes=num_classes, base_channels=config.base_channels).to(device)
        pos_weights = class_pos_weights_from_manifest(manifest, class_ids).to(device).view(-1, 1, 1, 1)
    else:
        model = UNet2D(in_channels=1, num_classes=num_classes, base_channels=config.base_channels).to(device)
        pos_weights = class_pos_weights_from_manifest(manifest, class_ids).to(device).view(-1, 1, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")
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
            progress_desc=f"Epoch {epoch}/{config.epochs} train",
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
                    progress_desc=f"Epoch {epoch}/{config.epochs} val",
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
            "train_bce_loss": round(float(train_metrics["bce_loss"]), 6),
            "train_dice_loss": round(float(train_metrics["dice_loss"]), 6),
            "train_accuracy": round(float(train_metrics["accuracy"]), 6),
            "train_mean_iou": round(float(train_metrics["mean_iou"]), 6),
            "train_mean_dice": round(float(train_metrics["mean_dice"]), 6),
        }
        if val_metrics is not None:
            row.update(
                {
                    "val_loss": round(float(val_metrics["loss"]), 6),
                    "val_bce_loss": round(float(val_metrics["bce_loss"]), 6),
                    "val_dice_loss": round(float(val_metrics["dice_loss"]), 6),
                    "val_accuracy": round(float(val_metrics["accuracy"]), 6),
                    "val_mean_iou": round(float(val_metrics["mean_iou"]), 6),
                    "val_mean_dice": round(float(val_metrics["mean_dice"]), 6),
                }
            )
        history.append(row)

        if val_metrics is None:
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"train_miou={train_metrics['mean_iou']:.4f} "
                f"train_mdice={train_metrics['mean_dice']:.4f}"
            )
        else:
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} train_miou={train_metrics['mean_iou']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_miou={val_metrics['mean_iou']:.4f} val_mdice={val_metrics['mean_dice']:.4f}"
            )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_version": config.model_version,
        "model_type": "unet3d" if config.train_mode == "3d" else "unet2d",
        "task_type": "multilabel_segmentation",
        "model_config": {
            "in_channels": 1,
            "num_classes": num_classes,
            "base_channels": config.base_channels,
            "train_mode": config.train_mode,
        },
        "class_channels": [
            {"channel_index": int(index), "class_id": int(class_id), "name": str(class_names[index])}
            for index, class_id in enumerate(class_ids)
        ],
        "class_names": {str(class_id): class_names[index] for index, class_id in enumerate(class_ids)},
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
        "task_type": "multilabel_segmentation",
        "roi_gated": True,
        "train_mode": config.train_mode,
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "history": history,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        # Backward-compatible aliases used by existing smoke tests.
        "train_slices": len(train_dataset),
        "val_slices": len(val_dataset),
        "image_size": config.image_size if config.train_mode == "2d" else None,
        "patch_size": list(config.patch_size) if config.train_mode == "3d" else None,
        "class_channels": checkpoint["class_channels"],
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
        f"train_samples={metrics['train_samples']} val_samples={metrics['val_samples']}"
    )


if __name__ == "__main__":
    main()
