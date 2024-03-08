"""Train the legacy VGG11 U-Net pipeline used by the old training notebook.

This script reproduces the notebook flow on exported NIfTI data:
1) Read volumes from:
   - <data_root>/dataset/*.nii.gz
   - <data_root>/mask/*mask.nii
2) Convert matched volumes to per-slice PNG pairs.
3) Split into train/val/test.
4) Train VGG11-encoder U-Net with NLL loss.
5) Save checkpoint + metrics JSON.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import nibabel
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional
    _tqdm = None


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = CTSCAN_ROOT / "data" / "legacy_compatible"
DEFAULT_WORK_DIR = CTSCAN_ROOT / "data" / "legacy_compatible_png"
DEFAULT_OUTPUT_PATH = CTSCAN_ROOT / "model" / "legacy_vgg11_unet.pt"
DEFAULT_METRICS_PATH = CTSCAN_ROOT / "model" / "legacy_vgg11_unet.metrics.json"
DEFAULT_LOG_PATH = CTSCAN_ROOT / "model" / "legacy_vgg11_unet.train.log"


def progress_iter(iterable, total: int | None, desc: str, unit: str = "item"):
    if _tqdm is not None:
        yield from _tqdm(iterable, total=total, desc=desc, unit=unit)
        return
    for item in iterable:
        yield item


@dataclass
class TrainConfig:
    data_root: Path
    work_dir: Path
    output_path: Path
    metrics_path: Path
    log_path: Path
    resume_path: Path | None
    model_version: str
    epochs: int
    batch_size: int
    learning_rate: float
    image_size: int
    seed: int
    num_workers: int
    device: str
    overwrite_workdir: bool
    skip_existing_png: bool
    max_volumes: int


class LegacyLungDataset(Dataset):
    def __init__(self, names: list[str], images_dir: Path, masks_dir: Path, image_size: int):
        self.names = list(names)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[index]
        image = Image.open(self.images_dir / f"{name}.png").convert("P")
        mask = Image.open(self.masks_dir / f"{name}.png")

        image_tensor = torchvision.transforms.functional.to_tensor(image) - 0.5
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()

        if self.image_size > 0:
            target_hw = (self.image_size, self.image_size)
        else:
            target_hw = None
        if target_hw is not None and tuple(image_tensor.shape[-2:]) != target_hw:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=target_hw,
                mode="nearest",
            ).squeeze(0).squeeze(0).long()
        return image_tensor, mask_tensor


class Block(nn.Module):
    def __init__(self, in_channels: int, mid_channel: int, out_channels: int, batch_norm: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, padding=1)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(mid_channel)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x


class LegacyUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool = True, upscale_mode: str = "bilinear"):
        super().__init__()
        self.upscale_mode = upscale_mode
        self.init_conv = nn.Conv2d(in_channels, 3, 1)

        try:
            encoder = torchvision.models.vgg11(pretrained=True).features
        except TypeError:
            weights = torchvision.models.VGG11_Weights.IMAGENET1K_V1
            encoder = torchvision.models.vgg11(weights=weights).features
        self.conv1 = encoder[0]   # 64
        self.conv2 = encoder[3]   # 128
        self.conv3 = encoder[6]   # 256
        self.conv3s = encoder[8]  # 256
        self.conv4 = encoder[11]  # 512
        self.conv4s = encoder[13]  # 512
        self.conv5 = encoder[16]  # 512
        self.conv5s = encoder[18]  # 512

        self.center = Block(512, 512, 256, batch_norm=batch_norm)
        self.dec5 = Block(512 + 256, 512, 256, batch_norm=batch_norm)
        self.dec4 = Block(512 + 256, 512, 128, batch_norm=batch_norm)
        self.dec3 = Block(256 + 128, 256, 64, batch_norm=batch_norm)
        self.dec2 = Block(128 + 64, 128, 32, batch_norm=batch_norm)
        self.dec1 = Block(64 + 32, 64, 32, batch_norm=batch_norm)
        self.out = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)

    def up(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode=self.upscale_mode)

    @staticmethod
    def down(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_conv = F.relu(self.init_conv(x), inplace=True)

        enc1 = F.relu(self.conv1(init_conv), inplace=True)
        enc2 = F.relu(self.conv2(self.down(enc1)), inplace=True)
        enc3 = F.relu(self.conv3(self.down(enc2)), inplace=True)
        enc3 = F.relu(self.conv3s(enc3), inplace=True)
        enc4 = F.relu(self.conv4(self.down(enc3)), inplace=True)
        enc4 = F.relu(self.conv4s(enc4), inplace=True)
        enc5 = F.relu(self.conv5(self.down(enc4)), inplace=True)
        enc5 = F.relu(self.conv5s(enc5), inplace=True)

        center = self.center(self.down(enc5))
        dec5 = self.dec5(torch.cat([self.up(center, enc5.size()[-2:]), enc5], dim=1))
        dec4 = self.dec4(torch.cat([self.up(dec5, enc4.size()[-2:]), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], dim=1))
        return self.out(dec1)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train legacy VGG11 U-Net like the old notebook.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--resume-path", type=Path, default=None)
    parser.add_argument("--model-version", type=str, default="legacy-vgg11-unet-0.1.0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--image-size", type=int, default=0, help="Resize PNG slices to NxN before batching. Default 0 keeps original notebook behavior.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite-workdir", action="store_true")
    parser.add_argument("--skip-existing-png", action="store_true")
    parser.add_argument("--max-volumes", type=int, default=0)
    args = parser.parse_args()

    return TrainConfig(
        data_root=args.data_root.resolve(),
        work_dir=args.work_dir.resolve(),
        output_path=args.output_path.resolve(),
        metrics_path=args.metrics_path.resolve(),
        log_path=args.log_path.resolve(),
        resume_path=args.resume_path.resolve() if args.resume_path is not None else None,
        model_version=str(args.model_version),
        epochs=max(int(args.epochs), 1),
        batch_size=max(int(args.batch_size), 1),
        learning_rate=float(args.learning_rate),
        image_size=max(int(args.image_size), 0),
        seed=int(args.seed),
        num_workers=max(int(args.num_workers), 0),
        device=str(args.device).strip().lower(),
        overwrite_workdir=bool(args.overwrite_workdir),
        skip_existing_png=bool(args.skip_existing_png),
        max_volumes=max(int(args.max_volumes), 0),
    )


def epoch_checkpoint_path(output_path: Path, epoch: int) -> Path:
    if epoch < 1:
        raise ValueError("epoch must be >= 1")
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.epoch{epoch:03d}{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.epoch{epoch:03d}")


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


def volume_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def match_volumes(data_root: Path, max_volumes: int) -> list[tuple[str, Path, Path]]:
    dataset_dir = data_root / "dataset"
    mask_dir = data_root / "mask"
    if not dataset_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Expected dataset and mask dirs under {data_root}")

    matches: list[tuple[str, Path, Path]] = []
    for image_path in sorted(dataset_dir.glob("*.nii.gz")):
        volume_id = volume_id_from_path(image_path)
        mask_path = mask_dir / f"{volume_id}mask.nii"
        if not mask_path.exists():
            continue
        matches.append((volume_id, image_path, mask_path))
        if max_volumes > 0 and len(matches) >= max_volumes:
            break
    return matches


def normalize_slice_to_uint8(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - mn) / (mx - mn)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def convert_volumes_to_png(
    matches: list[tuple[str, Path, Path]],
    images_dir: Path,
    masks_dir: Path,
    skip_existing_png: bool,
) -> list[str]:
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    for volume_id, image_path, mask_path in progress_iter(matches, total=len(matches), desc="Convert NIfTI", unit="vol"):
        image_xyz = np.asanyarray(nibabel.load(str(image_path)).dataobj).astype(np.float32)
        mask_xyz = np.asanyarray(nibabel.load(str(mask_path)).dataobj).astype(np.uint8)
        if image_xyz.shape != mask_xyz.shape or image_xyz.ndim != 3:
            continue

        z_slices = int(image_xyz.shape[2])
        for z in range(z_slices):
            stem = f"{volume_id}_z{z + 1:03d}"
            image_png = images_dir / f"{stem}.png"
            mask_png = masks_dir / f"{stem}.png"
            names.append(stem)

            if skip_existing_png and image_png.exists() and mask_png.exists():
                continue

            image_uint8 = normalize_slice_to_uint8(image_xyz[:, :, z])
            mask_uint8 = np.asarray(mask_xyz[:, :, z], dtype=np.uint8)
            Image.fromarray(image_uint8, mode="L").save(image_png)
            Image.fromarray(mask_uint8, mode="L").save(mask_png)

    # Keep only names with both files present.
    final_names = [name for name in sorted(set(names)) if (images_dir / f"{name}.png").exists() and (masks_dir / f"{name}.png").exists()]
    return final_names


def existing_png_names(images_dir: Path, masks_dir: Path) -> list[str]:
    if not images_dir.exists() or not masks_dir.exists():
        return []
    names = []
    for mask_png in sorted(masks_dir.glob("*.png")):
        stem = mask_png.stem
        if (images_dir / f"{stem}.png").exists():
            names.append(stem)
    return names


def split_names(names: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    if not names:
        return [], [], []
    train_names, test_names = train_test_split(list(names), test_size=0.2, random_state=seed)
    train_names, val_names = train_test_split(list(train_names), test_size=0.1, random_state=seed)
    return list(train_names), list(val_names), list(test_names)


def write_split_json(path: Path, train_names: list[str], val_names: list[str], test_names: list[str]) -> None:
    payload = {
        "train": train_names,
        "val": val_names,
        "test": test_names,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_class_jaccard(y_true: torch.Tensor, y_pred: torch.Tensor, classes: int) -> list[float]:
    eps = 1e-6
    values: list[float] = []
    for class_id in range(classes):
        true_c = (y_true == class_id)
        pred_c = (y_pred == class_id)
        intersection = float((true_c & pred_c).sum().item())
        union = float((true_c | pred_c).sum().item())
        values.append(intersection / (union + eps))
    return values


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    dataset_size: int,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    classes: int,
    nominal_batch_size: int,
    desc: str,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    jaccard_sum = np.zeros(classes, dtype=np.float64)
    count = 0

    iterator = progress_iter(loader, total=len(loader), desc=desc, unit="batch")
    for images, masks in iterator:
        images = images.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.long)

        with torch.set_grad_enabled(training):
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(log_probs, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * nominal_batch_size
        preds = torch.argmax(log_probs, dim=1)
        batch_j = compute_class_jaccard(masks.detach().cpu(), preds.detach().cpu(), classes=classes)
        jaccard_sum += np.asarray(batch_j, dtype=np.float64)
        count += 1

    if dataset_size <= 0 or count <= 0:
        return {"loss": 0.0, "jaccard": [0.0] * classes, "dice": [0.0] * classes}

    mean_j = (jaccard_sum / max(count, 1)).tolist()
    mean_d = [float((2.0 * j) / (1.0 + j + 1e-6)) for j in mean_j]
    return {
        "loss": float(total_loss / dataset_size),
        "jaccard": [float(v) for v in mean_j],
        "dice": [float(v) for v in mean_d],
    }


def train(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    print(f"device={device}")

    if config.overwrite_workdir and config.work_dir.exists():
        shutil.rmtree(config.work_dir)
    config.work_dir.mkdir(parents=True, exist_ok=True)

    images_dir = config.work_dir / "images"
    masks_dir = config.work_dir / "masks"
    split_path = config.work_dir / "splits.json"

    names: list[str]
    matches: list[tuple[str, Path, Path]] = []
    cached_names = existing_png_names(images_dir, masks_dir) if config.skip_existing_png else []
    if cached_names:
        names = cached_names
        print(f"using_existing_png_slices={len(names)}")
    else:
        matches = match_volumes(config.data_root, config.max_volumes)
        if not matches:
            raise RuntimeError(f"No matched image/mask NIfTI pairs found under {config.data_root}")
        print(f"matched_volumes={len(matches)}")

        names = convert_volumes_to_png(
            matches=matches,
            images_dir=images_dir,
            masks_dir=masks_dir,
            skip_existing_png=config.skip_existing_png,
        )
    if not names:
        raise RuntimeError("No PNG slices created.")
    print(f"png_slices={len(names)}")

    train_names, val_names, test_names = split_names(names, seed=42)
    write_split_json(split_path, train_names, val_names, test_names)
    print(f"train={len(train_names)} val={len(val_names)} test={len(test_names)}")

    train_ds = LegacyLungDataset(train_names, images_dir, masks_dir, image_size=config.image_size)
    val_ds = LegacyLungDataset(val_names, images_dir, masks_dir, image_size=config.image_size)
    test_ds = LegacyLungDataset(test_names, images_dir, masks_dir, image_size=config.image_size)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = LegacyUNet(in_channels=1, out_channels=4, batch_norm=True, upscale_mode="bilinear").to(device)
    if config.resume_path is not None:
        state_dict = torch.load(config.resume_path, map_location=torch.device("cpu"))
        if not isinstance(state_dict, dict):
            raise ValueError(f"resume checkpoint must be a raw state_dict: {config.resume_path}")
        model.load_state_dict(state_dict, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    with config.log_path.open("w", encoding="utf-8") as log_handle:
        for epoch in range(1, config.epochs + 1):
            start = time.time()
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                dataset_size=len(train_ds),
                optimizer=optimizer,
                device=device,
                classes=4,
                nominal_batch_size=config.batch_size,
                desc=f"Epoch {epoch}/{config.epochs} train",
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    dataset_size=len(val_ds),
                    optimizer=None,
                    device=device,
                    classes=4,
                    nominal_batch_size=config.batch_size,
                    desc=f"Epoch {epoch}/{config.epochs} val",
                ) if len(val_ds) > 0 else {"loss": 0.0, "jaccard": [0.0] * 4, "dice": [0.0] * 4}

            elapsed = time.time() - start
            row = {
                "epoch": epoch,
                "elapsed_sec": float(elapsed),
                "train_loss": float(train_metrics["loss"]),
                "val_loss": float(val_metrics["loss"]),
                "train_jaccard": train_metrics["jaccard"],
                "val_jaccard": val_metrics["jaccard"],
                "train_dice": train_metrics["dice"],
                "val_dice": val_metrics["dice"],
            }
            history.append(row)
            message = (
                f"Epoch: {epoch}/{config.epochs}, Time spent: {elapsed}, "
                f"Training loss: {train_metrics['loss']}, Validation loss: {val_metrics['loss']}"
            )
            print(message)
            log_handle.write(message + "\n")
            log_handle.flush()

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = float(val_metrics["loss"])
                best_epoch = epoch
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                config.output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, config.output_path)
                print("Model saved")
                log_handle.write("Model saved\n")
                log_handle.flush()

            epoch_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(epoch_state, epoch_checkpoint_path(config.output_path, epoch))

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            dataset_size=len(test_ds),
            optimizer=None,
            device=device,
            classes=4,
            nominal_batch_size=config.batch_size,
            desc="Test",
        ) if len(test_ds) > 0 else {"loss": 0.0, "jaccard": [0.0] * 4, "dice": [0.0] * 4}

    if not config.output_path.exists():
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, config.output_path)

    metrics = {
        "model_version": config.model_version,
        "device": str(device),
        "matched_volumes": len(matches),
        "png_slices": len(names),
        "train_slices": len(train_ds),
        "val_slices": len(val_ds),
        "test_slices": len(test_ds),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "history": history,
        "test": test_metrics,
        "split_path": str(split_path),
        "log_path": str(config.log_path),
        "class_ids": [0, 1, 2, 3],
        "image_size": int(config.image_size),
        "epoch_checkpoint_pattern": str(epoch_checkpoint_path(config.output_path, 1)).replace("epoch001", "epochNNN"),
        "resume_path": str(config.resume_path) if config.resume_path is not None else None,
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
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
