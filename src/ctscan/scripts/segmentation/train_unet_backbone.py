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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional
    _tqdm = None

import segmentation_models_pytorch as smp


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SLICE_DIR = CTSCAN_ROOT / "data" / "ctscan" / "processed" / "slice_dataset"
DEFAULT_OUTPUT_PATH = CTSCAN_ROOT / "model" / "unet_backbone.pt"
DEFAULT_METRICS_PATH = CTSCAN_ROOT / "model" / "unet_backbone.metrics.json"
PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    "default": {},
    "legacy_png_best": {
        "architecture": "fpn",
        "encoder_name": "efficientnet-b0",
        "classes": 4,
        "image_size": 320,
        "batch_size": 6,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "loss": "lovasz_ce",
        "class_weight_mode": "none",
        "scheduler": "none",
        "sampler": "rare_fg",
        "augmentation": "none",
        "gradient_accumulation_steps": 1,
        "tversky_alpha": 0.3,
        "tversky_beta": 0.7,
        "ce_label_smoothing": 0.0,
        "selection_metric": "val_mean_dice_fg",
    },
}


@dataclass
class TrainConfig:
    slice_dir: Path
    output_path: Path
    metrics_path: Path
    model_version: str
    architecture: str
    encoder_name: str
    encoder_weights: str | None
    classes: int
    in_channels: int
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer_name: str
    loss_name: str
    class_weight_mode: str
    scheduler_name: str
    sampler_name: str
    augmentation_name: str
    gradient_accumulation_steps: int
    tversky_alpha: float
    tversky_beta: float
    ce_label_smoothing: float
    selection_metric: str
    num_workers: int
    seed: int
    device: str
    max_train_batches: int
    max_val_batches: int
    max_test_batches: int


class SlicePairDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]], image_size: int, augmentation_name: str = "none"):
        self.root = root
        self.image_size = int(image_size)
        self.rows = list(rows)
        self.augmentation_name = str(augmentation_name).strip().lower()

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

        if self.augmentation_name == "light":
            if torch.rand(1).item() < 0.5:
                image_t = torch.flip(image_t, dims=(-1,))
                mask_t = torch.flip(mask_t, dims=(-1,))
            scale = 1.0 + ((torch.rand(1).item() * 0.2) - 0.1)
            bias = (torch.rand(1).item() * 0.1) - 0.05
            image_t = torch.clamp((image_t * scale) + bias, 0.0, 1.0)
        elif self.augmentation_name not in {"", "none"}:
            raise ValueError(f"unsupported augmentation: {self.augmentation_name}")

        return image_t, mask_t


def parse_args() -> TrainConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--preset", type=str, default="default", choices=sorted(PRESET_DEFAULTS))
    preset_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Train pretrained-backbone U-Net on slice PNG pairs.")
    parser.add_argument("--preset", type=str, default="default", choices=sorted(PRESET_DEFAULTS))
    parser.add_argument("--slice-dir", type=Path, default=DEFAULT_SLICE_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--model-version", type=str, default="0.1.0-backbone")
    parser.add_argument("--architecture", type=str, default="unet")
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--loss", type=str, default="dice_ce")
    parser.add_argument("--class-weight-mode", type=str, default="none")
    parser.add_argument("--scheduler", type=str, default="none")
    parser.add_argument("--sampler", type=str, default="none")
    parser.add_argument("--augmentation", type=str, default="none")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--ce-label-smoothing", type=float, default=0.0)
    parser.add_argument("--selection-metric", type=str, default="val_mean_dice_fg")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.set_defaults(**PRESET_DEFAULTS[preset_args.preset])
    args = parser.parse_args()

    encoder_weights = None if str(args.encoder_weights).strip().lower() in {"", "none"} else str(args.encoder_weights)
    return TrainConfig(
        slice_dir=args.slice_dir.resolve(),
        output_path=args.output_path.resolve(),
        metrics_path=args.metrics_path.resolve(),
        model_version=str(args.model_version),
        architecture=str(args.architecture).strip().lower(),
        encoder_name=str(args.encoder_name),
        encoder_weights=encoder_weights,
        classes=max(int(args.classes), 2),
        in_channels=max(int(args.in_channels), 1),
        image_size=max(int(args.image_size), 64),
        batch_size=max(int(args.batch_size), 1),
        epochs=max(int(args.epochs), 1),
        learning_rate=float(args.learning_rate),
        weight_decay=max(float(args.weight_decay), 0.0),
        optimizer_name=str(args.optimizer).strip().lower(),
        loss_name=str(args.loss).strip().lower(),
        class_weight_mode=str(args.class_weight_mode).strip().lower(),
        scheduler_name=str(args.scheduler).strip().lower(),
        sampler_name=str(args.sampler).strip().lower(),
        augmentation_name=str(args.augmentation).strip().lower(),
        gradient_accumulation_steps=max(int(args.gradient_accumulation_steps), 1),
        tversky_alpha=float(args.tversky_alpha),
        tversky_beta=float(args.tversky_beta),
        ce_label_smoothing=max(float(args.ce_label_smoothing), 0.0),
        selection_metric=str(args.selection_metric).strip().lower(),
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


def _rows_from_csv(split_csv: Path) -> list[dict[str, str]]:
    if not split_csv.exists():
        return []
    with split_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rows_from_legacy_split_json(root: Path, split_name: str) -> list[dict[str, str]]:
    split_json = root / "splits.json"
    if not split_json.exists():
        return []
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    stems = payload.get(split_name, [])
    if not isinstance(stems, list):
        return []
    rows: list[dict[str, str]] = []
    for stem in stems:
        stem_value = str(stem).strip()
        if not stem_value:
            continue
        rows.append(
            {
                "image": f"images/{stem_value}.png",
                "mask": f"masks/{stem_value}.png",
            }
        )
    return rows


def load_split_rows(root: Path, split_name: str) -> list[dict[str, str]]:
    split_csv = root / "splits" / f"{split_name}.csv"
    rows = _rows_from_csv(split_csv)
    if rows:
        return rows
    return _rows_from_legacy_split_json(root, split_name)


def build_model(config: TrainConfig) -> nn.Module:
    builders = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "fpn": smp.FPN,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "manet": smp.MAnet,
    }
    if config.architecture not in builders:
        raise ValueError(f"unsupported architecture: {config.architecture}")
    return builders[config.architecture](
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        classes=config.classes,
    )


class DiceCELoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.label_smoothing = float(label_smoothing)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        ) + self.dice(logits, target)


class TverskyCELoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        alpha: float = 0.3,
        beta: float = 0.7,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.tversky = smp.losses.TverskyLoss(
            mode="multiclass",
            from_logits=True,
            alpha=alpha,
            beta=beta,
        )
        self.label_smoothing = float(label_smoothing)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        ) + self.tversky(logits, target)


class LovaszCELoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.lovasz = smp.losses.LovaszLoss(mode="multiclass", per_image=False)
        self.label_smoothing = float(label_smoothing)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        ) + self.lovasz(logits, target)


class MulticlassFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = torch.pow(1.0 - pt, self.gamma) * ce
        return loss.mean()


def build_loss(
    name: str,
    class_weights: torch.Tensor | None = None,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    ce_label_smoothing: float = 0.0,
) -> nn.Module:
    loss_name = str(name).strip().lower()
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ce_label_smoothing)
    if loss_name == "dice_ce":
        return DiceCELoss(class_weights=class_weights, label_smoothing=ce_label_smoothing)
    if loss_name == "tversky_ce":
        return TverskyCELoss(
            class_weights=class_weights,
            alpha=tversky_alpha,
            beta=tversky_beta,
            label_smoothing=ce_label_smoothing,
        )
    if loss_name == "lovasz_ce":
        return LovaszCELoss(class_weights=class_weights, label_smoothing=ce_label_smoothing)
    if loss_name == "focal":
        return MulticlassFocalLoss()
    raise ValueError(f"unsupported loss: {name}")


def build_optimizer(name: str, params, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    optimizer_name = str(name).strip().lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer: {name}")


def build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    steps_per_epoch: int,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_name = str(name).strip().lower()
    if scheduler_name in {"", "none"}:
        return None
    if scheduler_name == "onecycle":
        total_steps = max(int(steps_per_epoch), 1) * max(int(epochs), 1)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )
    raise ValueError(f"unsupported scheduler: {name}")


def metric_direction(name: str) -> int:
    metric_name = str(name).strip().lower()
    if metric_name.endswith("loss"):
        return -1
    return 1


def metric_value(row: dict[str, float], name: str) -> float:
    return float(row.get(name, 0.0))


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


def compute_class_weights(root: Path, rows: list[dict[str, str]], classes: int, mode: str) -> torch.Tensor | None:
    weight_mode = str(mode).strip().lower()
    if weight_mode in {"", "none"}:
        return None
    if weight_mode not in {"inverse", "inverse_sqrt"}:
        raise ValueError(f"unsupported class weight mode: {mode}")

    counts = np.zeros(classes, dtype=np.float64)
    for row in rows:
        mask_arr = np.asarray(Image.open(root / row["mask"]).convert("L"), dtype=np.int64)
        bincount = np.bincount(mask_arr.reshape(-1), minlength=classes)
        counts[:classes] += bincount[:classes]

    counts = np.maximum(counts, 1.0)
    freqs = counts / counts.sum()
    if weight_mode == "inverse":
        weights = 1.0 / freqs
    else:
        weights = 1.0 / np.sqrt(freqs)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(root: Path, rows: list[dict[str, str]], classes: int, mode: str) -> list[float] | None:
    sampler_mode = str(mode).strip().lower()
    if sampler_mode in {"", "none"}:
        return None
    if sampler_mode != "rare_fg":
        raise ValueError(f"unsupported sampler: {mode}")

    row_labels: list[set[int]] = []
    class_counts = np.zeros(classes, dtype=np.int64)
    for row in rows:
        mask_arr = np.asarray(Image.open(root / row["mask"]).convert("L"), dtype=np.int64)
        labels = {int(value) for value in np.unique(mask_arr).tolist() if int(value) > 0}
        row_labels.append(labels)
        for label in labels:
            if label < classes:
                class_counts[label] += 1

    fg_counts = class_counts[1:]
    valid_counts = fg_counts[fg_counts > 0]
    if valid_counts.size == 0:
        return [1.0] * len(rows)
    reference = float(valid_counts.max())

    weights: list[float] = []
    for labels in row_labels:
        weight = 1.0
        for label in labels:
            if 0 < label < classes and class_counts[label] > 0:
                weight += float(np.sqrt(reference / float(class_counts[label])))
        weights.append(weight)
    return weights


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    gradient_accumulation_steps: int,
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
    grad_steps = max(int(gradient_accumulation_steps), 1)
    if training:
        optimizer.zero_grad(set_to_none=True)

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
                (loss / grad_steps).backward()
                reached_limit = max_batches > 0 and batch_idx >= max_batches
                should_step = (batch_idx % grad_steps == 0) or reached_limit or (batch_idx == total_batches)
                if should_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

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

    train_rows = load_split_rows(config.slice_dir, "train")
    val_rows = load_split_rows(config.slice_dir, "val")
    test_rows = load_split_rows(config.slice_dir, "test")

    train_ds = SlicePairDataset(config.slice_dir, train_rows, config.image_size, augmentation_name=config.augmentation_name)
    val_ds = SlicePairDataset(config.slice_dir, val_rows, config.image_size)
    test_ds = SlicePairDataset(config.slice_dir, test_rows, config.image_size)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training rows found under {config.slice_dir}")

    sample_weights = compute_sample_weights(config.slice_dir, train_rows, config.classes, config.sampler_name)
    train_sampler = None
    train_shuffle = True
    if sample_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_model(config).to(device)
    optimizer = build_optimizer(config.optimizer_name, model.parameters(), config.learning_rate, config.weight_decay)
    scheduler = build_scheduler(
        config.scheduler_name,
        optimizer,
        config.learning_rate,
        steps_per_epoch=max(int(np.ceil(len(train_loader) / config.gradient_accumulation_steps)), 1),
        epochs=config.epochs,
    )
    class_weights = compute_class_weights(config.slice_dir, train_rows, config.classes, config.class_weight_mode)
    criterion = build_loss(
        config.loss_name,
        class_weights=class_weights.to(device) if class_weights is not None else None,
        tversky_alpha=config.tversky_alpha,
        tversky_beta=config.tversky_beta,
        ce_label_smoothing=config.ce_label_smoothing,
    )

    history: list[dict[str, float]] = []
    best_score: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    selection_direction = metric_direction(config.selection_metric)

    for epoch in range(1, config.epochs + 1):
        train_m = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
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
                scheduler=None,
                gradient_accumulation_steps=1,
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

        score = metric_value(row, config.selection_metric)
        should_update = best_score is None
        if best_score is not None:
            if selection_direction > 0:
                should_update = score >= best_score
            else:
                should_update = score <= best_score
        if should_update:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    checkpoint = {
        "model_version": config.model_version,
        "model_type": "unet_pretrained_backbone",
        "architecture": config.architecture,
        "encoder_name": config.encoder_name,
        "encoder_weights": config.encoder_weights,
        "optimizer_name": config.optimizer_name,
        "loss_name": config.loss_name,
        "class_weight_mode": config.class_weight_mode,
        "scheduler_name": config.scheduler_name,
        "sampler_name": config.sampler_name,
        "augmentation_name": config.augmentation_name,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "tversky_alpha": config.tversky_alpha,
        "tversky_beta": config.tversky_beta,
        "ce_label_smoothing": config.ce_label_smoothing,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "sample_weights_summary": {
            "min": float(min(sample_weights)) if sample_weights else None,
            "max": float(max(sample_weights)) if sample_weights else None,
            "mean": float(np.mean(sample_weights)) if sample_weights else None,
        },
        "selection_metric": config.selection_metric,
        "in_channels": config.in_channels,
        "classes": config.classes,
        "image_size": config.image_size,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
        "state_dict": best_state,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, config.output_path)

    model.load_state_dict(best_state)
    with torch.no_grad():
        test_m = run_epoch(
            model=model,
            loader=test_loader,
            optimizer=None,
            scheduler=None,
            gradient_accumulation_steps=1,
            criterion=criterion,
            device=device,
            classes=config.classes,
            max_batches=config.max_test_batches,
            progress_desc="Test",
        ) if len(test_ds) > 0 else {"loss": 0.0, "mean_iou": 0.0, "mean_dice": 0.0, "mean_iou_fg": 0.0, "mean_dice_fg": 0.0}

    metrics = {
        "model_version": config.model_version,
        "architecture": config.architecture,
        "encoder_name": config.encoder_name,
        "encoder_weights": config.encoder_weights,
        "optimizer_name": config.optimizer_name,
        "loss_name": config.loss_name,
        "class_weight_mode": config.class_weight_mode,
        "scheduler_name": config.scheduler_name,
        "sampler_name": config.sampler_name,
        "augmentation_name": config.augmentation_name,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "tversky_alpha": config.tversky_alpha,
        "tversky_beta": config.tversky_beta,
        "ce_label_smoothing": config.ce_label_smoothing,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "sample_weights_summary": {
            "min": float(min(sample_weights)) if sample_weights else None,
            "max": float(max(sample_weights)) if sample_weights else None,
            "mean": float(np.mean(sample_weights)) if sample_weights else None,
        },
        "selection_metric": config.selection_metric,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "device": str(device),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "test_rows": len(test_ds),
        "best_epoch": best_epoch,
        "best_score": best_score,
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
