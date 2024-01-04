"""Train the chest CT nodule classifier on canonical real patches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

CTSCAN_ROOT = Path(__file__).resolve().parents[2]
if str(CTSCAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CTSCAN_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.nodules import create_model, save_model_bundle


HIGH_RISK_THRESHOLD = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train chest CT nodule classifier.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=CTSCAN_ROOT / "data" / "ctscan" / "processed" / "nodules_training.npz",
        help="Path to the canonical training dataset npz.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=CTSCAN_ROOT / "models" / "nodules.pt",
        help="Path to the output state_dict bundle.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--model-version", type=str, default="0.5.0")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_training_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found at {path}. Run: `python scripts/nodules/download_data.py`")
    bundle = np.load(path)
    required = {"patches", "nodule_target", "malignancy_target", "malignancy_mask", "series_ids"}
    missing = required - set(bundle.files)
    if missing:
        raise ValueError(f"Training dataset missing arrays: {sorted(missing)}")
    patches = torch.tensor(bundle["patches"].astype(np.float32), dtype=torch.float32)
    nodule_target = torch.tensor(bundle["nodule_target"], dtype=torch.float32).unsqueeze(1)
    malignancy_target = torch.tensor(bundle["malignancy_target"], dtype=torch.float32).unsqueeze(1)
    malignancy_mask = torch.tensor(bundle["malignancy_mask"], dtype=torch.float32).unsqueeze(1)
    series_ids = bundle["series_ids"].astype(str)
    return patches, nodule_target, malignancy_target, malignancy_mask, series_ids


def split_dataset(
    patches: torch.Tensor,
    nodule_target: torch.Tensor,
    malignancy_target: torch.Tensor,
    malignancy_mask: torch.Tensor,
    series_ids: np.ndarray,
    split_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    unique_series = np.unique(series_ids)
    rng = np.random.default_rng(split_seed)

    malignant_series: list[str] = []
    other_series: list[str] = []
    for series_id in unique_series:
        series_mask = series_ids == series_id
        series_high_risk = bool(((malignancy_target[series_mask] >= HIGH_RISK_THRESHOLD) & (malignancy_mask[series_mask] > 0.5)).any().item())
        if series_high_risk:
            malignant_series.append(series_id)
        else:
            other_series.append(series_id)

    rng.shuffle(malignant_series)
    rng.shuffle(other_series)

    def _take_train(bucket: list[str]) -> set[str]:
        if not bucket:
            return set()
        split = max(1, int(round(0.8 * len(bucket))))
        if split >= len(bucket) and len(bucket) > 1:
            split = len(bucket) - 1
        return set(bucket[:split])

    train_series = _take_train(malignant_series) | _take_train(other_series)
    train_mask = np.array([series_id in train_series for series_id in series_ids], dtype=bool)
    val_mask = ~train_mask
    if not val_mask.any():
        val_mask = train_mask.copy()

    return (
        patches[train_mask],
        nodule_target[train_mask],
        malignancy_target[train_mask],
        malignancy_mask[train_mask],
        patches[val_mask],
        nodule_target[val_mask],
        malignancy_target[val_mask],
        malignancy_mask[val_mask],
    )


def auc_from_scores(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    target = y_true.reshape(-1).detach().cpu()
    score = y_score.reshape(-1).detach().cpu()
    pos_mask = target >= 0.5
    n_pos = int(pos_mask.sum().item())
    n_neg = int((~pos_mask).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    score_series = pd.Series(score.numpy())
    ranks = score_series.rank(method="average")
    sum_ranks_pos = float(ranks[pos_mask.numpy()].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def sigmoid_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = target * prob + (1.0 - target) * (1.0 - prob)
    alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
    loss = alpha_t * ((1.0 - p_t) ** gamma) * bce
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


class PatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        patches: torch.Tensor,
        nodule_target: torch.Tensor,
        malignancy_target: torch.Tensor,
        malignancy_mask: torch.Tensor,
        patch_mean: float,
        patch_std: float,
        augment: bool,
        seed: int,
    ) -> None:
        self.patches = patches
        self.nodule_target = nodule_target
        self.malignancy_target = malignancy_target
        self.malignancy_mask = malignancy_mask
        self.patch_mean = patch_mean
        self.patch_std = max(patch_std, 1e-6)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patch = self.patches[index].clone().float()
        if self.augment:
            for axis in (1, 2, 3):
                if float(self.rng.random()) < 0.5:
                    patch = torch.flip(patch, dims=(axis,))
            if float(self.rng.random()) < 0.35:
                patch = patch + torch.randn_like(patch) * 12.0
            if float(self.rng.random()) < 0.35:
                patch = patch * float(self.rng.uniform(0.92, 1.08))
            if float(self.rng.random()) < 0.35:
                patch = patch + float(self.rng.normal(0.0, 18.0))
        patch = (patch - self.patch_mean) / self.patch_std
        return patch, self.nodule_target[index], self.malignancy_target[index], self.malignancy_mask[index]


@dataclass
class TrainingResult:
    model: torch.nn.Module
    patch_mean: float
    patch_std: float
    nodule_accuracy: float
    nodule_auc: float
    malignancy_auc: float
    nodule_sensitivity: float
    nodule_specificity: float


def train_model(
    x_train: torch.Tensor,
    y_train_nodule: torch.Tensor,
    y_train_malignancy: torch.Tensor,
    y_train_malignancy_mask: torch.Tensor,
    x_val: torch.Tensor,
    y_val_nodule: torch.Tensor,
    y_val_malignancy: torch.Tensor,
    y_val_malignancy_mask: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    patience: int,
) -> TrainingResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    patch_mean = float(x_train.mean().item())
    patch_std = float(x_train.std().item())
    model = create_model().to(device)

    train_dataset = PatchDataset(
        x_train,
        y_train_nodule,
        y_train_malignancy,
        y_train_malignancy_mask,
        patch_mean=patch_mean,
        patch_std=patch_std,
        augment=True,
        seed=seed,
    )
    val_dataset = PatchDataset(
        x_val,
        y_val_nodule,
        y_val_malignancy,
        y_val_malignancy_mask,
        patch_mean=patch_mean,
        patch_std=patch_std,
        augment=False,
        seed=seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    pos_nodule = max(float(y_train_nodule.sum().item()), 1.0)
    neg_nodule = max(float((1.0 - y_train_nodule).sum().item()), 1.0)
    focal_alpha = min(0.85, max(0.2, neg_nodule / (neg_nodule + pos_nodule)))

    malignant_train = y_train_malignancy[y_train_malignancy_mask > 0.5]
    if malignant_train.numel():
        malignant_positive = max(float((malignant_train >= HIGH_RISK_THRESHOLD).sum().item()), 1.0)
        malignant_negative = max(float((malignant_train < HIGH_RISK_THRESHOLD).sum().item()), 1.0)
        malignancy_pos_weight = malignant_negative / malignant_positive
    else:
        malignancy_pos_weight = 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_state = None
    best_score = float("-inf")
    best_metrics = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        for patches, nodule_target, malignancy_target, malignancy_mask in train_loader:
            patches = patches.to(device)
            nodule_target = nodule_target.to(device)
            malignancy_target = malignancy_target.to(device)
            malignancy_mask = malignancy_mask.to(device)

            optimizer.zero_grad()
            logits = model(patches)
            nodule_logits = logits[:, :1]
            malignancy_logits = logits[:, 1:]

            nodule_loss = sigmoid_focal_loss(
                nodule_logits,
                nodule_target,
                alpha=focal_alpha,
                gamma=2.0,
            )

            labeled_mask = malignancy_mask > 0.5
            if labeled_mask.any():
                malignancy_loss = F.binary_cross_entropy_with_logits(
                    malignancy_logits[labeled_mask],
                    malignancy_target[labeled_mask],
                    pos_weight=torch.tensor([malignancy_pos_weight], device=device),
                )
            else:
                malignancy_loss = torch.zeros((), device=device)

            loss = nodule_loss + 0.8 * malignancy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            nodule_probs: list[torch.Tensor] = []
            malignancy_probs: list[torch.Tensor] = []
            nodule_truth: list[torch.Tensor] = []
            malignancy_truth: list[torch.Tensor] = []
            malignancy_masks: list[torch.Tensor] = []
            for patches, nodule_target, malignancy_target, malignancy_mask in val_loader:
                patches = patches.to(device)
                logits = model(patches)
                nodule_probs.append(torch.sigmoid(logits[:, :1]).cpu())
                malignancy_probs.append(torch.sigmoid(logits[:, 1:]).cpu())
                nodule_truth.append(nodule_target.cpu())
                malignancy_truth.append(malignancy_target.cpu())
                malignancy_masks.append(malignancy_mask.cpu())

            nodule_prob = torch.cat(nodule_probs, dim=0)
            malignancy_prob = torch.cat(malignancy_probs, dim=0)
            nodule_true = torch.cat(nodule_truth, dim=0)
            malignancy_true = torch.cat(malignancy_truth, dim=0)
            malignancy_mask = torch.cat(malignancy_masks, dim=0)

            nodule_pred = (nodule_prob >= 0.5).float()
            nodule_accuracy = float((nodule_pred == nodule_true).float().mean().item())
            true_positive = float(((nodule_pred == 1.0) & (nodule_true == 1.0)).sum().item())
            false_negative = float(((nodule_pred == 0.0) & (nodule_true == 1.0)).sum().item())
            true_negative = float(((nodule_pred == 0.0) & (nodule_true == 0.0)).sum().item())
            false_positive = float(((nodule_pred == 1.0) & (nodule_true == 0.0)).sum().item())
            nodule_sensitivity = true_positive / max(true_positive + false_negative, 1.0)
            nodule_specificity = true_negative / max(true_negative + false_positive, 1.0)
            nodule_auc = float(auc_from_scores(nodule_true, nodule_prob))

            malignant_only = malignancy_mask > 0.5
            malignancy_auc = float("nan")
            if malignant_only.any():
                malignancy_binary = (malignancy_true[malignant_only] >= HIGH_RISK_THRESHOLD).float()
                malignancy_auc = float(auc_from_scores(malignancy_binary, malignancy_prob[malignant_only]))

            composite_score = (
                (0.0 if np.isnan(malignancy_auc) else 0.45 * malignancy_auc)
                + (0.40 * (0.0 if np.isnan(nodule_auc) else nodule_auc))
                + (0.10 * nodule_sensitivity)
                + (0.05 * nodule_specificity)
            )

        if composite_score > best_score:
            best_score = composite_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = (nodule_accuracy, nodule_auc, malignancy_auc, nodule_sensitivity, nodule_specificity)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

        print(
            f"epoch={epoch + 1} "
            f"nodule_acc={nodule_accuracy:.4f} "
            f"nodule_auc={nodule_auc:.4f} "
            f"nodule_sens={nodule_sensitivity:.4f} "
            f"nodule_spec={nodule_specificity:.4f} "
            f"mal_auc={malignancy_auc:.4f}"
        )

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    model = model.to("cpu")
    nodule_accuracy, nodule_auc, malignancy_auc, nodule_sensitivity, nodule_specificity = best_metrics
    return TrainingResult(
        model=model,
        patch_mean=patch_mean,
        patch_std=patch_std,
        nodule_accuracy=nodule_accuracy,
        nodule_auc=nodule_auc,
        malignancy_auc=malignancy_auc,
        nodule_sensitivity=nodule_sensitivity,
        nodule_specificity=nodule_specificity,
    )


def main() -> None:
    args = parse_args()
    patches, nodule_target, malignancy_target, malignancy_mask, series_ids = load_training_dataset(args.input_path)
    split = split_dataset(
        patches,
        nodule_target,
        malignancy_target,
        malignancy_mask,
        series_ids=series_ids,
        split_seed=args.split_seed,
    )

    result = train_model(
        *split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        patience=args.patience,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model_bundle(
        path=args.output_path,
        model=result.model,
        patch_mean=result.patch_mean,
        patch_std=result.patch_std,
        model_version=args.model_version,
        nodule_accuracy=result.nodule_accuracy,
        nodule_auc=result.nodule_auc,
        malignancy_auc=result.malignancy_auc,
        dataset_rows=int(patches.shape[0]),
    )

    print(f"Saved model to: {args.output_path}")
    print(f"Nodule accuracy: {result.nodule_accuracy:.4f}")
    print(f"Nodule AUROC: {result.nodule_auc:.4f}")
    print(f"Nodule sensitivity: {result.nodule_sensitivity:.4f}")
    print(f"Nodule specificity: {result.nodule_specificity:.4f}")
    print(f"Malignancy AUROC: {result.malignancy_auc:.4f}")


if __name__ == "__main__":
    main()
