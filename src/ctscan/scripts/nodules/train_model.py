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
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.nodules import create_model, save_model_bundle


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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--model-version", type=str, default="0.4.0")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_training_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found at {path}. Run: `python scripts/nodules/download_data.py`")
    bundle = np.load(path)
    required = {"patches", "nodule_target", "malignancy_target", "series_ids"}
    missing = required - set(bundle.files)
    if missing:
        raise ValueError(f"Training dataset missing arrays: {sorted(missing)}")
    patches = torch.tensor(bundle["patches"].astype(np.float32), dtype=torch.float32)
    nodule_target = torch.tensor(bundle["nodule_target"], dtype=torch.float32).unsqueeze(1)
    malignancy_target = torch.tensor(bundle["malignancy_target"], dtype=torch.float32).unsqueeze(1)
    series_ids = bundle["series_ids"].astype(str)
    return patches, nodule_target, malignancy_target, series_ids


def split_dataset(
    patches: torch.Tensor,
    nodule_target: torch.Tensor,
    malignancy_target: torch.Tensor,
    series_ids: np.ndarray,
    split_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    unique_series = np.unique(series_ids)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(unique_series)
    split = max(1, int(0.8 * len(unique_series)))
    train_series = set(unique_series[:split].tolist())
    train_mask = np.array([series_id in train_series for series_id in series_ids], dtype=bool)
    val_mask = ~train_mask
    if not val_mask.any():
        val_mask = train_mask.copy()
    return (
        patches[train_mask],
        nodule_target[train_mask],
        malignancy_target[train_mask],
        patches[val_mask],
        nodule_target[val_mask],
        malignancy_target[val_mask],
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


class PatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        patches: torch.Tensor,
        nodule_target: torch.Tensor,
        malignancy_target: torch.Tensor,
        patch_mean: float,
        patch_std: float,
        augment: bool,
        seed: int,
    ) -> None:
        self.patches = patches
        self.nodule_target = nodule_target
        self.malignancy_target = malignancy_target
        self.patch_mean = patch_mean
        self.patch_std = max(patch_std, 1e-6)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch = self.patches[index].clone().float()
        if self.augment:
            for axis in (1, 2, 3):
                if float(self.rng.random()) < 0.5:
                    patch = torch.flip(patch, dims=(axis,))
            if float(self.rng.random()) < 0.5:
                noise = torch.randn_like(patch) * 15.0
                patch = patch + noise
            if float(self.rng.random()) < 0.5:
                patch = patch + float(self.rng.normal(0.0, 25.0))
        patch = (patch - self.patch_mean) / self.patch_std
        return patch, self.nodule_target[index], self.malignancy_target[index]


@dataclass
class TrainingResult:
    model: torch.nn.Module
    patch_mean: float
    patch_std: float
    nodule_accuracy: float
    nodule_auc: float
    malignancy_auc: float


def train_model(
    x_train: torch.Tensor,
    y_train_nodule: torch.Tensor,
    y_train_malignancy: torch.Tensor,
    x_val: torch.Tensor,
    y_val_nodule: torch.Tensor,
    y_val_malignancy: torch.Tensor,
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
        patch_mean=patch_mean,
        patch_std=patch_std,
        augment=True,
        seed=seed,
    )
    val_dataset = PatchDataset(
        x_val,
        y_val_nodule,
        y_val_malignancy,
        patch_mean=patch_mean,
        patch_std=patch_std,
        augment=False,
        seed=seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    pos_nodule = max(float(y_train_nodule.sum().item()), 1.0)
    neg_nodule = max(float((1.0 - y_train_nodule).sum().item()), 1.0)
    pos_malignancy = max(float(y_train_malignancy.sum().item()), 1.0)
    neg_malignancy = max(float((1.0 - y_train_malignancy).sum().item()), 1.0)

    nodule_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_nodule / pos_nodule], device=device))
    malignancy_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_malignancy / pos_malignancy], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_state = None
    best_score = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        for patches, nodule_target, malignancy_target in train_loader:
            patches = patches.to(device)
            nodule_target = nodule_target.to(device)
            malignancy_target = malignancy_target.to(device)

            optimizer.zero_grad()
            logits = model(patches)
            nodule_loss = nodule_loss_fn(logits[:, :1], nodule_target)
            malignancy_loss = malignancy_loss_fn(logits[:, 1:], malignancy_target)
            loss = nodule_loss + malignancy_loss
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
            for patches, nodule_target, malignancy_target in val_loader:
                patches = patches.to(device)
                logits = model(patches)
                nodule_probs.append(torch.sigmoid(logits[:, :1]).cpu())
                malignancy_probs.append(torch.sigmoid(logits[:, 1:]).cpu())
                nodule_truth.append(nodule_target.cpu())
                malignancy_truth.append(malignancy_target.cpu())

            nodule_prob = torch.cat(nodule_probs, dim=0)
            malignancy_prob = torch.cat(malignancy_probs, dim=0)
            nodule_true = torch.cat(nodule_truth, dim=0)
            malignancy_true = torch.cat(malignancy_truth, dim=0)
            nodule_pred = (nodule_prob >= 0.5).float()
            nodule_accuracy = float((nodule_pred == nodule_true).float().mean().item())
            nodule_auc = float(auc_from_scores(nodule_true, nodule_prob))
            malignancy_auc = float(auc_from_scores(malignancy_true, malignancy_prob))
            composite_score = malignancy_auc + 0.5 * nodule_auc + 0.25 * nodule_accuracy

        if composite_score > best_score:
            best_score = composite_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = (nodule_accuracy, nodule_auc, malignancy_auc)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

        print(
            f"epoch={epoch + 1} "
            f"nodule_acc={nodule_accuracy:.4f} "
            f"nodule_auc={nodule_auc:.4f} "
            f"mal_auc={malignancy_auc:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    model = model.to("cpu")
    nodule_accuracy, nodule_auc, malignancy_auc = best_metrics
    return TrainingResult(
        model=model,
        patch_mean=patch_mean,
        patch_std=patch_std,
        nodule_accuracy=nodule_accuracy,
        nodule_auc=nodule_auc,
        malignancy_auc=malignancy_auc,
    )


def main() -> None:
    args = parse_args()
    patches, nodule_target, malignancy_target, series_ids = load_training_dataset(args.input_path)
    x_train, y_train_nodule, y_train_malignancy, x_val, y_val_nodule, y_val_malignancy = split_dataset(
        patches,
        nodule_target,
        malignancy_target,
        series_ids=series_ids,
        split_seed=args.split_seed,
    )

    result = train_model(
        x_train=x_train,
        y_train_nodule=y_train_nodule,
        y_train_malignancy=y_train_malignancy,
        x_val=x_val,
        y_val_nodule=y_val_nodule,
        y_val_malignancy=y_val_malignancy,
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
    print(f"Malignancy AUROC: {result.malignancy_auc:.4f}")


if __name__ == "__main__":
    main()
