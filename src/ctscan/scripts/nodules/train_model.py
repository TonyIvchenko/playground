"""Train the chest CT nodule patch classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

CTSCAN_ROOT = Path(__file__).resolve().parents[2]
if str(CTSCAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CTSCAN_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--model-version", type=str, default="0.1.0")
    return parser.parse_args()


def load_training_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found at {path}. Run: `python scripts/nodules/download_data.py`")
    bundle = np.load(path)
    required = {"patches", "nodule_target", "malignancy_target"}
    missing = required - set(bundle.files)
    if missing:
        raise ValueError(f"Training dataset missing arrays: {sorted(missing)}")
    patches = torch.tensor(bundle["patches"], dtype=torch.float32)
    nodule_target = torch.tensor(bundle["nodule_target"], dtype=torch.float32).unsqueeze(1)
    malignancy_target = torch.tensor(bundle["malignancy_target"], dtype=torch.float32).unsqueeze(1)
    return patches, nodule_target, malignancy_target


def split_dataset(
    patches: torch.Tensor,
    nodule_target: torch.Tensor,
    malignancy_target: torch.Tensor,
    split_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(patches.shape[0], generator=generator)
    split = max(1, int(0.8 * len(indices)))
    train_idx = indices[:split]
    val_idx = indices[split:]
    if len(val_idx) == 0:
        val_idx = train_idx.clone()
    return (
        patches[train_idx],
        nodule_target[train_idx],
        malignancy_target[train_idx],
        patches[val_idx],
        nodule_target[val_idx],
        malignancy_target[val_idx],
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
    seed: int,
) -> tuple[torch.nn.Module, float, float, float, float]:
    torch.manual_seed(seed)
    patch_mean = float(x_train.mean().item())
    patch_std = float(x_train.std().item())
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(x_train, y_train_nodule, y_train_malignancy), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for patches, nodule_target, malignancy_target in loader:
            optimizer.zero_grad()
            logits = model((patches - patch_mean) / max(patch_std, 1e-6))
            nodule_loss = loss_fn(logits[:, :1], nodule_target)
            malignancy_loss = loss_fn(logits[:, 1:], malignancy_target)
            loss = nodule_loss + malignancy_loss
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model((x_val - patch_mean) / max(patch_std, 1e-6))
        nodule_prob = torch.sigmoid(logits[:, :1])
        malignancy_prob = torch.sigmoid(logits[:, 1:])
        nodule_pred = (nodule_prob >= 0.5).float()
        nodule_accuracy = float((nodule_pred == y_val_nodule).float().mean().item())
        malignancy_auc = float(auc_from_scores(y_val_malignancy, malignancy_prob))

    return model, patch_mean, patch_std, nodule_accuracy, malignancy_auc


def main() -> None:
    args = parse_args()
    patches, nodule_target, malignancy_target = load_training_dataset(args.input_path)
    x_train, y_train_nodule, y_train_malignancy, x_val, y_val_nodule, y_val_malignancy = split_dataset(
        patches,
        nodule_target,
        malignancy_target,
        split_seed=args.split_seed,
    )

    model, patch_mean, patch_std, nodule_accuracy, malignancy_auc = train_model(
        x_train=x_train,
        y_train_nodule=y_train_nodule,
        y_train_malignancy=y_train_malignancy,
        x_val=x_val,
        y_val_nodule=y_val_nodule,
        y_val_malignancy=y_val_malignancy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model_bundle(
        path=args.output_path,
        model=model,
        patch_mean=patch_mean,
        patch_std=patch_std,
        model_version=args.model_version,
        nodule_accuracy=nodule_accuracy,
        malignancy_auc=malignancy_auc,
        dataset_rows=int(patches.shape[0]),
    )

    print(f"Saved model to: {args.output_path}")
    print(f"Nodule accuracy: {nodule_accuracy:.4f}")
    print(f"Malignancy AUROC: {malignancy_auc:.4f}")


if __name__ == "__main__":
    main()
