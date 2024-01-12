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
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--model-version", type=str, default="0.5.0")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--train-negative-ratio", type=float, default=1.5)
    return parser.parse_args()


def get_device(device_name: str = "auto") -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_training_dataset(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found at {path}. Run: `python scripts/nodules/download_data.py`")
    bundle = np.load(path)
    required = {
        "patches",
        "nodule_target",
        "malignancy_target",
        "malignancy_mask",
        "nodule_weight",
        "malignancy_weight",
        "series_ids",
    }
    missing = required - set(bundle.files)
    if missing:
        raise ValueError(f"Training dataset missing arrays: {sorted(missing)}")
    patches = torch.tensor(bundle["patches"].astype(np.float32), dtype=torch.float32)
    nodule_target = torch.tensor(bundle["nodule_target"], dtype=torch.float32).unsqueeze(1)
    malignancy_target = torch.tensor(bundle["malignancy_target"], dtype=torch.float32).unsqueeze(1)
    malignancy_mask = torch.tensor(bundle["malignancy_mask"], dtype=torch.float32).unsqueeze(1)
    nodule_weight = torch.tensor(bundle["nodule_weight"], dtype=torch.float32).unsqueeze(1)
    malignancy_weight = torch.tensor(bundle["malignancy_weight"], dtype=torch.float32).unsqueeze(1)
    series_ids = bundle["series_ids"].astype(str)
    return patches, nodule_target, malignancy_target, malignancy_mask, nodule_weight, malignancy_weight, series_ids


def split_dataset(
    patches: torch.Tensor,
    nodule_target: torch.Tensor,
    malignancy_target: torch.Tensor,
    malignancy_mask: torch.Tensor,
    nodule_weight: torch.Tensor,
    malignancy_weight: torch.Tensor,
    series_ids: np.ndarray,
    split_seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
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
        nodule_weight[train_mask],
        malignancy_weight[train_mask],
        patches[val_mask],
        nodule_target[val_mask],
        malignancy_target[val_mask],
        malignancy_mask[val_mask],
        nodule_weight[val_mask],
        malignancy_weight[val_mask],
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


def augment_batch(batch: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    if float(rng.random()) < 0.5:
        batch = torch.flip(batch, dims=(2,))
    if float(rng.random()) < 0.5:
        batch = torch.flip(batch, dims=(3,))
    if float(rng.random()) < 0.5:
        batch = torch.flip(batch, dims=(4,))
    if float(rng.random()) < 0.35:
        batch = batch + torch.randn_like(batch) * 0.05
    if float(rng.random()) < 0.35:
        batch = batch * float(rng.uniform(0.92, 1.08))
    if float(rng.random()) < 0.35:
        batch = batch + float(rng.normal(0.0, 0.08))
    return batch


def sample_epoch_indices(
    nodule_target: torch.Tensor,
    nodule_weight: torch.Tensor,
    negative_ratio: float,
    seed: int,
    epoch: int,
) -> torch.Tensor:
    target_cpu = nodule_target.reshape(-1).detach().cpu()
    weight_cpu = nodule_weight.reshape(-1).detach().cpu()
    positive_idx = torch.nonzero(target_cpu >= 0.5, as_tuple=False).reshape(-1)
    negative_idx = torch.nonzero(target_cpu < 0.5, as_tuple=False).reshape(-1)
    if negative_ratio <= 0.0 or negative_idx.numel() == 0:
        epoch_idx = positive_idx.clone()
    else:
        negative_count = min(int(max(1, round(float(positive_idx.numel()) * negative_ratio))), int(negative_idx.numel()))
        negative_probs = weight_cpu[negative_idx]
        if float(negative_probs.sum().item()) <= 0.0:
            negative_probs = torch.ones_like(negative_probs)
        generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        sampled_negative_positions = torch.multinomial(negative_probs, num_samples=negative_count, replacement=False, generator=generator)
        epoch_idx = torch.cat([positive_idx, negative_idx[sampled_negative_positions]], dim=0)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000 + epoch)
    perm = torch.randperm(epoch_idx.numel(), generator=generator)
    return epoch_idx[perm]


def iterate_minibatches(indices: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    return [indices[start : start + batch_size] for start in range(0, int(indices.numel()), batch_size)]


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
    y_train_nodule_weight: torch.Tensor,
    y_train_malignancy_weight: torch.Tensor,
    x_val: torch.Tensor,
    y_val_nodule: torch.Tensor,
    y_val_malignancy: torch.Tensor,
    y_val_malignancy_mask: torch.Tensor,
    y_val_nodule_weight: torch.Tensor,
    y_val_malignancy_weight: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    patience: int,
    device_name: str = "auto",
    train_negative_ratio: float = 1.5,
) -> TrainingResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device(device_name)
    patch_mean = float(x_train.mean().item())
    patch_std = float(x_train.std().item())
    model = create_model().to(device)
    train_patches = ((x_train - patch_mean) / max(patch_std, 1e-6)).to(device)
    train_nodule_target = y_train_nodule.to(device)
    train_malignancy_target = y_train_malignancy.to(device)
    train_malignancy_mask = y_train_malignancy_mask.to(device)
    train_nodule_weight = y_train_nodule_weight.to(device)
    train_malignancy_weight = y_train_malignancy_weight.to(device)

    val_patches = ((x_val - patch_mean) / max(patch_std, 1e-6)).to(device)
    val_nodule_target = y_val_nodule.to(device)
    val_malignancy_target = y_val_malignancy.to(device)
    val_malignancy_mask = y_val_malignancy_mask.to(device)

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
        epoch_rng = np.random.default_rng(seed + epoch)
        epoch_indices = sample_epoch_indices(
            nodule_target=y_train_nodule,
            nodule_weight=y_train_nodule_weight,
            negative_ratio=train_negative_ratio,
            seed=seed,
            epoch=epoch,
        ).to(device)
        for batch_indices in iterate_minibatches(epoch_indices, batch_size=batch_size):
            patches = augment_batch(train_patches.index_select(0, batch_indices), rng=epoch_rng)
            nodule_target = train_nodule_target.index_select(0, batch_indices)
            malignancy_target = train_malignancy_target.index_select(0, batch_indices)
            malignancy_mask = train_malignancy_mask.index_select(0, batch_indices)
            nodule_weight = train_nodule_weight.index_select(0, batch_indices)
            malignancy_weight = train_malignancy_weight.index_select(0, batch_indices)
            optimizer.zero_grad()
            logits = model(patches)
            nodule_logits = logits[:, :1]
            malignancy_logits = logits[:, 1:]

            nodule_loss = sigmoid_focal_loss(
                nodule_logits,
                nodule_target,
                alpha=focal_alpha,
                gamma=2.0,
                sample_weight=nodule_weight,
            )

            labeled_mask = malignancy_mask > 0.5
            if labeled_mask.any():
                malignancy_loss = F.binary_cross_entropy_with_logits(
                    malignancy_logits[labeled_mask],
                    malignancy_target[labeled_mask],
                    pos_weight=torch.tensor([malignancy_pos_weight], device=device),
                    reduction="none",
                )
                malignancy_loss = (malignancy_loss * malignancy_weight[labeled_mask]).mean()
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
            val_indices = torch.arange(int(val_patches.shape[0]), device=device)
            for batch_indices in iterate_minibatches(val_indices, batch_size=max(batch_size, 32)):
                patches = val_patches.index_select(0, batch_indices)
                nodule_target = val_nodule_target.index_select(0, batch_indices)
                malignancy_target = val_malignancy_target.index_select(0, batch_indices)
                malignancy_mask = val_malignancy_mask.index_select(0, batch_indices)
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
            f"mal_auc={malignancy_auc:.4f}",
            flush=True,
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
    (
        patches,
        nodule_target,
        malignancy_target,
        malignancy_mask,
        nodule_weight,
        malignancy_weight,
        series_ids,
    ) = load_training_dataset(args.input_path)
    split = split_dataset(
        patches,
        nodule_target,
        malignancy_target,
        malignancy_mask,
        nodule_weight,
        malignancy_weight,
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
        device_name=args.device,
        train_negative_ratio=args.train_negative_ratio,
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
        nodule_sensitivity=result.nodule_sensitivity,
        nodule_specificity=result.nodule_specificity,
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
