from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


PATCH_SHAPE = (16, 16, 16)


class NodulePatchNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def create_model() -> NodulePatchNet:
    return NodulePatchNet()


def save_model_bundle(
    path: Path,
    model: NodulePatchNet,
    patch_mean: float,
    patch_std: float,
    model_version: str,
    nodule_accuracy: float,
    malignancy_auc: float,
    dataset_rows: int,
) -> None:
    bundle = {
        "state_dict": model.state_dict(),
        "patch_mean": float(patch_mean),
        "patch_std": float(max(patch_std, 1e-6)),
        "model_version": model_version,
        "nodule_accuracy": float(nodule_accuracy),
        "malignancy_auc": float(malignancy_auc),
        "dataset_rows": int(dataset_rows),
        "patch_shape": list(PATCH_SHAPE),
    }
    torch.save(bundle, path)


def load_model_bundle(path: Path) -> tuple[NodulePatchNet, float, float, str, dict[str, float]]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model = create_model()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    metrics = {
        "nodule_accuracy": float(bundle.get("nodule_accuracy", 0.0)),
        "malignancy_auc": float(bundle.get("malignancy_auc", 0.0)),
        "dataset_rows": float(bundle.get("dataset_rows", 0)),
    }
    return (
        model,
        float(bundle.get("patch_mean", 0.0)),
        float(bundle.get("patch_std", 1.0)),
        str(bundle.get("model_version", "unknown")),
        metrics,
    )


def predict_logits(
    model: NodulePatchNet,
    patches: torch.Tensor,
    patch_mean: float,
    patch_std: float,
) -> torch.Tensor:
    if patches.ndim != 5:
        raise ValueError(f"Expected tensor of shape [batch, channel, depth, height, width], got {patches.shape}")
    normalized = (patches - patch_mean) / max(patch_std, 1e-6)
    with torch.no_grad():
        return model(normalized)
