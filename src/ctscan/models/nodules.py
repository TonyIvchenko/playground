from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


PATCH_SHAPE = (24, 24, 24)


class LegacyPatchNet(nn.Module):
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


class SqueezeExcite3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.proj(self.pool(x))


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels)
        self.se = SqueezeExcite3D(out_channels)
        self.act = nn.SiLU()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.se(self.norm2(self.conv2(x)))
        return self.act(x + residual)


class NoduleResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.SiLU(),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock3D(32, 32),
            ResidualBlock3D(32, 32),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock3D(32, 64, stride=2),
            ResidualBlock3D(64, 64),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock3D(64, 128, stride=2),
            ResidualBlock3D(128, 128),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock3D(128, 192, stride=2),
            ResidualBlock3D(192, 192),
        )
        self.pool = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=192),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Dropout(p=0.25),
        )
        self.nodule_head = nn.Linear(96, 1)
        self.malignancy_head = nn.Linear(96, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        embedding = self.embedding(self.pool(x))
        return torch.cat([self.nodule_head(embedding), self.malignancy_head(embedding)], dim=1)


def create_model() -> NoduleResNet:
    return NoduleResNet()


def save_model_bundle(
    path: Path,
    model: nn.Module,
    patch_mean: float,
    patch_std: float,
    model_version: str,
    nodule_accuracy: float,
    nodule_auc: float,
    malignancy_auc: float,
    nodule_sensitivity: float,
    nodule_specificity: float,
    dataset_rows: int,
) -> None:
    bundle = {
        "state_dict": model.state_dict(),
        "patch_mean": float(patch_mean),
        "patch_std": float(max(patch_std, 1e-6)),
        "model_version": model_version,
        "nodule_accuracy": float(nodule_accuracy),
        "nodule_auc": float(nodule_auc),
        "malignancy_auc": float(malignancy_auc),
        "nodule_sensitivity": float(nodule_sensitivity),
        "nodule_specificity": float(nodule_specificity),
        "dataset_rows": int(dataset_rows),
        "patch_shape": list(PATCH_SHAPE),
    }
    torch.save(bundle, path)


def load_model_bundle(path: Path) -> tuple[nn.Module, float, float, str, dict[str, float | list[int]]]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = bundle["state_dict"]
    if any(key.startswith("encoder.") for key in state_dict):
        model = LegacyPatchNet()
    else:
        model = create_model()
    model.load_state_dict(state_dict)
    model.eval()
    metrics: dict[str, float | list[int]] = {
        "nodule_accuracy": float(bundle.get("nodule_accuracy", 0.0)),
        "nodule_auc": float(bundle.get("nodule_auc", 0.0)),
        "malignancy_auc": float(bundle.get("malignancy_auc", 0.0)),
        "nodule_sensitivity": float(bundle.get("nodule_sensitivity", 0.0)),
        "nodule_specificity": float(bundle.get("nodule_specificity", 0.0)),
        "dataset_rows": float(bundle.get("dataset_rows", 0)),
        "patch_shape": [int(x) for x in bundle.get("patch_shape", PATCH_SHAPE)],
    }
    return (
        model,
        float(bundle.get("patch_mean", 0.0)),
        float(bundle.get("patch_std", 1.0)),
        str(bundle.get("model_version", "unknown")),
        metrics,
    )


def predict_logits(
    model: nn.Module,
    patches: torch.Tensor,
    patch_mean: float,
    patch_std: float,
) -> torch.Tensor:
    if patches.ndim != 5:
        raise ValueError(f"Expected tensor of shape [batch, channel, depth, height, width], got {patches.shape}")
    normalized = (patches - patch_mean) / max(patch_std, 1e-6)
    with torch.no_grad():
        return model(normalized)
