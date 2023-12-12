from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


FEATURE_NAMES = ["temp_c", "humidity_pct", "wind_kph", "ffmc", "dmc", "drought_code", "isi"]


class WildfiresMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_model() -> WildfiresMLP:
    return WildfiresMLP()


def save_model_bundle(
    path: Path,
    model: WildfiresMLP,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    model_version: str,
    val_accuracy: float,
    dataset_rows: int,
    val_balanced_accuracy: float | None = None,
    val_auc: float | None = None,
) -> None:
    bundle: dict[str, object] = {
        "state_dict": model.state_dict(),
        "feature_mean": feature_mean.squeeze(0).tolist(),
        "feature_std": feature_std.squeeze(0).tolist(),
        "model_version": model_version,
        "feature_names": FEATURE_NAMES,
        "val_accuracy": val_accuracy,
        "dataset_rows": dataset_rows,
    }
    if val_balanced_accuracy is not None:
        bundle["val_balanced_accuracy"] = val_balanced_accuracy
    if val_auc is not None:
        bundle["val_auc"] = val_auc
    torch.save(bundle, path)


def load_model_bundle(path: Path) -> tuple[WildfiresMLP, torch.Tensor, torch.Tensor, str]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model = create_model()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    feature_mean = torch.tensor(bundle["feature_mean"], dtype=torch.float32)
    feature_std = torch.tensor(bundle["feature_std"], dtype=torch.float32).clamp_min(1e-6)
    model_version = str(bundle.get("model_version", "unknown"))
    return model, feature_mean, feature_std, model_version
