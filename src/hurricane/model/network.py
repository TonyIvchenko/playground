from __future__ import annotations

import torch
from torch import nn


FEATURE_NAMES = [
    "vmax_kt",
    "min_pressure_mb",
    "lat",
    "lon",
    "month",
    "month_sin",
    "month_cos",
    "abs_lat",
    "pressure_deficit",
    "dvmax_6h",
    "dpres_6h",
]


class HurricaneMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_model() -> HurricaneMLP:
    return HurricaneMLP()
