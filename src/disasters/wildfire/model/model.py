from __future__ import annotations

import torch
from torch import nn


FEATURE_NAMES = ["temp_c", "humidity_pct", "wind_kph", "ffmc", "dmc", "drought_code", "isi"]


class WildfireMLP(nn.Module):
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


def create_model() -> WildfireMLP:
    return WildfireMLP()
