from __future__ import annotations

import torch
from torch import nn


FEATURE_NAMES = ["temp_c", "humidity_pct", "wind_kph", "drought_index"]


class WildfireMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_model() -> WildfireMLP:
    return WildfireMLP()
