from __future__ import annotations

import torch
from torch import nn


FEATURE_NAMES = ["temp_c", "humidity_pct", "wind_kph", "drought_index"]


class WildfireMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_model() -> WildfireMLP:
    return WildfireMLP()
