from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src.disasters.scripts.wildfires.generate_tiles as wildfires_tiles


def test_basic_helpers_and_build_frames():
    frames = wildfires_tiles.build_frames(2001, 2002)
    assert len(frames) == 24
    assert frames[0] == (2001, 1)
    assert frames[-1] == (2002, 12)

    kernel = wildfires_tiles.gaussian_kernel(radius=2, sigma=1.1)
    assert kernel.shape == (5, 5)
    assert float(kernel.max()) > 0.0

    inside = wildfires_tiles.latlon_to_grid(36.0, 0.0)
    outside = wildfires_tiles.latlon_to_grid(-99.0, 0.0)
    assert inside is not None
    assert outside is None


def test_accumulate_points_and_predict_probabilities(monkeypatch):
    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 1), dtype=torch.float32)

    def fake_load_model_bundle(_path: Path):
        return DummyModel(), torch.zeros(7), torch.ones(7), "test"

    monkeypatch.setattr(wildfires_tiles, "load_model_bundle", fake_load_model_bundle)

    df = pd.DataFrame(
        {
            "year": [2018.0, 2018.0],
            "month": [7.0, 8.0],
            "lat": [36.5, 36.6],
            "lon": [0.2, 0.3],
            "temp_c": [30.0, 31.0],
            "humidity_pct": [25.0, 20.0],
            "wind_kph": [12.0, 10.0],
            "ffmc": [90.0, 91.0],
            "dmc": [100.0, 101.0],
            "drought_code": [350.0, 360.0],
            "isi": [8.0, 9.0],
            "source_weight": [1.0, 1.0],
        }
    )

    scored = wildfires_tiles.predict_probabilities(df, model_path=Path("dummy.pt"))
    assert np.allclose(scored["prob"].to_numpy(), 0.5)

    kernel = wildfires_tiles.gaussian_kernel(radius=1, sigma=1.0)
    summed, counted = wildfires_tiles.accumulate_points(scored, kernel)
    assert summed.shape == (wildfires_tiles.GRID_H, wildfires_tiles.GRID_W)
    assert counted.shape == (wildfires_tiles.GRID_H, wildfires_tiles.GRID_W)
    assert float(counted.max()) > 0.0
