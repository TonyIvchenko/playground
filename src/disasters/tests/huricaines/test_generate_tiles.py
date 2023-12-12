from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src.disasters.scripts.huricaines.generate_tiles as huricaines_tiles


def test_helper_functions_and_frames():
    frames = huricaines_tiles.build_frames(2000, 2000)
    assert len(frames) == 12
    assert frames[0] == (2000, 1)
    assert frames[-1] == (2000, 12)

    kernel = huricaines_tiles.gaussian_kernel(radius=2, sigma=1.2)
    assert kernel.shape == (5, 5)
    assert float(kernel.max()) > 0.0

    assert huricaines_tiles.latlon_to_grid(20.0, -60.0) is not None
    assert huricaines_tiles.latlon_to_grid(-50.0, -60.0) is None


def test_load_points_and_accumulate(monkeypatch, tmp_path: Path):
    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 1), dtype=torch.float32)

    def fake_load_model_bundle(_path: Path):
        return DummyModel(), torch.zeros(11), torch.ones(11), "test"

    monkeypatch.setattr(huricaines_tiles, "load_model_bundle", fake_load_model_bundle)

    csv_path = tmp_path / "tracks.csv"
    pd.DataFrame(
        [
            {
                "storm_id": "AL012000",
                "iso_time": "2000-08-01 00:00:00",
                "lat": 20.0,
                "lon": -60.0,
                "vmax_kt": 55.0,
                "min_pressure_mb": 1002.0,
            },
            {
                "storm_id": "AL012000",
                "iso_time": "2000-08-01 06:00:00",
                "lat": 20.5,
                "lon": -60.2,
                "vmax_kt": 60.0,
                "min_pressure_mb": 998.0,
            },
        ]
    ).to_csv(csv_path, index=False)

    points = huricaines_tiles.load_points(csv_path, model_path=Path("dummy.pt"))
    assert not points.empty
    assert np.allclose(points["prob"].to_numpy(), 0.5)

    kernel = huricaines_tiles.gaussian_kernel(radius=1, sigma=1.0)
    summed, counted = huricaines_tiles.accumulate_points(points, kernel)
    assert summed.shape == (huricaines_tiles.GRID_H, huricaines_tiles.GRID_W)
    assert counted.shape == (huricaines_tiles.GRID_H, huricaines_tiles.GRID_W)
    assert float(counted.max()) > 0.0
