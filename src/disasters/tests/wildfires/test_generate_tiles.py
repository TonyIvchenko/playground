from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import src.disasters.scripts.wildfires.generate_tiles as wildfires_tiles


def test_basic_helpers_and_build_frames():
    frames = wildfires_tiles.build_frames(2001, 2002)
    assert len(frames) == 24
    assert frames[0] == (2001, 1)
    assert frames[-1] == (2002, 12)

    kernel = wildfires_tiles.gaussian_kernel(radius=2, sigma=1.1)
    assert kernel.shape == (5, 5)
    assert float(kernel.max()) > 0.0

    inside = wildfires_tiles.latlon_to_grid(36.0, -100.0)
    outside = wildfires_tiles.latlon_to_grid(-99.0, 0.0)
    assert inside is not None
    assert outside is None


def test_load_us_points_normalizes_and_filters(tmp_path: Path):
    input_csv = tmp_path / "us_points.csv"
    pd.DataFrame(
        {
            "year": [2018, 2019, 2019],
            "month": [7, 8, 13],  # last row invalid month
            "lat": [36.5, 40.1, 91.0],  # last row invalid lat
            "lon": [-120.2, -90.3, -75.0],
            "fire_size": [120.0, 800.0, 300.0],
        }
    ).to_csv(input_csv, index=False)

    out = wildfires_tiles.load_us_points(input_csv)

    assert len(out) == 2
    assert set(out.columns) == {"year", "month", "lat", "lon", "prob_weight", "conf_weight", "source_weight"}
    assert float(out["prob_weight"].min()) >= 0.05
    assert float(out["prob_weight"].max()) <= 1.0
    assert float(out["conf_weight"].min()) >= 0.25
    assert float(out["conf_weight"].max()) <= 1.0


def test_build_frame_outputs_nonzero():
    points = pd.DataFrame(
        {
            "year": [2018.0, 2018.0],
            "month": [7.0, 8.0],
            "lat": [36.5, 36.6],
            "lon": [-120.2, -120.1],
            "prob_weight": [0.5, 0.8],
            "conf_weight": [0.6, 0.9],
            "source_weight": [1.0, 1.0],
        }
    )
    kernel = wildfires_tiles.gaussian_kernel(radius=1, sigma=1.0)
    risk, conf = wildfires_tiles._build_frame(
        points=points,
        kernel=kernel,
        fallback_risk=np.zeros((wildfires_tiles.GRID_H, wildfires_tiles.GRID_W), dtype=np.float32),
        fallback_conf=np.zeros((wildfires_tiles.GRID_H, wildfires_tiles.GRID_W), dtype=np.float32),
    )
    assert risk.shape == (wildfires_tiles.GRID_H, wildfires_tiles.GRID_W)
    assert conf.shape == (wildfires_tiles.GRID_H, wildfires_tiles.GRID_W)
    assert float(risk.max()) > 0.0
    assert float(conf.max()) > 0.0
