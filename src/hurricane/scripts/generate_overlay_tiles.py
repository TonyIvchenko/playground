"""Pre-generate hurricane yearly risk overlay tiles for Google Maps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image
import torch

from hurricane.model import load_model_bundle


TRAIN_END_YEAR = 2018
EVAL_END_YEAR = 2023
TILE_SIZE = 256

LOW_COLOR = (46, 204, 113, 110)
MID_COLOR = (241, 196, 15, 125)
HIGH_COLOR = (231, 76, 60, 145)


def tile_center_lat_lon(x: int, y: int, z: int) -> tuple[float, float]:
    n = 2 ** z
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ((y + 0.5) / n))))
    lat = math.degrees(lat_rad)
    return lat, lon


def synthetic_features(lat: float, lon: float, year: int) -> list[float]:
    year_offset = year - 2000
    month = 9.0
    month_angle = 2.0 * math.pi * month / 12.0

    tropical_band = max(0.0, 1.0 - abs(lat - 20.0) / 28.0)
    atlantic_focus = max(0.0, 1.0 - abs(lon + 60.0) / 55.0)
    pacific_focus = 0.35 * max(0.0, 1.0 - abs(lon + 140.0) / 45.0)
    basin_signal = max(0.0, min(1.0, tropical_band * (atlantic_focus + pacific_focus)))

    vmax_kt = 40.0 + 65.0 * basin_signal + 0.25 * year_offset
    min_pressure_mb = 1007.0 - 38.0 * basin_signal - 0.15 * year_offset
    dvmax_6h = 1.0 + 8.0 * basin_signal + 0.03 * year_offset
    dpres_6h = -1.0 - 6.0 * basin_signal - 0.02 * year_offset

    return [
        max(0.0, min(200.0, vmax_kt)),
        max(850.0, min(1025.0, min_pressure_mb)),
        lat,
        lon,
        month,
        math.sin(month_angle),
        math.cos(month_angle),
        abs(lat),
        1010.0 - min_pressure_mb,
        max(-60.0, min(60.0, dvmax_6h)),
        max(-60.0, min(60.0, dpres_6h)),
    ]


def classify(probability: float) -> tuple[int, int, int, int]:
    if probability < 0.33:
        return LOW_COLOR
    if probability < 0.66:
        return MID_COLOR
    return HIGH_COLOR


def render_tile(color: tuple[int, int, int, int], output_path: Path) -> None:
    image = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate precomputed hurricane map overlay tiles.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("src/hurricane/model/hurricane_model.pt"),
        help="Path to hurricane model artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/hurricane/tiles"),
        help="Directory where map tiles and config will be written.",
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2030)
    parser.add_argument("--zoom", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, feature_mean, feature_std, model_version = load_model_bundle(args.model_path)

    n = 2 ** args.zoom
    for year in range(args.start_year, args.end_year + 1):
        for x in range(n):
            for y in range(n):
                lat, lon = tile_center_lat_lon(x, y, args.zoom)
                features = synthetic_features(lat=lat, lon=lon, year=year)
                x_tensor = torch.tensor([features], dtype=torch.float32)
                x_tensor = (x_tensor - feature_mean) / feature_std
                with torch.no_grad():
                    probability = float(torch.sigmoid(model(x_tensor))[0, 0].item())
                color = classify(probability)
                tile_path = args.output_dir / str(year) / str(args.zoom) / str(x) / f"{y}.png"
                render_tile(color=color, output_path=tile_path)

    config = {
        "service": "hurricane",
        "start_year": args.start_year,
        "end_year": args.end_year,
        "zoom": args.zoom,
        "training_end_year": TRAIN_END_YEAR,
        "eval_end_year": EVAL_END_YEAR,
        "center_lat": 23.0,
        "center_lon": -60.0,
        "model_version": model_version,
    }
    config_path = args.output_dir / "overlay_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote hurricane overlay tiles to: {args.output_dir}")
    print(f"Wrote config to: {config_path}")


if __name__ == "__main__":
    main()
