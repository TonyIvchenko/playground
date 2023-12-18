"""Generate a precomputed monthly wildfires overlay cube (US-focused, 2000-2030)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

DISASTERS_ROOT = Path(__file__).resolve().parents[2]
if str(DISASTERS_ROOT) not in sys.path:
    sys.path.insert(0, str(DISASTERS_ROOT))

import numpy as np
import pandas as pd


TRAIN_END_YEAR = 2018
EVAL_END_YEAR = 2023

LAT_MIN = 24.0
LAT_MAX = 50.0
LON_MIN = -125.0
LON_MAX = -66.0
GRID_H = 220
GRID_W = 360


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate wildfires monthly overlay cube.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DISASTERS_ROOT / "data" / "wildfires" / "raw" / "wildfires_us_overlay.csv",
        help="Path to US wildfire overlay points CSV from download_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DISASTERS_ROOT / "tiles" / "wildfires",
        help="Output directory for overlay + metadata.",
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2030)
    parser.add_argument("--zoom-min", type=int, default=4)
    parser.add_argument("--zoom-max", type=int, default=8)
    return parser.parse_args()


def load_us_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"US wildfire points csv not found: {path}")

    raw = pd.read_csv(path)
    required = {"year", "month", "lat", "lon", "fire_size"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"US wildfire points csv is missing columns: {missing}")

    points = pd.DataFrame(
        {
            "year": pd.to_numeric(raw["year"], errors="coerce"),
            "month": pd.to_numeric(raw["month"], errors="coerce"),
            "lat": pd.to_numeric(raw["lat"], errors="coerce"),
            "lon": pd.to_numeric(raw["lon"], errors="coerce"),
            "fire_size": pd.to_numeric(raw["fire_size"], errors="coerce"),
        }
    )
    points = points.dropna().reset_index(drop=True)
    points = points[
        points["month"].between(1.0, 12.0)
        & points["lat"].between(LAT_MIN, LAT_MAX)
        & points["lon"].between(LON_MIN, LON_MAX)
        & (points["fire_size"] > 0.0)
    ].reset_index(drop=True)

    if points.empty:
        return pd.DataFrame(columns=["year", "month", "lat", "lon", "prob_weight", "conf_weight", "source_weight"])

    size_cap = float(points["fire_size"].quantile(0.98))
    size_cap = max(size_cap, 10.0)
    size_norm = np.log1p(points["fire_size"]) / np.log1p(size_cap)
    size_norm = np.clip(size_norm, 0.0, 1.0)

    points["prob_weight"] = np.clip(0.08 + 0.92 * size_norm, 0.05, 1.0)
    points["conf_weight"] = np.clip(0.25 + 0.75 * size_norm, 0.25, 1.0)
    points["source_weight"] = 1.0
    return points[["year", "month", "lat", "lon", "prob_weight", "conf_weight", "source_weight"]]


def gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    coords = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(coords, coords)
    kernel = np.exp(-((xx**2 + yy**2) / (2.0 * sigma**2)))
    return kernel.astype(np.float32)


def latlon_to_grid(lat: float, lon: float) -> tuple[int, int] | None:
    if lat < LAT_MIN or lat > LAT_MAX or lon < LON_MIN or lon > LON_MAX:
        return None
    iy = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * (GRID_H - 1))
    ix = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * (GRID_W - 1))
    iy = max(0, min(GRID_H - 1, iy))
    ix = max(0, min(GRID_W - 1, ix))
    return iy, ix


def _accumulate_weighted(points: pd.DataFrame, kernel: np.ndarray, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_radius = kernel.shape[0] // 2
    weighted_sum = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    weighted_count = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    hit_count = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    if points.empty:
        return weighted_sum, weighted_count, hit_count

    for row in points.itertuples(index=False):
        idx = latlon_to_grid(float(row.lat), float(row.lon))
        if idx is None:
            continue
        iy, ix = idx
        point_weight = float(row.source_weight)
        point_value = float(getattr(row, value_col))

        y0 = max(0, iy - k_radius)
        y1 = min(GRID_H, iy + k_radius + 1)
        x0 = max(0, ix - k_radius)
        x1 = min(GRID_W, ix + k_radius + 1)

        ky0 = y0 - (iy - k_radius)
        ky1 = ky0 + (y1 - y0)
        kx0 = x0 - (ix - k_radius)
        kx1 = kx0 + (x1 - x0)

        k_slice = kernel[ky0:ky1, kx0:kx1] * point_weight
        weighted_sum[y0:y1, x0:x1] += point_value * k_slice
        weighted_count[y0:y1, x0:x1] += k_slice
        hit_count[y0:y1, x0:x1] += point_weight

    return weighted_sum, weighted_count, hit_count


def _build_frame(points: pd.DataFrame, kernel: np.ndarray, fallback_risk: np.ndarray, fallback_conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    risk_sum, risk_count, risk_hits = _accumulate_weighted(points, kernel, "prob_weight")
    conf_sum, conf_count, conf_hits = _accumulate_weighted(points, kernel, "conf_weight")

    if float(risk_count.max()) <= 0.0:
        return fallback_risk.copy(), fallback_conf.copy()

    risk = np.divide(risk_sum, risk_count, out=fallback_risk.copy(), where=risk_count > 1e-6)
    conf_mean = np.divide(conf_sum, conf_count, out=fallback_conf.copy(), where=conf_count > 1e-6)

    hits = np.maximum(risk_hits, conf_hits)
    density = np.clip(np.log1p(hits) / np.log1p(max(float(hits.max()), 1.0)), 0.0, 1.0)
    confidence = np.clip(0.35 * conf_mean + 0.65 * density, 0.0, 1.0)
    return risk.astype(np.float32), confidence.astype(np.float32)


def build_frames(start_year: int, end_year: int) -> list[tuple[int, int]]:
    frames: list[tuple[int, int]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            frames.append((year, month))
    return frames


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points_df = load_us_points(args.input_csv)
    frames = build_frames(args.start_year, args.end_year)
    frame_labels = [f"{y:04d}-{m:02d}" for y, m in frames]

    kernel = gaussian_kernel(radius=2, sigma=1.15)

    month_clim_risk: dict[int, np.ndarray] = {}
    month_clim_conf: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        month_points = points_df[points_df["month"] == float(month)]
        risk, conf = _build_frame(
            points=month_points,
            kernel=kernel,
            fallback_risk=np.zeros((GRID_H, GRID_W), dtype=np.float32),
            fallback_conf=np.zeros((GRID_H, GRID_W), dtype=np.float32),
        )
        month_clim_risk[month] = risk
        month_clim_conf[month] = conf

    risk_cube = np.zeros((len(frames), GRID_H, GRID_W), dtype=np.float32)
    conf_cube = np.zeros((len(frames), GRID_H, GRID_W), dtype=np.float32)

    max_hist_year = int(points_df["year"].max()) if not points_df.empty else EVAL_END_YEAR
    yearly_counts = points_df.groupby(points_df["year"].astype(int)).size() if not points_df.empty else pd.Series(dtype=float)
    trend_per_year = 0.0
    if len(yearly_counts) >= 3:
        x = yearly_counts.index.values.astype(float)
        y = yearly_counts.values.astype(float)
        trend_per_year = float(np.polyfit(x, y, 1)[0] / max(float(y.mean()), 1.0))

    for i, (year, month) in enumerate(frames):
        frame_points = points_df[(points_df["year"] == float(year)) & (points_df["month"] == float(month))]
        frame_risk, frame_conf = _build_frame(
            points=frame_points,
            kernel=kernel,
            fallback_risk=month_clim_risk[month],
            fallback_conf=month_clim_conf[month],
        )

        if not frame_points.empty:
            frame_conf = np.clip(0.7 * frame_conf + 0.3 * month_clim_conf[month], 0.0, 1.0)

        if year > max_hist_year:
            years_ahead = year - max_hist_year
            frame_risk = np.clip(frame_risk * (1.0 + 0.012 * years_ahead) + trend_per_year * years_ahead * 0.25, 0.0, 1.0)
            frame_conf = np.clip(frame_conf * (1.0 - 0.003 * years_ahead), 0.18, 1.0)

        risk_cube[i] = frame_risk.astype(np.float32)
        conf_cube[i] = frame_conf.astype(np.float32)

    cube_path = args.output_dir / "overlay.npz"
    np.savez_compressed(
        cube_path,
        risk=risk_cube.astype(np.float16),
        confidence=conf_cube.astype(np.float16),
        activity=conf_cube.astype(np.float16),
        frames=np.array(frame_labels),
    )

    config = {
        "service": "wildfires",
        "start_year": args.start_year,
        "end_year": args.end_year,
        "zoom_min": args.zoom_min,
        "zoom_max": args.zoom_max,
        "training_end_year": TRAIN_END_YEAR,
        "eval_end_year": EVAL_END_YEAR,
        "center_lat": 39.5,
        "center_lon": -98.35,
        "lat_min": LAT_MIN,
        "lat_max": LAT_MAX,
        "lon_min": LON_MIN,
        "lon_max": LON_MAX,
        "grid_h": GRID_H,
        "grid_w": GRID_W,
        "frame_count": len(frame_labels),
    }
    config_path = args.output_dir / "overlay.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Wrote wildfires overlay to: {cube_path}")
    print(f"Wrote wildfires overlay metadata to: {config_path}")


if __name__ == "__main__":
    main()
