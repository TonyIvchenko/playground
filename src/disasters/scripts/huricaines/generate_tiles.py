"""Generate a precomputed monthly huricaines overlay cube (2000-2030)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.disasters.models.huricaines import load_model_bundle


TRAIN_END_YEAR = 2018
EVAL_END_YEAR = 2023

LAT_MIN = 0.0
LAT_MAX = 55.0
LON_MIN = -120.0
LON_MAX = 20.0
GRID_H = 220
GRID_W = 320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate huricaines monthly overlay cube.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("src/disasters/models/huricaines.pt"),
        help="Path to huricaines model artifact.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("src/disasters/data/huricaines/raw/huricaines_tracks_merged.csv"),
        help="Path to merged huricaines tracks csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/disasters/tiles/huricaines"),
        help="Output directory for overlay + metadata.",
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2030)
    parser.add_argument("--zoom-min", type=int, default=4)
    parser.add_argument("--zoom-max", type=int, default=8)
    return parser.parse_args()


def build_frames(start_year: int, end_year: int) -> list[tuple[int, int]]:
    frames: list[tuple[int, int]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            frames.append((year, month))
    return frames


def load_points(path: Path, model_path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Huricaines tracks file not found: {path}")

    usecols = ["storm_id", "iso_time", "lat", "lon", "vmax_kt", "min_pressure_mb"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    df["storm_id"] = df["storm_id"].astype(str).str.strip()
    df["iso_time"] = pd.to_datetime(df["iso_time"], errors="coerce")
    for col in ["lat", "lon", "vmax_kt", "min_pressure_mb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["storm_id", "iso_time", "vmax_kt", "lat", "lon"]).copy()
    df = df[df["vmax_kt"].between(0.0, 200.0)]
    df = df[df["lat"].between(-5.0, 70.0)]
    df = df[df["lon"].between(-120.0, 20.0)]
    df = df.sort_values(["storm_id", "iso_time"]).reset_index(drop=True)

    df["month"] = df["iso_time"].dt.month.astype(float)
    angle = 2.0 * math.pi * df["month"] / 12.0
    df["month_sin"] = angle.map(math.sin)
    df["month_cos"] = angle.map(math.cos)
    df["abs_lat"] = df["lat"].abs()

    pressure_median = float(df["min_pressure_mb"].median())
    if not math.isfinite(pressure_median):
        pressure_median = 1000.0
    df["min_pressure_mb"] = df["min_pressure_mb"].fillna(pressure_median)
    df["pressure_deficit"] = 1010.0 - df["min_pressure_mb"]

    df["prev_time"] = df.groupby("storm_id", sort=False)["iso_time"].shift(1)
    df["prev_vmax"] = df.groupby("storm_id", sort=False)["vmax_kt"].shift(1)
    df["prev_pressure"] = df.groupby("storm_id", sort=False)["min_pressure_mb"].shift(1)

    dt_hours = (df["iso_time"] - df["prev_time"]).dt.total_seconds() / 3600.0
    step = (dt_hours / 6.0).where(dt_hours > 0.0)
    df["dvmax_6h"] = ((df["vmax_kt"] - df["prev_vmax"]) / step).fillna(0.0).clip(-60.0, 60.0)
    df["dpres_6h"] = ((df["min_pressure_mb"] - df["prev_pressure"]) / step).fillna(0.0).clip(-60.0, 60.0)

    feature_cols = [
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

    model, feature_mean, feature_std, _ = load_model_bundle(model_path)
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    x = (x - feature_mean) / feature_std
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).reshape(-1).cpu().numpy()

    out = pd.DataFrame(
        {
            "year": df["iso_time"].dt.year.astype(float),
            "month": df["iso_time"].dt.month.astype(float),
            "lat": df["lat"].astype(float),
            "lon": df["lon"].astype(float),
            "prob": probs.astype(float),
            "source_weight": 1.0,
        }
    )
    out = out[(out["lat"].between(LAT_MIN, LAT_MAX)) & (out["lon"].between(LON_MIN, LON_MAX))].reset_index(drop=True)
    return out


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


def accumulate_points(points: pd.DataFrame, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_radius = kernel.shape[0] // 2
    weighted_sum = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    weighted_count = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    if points.empty:
        return weighted_sum, weighted_count

    for row in points.itertuples(index=False):
        idx = latlon_to_grid(float(row.lat), float(row.lon))
        if idx is None:
            continue
        iy, ix = idx
        point_weight = float(row.source_weight)
        point_prob = float(row.prob)

        y0 = max(0, iy - k_radius)
        y1 = min(GRID_H, iy + k_radius + 1)
        x0 = max(0, ix - k_radius)
        x1 = min(GRID_W, ix + k_radius + 1)

        ky0 = y0 - (iy - k_radius)
        ky1 = ky0 + (y1 - y0)
        kx0 = x0 - (ix - k_radius)
        kx1 = kx0 + (x1 - x0)

        k_slice = kernel[ky0:ky1, kx0:kx1] * point_weight
        weighted_sum[y0:y1, x0:x1] += point_prob * k_slice
        weighted_count[y0:y1, x0:x1] += k_slice

    return weighted_sum, weighted_count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points_df = load_points(args.input_csv, model_path=args.model_path)
    frames = build_frames(args.start_year, args.end_year)
    frame_labels = [f"{y:04d}-{m:02d}" for y, m in frames]

    kernel = gaussian_kernel(radius=2, sigma=1.2)

    # Monthly climatology from historical observations.
    month_clim_risk: dict[int, np.ndarray] = {}
    month_clim_activity: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        base_points = points_df[points_df["month"] == float(month)]
        s, c = accumulate_points(base_points, kernel)
        risk = np.divide(s, c, out=np.zeros_like(s, dtype=np.float32), where=c > 1e-6).astype(np.float32)
        activity = np.clip(np.log1p(c) / np.log1p(max(float(c.max()), 1.0)), 0.0, 1.0).astype(np.float32)
        month_clim_risk[month] = risk
        month_clim_activity[month] = activity

    max_hist_year = int(points_df["year"].max()) if not points_df.empty else EVAL_END_YEAR
    yearly_mean = points_df.groupby(points_df["year"].astype(int))["prob"].mean() if not points_df.empty else pd.Series(dtype=float)
    trend_per_year = 0.0
    if len(yearly_mean) >= 3:
        x = yearly_mean.index.values.astype(float)
        y = yearly_mean.values.astype(float)
        trend_per_year = float(np.polyfit(x, y, 1)[0])

    risk_cube = np.zeros((len(frames), GRID_H, GRID_W), dtype=np.float32)
    activity_cube = np.zeros((len(frames), GRID_H, GRID_W), dtype=np.float32)

    for i, (year, month) in enumerate(frames):
        monthly_points = points_df[(points_df["year"] == float(year)) & (points_df["month"] == float(month))]
        s, c = accumulate_points(monthly_points, kernel)

        if float(c.max()) > 0.0:
            frame_risk = np.divide(s, c, out=month_clim_risk[month].copy(), where=c > 1e-6)
            frame_activity = np.clip(np.log1p(c) / np.log1p(max(float(c.max()), 1.0)), 0.0, 1.0)
            frame_activity = 0.7 * frame_activity + 0.3 * month_clim_activity[month]
        else:
            frame_risk = month_clim_risk[month].copy()
            frame_activity = month_clim_activity[month].copy()

        if year > max_hist_year:
            years_ahead = year - max_hist_year
            frame_risk = np.clip(frame_risk + (trend_per_year * years_ahead * 0.6), 0.0, 1.0)
            frame_activity = np.clip(frame_activity * (1.0 + 0.004 * years_ahead), 0.0, 1.0)

        risk_cube[i] = frame_risk.astype(np.float32)
        activity_cube[i] = frame_activity.astype(np.float32)

    cube_path = args.output_dir / "overlay.npz"
    np.savez_compressed(
        cube_path,
        risk=risk_cube.astype(np.float16),
        activity=activity_cube.astype(np.float16),
        frames=np.array(frame_labels),
    )

    model_bundle = torch.load(args.model_path, map_location="cpu", weights_only=True)
    config = {
        "service": "huricaines",
        "start_year": args.start_year,
        "end_year": args.end_year,
        "zoom_min": args.zoom_min,
        "zoom_max": args.zoom_max,
        "training_end_year": TRAIN_END_YEAR,
        "eval_end_year": EVAL_END_YEAR,
        "center_lat": 24.0,
        "center_lon": -60.0,
        "lat_min": LAT_MIN,
        "lat_max": LAT_MAX,
        "lon_min": LON_MIN,
        "lon_max": LON_MAX,
        "grid_h": GRID_H,
        "grid_w": GRID_W,
        "model_version": str(model_bundle.get("model_version", "unknown")),
        "frame_count": len(frame_labels),
    }
    config_path = args.output_dir / "overlay.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Wrote huricaines overlay to: {cube_path}")
    print(f"Wrote huricaines overlay metadata to: {config_path}")


if __name__ == "__main__":
    main()
