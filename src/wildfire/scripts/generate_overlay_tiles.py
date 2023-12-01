"""Generate a precomputed monthly wildfire overlay cube (2000-2030)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.wildfire.model import load_model_bundle


TRAIN_END_YEAR = 2018
EVAL_END_YEAR = 2023

LAT_MIN = 30.0
LAT_MAX = 45.5
LON_MIN = -12.5
LON_MAX = 12.5
GRID_H = 180
GRID_W = 260


MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate wildfire monthly overlay cube.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("src/wildfire/model/wildfire_model.pt"),
        help="Path to wildfire model artifact.",
    )
    parser.add_argument(
        "--forest-path",
        type=Path,
        default=Path("src/wildfire/data/raw/forestfires_uci.csv"),
        help="Path to UCI forest fires csv.",
    )
    parser.add_argument(
        "--algerian-path",
        type=Path,
        default=Path("src/wildfire/data/raw/algerian_forest_fires.csv"),
        help="Path to Algerian forest fires csv-like file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/wildfire/tiles"),
        help="Output directory for overlay cube + config.",
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2030)
    parser.add_argument("--zoom-min", type=int, default=4)
    parser.add_argument("--zoom-max", type=int, default=8)
    return parser.parse_args()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def load_forest_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Forest fires csv not found: {path}")

    df = pd.read_csv(path)
    month = df["month"].astype(str).str.lower().str[:3].map(MONTH_MAP)

    # Approximate coordinates for Montesinho park, Portugal where this grid comes from.
    lat = 41.5 + (pd.to_numeric(df["Y"], errors="coerce") - 2.0) * 0.08
    lon = -8.2 + (pd.to_numeric(df["X"], errors="coerce") - 1.0) * 0.08

    out = pd.DataFrame(
        {
            "year": np.nan,
            "month": pd.to_numeric(month, errors="coerce"),
            "lat": pd.to_numeric(lat, errors="coerce"),
            "lon": pd.to_numeric(lon, errors="coerce"),
            "temp_c": pd.to_numeric(df["temp"], errors="coerce"),
            "humidity_pct": pd.to_numeric(df["RH"], errors="coerce"),
            "wind_kph": pd.to_numeric(df["wind"], errors="coerce"),
            "ffmc": pd.to_numeric(df["FFMC"], errors="coerce"),
            "dmc": pd.to_numeric(df["DMC"], errors="coerce"),
            "drought_code": pd.to_numeric(df["DC"], errors="coerce"),
            "isi": pd.to_numeric(df["ISI"], errors="coerce"),
            "source_weight": 0.55,
        }
    )
    out = out.dropna().reset_index(drop=True)
    out = out[out["month"].between(1, 12)]
    return out


def load_algerian_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Algerian dataset not found: {path}")

    rows: list[dict[str, float]] = []
    region = "bejaia"

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            low = text.lower()
            if "bejaia region dataset" in low:
                region = "bejaia"
                continue
            if "sidi-bel abbes region dataset" in low or "sidi bel-abbes region dataset" in low:
                region = "sidi_bel_abbes"
                continue
            if low.startswith("day,month,year"):
                continue

            parts = [p.strip() for p in text.split(",")]
            if len(parts) < 14:
                continue

            try:
                day = int(parts[0])
                month = int(parts[1])
                year = int(parts[2])
                temp_c = float(parts[3])
                humidity_pct = float(parts[4])
                wind_kph = float(parts[5])
                ffmc = float(parts[7])
                dmc = float(parts[8])
                drought_code = float(parts[9])
                isi = float(parts[10])
            except ValueError:
                continue

            if region == "bejaia":
                base_lat, base_lon = 36.75, 5.05
            else:
                base_lat, base_lon = 34.68, -0.63

            # Light deterministic spatial jitter to avoid full overlap.
            lat = base_lat + ((day % 7) - 3) * 0.05
            lon = base_lon + ((day % 5) - 2) * 0.05

            rows.append(
                {
                    "year": float(year),
                    "month": float(month),
                    "lat": float(lat),
                    "lon": float(lon),
                    "temp_c": temp_c,
                    "humidity_pct": humidity_pct,
                    "wind_kph": wind_kph,
                    "ffmc": ffmc,
                    "dmc": dmc,
                    "drought_code": drought_code,
                    "isi": isi,
                    "source_weight": 1.0,
                }
            )

    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def predict_probabilities(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    model, feature_mean, feature_std, _ = load_model_bundle(model_path)
    feature_cols = ["temp_c", "humidity_pct", "wind_kph", "ffmc", "dmc", "drought_code", "isi"]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    x = (x - feature_mean) / feature_std
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).reshape(-1).cpu().numpy()

    out = df.copy()
    out["prob"] = probs
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


def build_frames(start_year: int, end_year: int) -> list[tuple[int, int]]:
    frames: list[tuple[int, int]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            frames.append((year, month))
    return frames


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forest_df = load_forest_points(args.forest_path)
    algerian_df = load_algerian_points(args.algerian_path)
    points_df = pd.concat([forest_df, algerian_df], ignore_index=True).dropna().reset_index(drop=True)
    points_df = predict_probabilities(points_df, model_path=args.model_path)

    frames = build_frames(args.start_year, args.end_year)
    frame_labels = [f"{y:04d}-{m:02d}" for y, m in frames]

    kernel = gaussian_kernel(radius=2, sigma=1.15)

    # Monthly climatology from all available source points.
    month_clim_risk: dict[int, np.ndarray] = {}
    month_clim_activity: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        base_points = points_df[points_df["month"] == float(month)]
        s, c = accumulate_points(base_points, kernel)
        risk = np.divide(s, c, out=np.zeros_like(s, dtype=np.float32), where=c > 1e-6).astype(np.float32)
        activity = np.clip(np.log1p(c) / np.log1p(max(float(c.max()), 1.0)), 0.0, 1.0).astype(np.float32)
        month_clim_risk[month] = risk
        month_clim_activity[month] = activity

    risk_cube = np.zeros((len(frames), GRID_H, GRID_W), dtype=np.float32)
    activity_cube = np.zeros((len(frames), GRID_H, GRID_W), dtype=np.float32)

    for i, (year, month) in enumerate(frames):
        # Use year-specific Algerian observations when available, always blended with monthly climatology.
        year_points = points_df[(points_df["year"] == float(year)) & (points_df["month"] == float(month))]
        s, c = accumulate_points(year_points, kernel)
        if float(c.max()) > 0.0:
            frame_risk = np.divide(s, c, out=month_clim_risk[month].copy(), where=c > 1e-6)
            frame_activity = np.clip(np.log1p(c) / np.log1p(max(float(c.max()), 1.0)), 0.0, 1.0)
            frame_activity = 0.65 * frame_activity + 0.35 * month_clim_activity[month]
        else:
            frame_risk = month_clim_risk[month].copy()
            frame_activity = month_clim_activity[month].copy()

        # Light forward trend for inference years.
        if year > EVAL_END_YEAR:
            years_ahead = year - EVAL_END_YEAR
            frame_risk = np.clip(frame_risk * (1.0 + 0.006 * years_ahead), 0.0, 1.0)
            frame_activity = np.clip(frame_activity * (1.0 + 0.004 * years_ahead), 0.0, 1.0)

        risk_cube[i] = frame_risk.astype(np.float32)
        activity_cube[i] = frame_activity.astype(np.float32)

    cube_path = args.output_dir / "overlay_cube.npz"
    np.savez_compressed(
        cube_path,
        risk=risk_cube.astype(np.float16),
        activity=activity_cube.astype(np.float16),
        frames=np.array(frame_labels),
    )

    model_bundle = torch.load(args.model_path, map_location="cpu", weights_only=True)
    config = {
        "service": "wildfire",
        "start_year": args.start_year,
        "end_year": args.end_year,
        "zoom_min": args.zoom_min,
        "zoom_max": args.zoom_max,
        "training_end_year": TRAIN_END_YEAR,
        "eval_end_year": EVAL_END_YEAR,
        "center_lat": 36.5,
        "center_lon": 0.5,
        "lat_min": LAT_MIN,
        "lat_max": LAT_MAX,
        "lon_min": LON_MIN,
        "lon_max": LON_MAX,
        "grid_h": GRID_H,
        "grid_w": GRID_W,
        "model_version": str(model_bundle.get("model_version", "unknown")),
        "frame_count": len(frame_labels),
    }
    config_path = args.output_dir / "overlay_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Wrote wildfire overlay cube to: {cube_path}")
    print(f"Wrote wildfire overlay config to: {config_path}")


if __name__ == "__main__":
    main()
