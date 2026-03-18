from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
from PIL import Image, ImageFilter

from common import (
    CELL_PAINT_RADIUS,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MODEL_BUNDLE_PATH,
    OVERLAY_HEIGHT,
    OVERLAY_JSON_PATH,
    OVERLAY_NPZ_PATH,
    OVERLAY_WIDTH,
    TILES_DIR,
    current_month,
    dow_sin_cos,
    ensure_dirs,
    hour_sin_cos,
    month_sin_cos,
    weekly_frame_labels,
)


def feature_matrix(
    candidate_lats: np.ndarray,
    candidate_lons: np.ndarray,
    cell_total_counts: np.ndarray,
    cell_hour_counts: np.ndarray,
    frame_idx: int,
    month: int,
) -> np.ndarray:
    day_of_week = frame_idx // 24 + 1
    hour = frame_idx % 24
    hour_s, hour_c = hour_sin_cos(hour)
    dow_s, dow_c = dow_sin_cos(day_of_week)
    month_s, month_c = month_sin_cos(month)
    same_hour = cell_hour_counts[:, frame_idx]
    return np.column_stack(
        [
            candidate_lats,
            candidate_lons,
            np.full_like(candidate_lats, hour_s),
            np.full_like(candidate_lats, hour_c),
            np.full_like(candidate_lats, dow_s),
            np.full_like(candidate_lats, dow_c),
            np.full_like(candidate_lats, month_s),
            np.full_like(candidate_lats, month_c),
            np.log1p(cell_total_counts),
            np.log1p(same_hour),
            np.divide(same_hour, np.maximum(cell_total_counts, 1.0)),
        ]
    ).astype(np.float32)


def normalize_probs(probabilities: np.ndarray) -> np.ndarray:
    upper = np.quantile(probabilities, 0.995)
    if upper <= 0:
        return np.zeros_like(probabilities, dtype=np.float32)
    return np.clip(probabilities / upper, 0.0, 1.0).astype(np.float32)


def paint_frame(
    lats: np.ndarray,
    lons: np.ndarray,
    risk_values: np.ndarray,
    confidence_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    risk_grid = np.zeros((OVERLAY_HEIGHT, OVERLAY_WIDTH), dtype=np.float32)
    conf_grid = np.zeros_like(risk_grid)

    row = ((LAT_MAX - lats) / (LAT_MAX - LAT_MIN) * (OVERLAY_HEIGHT - 1)).astype(int)
    col = ((lons - LON_MIN) / (LON_MAX - LON_MIN) * (OVERLAY_WIDTH - 1)).astype(int)
    row = np.clip(row, 0, OVERLAY_HEIGHT - 1)
    col = np.clip(col, 0, OVERLAY_WIDTH - 1)

    for r, c, risk, conf in zip(row, col, risk_values, confidence_values, strict=False):
        r0 = max(0, r - CELL_PAINT_RADIUS)
        r1 = min(OVERLAY_HEIGHT, r + CELL_PAINT_RADIUS + 1)
        c0 = max(0, c - CELL_PAINT_RADIUS)
        c1 = min(OVERLAY_WIDTH, c + CELL_PAINT_RADIUS + 1)
        risk_grid[r0:r1, c0:c1] = np.maximum(risk_grid[r0:r1, c0:c1], risk)
        conf_grid[r0:r1, c0:c1] = np.maximum(conf_grid[r0:r1, c0:c1], conf)

    risk_blurred = np.asarray(
        Image.fromarray(np.clip(risk_grid * 255.0, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=1.8)
        ),
        dtype=np.float32,
    ) / 255.0
    conf_blurred = np.asarray(
        Image.fromarray(np.clip(conf_grid * 255.0, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=1.8)
        ),
        dtype=np.float32,
    ) / 255.0
    return risk_blurred, conf_blurred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=int, default=current_month())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    model = bundle["model"]
    candidate_lats = bundle["candidate_lats"].astype(np.float32)
    candidate_lons = bundle["candidate_lons"].astype(np.float32)
    cell_total_counts = bundle["cell_total_counts"].astype(np.float32)
    cell_hour_counts = bundle["cell_hour_counts"].astype(np.float32)
    confidence_values = np.log1p(cell_total_counts) / np.log1p(max(1.0, float(cell_total_counts.max())))

    raw_probs = []
    for frame_idx in range(24 * 7):
        features = feature_matrix(
            candidate_lats=candidate_lats,
            candidate_lons=candidate_lons,
            cell_total_counts=cell_total_counts,
            cell_hour_counts=cell_hour_counts,
            frame_idx=frame_idx,
            month=args.month,
        )
        raw_probs.append(model.predict_proba(features)[:, 1].astype(np.float32))

    raw_probs = np.stack(raw_probs, axis=0)
    normalized = normalize_probs(raw_probs)
    risk_cube = np.zeros((24 * 7, OVERLAY_HEIGHT, OVERLAY_WIDTH), dtype=np.float32)
    conf_cube = np.zeros_like(risk_cube)
    for frame_idx in range(24 * 7):
        risk_grid, conf_grid = paint_frame(
            lats=candidate_lats,
            lons=candidate_lons,
            risk_values=normalized[frame_idx],
            confidence_values=confidence_values,
        )
        risk_cube[frame_idx] = risk_grid
        conf_cube[frame_idx] = conf_grid

    np.savez_compressed(
        OVERLAY_NPZ_PATH,
        risk=risk_cube,
        confidence=conf_cube,
        frames=np.array(weekly_frame_labels()),
    )
    OVERLAY_JSON_PATH.write_text(
        json.dumps(
            {
                "timeline_type": "weekly_cycle",
                "month": int(args.month),
                "zoom_min": 3,
                "zoom_max": 9,
                "center_lat": 39.5,
                "center_lon": -98.35,
                "lat_min": LAT_MIN,
                "lat_max": LAT_MAX,
                "lon_min": LON_MIN,
                "lon_max": LON_MAX,
                "model_version": bundle.get("model_version", "0.1.0"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OVERLAY_NPZ_PATH}")
    print(f"wrote {OVERLAY_JSON_PATH}")


if __name__ == "__main__":
    main()
