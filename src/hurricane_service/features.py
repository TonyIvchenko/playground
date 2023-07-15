"""Feature engineering for hurricane intensity-risk inference."""

from __future__ import annotations

from typing import Dict

from .constants import FEATURE_COLUMNS
from .schemas import HistoryPoint, PredictRequest


_REQUIRED_HISTORY_HOURS = (6, 12, 24)


def _index_history(points: list[HistoryPoint]) -> Dict[int, HistoryPoint]:
    indexed: Dict[int, HistoryPoint] = {}
    for point in points:
        indexed[point.hours_ago] = point

    missing = [hour for hour in _REQUIRED_HISTORY_HOURS if hour not in indexed]
    if missing:
        raise ValueError(f"history_24h missing required snapshots: {missing}")
    return indexed


def build_feature_row(payload: PredictRequest) -> dict[str, float]:
    current = payload.storm_state
    history = _index_history(payload.history_24h)

    row = {
        "lat": float(current.lat),
        "lon": float(current.lon),
        "vmax_kt": float(current.vmax_kt),
        "mslp_mb": float(current.mslp_mb),
        "motion_dir_deg": float(current.motion_dir_deg),
        "motion_speed_kt": float(current.motion_speed_kt),
        "lag6_vmax_kt": float(history[6].vmax_kt),
        "lag12_vmax_kt": float(history[12].vmax_kt),
        "lag24_vmax_kt": float(history[24].vmax_kt),
        "lag6_mslp_mb": float(history[6].mslp_mb),
        "lag12_mslp_mb": float(history[12].mslp_mb),
        "lag24_mslp_mb": float(history[24].mslp_mb),
        "delta24_vmax_kt": float(current.vmax_kt - history[24].vmax_kt),
        "delta24_mslp_mb": float(current.mslp_mb - history[24].mslp_mb),
        "sst_c": float(payload.environment.sst_c),
        "ohc_kj_cm2": float(payload.environment.ohc_kj_cm2),
        "shear_200_850_kt": float(payload.environment.shear_200_850_kt),
        "midlevel_rh_pct": float(payload.environment.midlevel_rh_pct),
        "vorticity_850_s-1": float(payload.environment.vorticity_850_s_1),
    }

    missing_columns = [name for name in FEATURE_COLUMNS if name not in row]
    if missing_columns:
        raise ValueError(f"feature row missing columns: {missing_columns}")

    return row


def to_feature_vector(feature_row: dict[str, float]) -> list[float]:
    return [float(feature_row[name]) for name in FEATURE_COLUMNS]
