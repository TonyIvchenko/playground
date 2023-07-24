"""Feature engineering for wildfire ignition-risk inference."""

from __future__ import annotations

from datetime import date
import math

from .constants import FEATURE_COLUMNS
from .schemas import Conditions, PredictRequest


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _season_phase(day_of_year: int) -> float:
    return (2.0 * math.pi * day_of_year) / 365.0


def _baseline_conditions(lat: float, lon: float, day_of_year: int) -> dict[str, float]:
    phase = _season_phase(day_of_year)
    lat_shift = abs(lat - 37.0)

    temp_c = 17.0 + 13.0 * math.sin(phase - 1.1) - 0.12 * lat_shift
    humidity = 58.0 - 19.0 * math.sin(phase - 1.1) + 0.20 * lat_shift
    wind = 11.0 + 7.0 * abs(math.sin(math.radians(lon * 1.7))) + 2.0 * math.cos(phase)
    precip = 10.0 + 11.0 * math.cos(phase - 0.3) - 0.03 * abs(lon + 100.0)
    drought = 0.45 + 0.28 * math.sin(phase - 2.1) + 0.002 * abs(lon + 100.0)

    humidity = _clamp(humidity, 8.0, 98.0)
    wind = _clamp(wind, 0.0, 70.0)
    precip = _clamp(precip, 0.0, 90.0)
    drought = _clamp(drought, 0.0, 1.0)

    fuel_moisture = 72.0 - 34.0 * drought - 0.30 * temp_c + 0.16 * humidity
    fuel_moisture = _clamp(fuel_moisture, 2.0, 90.0)
    vegetation_dryness = _clamp(1.0 - (fuel_moisture / 100.0) + 0.22 * drought, 0.0, 1.0)

    human_activity = _clamp(
        0.40 + 0.23 * math.cos(math.radians(lat * 2.0)) * math.cos(math.radians(lon * 0.6)),
        0.0,
        1.0,
    )
    elevation = _clamp(
        650.0
        + 580.0 * math.sin(math.radians(lat * 2.7))
        + 330.0 * math.cos(math.radians(lon * 1.4)),
        -200.0,
        4000.0,
    )
    slope = _clamp(
        6.0 + 18.0 * abs(math.sin(math.radians(lat * 3.1)) * math.cos(math.radians(lon * 1.9))),
        0.0,
        45.0,
    )

    return {
        "temp_c": float(temp_c),
        "relative_humidity_pct": float(humidity),
        "wind_speed_kph": float(wind),
        "precip_7d_mm": float(precip),
        "drought_index": float(drought),
        "fuel_moisture_pct": float(fuel_moisture),
        "vegetation_dryness": float(vegetation_dryness),
        "human_activity_index": float(human_activity),
        "elevation_m": float(elevation),
        "slope_deg": float(slope),
    }


def _merge_conditions(baseline: dict[str, float], overrides: Conditions | None) -> dict[str, float]:
    if overrides is None:
        return baseline

    override_values = overrides.model_dump(exclude_none=True)
    merged = dict(baseline)
    merged.update({key: float(value) for key, value in override_values.items()})
    return merged


def day_features(forecast_date: date) -> tuple[int, float, float]:
    day_of_year = int(forecast_date.timetuple().tm_yday)
    phase = _season_phase(day_of_year)
    return day_of_year, math.sin(phase), math.cos(phase)


def build_feature_row(payload: PredictRequest) -> dict[str, float]:
    lat = float(payload.location.lat)
    lon = float(payload.location.lon)
    day_of_year, sin_doy, cos_doy = day_features(payload.forecast_date)

    baseline = _baseline_conditions(lat=lat, lon=lon, day_of_year=day_of_year)
    merged = _merge_conditions(baseline=baseline, overrides=payload.conditions)

    row = {
        "lat": lat,
        "lon": lon,
        "day_of_year": float(day_of_year),
        "sin_doy": float(sin_doy),
        "cos_doy": float(cos_doy),
        "temp_c": merged["temp_c"],
        "relative_humidity_pct": merged["relative_humidity_pct"],
        "wind_speed_kph": merged["wind_speed_kph"],
        "precip_7d_mm": merged["precip_7d_mm"],
        "drought_index": merged["drought_index"],
        "fuel_moisture_pct": merged["fuel_moisture_pct"],
        "vegetation_dryness": merged["vegetation_dryness"],
        "human_activity_index": merged["human_activity_index"],
        "elevation_m": merged["elevation_m"],
        "slope_deg": merged["slope_deg"],
    }

    missing_columns = [name for name in FEATURE_COLUMNS if name not in row]
    if missing_columns:
        raise ValueError(f"feature row missing columns: {missing_columns}")

    return row


def to_feature_vector(feature_row: dict[str, float]) -> list[float]:
    return [float(feature_row[column]) for column in FEATURE_COLUMNS]
