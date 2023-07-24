"""Shared constants for wildfire ignition-risk training and inference."""

FEATURE_COLUMNS = [
    "lat",
    "lon",
    "day_of_year",
    "sin_doy",
    "cos_doy",
    "temp_c",
    "relative_humidity_pct",
    "wind_speed_kph",
    "precip_7d_mm",
    "drought_index",
    "fuel_moisture_pct",
    "vegetation_dryness",
    "human_activity_index",
    "elevation_m",
    "slope_deg",
]

TARGET_COLUMN = "ignition_next_24h"
DEFAULT_MODEL_FILENAME = "model_bundle.joblib"

RISK_DRIVER_FEATURES = [
    "temp_c",
    "relative_humidity_pct",
    "wind_speed_kph",
    "precip_7d_mm",
    "drought_index",
    "fuel_moisture_pct",
    "vegetation_dryness",
    "human_activity_index",
    "slope_deg",
]

RISK_DIRECTIONS = {
    "temp_c": 1.0,
    "relative_humidity_pct": -1.0,
    "wind_speed_kph": 1.0,
    "precip_7d_mm": -1.0,
    "drought_index": 1.0,
    "fuel_moisture_pct": -1.0,
    "vegetation_dryness": 1.0,
    "human_activity_index": 1.0,
    "slope_deg": 1.0,
}
