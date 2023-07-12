"""Shared constants for hurricane intensity-risk training and inference."""

FEATURE_COLUMNS = [
    "lat",
    "lon",
    "vmax_kt",
    "mslp_mb",
    "motion_dir_deg",
    "motion_speed_kt",
    "lag6_vmax_kt",
    "lag12_vmax_kt",
    "lag24_vmax_kt",
    "lag6_mslp_mb",
    "lag12_mslp_mb",
    "lag24_mslp_mb",
    "delta24_vmax_kt",
    "delta24_mslp_mb",
    "sst_c",
    "ohc_kj_cm2",
    "shear_200_850_kt",
    "midlevel_rh_pct",
    "vorticity_850_s-1",
]

RI_TARGET_COLUMN = "ri_next_24h"
INTENSITY_TARGET_COLUMNS = {
    "24h": "vmax_tplus24_kt",
    "48h": "vmax_tplus48_kt",
}

SUPPORTED_HORIZONS = ("24h", "48h")
SUPPORTED_QUANTILES = (0.1, 0.5, 0.9)
DEFAULT_MODEL_FILENAME = "model_bundle.joblib"
