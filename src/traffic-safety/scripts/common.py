from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math


SERVICE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SERVICE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "fars"
NOAA_RAW_DIR = DATA_DIR / "raw" / "noaa"
NOAA_ISD_LITE_DIR = NOAA_RAW_DIR / "isd-lite"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_WEATHER_DIR = PROCESSED_DIR / "weather"
WEATHER_HOURLY_DIR = PROCESSED_WEATHER_DIR / "hourly"
MODELS_DIR = SERVICE_DIR / "models"
TILES_DIR = SERVICE_DIR / "tiles"

ACCIDENTS_CLEAN_PATH = PROCESSED_DIR / "accidents_clean.csv.gz"
CANDIDATE_CELLS_PATH = PROCESSED_DIR / "candidate_cells.csv.gz"
WEEKLY_COUNTS_PATH = PROCESSED_DIR / "weekly_counts.csv.gz"
STATION_HISTORY_PATH = NOAA_RAW_DIR / "isd-history.csv"
REPRESENTATIVE_STATIONS_PATH = PROCESSED_WEATHER_DIR / "representative_stations.csv.gz"
CELL_WEATHER_STATIONS_PATH = PROCESSED_WEATHER_DIR / "cell_weather_stations.csv.gz"
WEATHER_CLIMATOLOGY_PATH = PROCESSED_WEATHER_DIR / "weather_climatology.csv.gz"
MODEL_BUNDLE_PATH = MODELS_DIR / "traffic_safety.joblib"
OVERLAY_NPZ_PATH = TILES_DIR / "overlay.npz"
OVERLAY_JSON_PATH = TILES_DIR / "overlay.json"

DEFAULT_YEARS = list(range(2016, 2025))
DEFAULT_TRAIN_YEARS = list(range(2018, 2024))
DEFAULT_EVAL_YEAR = 2024
H3_RESOLUTION = 5
NEGATIVE_RATIO = 4
RANDOM_SEED = 42
WEATHER_REPRESENTATION_H3_RES = 2

LAT_MIN = 18.0
LAT_MAX = 72.0
LON_MIN = -179.0
LON_MAX = -66.0
OVERLAY_HEIGHT = 360
OVERLAY_WIDTH = 760
CELL_PAINT_RADIUS = 2

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    NOAA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    NOAA_ISD_LITE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    WEATHER_HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)


def fars_zip_url(year: int) -> str:
    return (
        "https://static.nhtsa.gov/nhtsa/downloads/FARS/"
        f"{year}/National/FARS{year}NationalCSV.zip"
    )


def fars_zip_path(year: int) -> Path:
    return RAW_DIR / f"FARS{year}NationalCSV.zip"


def noaa_isd_lite_url(station_id: str, year: int) -> str:
    return f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{station_id}-{year}.gz"


def noaa_isd_lite_path(station_id: str, year: int) -> Path:
    return NOAA_ISD_LITE_DIR / str(year) / f"{station_id}-{year}.gz"


def month_sin_cos(month: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * float(month) / 12.0
    return math.sin(angle), math.cos(angle)


def hour_sin_cos(hour: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * float(hour) / 24.0
    return math.sin(angle), math.cos(angle)


def dow_sin_cos(day_of_week: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * float(day_of_week - 1) / 7.0
    return math.sin(angle), math.cos(angle)


def local_hour_of_week_label(frame_idx: int) -> str:
    day_idx = frame_idx // 24
    hour = frame_idx % 24
    return f"{WEEKDAY_LABELS[day_idx]} {hour:02d}:00"


def weekly_frame_labels() -> list[str]:
    return [local_hour_of_week_label(idx) for idx in range(24 * 7)]


def weekly_ticks() -> list[dict[str, int | str]]:
    return [
        {"label": weekday, "frame_idx": idx * 24}
        for idx, weekday in enumerate(WEEKDAY_LABELS)
    ]


def current_month() -> int:
    return datetime.now().month
