"""Train wildfire ignition-risk classifier and serialize a model bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wildfire_service.constants import FEATURE_COLUMNS, TARGET_COLUMN
from wildfire_service.model_bundle import ModelBundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT / "data" / "sample" / "wildfire_training_sample.csv",
        help="Training dataset CSV path",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "wildfire" / "model_bundle.joblib",
        help="Output model bundle path",
    )
    parser.add_argument(
        "--model-version",
        default="2026.03.v1",
        help="Model version label written into bundle metadata",
    )
    parser.add_argument(
        "--allow-demo-data",
        action="store_true",
        help="Generate synthetic demo data if --input-csv does not exist",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1200,
        help="Number of rows to generate when --allow-demo-data is enabled",
    )
    return parser.parse_args()


def _clamp(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.minimum(np.maximum(values, lower), upper)


def build_demo_dataset(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    lat = rng.uniform(24.5, 49.3, size=rows)
    lon = rng.uniform(-124.8, -67.0, size=rows)
    day_of_year = rng.integers(1, 366, size=rows)

    phase = (2.0 * np.pi * day_of_year) / 365.0
    sin_doy = np.sin(phase)
    cos_doy = np.cos(phase)

    lat_shift = np.abs(lat - 37.0)

    temp = 17.0 + 13.0 * np.sin(phase - 1.1) - 0.12 * lat_shift + rng.normal(0, 2.7, size=rows)
    humidity = 58.0 - 19.0 * np.sin(phase - 1.1) + 0.20 * lat_shift + rng.normal(0, 8.0, size=rows)
    wind = 11.0 + 7.0 * np.abs(np.sin(np.radians(lon * 1.7))) + 2.0 * np.cos(phase)
    wind += rng.normal(0, 3.0, size=rows)
    precip = 10.0 + 11.0 * np.cos(phase - 0.3) - 0.03 * np.abs(lon + 100.0)
    precip += rng.normal(0, 4.0, size=rows)
    drought = 0.45 + 0.28 * np.sin(phase - 2.1) + 0.002 * np.abs(lon + 100.0)
    drought += rng.normal(0, 0.08, size=rows)

    humidity = _clamp(humidity, 8.0, 98.0)
    wind = _clamp(wind, 0.0, 70.0)
    precip = _clamp(precip, 0.0, 90.0)
    drought = _clamp(drought, 0.0, 1.0)

    fuel_moisture = 72.0 - 34.0 * drought - 0.30 * temp + 0.16 * humidity
    fuel_moisture += rng.normal(0, 4.5, size=rows)
    fuel_moisture = _clamp(fuel_moisture, 2.0, 90.0)

    vegetation_dryness = _clamp(1.0 - (fuel_moisture / 100.0) + 0.22 * drought, 0.0, 1.0)

    human_activity = 0.40 + 0.23 * np.cos(np.radians(lat * 2.0)) * np.cos(np.radians(lon * 0.6))
    human_activity += rng.normal(0, 0.07, size=rows)
    human_activity = _clamp(human_activity, 0.0, 1.0)

    elevation = (
        650.0
        + 580.0 * np.sin(np.radians(lat * 2.7))
        + 330.0 * np.cos(np.radians(lon * 1.4))
        + rng.normal(0, 120.0, size=rows)
    )
    elevation = _clamp(elevation, -200.0, 4000.0)

    slope = 6.0 + 18.0 * np.abs(np.sin(np.radians(lat * 3.1)) * np.cos(np.radians(lon * 1.9)))
    slope += rng.normal(0, 2.0, size=rows)
    slope = _clamp(slope, 0.0, 45.0)

    signal = (
        0.085 * (temp - 22.0)
        - 0.030 * (humidity - 35.0)
        + 0.070 * (wind - 16.0)
        - 0.018 * (precip - 3.0)
        + 2.2 * (drought - 0.5)
        - 0.050 * (fuel_moisture - 15.0)
        + 2.0 * (vegetation_dryness - 0.5)
        + 1.3 * (human_activity - 0.5)
        + 0.018 * (slope - 12.0)
        + 0.8 * sin_doy
        - 1.9
    )

    probability = 1.0 / (1.0 + np.exp(-signal))
    ignition = (rng.uniform(0, 1, size=rows) < probability).astype(int)

    # Keep demo data trainable for small row counts.
    min_positives = max(4, rows // 40)
    positive_count = int(ignition.sum())
    if positive_count < min_positives:
        top_idx = np.argsort(probability)[-min_positives:]
        ignition[top_idx] = 1
    if int(ignition.sum()) >= rows:
        low_idx = np.argsort(probability)[:max(1, rows // 10)]
        ignition[low_idx] = 0

    frame = pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "day_of_year": day_of_year.astype(float),
            "sin_doy": sin_doy,
            "cos_doy": cos_doy,
            "temp_c": temp,
            "relative_humidity_pct": humidity,
            "wind_speed_kph": wind,
            "precip_7d_mm": precip,
            "drought_index": drought,
            "fuel_moisture_pct": fuel_moisture,
            "vegetation_dryness": vegetation_dryness,
            "human_activity_index": human_activity,
            "elevation_m": elevation,
            "slope_deg": slope,
            TARGET_COLUMN: ignition,
        }
    )
    return frame


def load_frame(path: Path, allow_demo_data: bool, rows: int) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    if not allow_demo_data:
        raise FileNotFoundError(f"Input dataset not found: {path}")

    frame = build_demo_dataset(rows=rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def validate_columns(frame: pd.DataFrame) -> None:
    expected = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def train_model(frame: pd.DataFrame) -> tuple[ModelBundle, dict[str, float]]:
    validate_columns(frame)

    X = frame[FEATURE_COLUMNS].astype(float).to_numpy()
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Dataset must contain at least two classes for training")
    if int(counts.min()) < 2:
        raise ValueError("Dataset must contain at least two samples per class for stratified split")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classifier = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    classifier.fit(X_train, y_train)

    test_prob = classifier.predict_proba(X_test)[:, 1]
    roc_auc = 0.5
    if len(np.unique(y_test)) > 1:
        roc_auc = float(roc_auc_score(y_test, test_prob))

    metrics = {
        "roc_auc": roc_auc,
        "brier": float(brier_score_loss(y_test, test_prob)),
        "positive_rate": float(y.mean()),
    }

    feature_mean = frame[FEATURE_COLUMNS].mean().to_dict()
    feature_std = frame[FEATURE_COLUMNS].std(ddof=0).replace(0.0, 1.0).to_dict()

    metadata = {
        "model_version": "",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(frame.shape[0]),
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "feature_importances": {
            feature: float(importance)
            for feature, importance in zip(FEATURE_COLUMNS, classifier.feature_importances_)
        },
        "feature_mean": {k: float(v) for k, v in feature_mean.items()},
        "feature_std": {k: float(v) for k, v in feature_std.items()},
    }

    bundle = ModelBundle(
        ignition_classifier=classifier,
        metadata=metadata,
        feature_columns=list(FEATURE_COLUMNS),
    )
    return bundle, metrics


def main() -> None:
    args = parse_args()
    frame = load_frame(args.input_csv, allow_demo_data=args.allow_demo_data, rows=args.rows)
    bundle, metrics = train_model(frame)
    bundle.metadata["model_version"] = args.model_version

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.save(args.output_path)

    print(f"Saved model bundle: {args.output_path}")
    print(f"Rows used: {len(frame)}")
    for key, value in sorted(metrics.items()):
        print(f"{key}={value:.4f}")


if __name__ == "__main__":
    main()
