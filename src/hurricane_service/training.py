"""Training utilities shared by scripts and runtime helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

from .constants import (
    FEATURE_COLUMNS,
    INTENSITY_TARGET_COLUMNS,
    RI_TARGET_COLUMN,
    SUPPORTED_QUANTILES,
)
from .model_bundle import ModelBundle


def build_demo_dataset(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    lat = rng.uniform(8.0, 35.0, size=rows)
    lon = rng.uniform(-95.0, -10.0, size=rows)
    vmax = rng.uniform(20.0, 120.0, size=rows)
    mslp = 1020.0 - (vmax * 0.35) + rng.normal(0, 5.0, size=rows)
    motion_dir = rng.uniform(0, 359.9, size=rows)
    motion_speed = rng.uniform(1.0, 25.0, size=rows)

    lag24_vmax = np.maximum(10.0, vmax - rng.uniform(-10, 25, size=rows))
    lag12_vmax = np.maximum(10.0, vmax - rng.uniform(-6, 14, size=rows))
    lag6_vmax = np.maximum(10.0, vmax - rng.uniform(-4, 9, size=rows))

    lag24_mslp = mslp + rng.uniform(-6, 10, size=rows)
    lag12_mslp = mslp + rng.uniform(-4, 7, size=rows)
    lag6_mslp = mslp + rng.uniform(-2, 5, size=rows)

    sst = rng.uniform(24.0, 31.5, size=rows)
    ohc = rng.uniform(10.0, 140.0, size=rows)
    shear = rng.uniform(2.0, 35.0, size=rows)
    rh = rng.uniform(35.0, 95.0, size=rows)
    vort = rng.uniform(0.00002, 0.00035, size=rows)

    delta24_vmax = vmax - lag24_vmax
    delta24_mslp = mslp - lag24_mslp

    ri_signal = (
        0.10 * delta24_vmax
        + 0.09 * (sst - 26.0)
        + 0.015 * (ohc - 40.0)
        - 0.08 * shear
        + 0.04 * (rh - 55.0)
    )
    ri_prob = 1.0 / (1.0 + np.exp(-(ri_signal / 4.5)))
    ri_target = (rng.uniform(0, 1, size=rows) < ri_prob).astype(int)

    vmax24 = vmax + (ri_target * rng.uniform(8, 22, size=rows)) + rng.normal(0, 6, size=rows)
    vmax48 = vmax24 + (ri_target * rng.uniform(5, 18, size=rows)) + rng.normal(0, 7, size=rows)

    return pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "vmax_kt": vmax,
            "mslp_mb": mslp,
            "motion_dir_deg": motion_dir,
            "motion_speed_kt": motion_speed,
            "lag6_vmax_kt": lag6_vmax,
            "lag12_vmax_kt": lag12_vmax,
            "lag24_vmax_kt": lag24_vmax,
            "lag6_mslp_mb": lag6_mslp,
            "lag12_mslp_mb": lag12_mslp,
            "lag24_mslp_mb": lag24_mslp,
            "delta24_vmax_kt": delta24_vmax,
            "delta24_mslp_mb": delta24_mslp,
            "sst_c": sst,
            "ohc_kj_cm2": ohc,
            "shear_200_850_kt": shear,
            "midlevel_rh_pct": rh,
            "vorticity_850_s-1": vort,
            RI_TARGET_COLUMN: ri_target,
            INTENSITY_TARGET_COLUMNS["24h"]: vmax24,
            INTENSITY_TARGET_COLUMNS["48h"]: vmax48,
        }
    )


def validate_columns(frame: pd.DataFrame) -> None:
    expected = FEATURE_COLUMNS + [RI_TARGET_COLUMN] + list(INTENSITY_TARGET_COLUMNS.values())
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def train_models(frame: pd.DataFrame, model_version: str) -> tuple[ModelBundle, dict[str, float]]:
    validate_columns(frame)

    X = frame[FEATURE_COLUMNS].astype(float).to_numpy()
    y_ri = frame[RI_TARGET_COLUMN].astype(int).to_numpy()
    indices = np.arange(len(frame))

    train_idx, test_idx, y_train, y_test = train_test_split(
        indices,
        y_ri,
        test_size=0.2,
        random_state=42,
        stratify=y_ri,
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
    )
    classifier.fit(X_train, y_train)
    ri_probs = classifier.predict_proba(X_test)[:, 1]

    intensity_models: dict[str, dict[str, GradientBoostingRegressor]] = {}
    metrics: dict[str, float] = {
        "ri_roc_auc": float(roc_auc_score(y_test, ri_probs)),
    }

    for horizon, target_column in INTENSITY_TARGET_COLUMNS.items():
        y = frame[target_column].astype(float).to_numpy()
        y_train_reg = y[train_idx]
        y_test_reg = y[test_idx]
        horizon_models: dict[str, GradientBoostingRegressor] = {}

        for quantile in SUPPORTED_QUANTILES:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=quantile,
                n_estimators=250,
                learning_rate=0.05,
                random_state=42,
            )
            model.fit(X_train, y_train_reg)
            predictions = model.predict(X_test)
            metrics[f"mae_{horizon}_q{int(quantile * 100)}"] = float(
                mean_absolute_error(y_test_reg, predictions)
            )
            horizon_models[f"{quantile:.1f}"] = model

        intensity_models[horizon] = horizon_models

    metadata = {
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(frame.shape[0]),
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
    }

    bundle = ModelBundle(
        ri_classifier=classifier,
        intensity_models=intensity_models,
        metadata=metadata,
        feature_columns=list(FEATURE_COLUMNS),
    )
    return bundle, metrics
