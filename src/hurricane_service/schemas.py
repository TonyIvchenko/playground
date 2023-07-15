"""Data contracts for hurricane intensity-risk inference."""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class StormState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    vmax_kt: float = Field(..., ge=0.0)
    mslp_mb: float = Field(..., ge=800.0, le=1100.0)
    motion_dir_deg: float = Field(..., ge=0.0, lt=360.0)
    motion_speed_kt: float = Field(..., ge=0.0)


class HistoryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hours_ago: int = Field(..., ge=1, le=72)
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    vmax_kt: float = Field(..., ge=0.0)
    mslp_mb: float = Field(..., ge=800.0, le=1100.0)


class EnvironmentFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sst_c: float
    ohc_kj_cm2: float
    shear_200_850_kt: float
    midlevel_rh_pct: float = Field(..., ge=0.0, le=100.0)
    vorticity_850_s_1: float = Field(..., alias="vorticity_850_s-1")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storm_id: str = Field(..., min_length=2, max_length=32)
    issue_time: datetime
    storm_state: StormState
    history_24h: List[HistoryPoint] = Field(..., min_length=3)
    environment: EnvironmentFeatures


class QuantileTriplet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p10: float
    p50: float
    p90: float


class IntensityQuantiles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    h24: QuantileTriplet = Field(..., alias="24h")
    h48: QuantileTriplet = Field(..., alias="48h")


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    storm_id: str
    model_version: str
    ri_probability_24h: float
    vmax_quantiles_kt: IntensityQuantiles
    warnings: List[str]
