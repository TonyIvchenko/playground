"""Pydantic schemas for wildfire ignition-risk inference API."""

from __future__ import annotations

from datetime import date
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)


class Conditions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temp_c: float | None = None
    relative_humidity_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    wind_speed_kph: float | None = Field(default=None, ge=0.0)
    precip_7d_mm: float | None = Field(default=None, ge=0.0)
    drought_index: float | None = Field(default=None, ge=0.0, le=1.0)
    fuel_moisture_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    vegetation_dryness: float | None = Field(default=None, ge=0.0, le=1.0)
    human_activity_index: float | None = Field(default=None, ge=0.0, le=1.0)
    elevation_m: float | None = Field(default=None, ge=-500.0, le=6000.0)
    slope_deg: float | None = Field(default=None, ge=0.0, le=90.0)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(..., min_length=2, max_length=64)
    location: Location
    forecast_date: date
    conditions: Conditions | None = None


class TopDriver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    value: float
    direction: str
    score: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str
    model_version: str
    ignition_probability_24h: float
    risk_level: str
    top_drivers: List[TopDriver]
    warnings: List[str]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    model_loaded: bool
    model_version: str | None
    detail: str | None = None
