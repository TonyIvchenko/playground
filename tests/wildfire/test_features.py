import pytest

from wildfire_service.features import build_feature_row, day_features
from wildfire_service.schemas import PredictRequest


def sample_payload(conditions=None):
    return PredictRequest.model_validate(
        {
            "region_id": "sierra-foothills",
            "location": {"lat": 38.5, "lon": -121.1},
            "forecast_date": "2026-08-25",
            "conditions": conditions,
        }
    )


def test_build_feature_row_uses_baseline_when_conditions_missing():
    row = build_feature_row(sample_payload())
    day, sin_doy, cos_doy = day_features(sample_payload().forecast_date)

    assert row["day_of_year"] == float(day)
    assert row["sin_doy"] == pytest.approx(sin_doy)
    assert row["cos_doy"] == pytest.approx(cos_doy)
    assert 0.0 <= row["drought_index"] <= 1.0


def test_build_feature_row_applies_condition_overrides():
    payload = sample_payload(
        conditions={
            "temp_c": 39.0,
            "relative_humidity_pct": 13.0,
            "wind_speed_kph": 42.0,
            "drought_index": 0.91,
        }
    )
    row = build_feature_row(payload)

    assert row["temp_c"] == 39.0
    assert row["relative_humidity_pct"] == 13.0
    assert row["wind_speed_kph"] == 42.0
    assert row["drought_index"] == 0.91
