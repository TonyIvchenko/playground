import pytest

from hurricane_service.features import build_feature_row
from hurricane_service.schemas import PredictRequest


def sample_payload(history_24h=None):
    return PredictRequest.model_validate(
        {
            "storm_id": "AL09",
            "issue_time": "2026-08-25T12:00:00Z",
            "storm_state": {
                "lat": 22.4,
                "lon": -71.8,
                "vmax_kt": 70,
                "mslp_mb": 987,
                "motion_dir_deg": 300,
                "motion_speed_kt": 12,
            },
            "history_24h": history_24h
            or [
                {"hours_ago": 6, "lat": 22.0, "lon": -70.7, "vmax_kt": 65, "mslp_mb": 992},
                {
                    "hours_ago": 12,
                    "lat": 21.6,
                    "lon": -69.8,
                    "vmax_kt": 60,
                    "mslp_mb": 996,
                },
                {
                    "hours_ago": 24,
                    "lat": 20.9,
                    "lon": -68.4,
                    "vmax_kt": 55,
                    "mslp_mb": 1000,
                },
            ],
            "environment": {
                "sst_c": 29.1,
                "ohc_kj_cm2": 68.0,
                "shear_200_850_kt": 9.0,
                "midlevel_rh_pct": 66.0,
                "vorticity_850_s-1": 0.00018,
            },
        }
    )


def test_build_feature_row_returns_expected_values():
    row = build_feature_row(sample_payload())

    assert row["lat"] == 22.4
    assert row["lag24_vmax_kt"] == 55.0
    assert row["delta24_vmax_kt"] == 15.0
    assert row["delta24_mslp_mb"] == -13.0
    assert row["vorticity_850_s-1"] == 0.00018


def test_build_feature_row_requires_6_12_24_history_points():
    payload = sample_payload(
        history_24h=[
            {"hours_ago": 6, "lat": 22.0, "lon": -70.7, "vmax_kt": 65, "mslp_mb": 992},
            {"hours_ago": 12, "lat": 21.6, "lon": -69.8, "vmax_kt": 60, "mslp_mb": 996},
            {"hours_ago": 18, "lat": 21.2, "lon": -69.0, "vmax_kt": 58, "mslp_mb": 998},
        ]
    )

    with pytest.raises(ValueError):
        build_feature_row(payload)
