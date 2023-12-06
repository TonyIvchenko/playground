from src.disasters.main import (
    HURRICANE_MODEL_VERSION,
    WILDFIRE_MODEL_VERSION,
    predict_hurricane,
    predict_wildfire,
)


def test_predict_wildfire_shape_and_values():
    result = predict_wildfire(
        region_id="norcal",
        temp_c=34,
        humidity_pct=22,
        wind_kph=28,
        ffmc=92.0,
        dmc=180.0,
        drought_code=640.0,
        isi=12.0,
    )
    assert result["region_id"] == "norcal"
    assert result["model_version"] == WILDFIRE_MODEL_VERSION
    assert 0.0 <= result["ignition_probability_24h"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "high", "extreme"}


def test_predict_hurricane_shape_and_values():
    result = predict_hurricane(
        storm_id="AL09",
        vmax_kt=70,
        min_pressure_mb=980,
        lat=22.5,
        lon=-65.0,
        month=9,
        dvmax_6h=5.0,
        dpres_6h=-3.0,
    )
    assert result["storm_id"] == "AL09"
    assert result["model_version"] == HURRICANE_MODEL_VERSION
    assert 0.0 <= result["ri_probability_24h"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "high", "extreme"}
