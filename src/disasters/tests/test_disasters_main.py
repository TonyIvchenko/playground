from src.disasters.main import (
    HIRICAINES_MODEL_VERSION,
    WILDFIRES_MODEL_VERSION,
    predict_hiricaines,
    predict_wildfires,
)


def test_predict_wildfires_shape_and_values():
    result = predict_wildfires(
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
    assert result["model_version"] == WILDFIRES_MODEL_VERSION
    assert 0.0 <= result["ignition_probability_24h"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "high", "extreme"}


def test_predict_hiricaines_shape_and_values():
    result = predict_hiricaines(
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
    assert result["model_version"] == HIRICAINES_MODEL_VERSION
    assert 0.0 <= result["ri_probability_24h"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "high", "extreme"}
