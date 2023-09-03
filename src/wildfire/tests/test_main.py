from pathlib import Path
import sys

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wildfire.main import load_settings, predict


def test_load_settings_defaults_and_port_fallback():
    defaults = load_settings({})
    assert defaults.port == 8010

    from_port = load_settings({"PORT": "7860"})
    assert from_port.port == 7860


def test_load_settings_invalid_port_raises():
    with pytest.raises(ValueError):
        load_settings({"API_PORT": "bad"})


def test_predict_output_shape_and_values():
    result = predict(
        region_id="norcal",
        temp_c=34,
        humidity_pct=22,
        wind_kph=28,
        drought_index=0.82,
        model_version="test-v1",
    )
    assert result["region_id"] == "norcal"
    assert result["model_version"] == "test-v1"
    assert 0.0 <= result["ignition_probability_24h"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "high", "extreme"}
