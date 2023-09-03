from pathlib import Path
import sys

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hurricane.main import load_settings, predict


def test_load_settings_defaults_and_port_fallback():
    defaults = load_settings({})
    assert defaults.port == 8000

    from_port = load_settings({"PORT": "7860"})
    assert from_port.port == 7860


def test_load_settings_invalid_port_raises():
    with pytest.raises(ValueError):
        load_settings({"API_PORT": "bad"})


def test_predict_output_shape_and_values():
    result = predict(
        storm_id="AL09",
        vmax_kt=70,
        shear_kt=10,
        sst_c=29.2,
        model_version="test-v1",
    )
    assert result["storm_id"] == "AL09"
    assert result["model_version"] == "test-v1"
    assert 0.0 <= result["ri_probability_24h"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "high", "extreme"}
