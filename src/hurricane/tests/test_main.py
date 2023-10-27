from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hurricane.main import MODEL_VERSION, predict


def test_predict_output_shape_and_values():
    result = predict(
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
    assert result["model_version"] == MODEL_VERSION
    assert 0.0 <= result["ri_probability_24h"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "high", "extreme"}
