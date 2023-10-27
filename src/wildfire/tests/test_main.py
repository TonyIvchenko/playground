from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wildfire.main import MODEL_VERSION, predict


def test_predict_output_shape_and_values():
    result = predict(
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
    assert result["model_version"] == MODEL_VERSION
    assert 0.0 <= result["ignition_probability_24h"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "high", "extreme"}
