from hurricane_service.constants import FEATURE_COLUMNS
from hurricane_service.model_bundle import ModelBundle


class FakeClassifier:
    def predict_proba(self, rows):
        assert len(rows) == 1
        return [[0.2, 0.8]]


class FakeRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, rows):
        assert len(rows) == 1
        return [self.value]


def test_model_bundle_predict_sorts_non_monotonic_quantiles():
    bundle = ModelBundle(
        ri_classifier=FakeClassifier(),
        intensity_models={
            "24h": {
                "0.1": FakeRegressor(90.0),
                "0.5": FakeRegressor(80.0),
                "0.9": FakeRegressor(100.0),
            },
            "48h": {
                "0.1": FakeRegressor(95.0),
                "0.5": FakeRegressor(105.0),
                "0.9": FakeRegressor(115.0),
            },
        },
        metadata={"model_version": "2026.03.v1"},
        feature_columns=list(FEATURE_COLUMNS),
    )

    feature_row = {name: 1.0 for name in FEATURE_COLUMNS}
    output = bundle.predict(feature_row)

    assert output["ri_probability_24h"] == 0.8
    assert output["vmax_quantiles_kt"]["24h"] == {"p10": 80.0, "p50": 90.0, "p90": 100.0}
    assert output["vmax_quantiles_kt"]["48h"] == {"p10": 95.0, "p50": 105.0, "p90": 115.0}
    assert output["model_version"] == "2026.03.v1"
    assert output["warnings"]


def test_model_bundle_payload_round_trip(tmp_path):
    bundle = ModelBundle(
        ri_classifier=FakeClassifier(),
        intensity_models={
            "24h": {"0.1": FakeRegressor(70), "0.5": FakeRegressor(80), "0.9": FakeRegressor(90)},
            "48h": {"0.1": FakeRegressor(72), "0.5": FakeRegressor(82), "0.9": FakeRegressor(92)},
        },
        metadata={"model_version": "demo"},
        feature_columns=list(FEATURE_COLUMNS),
    )

    target = tmp_path / "model_bundle.joblib"
    bundle.save(target)

    loaded = ModelBundle.load(target)
    assert loaded.model_version == "demo"
    assert loaded.feature_columns == FEATURE_COLUMNS
