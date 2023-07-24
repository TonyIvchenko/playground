from wildfire_service.constants import FEATURE_COLUMNS
from wildfire_service.model_bundle import ModelBundle


class FakeClassifier:
    def predict_proba(self, rows):
        assert len(rows) == 1
        return [[0.32, 0.68]]


def test_model_bundle_predict_returns_probability_and_drivers():
    metadata = {
        "model_version": "2026.03.v1",
        "feature_importances": {feature: 0.1 for feature in FEATURE_COLUMNS},
        "feature_mean": {feature: 0.0 for feature in FEATURE_COLUMNS},
        "feature_std": {feature: 1.0 for feature in FEATURE_COLUMNS},
    }
    bundle = ModelBundle(
        ignition_classifier=FakeClassifier(),
        metadata=metadata,
        feature_columns=list(FEATURE_COLUMNS),
    )

    row = {feature: 1.0 for feature in FEATURE_COLUMNS}
    result = bundle.predict(row)

    assert result["ignition_probability_24h"] == 0.68
    assert result["risk_level"] == "extreme"
    assert len(result["top_drivers"]) == 3
    assert result["model_version"] == "2026.03.v1"


def test_model_bundle_round_trip(tmp_path):
    metadata = {
        "model_version": "demo",
        "feature_importances": {feature: 0.1 for feature in FEATURE_COLUMNS},
        "feature_mean": {feature: 0.0 for feature in FEATURE_COLUMNS},
        "feature_std": {feature: 1.0 for feature in FEATURE_COLUMNS},
    }
    bundle = ModelBundle(
        ignition_classifier=FakeClassifier(),
        metadata=metadata,
        feature_columns=list(FEATURE_COLUMNS),
    )

    target = tmp_path / "model_bundle.joblib"
    bundle.save(target)

    loaded = ModelBundle.load(target)
    assert loaded.model_version == "demo"
    assert loaded.feature_columns == FEATURE_COLUMNS
