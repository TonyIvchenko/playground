"""Model bundle loading and scoring utilities for wildfire service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib

from .constants import FEATURE_COLUMNS, RISK_DIRECTIONS, RISK_DRIVER_FEATURES
from .features import to_feature_vector


class _ProbabilityModel(Protocol):
    def predict_proba(self, X: list[list[float]]) -> Any:
        ...


@dataclass
class ModelBundle:
    """Runtime container for wildfire probability model and metadata."""

    ignition_classifier: _ProbabilityModel
    metadata: dict[str, Any]
    feature_columns: list[str]

    @property
    def model_version(self) -> str:
        version = self.metadata.get("model_version")
        if not version:
            return "unknown"
        return str(version)

    def to_payload(self) -> dict[str, Any]:
        return {
            "ignition_classifier": self.ignition_classifier,
            "metadata": self.metadata,
            "feature_columns": self.feature_columns,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.to_payload(), target)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ModelBundle":
        required_keys = {"ignition_classifier", "metadata", "feature_columns"}
        missing = required_keys.difference(payload)
        if missing:
            raise ValueError(f"Invalid model bundle payload. Missing keys: {sorted(missing)}")

        feature_columns = list(payload["feature_columns"])
        if feature_columns != FEATURE_COLUMNS:
            raise ValueError("Model bundle feature columns do not match service feature contract")

        metadata = dict(payload["metadata"])
        for field in ("feature_importances", "feature_mean", "feature_std"):
            if field not in metadata:
                raise ValueError(f"Model bundle metadata missing required field: {field}")

        return cls(
            ignition_classifier=payload["ignition_classifier"],
            metadata=metadata,
            feature_columns=feature_columns,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        loaded = joblib.load(path)
        return cls.from_payload(loaded)

    def _risk_level(self, probability: float) -> str:
        if probability < 0.20:
            return "low"
        if probability < 0.40:
            return "moderate"
        if probability < 0.65:
            return "high"
        return "extreme"

    def _top_drivers(self, feature_row: dict[str, float], limit: int = 3) -> list[dict[str, float | str]]:
        importances = {
            key: float(value) for key, value in self.metadata.get("feature_importances", {}).items()
        }
        means = {key: float(value) for key, value in self.metadata.get("feature_mean", {}).items()}
        stds = {key: float(value) for key, value in self.metadata.get("feature_std", {}).items()}

        scored: list[dict[str, float | str]] = []
        for feature in RISK_DRIVER_FEATURES:
            value = float(feature_row[feature])
            mean = means.get(feature, 0.0)
            std = stds.get(feature, 1.0)
            if std <= 1e-6:
                std = 1.0

            z_score = (value - mean) / std
            direction_weight = RISK_DIRECTIONS.get(feature, 1.0)
            signed_effect = z_score * direction_weight
            score = abs(signed_effect) * importances.get(feature, 0.0)
            scored.append(
                {
                    "feature": feature,
                    "value": value,
                    "direction": "increase" if signed_effect >= 0 else "decrease",
                    "score": float(score),
                }
            )

        scored.sort(key=lambda item: float(item["score"]), reverse=True)
        return scored[:limit]

    def predict(self, feature_row: dict[str, float]) -> dict[str, Any]:
        row_vector = to_feature_vector(feature_row)
        vector = [row_vector]

        probability = float(self.ignition_classifier.predict_proba(vector)[0][1])
        return {
            "ignition_probability_24h": probability,
            "risk_level": self._risk_level(probability),
            "top_drivers": self._top_drivers(feature_row),
            "warnings": [],
            "model_version": self.model_version,
        }
