"""Model bundle load/save and prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib

from .constants import FEATURE_COLUMNS, INTENSITY_TARGET_COLUMNS, SUPPORTED_QUANTILES
from .features import to_feature_vector


class _ProbabilityModel(Protocol):
    def predict_proba(self, X: list[list[float]]) -> Any:
        ...


class _RegressionModel(Protocol):
    def predict(self, X: list[list[float]]) -> Any:
        ...


@dataclass
class ModelBundle:
    ri_classifier: _ProbabilityModel
    intensity_models: dict[str, dict[str, _RegressionModel]]
    metadata: dict[str, Any]
    feature_columns: list[str]

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "ri_classifier": self.ri_classifier,
            "intensity_models": self.intensity_models,
            "metadata": self.metadata,
            "feature_columns": self.feature_columns,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.to_payload(), target)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ModelBundle":
        required = {"ri_classifier", "intensity_models", "metadata", "feature_columns"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Invalid model bundle payload. Missing keys: {sorted(missing)}")

        feature_columns = list(payload["feature_columns"])
        if feature_columns != FEATURE_COLUMNS:
            raise ValueError("Model bundle feature columns do not match service feature contract")

        intensity_models = payload["intensity_models"]
        for horizon in INTENSITY_TARGET_COLUMNS:
            if horizon not in intensity_models:
                raise ValueError(f"Missing intensity horizon: {horizon}")
            for quantile in SUPPORTED_QUANTILES:
                key = f"{quantile:.1f}"
                if key not in intensity_models[horizon]:
                    raise ValueError(
                        f"Missing quantile model for horizon={horizon} quantile={key}"
                    )

        return cls(
            ri_classifier=payload["ri_classifier"],
            intensity_models=intensity_models,
            metadata=dict(payload["metadata"]),
            feature_columns=feature_columns,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        loaded = joblib.load(path)
        return cls.from_payload(loaded)

    def predict(self, feature_row: dict[str, float]) -> dict[str, Any]:
        row_vector = to_feature_vector(feature_row)
        batch = [row_vector]

        ri_probability = float(self.ri_classifier.predict_proba(batch)[0][1])
        warnings: list[str] = []
        quantiles: dict[str, dict[str, float]] = {}

        for horizon, models in self.intensity_models.items():
            p10 = float(models["0.1"].predict(batch)[0])
            p50 = float(models["0.5"].predict(batch)[0])
            p90 = float(models["0.9"].predict(batch)[0])

            sorted_values = sorted([p10, p50, p90])
            if sorted_values != [p10, p50, p90]:
                warnings.append(f"Non-monotonic quantiles corrected for horizon {horizon}")

            quantiles[horizon] = {
                "p10": sorted_values[0],
                "p50": sorted_values[1],
                "p90": sorted_values[2],
            }

        return {
            "ri_probability_24h": ri_probability,
            "vmax_quantiles_kt": quantiles,
            "warnings": warnings,
            "model_version": self.model_version,
        }
