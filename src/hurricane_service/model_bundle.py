"""Model bundle loading and scoring utilities."""

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
    """Runtime container for classifier and quantile regressors."""

    ri_classifier: _ProbabilityModel
    intensity_models: dict[str, dict[str, _RegressionModel]]
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
        required_keys = {"ri_classifier", "intensity_models", "metadata", "feature_columns"}
        missing = required_keys.difference(payload)
        if missing:
            raise ValueError(f"Invalid model bundle payload. Missing keys: {sorted(missing)}")

        feature_columns = list(payload["feature_columns"])
        if feature_columns != FEATURE_COLUMNS:
            raise ValueError("Model bundle feature columns do not match service feature contract")

        intensity_models = payload["intensity_models"]
        for horizon in INTENSITY_TARGET_COLUMNS:
            if horizon not in intensity_models:
                raise ValueError(f"Missing intensity horizon in model bundle: {horizon}")
            for quantile in SUPPORTED_QUANTILES:
                quantile_key = f"{quantile:.1f}"
                if quantile_key not in intensity_models[horizon]:
                    raise ValueError(
                        f"Missing quantile model for horizon={horizon} quantile={quantile_key}"
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
        vector = [row_vector]

        ri_prob = float(self.ri_classifier.predict_proba(vector)[0][1])
        quantiles: dict[str, dict[str, float]] = {}
        warnings: list[str] = []

        for horizon, models in self.intensity_models.items():
            p10 = float(models["0.1"].predict(vector)[0])
            p50 = float(models["0.5"].predict(vector)[0])
            p90 = float(models["0.9"].predict(vector)[0])
            ordered = sorted([p10, p50, p90])
            if ordered != [p10, p50, p90]:
                warnings.append(
                    f"Quantiles were non-monotonic for {horizon}; values were sorted before return"
                )
            quantiles[horizon] = {
                "p10": ordered[0],
                "p50": ordered[1],
                "p90": ordered[2],
            }

        return {
            "ri_probability_24h": ri_prob,
            "vmax_quantiles_kt": quantiles,
            "warnings": warnings,
            "model_version": self.model_version,
        }
