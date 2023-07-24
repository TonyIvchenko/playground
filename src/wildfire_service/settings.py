"""Environment-based settings for wildfire service runtime."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ServiceSettings:
    model_bundle_path: str = "/app/model/model_bundle.joblib"
    api_host: str = "0.0.0.0"
    api_port: int = 8010
    ui_path: str = "/ui"

    service_id: str = "wildfire-ignition-risk"
    service_name: str = "Wildfire Ignition-Risk Service"
    service_version: str = "0.1.0"


def _read_int(env: dict[str, str], key: str, default: int) -> int:
    raw_value = env.get(key)
    if raw_value in (None, ""):
        return default
    return int(raw_value)


def load_settings(env: dict[str, str] | None = None) -> ServiceSettings:
    values = os.environ if env is None else env
    return ServiceSettings(
        model_bundle_path=values.get("MODEL_BUNDLE_PATH", ServiceSettings.model_bundle_path),
        api_host=values.get("API_HOST", ServiceSettings.api_host),
        api_port=_read_int(values, "API_PORT", ServiceSettings.api_port),
        ui_path=values.get("UI_PATH", ServiceSettings.ui_path),
        service_id=values.get("SERVICE_ID", ServiceSettings.service_id),
        service_name=values.get("SERVICE_NAME", ServiceSettings.service_name),
        service_version=values.get("SERVICE_VERSION", ServiceSettings.service_version),
    )
