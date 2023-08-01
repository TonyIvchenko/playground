"""Minimal wildfire service with FastAPI endpoints and embedded Gradio UI."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr
from pydantic import BaseModel, ConfigDict, Field, ValidationError
import uvicorn


@dataclass(frozen=True)
class ServiceSettings:
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
        api_host=values.get("API_HOST", ServiceSettings.api_host),
        api_port=_read_int(values, "API_PORT", _read_int(values, "PORT", ServiceSettings.api_port)),
        ui_path=values.get("UI_PATH", ServiceSettings.ui_path),
        service_id=values.get("SERVICE_ID", ServiceSettings.service_id),
        service_name=values.get("SERVICE_NAME", ServiceSettings.service_name),
        service_version=values.get("SERVICE_VERSION", ServiceSettings.service_version),
    )


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(..., min_length=2, max_length=64)
    location: Location
    temp_c: float = 30.0
    humidity_pct: float = Field(30.0, ge=0.0, le=100.0)
    wind_kph: float = Field(20.0, ge=0.0)
    drought_index: float = Field(0.6, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str
    model_version: str
    ignition_probability_24h: float
    risk_level: str


def _risk_level(probability: float) -> str:
    if probability < 0.20:
        return "low"
    if probability < 0.40:
        return "moderate"
    if probability < 0.65:
        return "high"
    return "extreme"


def _score(payload: PredictRequest) -> float:
    signal = (
        0.09 * (payload.temp_c - 25.0)
        - 0.05 * (payload.humidity_pct - 30.0)
        + 0.08 * (payload.wind_kph - 20.0)
        + 2.0 * (payload.drought_index - 0.5)
        - 1.5
    )
    return 1.0 / (1.0 + math.exp(-signal))


def _predict(payload: PredictRequest, settings: ServiceSettings) -> PredictResponse:
    probability = _score(payload)
    return PredictResponse(
        region_id=payload.region_id,
        model_version=settings.service_version,
        ignition_probability_24h=probability,
        risk_level=_risk_level(probability),
    )


def _service_metadata(settings: ServiceSettings) -> dict[str, object]:
    example_request = {
        "region_id": "norcal-foothills",
        "location": {"lat": 38.7, "lon": -121.2},
        "temp_c": 34,
        "humidity_pct": 21,
        "wind_kph": 28,
        "drought_index": 0.82,
    }
    example_response = {
        "region_id": "norcal-foothills",
        "model_version": settings.service_version,
        "ignition_probability_24h": 0.61,
        "risk_level": "high",
    }
    return {
        "service_id": settings.service_id,
        "display_name": settings.service_name,
        "service_version": settings.service_version,
        "predict_endpoint": "/predict",
        "health_endpoint": "/health",
        "metadata_endpoint": "/service-metadata",
        "ui_endpoint": settings.ui_path,
        "request_schema": PredictRequest.model_json_schema(),
        "response_schema": PredictResponse.model_json_schema(),
        "example_request": example_request,
        "example_response": example_response,
    }


def _gradio_app(settings: ServiceSettings) -> gr.Blocks:
    metadata = _service_metadata(settings)

    def run_health() -> dict[str, object]:
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": settings.service_version,
            "detail": None,
        }

    def run_predict(request_json: str) -> tuple[dict[str, object] | None, str]:
        try:
            request = PredictRequest.model_validate_json(request_json)
        except ValidationError as exc:
            return None, json.dumps(exc.errors(), indent=2)

        response = _predict(request, settings)
        return response.model_dump(), ""

    with gr.Blocks(title=settings.service_name) as demo:
        gr.Markdown("# Wildfire Ignition-Risk Service")
        gr.Markdown("Minimal self-contained service for local and container runs.")

        with gr.Row():
            health_btn = gr.Button("Check Health")
            health_out = gr.JSON(label="Health")
        health_btn.click(fn=run_health, inputs=None, outputs=health_out)

        request_in = gr.Code(
            label="Predict Request JSON",
            language="json",
            value=json.dumps(metadata["example_request"], indent=2),
            lines=14,
        )
        predict_btn = gr.Button("Run Prediction")
        predict_out = gr.JSON(label="Predict Response")
        err_out = gr.Code(label="Validation Error", language="json", lines=6)
        predict_btn.click(fn=run_predict, inputs=request_in, outputs=[predict_out, err_out])

    return demo


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    cfg = load_settings() if settings is None else settings
    app = FastAPI(title=cfg.service_name, version=cfg.service_version)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": cfg.service_version,
            "detail": None,
        }

    @app.get("/service-metadata")
    def service_metadata() -> dict[str, object]:
        return _service_metadata(cfg)

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        return _predict(payload, cfg)

    if cfg.ui_path != "/":

        @app.get("/")
        def root_redirect() -> RedirectResponse:
            return RedirectResponse(url=cfg.ui_path)

    demo = _gradio_app(cfg)
    return gr.mount_gradio_app(app, demo, path=cfg.ui_path)


app = create_app()


if __name__ == "__main__":
    settings = load_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
