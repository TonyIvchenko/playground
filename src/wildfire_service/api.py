"""FastAPI + Gradio application for wildfire ignition-risk inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import gradio as gr
from pydantic import ValidationError

from .features import build_feature_row
from .model_bundle import ModelBundle
from .schemas import HealthResponse, PredictRequest, PredictResponse
from .settings import ServiceSettings, load_settings


@dataclass
class RuntimeState:
    settings: ServiceSettings
    model_bundle: ModelBundle | None = None
    model_key: str | None = None
    load_error: str | None = None


def _load_model_bundle(settings: ServiceSettings) -> tuple[ModelBundle, str]:
    bundle_path = Path(settings.model_bundle_path)
    bundle = ModelBundle.load(bundle_path)
    return bundle, str(bundle_path)


def _build_service_metadata(settings: ServiceSettings) -> dict[str, object]:
    request_example = {
        "region_id": "norcal-foothills",
        "location": {
            "lat": 38.72,
            "lon": -121.2,
        },
        "forecast_date": "2026-08-25",
        "conditions": {
            "temp_c": 34.0,
            "relative_humidity_pct": 21.0,
            "wind_speed_kph": 28.0,
            "precip_7d_mm": 0.5,
            "drought_index": 0.82,
            "fuel_moisture_pct": 12.0,
            "vegetation_dryness": 0.88,
            "human_activity_index": 0.67,
            "elevation_m": 480.0,
            "slope_deg": 18.0,
        },
    }

    response_example = {
        "region_id": "norcal-foothills",
        "model_version": "2026.03.v1",
        "ignition_probability_24h": 0.61,
        "risk_level": "high",
        "top_drivers": [
            {"feature": "wind_speed_kph", "value": 28.0, "direction": "increase", "score": 0.73},
            {
                "feature": "relative_humidity_pct",
                "value": 21.0,
                "direction": "increase",
                "score": 0.67,
            },
            {
                "feature": "fuel_moisture_pct",
                "value": 12.0,
                "direction": "increase",
                "score": 0.52,
            },
        ],
        "warnings": [],
    }

    return {
        "service_id": settings.service_id,
        "display_name": settings.service_name,
        "service_version": settings.service_version,
        "predict_endpoint": "/predict",
        "health_endpoint": "/health",
        "metadata_endpoint": "/service-metadata",
        "ui_endpoint": settings.ui_path,
        "tags": ["wildfire", "ignition", "risk", "tabular-ml", "gradio"],
        "request_schema": PredictRequest.model_json_schema(),
        "response_schema": PredictResponse.model_json_schema(),
        "example_request": request_example,
        "example_response": response_example,
    }


def _predict_from_payload(state: RuntimeState, payload: PredictRequest) -> PredictResponse:
    if state.model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "MODEL_UNAVAILABLE",
                "message": state.load_error or "Model bundle could not be loaded",
            },
        )

    feature_row = build_feature_row(payload)
    prediction = state.model_bundle.predict(feature_row)
    return PredictResponse(
        region_id=payload.region_id,
        model_version=prediction["model_version"],
        ignition_probability_24h=prediction["ignition_probability_24h"],
        risk_level=prediction["risk_level"],
        top_drivers=prediction["top_drivers"],
        warnings=prediction["warnings"],
    )


def _build_gradio_ui(state: RuntimeState, metadata: dict[str, object]) -> gr.Blocks:
    example_request = json.dumps(metadata["example_request"], indent=2)

    def run_health() -> dict[str, object]:
        loaded = state.model_bundle is not None
        response = HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            model_version=state.model_bundle.model_version if loaded else None,
            detail=None if loaded else state.load_error,
        )
        return response.model_dump()

    def run_predict(request_json: str) -> tuple[dict[str, object] | None, str]:
        try:
            payload = PredictRequest.model_validate_json(request_json)
        except ValidationError as exc:
            return None, json.dumps(exc.errors(), indent=2)
        except Exception as exc:  # pragma: no cover
            return None, str(exc)

        try:
            response = _predict_from_payload(state, payload)
        except HTTPException as exc:
            detail = exc.detail
            if isinstance(detail, dict):
                return None, json.dumps(detail, indent=2)
            return None, str(detail)

        return response.model_dump(), ""

    with gr.Blocks(title=str(metadata["display_name"])) as demo:
        gr.Markdown(
            "# Wildfire Ignition-Risk Service\n"
            "Self-contained service with FastAPI endpoints and built-in Gradio UI."
        )

        with gr.Row():
            health_button = gr.Button("Check Health", variant="secondary")
            health_output = gr.JSON(label="Health")

        health_button.click(fn=run_health, inputs=None, outputs=health_output)

        request_input = gr.Code(
            label="Predict Request JSON",
            language="json",
            value=example_request,
            lines=24,
        )
        predict_button = gr.Button("Run Prediction", variant="primary")
        response_output = gr.JSON(label="Predict Response")
        error_output = gr.Code(label="Validation / Runtime Error", language="json", lines=8)

        predict_button.click(
            fn=run_predict,
            inputs=request_input,
            outputs=[response_output, error_output],
        )

        with gr.Accordion("Service Metadata", open=False):
            gr.JSON(value=metadata, label="/service-metadata")

    return demo


def create_app(
    settings: ServiceSettings | None = None,
    loader: Callable[[ServiceSettings], tuple[ModelBundle, str]] | None = None,
) -> FastAPI:
    app_settings = load_settings() if settings is None else settings
    state = RuntimeState(settings=app_settings)
    load_fn = _load_model_bundle if loader is None else loader

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # pragma: no cover - exercised via API tests
        try:
            bundle, model_key = load_fn(app_settings)
            state.model_bundle = bundle
            state.model_key = model_key
            state.load_error = None
        except Exception as exc:
            state.model_bundle = None
            state.model_key = None
            state.load_error = str(exc)
        yield

    app = FastAPI(
        title=app_settings.service_name,
        version=app_settings.service_version,
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded = state.model_bundle is not None
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            model_version=state.model_bundle.model_version if loaded else None,
            detail=None if loaded else state.load_error,
        )

    @app.get("/service-metadata")
    def service_metadata() -> dict[str, object]:
        return _build_service_metadata(app_settings)

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        return _predict_from_payload(state, payload)

    if app_settings.ui_path != "/":

        @app.get("/")
        def root_redirect() -> RedirectResponse:
            return RedirectResponse(url=app_settings.ui_path)

    demo = _build_gradio_ui(state=state, metadata=_build_service_metadata(app_settings))
    app = gr.mount_gradio_app(app, demo, path=app_settings.ui_path)

    return app


app = create_app()
