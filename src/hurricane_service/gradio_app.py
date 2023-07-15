"""Gradio app for self-contained hurricane intensity-risk inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import gradio as gr

from .features import build_feature_row
from .model_bundle import ModelBundle
from .schemas import PredictRequest, PredictResponse
from .settings import ServiceSettings, load_settings


EXAMPLE_REQUEST = {
    "storm_id": "AL09",
    "issue_time": "2026-08-25T12:00:00Z",
    "storm_state": {
        "lat": 22.4,
        "lon": -71.8,
        "vmax_kt": 70,
        "mslp_mb": 987,
        "motion_dir_deg": 300,
        "motion_speed_kt": 12,
    },
    "history_24h": [
        {"hours_ago": 6, "lat": 22.0, "lon": -70.7, "vmax_kt": 65, "mslp_mb": 992},
        {"hours_ago": 12, "lat": 21.6, "lon": -69.8, "vmax_kt": 60, "mslp_mb": 996},
        {"hours_ago": 24, "lat": 20.9, "lon": -68.4, "vmax_kt": 55, "mslp_mb": 1000},
    ],
    "environment": {
        "sst_c": 29.1,
        "ohc_kj_cm2": 68.0,
        "shear_200_850_kt": 9.0,
        "midlevel_rh_pct": 66.0,
        "vorticity_850_s-1": 0.00018,
    },
}


@dataclass
class RuntimeState:
    settings: ServiceSettings
    model_bundle: ModelBundle | None
    load_error: str | None = None


def load_runtime_state(settings: ServiceSettings) -> RuntimeState:
    path = Path(settings.model_bundle_path)
    if not path.exists():
        return RuntimeState(
            settings=settings,
            model_bundle=None,
            load_error=f"Model bundle not found at {path}",
        )

    try:
        bundle = ModelBundle.load(path)
        return RuntimeState(settings=settings, model_bundle=bundle)
    except Exception as exc:
        return RuntimeState(settings=settings, model_bundle=None, load_error=str(exc))


def _status_text(state: RuntimeState) -> str:
    if state.model_bundle is None:
        return (
            "### Status: degraded\n"
            f"Model could not be loaded from `{state.settings.model_bundle_path}`.\n"
            f"Error: `{state.load_error or 'unknown'}`"
        )
    return (
        "### Status: ready\n"
        f"Model version: `{state.model_bundle.model_version}`\n"
        f"Bundle path: `{state.settings.model_bundle_path}`"
    )


def _predict_from_json(payload_text: str, state: RuntimeState) -> tuple[str, str]:
    if state.model_bundle is None:
        return (
            "Model unavailable",
            json.dumps(
                {
                    "error": "MODEL_UNAVAILABLE",
                    "message": state.load_error or "Model failed to load",
                },
                indent=2,
            ),
        )

    try:
        payload_dict = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return (
            "Invalid JSON payload",
            json.dumps({"error": "INVALID_JSON", "message": str(exc)}, indent=2),
        )

    try:
        request = PredictRequest.model_validate(payload_dict)
    except Exception as exc:
        return (
            "Invalid request schema",
            json.dumps({"error": "INVALID_SCHEMA", "message": str(exc)}, indent=2),
        )

    try:
        feature_row = build_feature_row(request)
        prediction = state.model_bundle.predict(feature_row)
        response = PredictResponse(
            storm_id=request.storm_id,
            model_version=prediction["model_version"],
            ri_probability_24h=prediction["ri_probability_24h"],
            vmax_quantiles_kt={
                "24h": prediction["vmax_quantiles_kt"]["24h"],
                "48h": prediction["vmax_quantiles_kt"]["48h"],
            },
            warnings=prediction["warnings"],
        )
        summary = (
            f"RI 24h probability: `{response.ri_probability_24h:.2%}` | "
            f"24h p50 vmax: `{response.vmax_quantiles_kt.h24.p50:.1f} kt` | "
            f"48h p50 vmax: `{response.vmax_quantiles_kt.h48.p50:.1f} kt`"
        )
        return summary, json.dumps(response.model_dump(by_alias=True), indent=2)
    except Exception as exc:
        return (
            "Inference error",
            json.dumps({"error": "INFERENCE_ERROR", "message": str(exc)}, indent=2),
        )


def create_demo(settings: ServiceSettings | None = None) -> gr.Blocks:
    resolved_settings = load_settings() if settings is None else settings
    state = load_runtime_state(resolved_settings)
    default_payload = json.dumps(EXAMPLE_REQUEST, indent=2)

    with gr.Blocks(title=resolved_settings.service_name) as demo:
        gr.Markdown(f"# {resolved_settings.service_name}")
        gr.Markdown(
            "Self-contained Gradio service for hurricane intensity-risk inference. "
            "This app can run locally in Docker or on Hugging Face Spaces."
        )

        status = gr.Markdown(_status_text(state))

        payload = gr.Code(
            label="Predict Request JSON",
            language="json",
            value=default_payload,
            lines=22,
        )
        predict_btn = gr.Button("Run Prediction", variant="primary")
        reset_btn = gr.Button("Reset Payload")

        summary = gr.Markdown(label="Prediction Summary")
        raw_response = gr.Code(label="Raw Response", language="json", lines=18)

        predict_btn.click(
            fn=lambda payload_text: _predict_from_json(payload_text, state),
            inputs=[payload],
            outputs=[summary, raw_response],
        )
        reset_btn.click(fn=lambda: default_payload, outputs=[payload])
        gr.Markdown(
            "Expected payload schema matches `PredictRequest` in `hurricane_service.schemas`."
        )

    return demo
