"""Minimal standalone Gradio app for hurricane intensity risk."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

import gradio as gr


@dataclass(frozen=True)
class ServiceSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    service_name: str = "Hurricane Intensity-Risk Service"
    service_version: str = "0.1.0"


def _read_int(env: dict[str, str], key: str, default: int) -> int:
    raw_value = env.get(key)
    if raw_value in (None, ""):
        return default
    return int(raw_value)


def load_settings(env: dict[str, str] | None = None) -> ServiceSettings:
    values = os.environ if env is None else env
    return ServiceSettings(
        host=values.get("API_HOST", ServiceSettings.host),
        port=_read_int(values, "API_PORT", _read_int(values, "PORT", ServiceSettings.port)),
        service_name=values.get("SERVICE_NAME", ServiceSettings.service_name),
        service_version=values.get("SERVICE_VERSION", ServiceSettings.service_version),
    )


def _risk_level(probability: float) -> str:
    if probability < 0.25:
        return "low"
    if probability < 0.50:
        return "moderate"
    if probability < 0.75:
        return "high"
    return "extreme"


def predict(storm_id: str, vmax_kt: float, shear_kt: float, sst_c: float, model_version: str) -> dict[str, object]:
    signal = 0.06 * (vmax_kt - 50.0) - 0.08 * (shear_kt - 12.0) + 0.12 * (sst_c - 27.0) - 0.35
    probability = 1.0 / (1.0 + math.exp(-signal))
    return {
        "storm_id": storm_id,
        "model_version": model_version,
        "ri_probability_24h": probability,
        "risk_level": _risk_level(probability),
    }


def build_demo(settings: ServiceSettings) -> gr.Blocks:
    with gr.Blocks(title=settings.service_name) as demo:
        gr.Markdown("# Hurricane Intensity-Risk Service")
        gr.Markdown(f"Model version: `{settings.service_version}`")

        storm_id = gr.Textbox(label="Storm ID", value="AL09")
        vmax_kt = gr.Number(label="Max Wind (kt)", value=70)
        shear_kt = gr.Number(label="Vertical Shear (kt)", value=10)
        sst_c = gr.Number(label="Sea Surface Temp (C)", value=29.2)
        output = gr.JSON(label="Prediction")

        run = gr.Button("Predict")
        run.click(
            fn=lambda sid, vmax, shear, sst: predict(
                storm_id=sid,
                vmax_kt=float(vmax),
                shear_kt=float(shear),
                sst_c=float(sst),
                model_version=settings.service_version,
            ),
            inputs=[storm_id, vmax_kt, shear_kt, sst_c],
            outputs=output,
        )

    return demo


def main() -> None:
    settings = load_settings()
    demo = build_demo(settings)
    demo.launch(server_name=settings.host, server_port=settings.port)


if __name__ == "__main__":
    main()
