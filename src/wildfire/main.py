"""Minimal standalone Gradio app for wildfire ignition risk."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

import gradio as gr


@dataclass(frozen=True)
class ServiceSettings:
    host: str = "0.0.0.0"
    port: int = 8010
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
        host=values.get("API_HOST", ServiceSettings.host),
        port=_read_int(values, "API_PORT", _read_int(values, "PORT", ServiceSettings.port)),
        service_name=values.get("SERVICE_NAME", ServiceSettings.service_name),
        service_version=values.get("SERVICE_VERSION", ServiceSettings.service_version),
    )


def _risk_level(probability: float) -> str:
    if probability < 0.20:
        return "low"
    if probability < 0.40:
        return "moderate"
    if probability < 0.65:
        return "high"
    return "extreme"


def predict(
    region_id: str,
    temp_c: float,
    humidity_pct: float,
    wind_kph: float,
    drought_index: float,
    model_version: str,
) -> dict[str, object]:
    signal = (
        0.09 * (temp_c - 25.0)
        - 0.05 * (humidity_pct - 30.0)
        + 0.08 * (wind_kph - 20.0)
        + 2.0 * (drought_index - 0.5)
        - 1.5
    )
    probability = 1.0 / (1.0 + math.exp(-signal))
    return {
        "region_id": region_id,
        "model_version": model_version,
        "ignition_probability_24h": probability,
        "risk_level": _risk_level(probability),
    }


def build_demo(settings: ServiceSettings) -> gr.Blocks:
    with gr.Blocks(title=settings.service_name) as demo:
        gr.Markdown("# Wildfire Ignition-Risk Service")
        gr.Markdown(f"Model version: `{settings.service_version}`")

        region_id = gr.Textbox(label="Region ID", value="norcal")
        temp_c = gr.Number(label="Temperature (C)", value=34)
        humidity_pct = gr.Number(label="Humidity (%)", value=22)
        wind_kph = gr.Number(label="Wind (kph)", value=28)
        drought_index = gr.Number(label="Drought Index (0-1)", value=0.82)
        output = gr.JSON(label="Prediction")

        run = gr.Button("Predict")
        run.click(
            fn=lambda rid, temp, humidity, wind, drought: predict(
                region_id=rid,
                temp_c=float(temp),
                humidity_pct=float(humidity),
                wind_kph=float(wind),
                drought_index=float(drought),
                model_version=settings.service_version,
            ),
            inputs=[region_id, temp_c, humidity_pct, wind_kph, drought_index],
            outputs=output,
        )

    return demo


def main() -> None:
    settings = load_settings()
    demo = build_demo(settings)
    demo.launch(server_name=settings.host, server_port=settings.port)


if __name__ == "__main__":
    main()
