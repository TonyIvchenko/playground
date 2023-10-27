"""Minimal hurricane Gradio app with globally loaded PyTorch model."""

from __future__ import annotations

from pathlib import Path
import math
import os

import gradio as gr
import torch

from hurricane.model import load_model_bundle


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", os.getenv("PORT", "8000")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent / "model" / "hurricane_model.pt")))
SERVICE_NAME = os.getenv("SERVICE_NAME", "Hurricane Intensity-Risk Service")

model, feature_mean, feature_std, MODEL_VERSION = load_model_bundle(MODEL_PATH)


def _risk_level(probability: float) -> str:
    if probability < 0.25:
        return "low"
    if probability < 0.50:
        return "moderate"
    if probability < 0.75:
        return "high"
    return "extreme"


def predict(
    storm_id: str,
    vmax_kt: float,
    min_pressure_mb: float,
    lat: float,
    lon: float,
    month: float,
    dvmax_6h: float,
    dpres_6h: float,
) -> dict[str, object]:
    month_angle = 2.0 * math.pi * float(month) / 12.0
    x = torch.tensor(
        [
            [
                float(vmax_kt),
                float(min_pressure_mb),
                float(lat),
                float(lon),
                float(month),
                math.sin(month_angle),
                math.cos(month_angle),
                abs(float(lat)),
                1010.0 - float(min_pressure_mb),
                float(dvmax_6h),
                float(dpres_6h),
            ]
        ],
        dtype=torch.float32,
    )
    x = (x - feature_mean) / feature_std

    with torch.no_grad():
        probability = float(torch.sigmoid(model(x))[0, 0].item())
        probability = max(0.0, min(1.0, probability))

    return {
        "storm_id": storm_id,
        "model_version": MODEL_VERSION,
        "ri_probability_24h": probability,
        "risk_level": _risk_level(probability),
    }


with gr.Blocks(title=SERVICE_NAME) as demo:
    gr.Markdown(f"# {SERVICE_NAME}")
    gr.Markdown(f"Model version: `{MODEL_VERSION}`")

    storm_id = gr.Textbox(label="Storm ID", value="AL09")
    vmax_kt = gr.Number(label="Max Wind (kt)", value=70)
    min_pressure_mb = gr.Number(label="Min Pressure (mb)", value=980)
    lat = gr.Number(label="Latitude", value=22.5)
    lon = gr.Number(label="Longitude", value=-65.0)
    month = gr.Number(label="Month", value=9)
    dvmax_6h = gr.Number(label="Recent Wind Change (kt per 6h)", value=5.0)
    dpres_6h = gr.Number(label="Recent Pressure Change (mb per 6h)", value=-3.0)
    output = gr.JSON(label="Prediction")
    run = gr.Button("Predict")
    run.click(
        fn=predict,
        inputs=[storm_id, vmax_kt, min_pressure_mb, lat, lon, month, dvmax_6h, dpres_6h],
        outputs=output,
    )


def main() -> None:
    demo.launch(server_name=HOST, server_port=PORT)


if __name__ == "__main__":
    main()
