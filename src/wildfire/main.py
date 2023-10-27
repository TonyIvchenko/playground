"""Minimal wildfire Gradio app with globally loaded PyTorch model."""

from __future__ import annotations

from pathlib import Path
import os

import gradio as gr
import torch

from wildfire.model import load_model_bundle


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", os.getenv("PORT", "8010")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent / "model" / "wildfire_model.pt")))
SERVICE_NAME = os.getenv("SERVICE_NAME", "Wildfire Ignition-Risk Service")

model, feature_mean, feature_std, MODEL_VERSION = load_model_bundle(MODEL_PATH)


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
    ffmc: float,
    dmc: float,
    drought_code: float,
    isi: float,
) -> dict[str, object]:
    x = torch.tensor(
        [[float(temp_c), float(humidity_pct), float(wind_kph), float(ffmc), float(dmc), float(drought_code), float(isi)]],
        dtype=torch.float32,
    )
    x = (x - feature_mean) / feature_std

    with torch.no_grad():
        probability = float(torch.sigmoid(model(x))[0, 0].item())
        probability = max(0.0, min(1.0, probability))

    return {
        "region_id": region_id,
        "model_version": MODEL_VERSION,
        "ignition_probability_24h": probability,
        "risk_level": _risk_level(probability),
    }


with gr.Blocks(title=SERVICE_NAME) as demo:
    gr.Markdown(f"# {SERVICE_NAME}")
    gr.Markdown(f"Model version: `{MODEL_VERSION}`")

    region_id = gr.Textbox(label="Region ID", value="norcal")
    temp_c = gr.Number(label="Temperature (C)", value=34)
    humidity_pct = gr.Number(label="Humidity (%)", value=22)
    wind_kph = gr.Number(label="Wind (kph)", value=28)
    ffmc = gr.Number(label="FFMC", value=92.0)
    dmc = gr.Number(label="DMC", value=180.0)
    drought_code = gr.Number(label="DC", value=640.0)
    isi = gr.Number(label="ISI", value=12.0)
    output = gr.JSON(label="Prediction")
    run = gr.Button("Predict")
    run.click(
        fn=predict,
        inputs=[region_id, temp_c, humidity_pct, wind_kph, ffmc, dmc, drought_code, isi],
        outputs=output,
    )


def main() -> None:
    demo.launch(server_name=HOST, server_port=PORT)


if __name__ == "__main__":
    main()
