"""Minimal wildfire Gradio app with globally loaded PyTorch model."""

from __future__ import annotations

import json
from pathlib import Path
import os

import gradio as gr
import torch


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", os.getenv("PORT", "8010")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent / "model" / "wildfire_model.pt")))
SERVICE_NAME = os.getenv("SERVICE_NAME", "Wildfire Ignition-Risk Service")


def _load_artifact(path: Path) -> tuple[torch.jit.ScriptModule, str]:
    extra_files: dict[str, str] = {"metadata.json": ""}
    loaded_model = torch.jit.load(str(path), map_location="cpu", _extra_files=extra_files)
    loaded_model.eval()

    model_version = "unknown"
    raw_metadata = extra_files.get("metadata.json", "")
    if isinstance(raw_metadata, bytes):
        raw_metadata = raw_metadata.decode("utf-8")
    if raw_metadata:
        metadata = json.loads(raw_metadata)
        model_version = str(metadata.get("model_version", "unknown"))
    return loaded_model, model_version


model, MODEL_VERSION = _load_artifact(MODEL_PATH)


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
) -> dict[str, object]:
    x = torch.tensor([[float(temp_c), float(humidity_pct), float(wind_kph), float(drought_index)]], dtype=torch.float32)

    with torch.no_grad():
        probability = float(model(x).reshape(-1)[0].item())
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
    drought_index = gr.Number(label="Drought Index (0-1)", value=0.82)
    output = gr.JSON(label="Prediction")
    run = gr.Button("Predict")
    run.click(
        fn=predict,
        inputs=[region_id, temp_c, humidity_pct, wind_kph, drought_index],
        outputs=output,
    )


def main() -> None:
    demo.launch(server_name=HOST, server_port=PORT)


if __name__ == "__main__":
    main()
