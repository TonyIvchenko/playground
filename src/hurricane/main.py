"""Minimal hurricane Gradio app with globally loaded PyTorch model."""

from __future__ import annotations

import json
from pathlib import Path
import os

import gradio as gr
import torch


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", os.getenv("PORT", "8000")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent / "model" / "hurricane_model.pt")))
SERVICE_NAME = os.getenv("SERVICE_NAME", "Hurricane Intensity-Risk Service")


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
) -> dict[str, object]:
    x = torch.tensor(
        [[float(vmax_kt), float(min_pressure_mb), float(lat), float(lon), float(month)]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        probability = float(model(x).reshape(-1)[0].item())
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
    output = gr.JSON(label="Prediction")
    run = gr.Button("Predict")
    run.click(
        fn=predict,
        inputs=[storm_id, vmax_kt, min_pressure_mb, lat, lon, month],
        outputs=output,
    )


def main() -> None:
    demo.launch(server_name=HOST, server_port=PORT)


if __name__ == "__main__":
    main()
