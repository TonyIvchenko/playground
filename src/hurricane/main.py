"""Minimal hurricane Gradio app with globally loaded PyTorch model."""

from __future__ import annotations

from pathlib import Path
import os

import gradio as gr
import torch
from torch import nn


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", os.getenv("PORT", "8000")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent / "model" / "hurricane_model.pt")))
SERVICE_NAME = os.getenv("SERVICE_NAME", "Hurricane Intensity-Risk Service")


class HurricaneMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model = HurricaneMLP()
model.load_state_dict(bundle["state_dict"])
model.eval()
feature_mean = torch.tensor(bundle["feature_mean"], dtype=torch.float32)
feature_std = torch.tensor(bundle["feature_std"], dtype=torch.float32).clamp_min(1e-6)
MODEL_VERSION = str(bundle.get("model_version", "unknown"))


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
    x = (x - feature_mean) / feature_std

    with torch.no_grad():
        probability = float(torch.sigmoid(model(x))[0, 0].item())

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
