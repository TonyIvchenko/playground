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


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Storm ID", value="AL09"),
        gr.Number(label="Max Wind (kt)", value=70),
        gr.Number(label="Min Pressure (mb)", value=980),
        gr.Number(label="Latitude", value=22.5),
        gr.Number(label="Longitude", value=-65.0),
        gr.Number(label="Month", value=9),
    ],
    outputs=gr.JSON(label="Prediction"),
    title=SERVICE_NAME,
    description=f"Model version: {MODEL_VERSION}",
)


def main() -> None:
    demo.launch(server_name=HOST, server_port=PORT)


if __name__ == "__main__":
    main()
