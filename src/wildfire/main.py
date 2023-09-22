"""Minimal wildfire Gradio app with globally loaded PyTorch model."""

from __future__ import annotations

from pathlib import Path
import os

import gradio as gr
import torch
from torch import nn


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", os.getenv("PORT", "8010")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent / "model" / "wildfire_model.pt")))
SERVICE_NAME = os.getenv("SERVICE_NAME", "Wildfire Ignition-Risk Service")


class WildfireMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model = WildfireMLP()
model.load_state_dict(bundle["state_dict"])
model.eval()
feature_mean = torch.tensor(bundle["feature_mean"], dtype=torch.float32)
feature_std = torch.tensor(bundle["feature_std"], dtype=torch.float32).clamp_min(1e-6)
MODEL_VERSION = str(bundle.get("model_version", "unknown"))


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
    x = (x - feature_mean) / feature_std

    with torch.no_grad():
        probability = float(torch.sigmoid(model(x))[0, 0].item())

    return {
        "region_id": region_id,
        "model_version": MODEL_VERSION,
        "ignition_probability_24h": probability,
        "risk_level": _risk_level(probability),
    }


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Region ID", value="norcal"),
        gr.Number(label="Temperature (C)", value=34),
        gr.Number(label="Humidity (%)", value=22),
        gr.Number(label="Wind (kph)", value=28),
        gr.Number(label="Drought Index (0-1)", value=0.82),
    ],
    outputs=gr.JSON(label="Prediction"),
    title=SERVICE_NAME,
    description=f"Model version: {MODEL_VERSION}",
)


def main() -> None:
    demo.launch(server_name=HOST, server_port=PORT)


if __name__ == "__main__":
    main()
