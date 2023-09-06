"""Standalone Gradio app backed by a trained PyTorch model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import gradio as gr
import torch
from torch import nn


def _default_model_path() -> str:
    return str(Path(__file__).resolve().parent / "model" / "wildfire_model.pt")


@dataclass(frozen=True)
class ServiceSettings:
    host: str = "0.0.0.0"
    port: int = 8010
    model_path: str = _default_model_path()
    service_name: str = "Wildfire Ignition-Risk Service"


@dataclass
class ModelRuntime:
    model: nn.Module
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    model_version: str


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
        model_path=values.get("MODEL_PATH", ServiceSettings.model_path),
        service_name=values.get("SERVICE_NAME", ServiceSettings.service_name),
    )


def load_runtime(model_path: str) -> ModelRuntime:
    artifact_path = Path(model_path).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    bundle = torch.load(artifact_path, map_location="cpu", weights_only=True)
    model = WildfireMLP()
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    feature_mean = torch.tensor(bundle["feature_mean"], dtype=torch.float32)
    feature_std = torch.tensor(bundle["feature_std"], dtype=torch.float32).clamp_min(1e-6)
    model_version = str(bundle.get("model_version", "unknown"))

    return ModelRuntime(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        model_version=model_version,
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
    runtime: ModelRuntime,
) -> dict[str, object]:
    features = torch.tensor([[temp_c, humidity_pct, wind_kph, drought_index]], dtype=torch.float32)
    normalized = (features - runtime.feature_mean) / runtime.feature_std

    with torch.no_grad():
        logits = runtime.model(normalized)
        probability = float(torch.sigmoid(logits)[0, 0].item())

    return {
        "region_id": region_id,
        "model_version": runtime.model_version,
        "ignition_probability_24h": probability,
        "risk_level": _risk_level(probability),
    }


def build_demo(settings: ServiceSettings, runtime: ModelRuntime) -> gr.Blocks:
    with gr.Blocks(title=settings.service_name) as demo:
        gr.Markdown("# Wildfire Ignition-Risk Service")
        gr.Markdown(f"Model version: `{runtime.model_version}`")

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
                runtime=runtime,
            ),
            inputs=[region_id, temp_c, humidity_pct, wind_kph, drought_index],
            outputs=output,
        )

    return demo


def main() -> None:
    settings = load_settings()
    runtime = load_runtime(settings.model_path)
    demo = build_demo(settings, runtime)
    demo.launch(server_name=settings.host, server_port=settings.port)


if __name__ == "__main__":
    main()
