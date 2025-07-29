"""Unified climate-risk service with wildfires and hurricanes overlays and inference."""

from __future__ import annotations

from functools import lru_cache
import html
import io
import json
import math
import os
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
import gradio as gr
import numpy as np
from PIL import Image
import torch
import uvicorn

try:
    from models.huricaines import load_model_bundle as load_huricaines_model_bundle
    from models.wildfires import load_model_bundle as load_wildfires_model_bundle
except ModuleNotFoundError:
    from src.disasters.models.huricaines import (
        load_model_bundle as load_huricaines_model_bundle,
    )
    from src.disasters.models.wildfires import (
        load_model_bundle as load_wildfires_model_bundle,
    )


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
GMAPS_API_KEY = os.getenv("GMAPS_API_KEY", "")
SERVICE_NAME = os.getenv("SERVICE_NAME", "Disasters")
HURRICANES_LABEL = "Hurricanes"
DISASTERS_STATIC_DIR = Path(__file__).resolve().parent / "static"
DISASTERS_STATIC_URL = "/disasters-static"
MAP_HEAD = """
<link rel="stylesheet" href="/disasters-static/map.css">
<script src="/disasters-static/map.js" defer></script>
""".strip()

WILDFIRES_MODEL_PATH = Path(
    os.getenv(
        "WILDFIRES_MODEL_PATH",
        str(Path(__file__).resolve().parent / "models" / "wildfires.pt"),
    )
)
HURICAINES_MODEL_PATH = Path(
    os.getenv(
        "HURICAINES_MODEL_PATH",
        str(Path(__file__).resolve().parent / "models" / "huricaines.pt"),
    )
)

WILDFIRES_TILES_DIR = Path(__file__).resolve().parent / "tiles" / "wildfires"
HURICAINES_TILES_DIR = Path(__file__).resolve().parent / "tiles" / "huricaines"

TILE_SIZE = 256
SAMPLE_SIZE = 64


def print_http_startup(service_name: str, host: str, port: int) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.service_startup import print_http_service_startup

    print_http_service_startup(service_name, host, port)


wildfires_model, wildfires_mean, wildfires_std, WILDFIRES_MODEL_VERSION = (
    load_wildfires_model_bundle(WILDFIRES_MODEL_PATH)
)
huricaines_model, huricaines_mean, huricaines_std, HURICAINES_MODEL_VERSION = (
    load_huricaines_model_bundle(HURICAINES_MODEL_PATH)
)

WILDFIRES_MODEL_BUNDLE = torch.load(
    WILDFIRES_MODEL_PATH, map_location="cpu", weights_only=True
)
HURICAINES_MODEL_BUNDLE = torch.load(
    HURICAINES_MODEL_PATH, map_location="cpu", weights_only=True
)


def _load_overlay(
    cube_path: Path,
    config_path: Path,
    default_config: dict[str, float | int | str],
    default_shape: tuple[int, int],
) -> dict[str, object]:
    config = dict(default_config)
    if config_path.exists():
        config.update(json.loads(config_path.read_text(encoding="utf-8")))

    if cube_path.exists():
        cube = np.load(cube_path)
        risk = cube["risk"].astype(np.float32)
        if "confidence" in cube:
            confidence = cube["confidence"].astype(np.float32)
        elif "activity" in cube:
            confidence = cube["activity"].astype(np.float32)
        else:
            confidence = np.zeros_like(risk, dtype=np.float32)
        frames = [str(x) for x in cube["frames"].tolist()]
    else:
        frames = [
            f"{year:04d}-{month:02d}"
            for year in range(int(config["start_year"]), int(config["end_year"]) + 1)
            for month in range(1, 13)
        ]
        risk = np.zeros(
            (len(frames), default_shape[0], default_shape[1]), dtype=np.float32
        )
        confidence = np.zeros_like(risk)

    return {
        "risk": risk,
        "confidence": confidence,
        "frames": frames,
        "config": config,
    }


WILDFIRES_OVERLAY = _load_overlay(
    cube_path=WILDFIRES_TILES_DIR / "overlay.npz",
    config_path=WILDFIRES_TILES_DIR / "overlay.json",
    default_config={
        "start_year": 2000,
        "end_year": 2030,
        "zoom_min": 4,
        "zoom_max": 8,
        "training_end_year": 2018,
        "eval_end_year": 2023,
        "center_lat": 39.5,
        "center_lon": -98.35,
        "lat_min": 24.0,
        "lat_max": 50.0,
        "lon_min": -125.0,
        "lon_max": -66.0,
    },
    default_shape=(220, 360),
)

HURICAINES_OVERLAY = _load_overlay(
    cube_path=HURICAINES_TILES_DIR / "overlay.npz",
    config_path=HURICAINES_TILES_DIR / "overlay.json",
    default_config={
        "start_year": 2000,
        "end_year": 2030,
        "zoom_min": 4,
        "zoom_max": 8,
        "training_end_year": 2018,
        "eval_end_year": 2023,
        "center_lat": 24.0,
        "center_lon": -60.0,
        "lat_min": 0.0,
        "lat_max": 55.0,
        "lon_min": -120.0,
        "lon_max": 20.0,
    },
    default_shape=(160, 220),
)

if len(WILDFIRES_OVERLAY["frames"]) <= len(HURICAINES_OVERLAY["frames"]):
    FRAMES = list(WILDFIRES_OVERLAY["frames"])
else:
    FRAMES = list(HURICAINES_OVERLAY["frames"])


OVERLAYS: dict[str, dict[str, object]] = {
    "wildfires": WILDFIRES_OVERLAY,
    "huricaines": HURICAINES_OVERLAY,
}


def _phase_for_frame(frame: str) -> str:
    year = int(frame[:4])
    train_end = min(
        int(WILDFIRES_OVERLAY["config"]["training_end_year"]),
        int(HURICAINES_OVERLAY["config"]["training_end_year"]),
    )
    eval_end = min(
        int(WILDFIRES_OVERLAY["config"]["eval_end_year"]),
        int(HURICAINES_OVERLAY["config"]["eval_end_year"]),
    )
    if year <= train_end:
        return "training"
    if year <= eval_end:
        return "eval"
    return "inference"


def _risk_level_wildfires(probability: float) -> str:
    if probability < 0.20:
        return "low"
    if probability < 0.40:
        return "moderate"
    if probability < 0.65:
        return "high"
    return "extreme"


def _risk_level_huricaines(probability: float) -> str:
    if probability < 0.25:
        return "low"
    if probability < 0.50:
        return "moderate"
    if probability < 0.75:
        return "high"
    return "extreme"


def _tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 2**z
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * y / n))))
    lat_bottom = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + 1) / n))))
    return float(lat_top), float(lat_bottom), float(lon_left), float(lon_right)


def _sample_layer(
    layer_grid: np.ndarray,
    config: dict[str, float | int | str],
    z: int,
    x: int,
    y: int,
) -> tuple[np.ndarray, np.ndarray]:
    lat_top, lat_bottom, lon_left, lon_right = _tile_bounds(z, x, y)
    lat_min = float(config["lat_min"])
    lat_max = float(config["lat_max"])
    lon_min = float(config["lon_min"])
    lon_max = float(config["lon_max"])

    if (
        lat_top < lat_min
        or lat_bottom > lat_max
        or lon_right < lon_min
        or lon_left > lon_max
    ):
        return np.zeros((SAMPLE_SIZE, SAMPLE_SIZE), dtype=np.float32), np.zeros(
            (SAMPLE_SIZE, SAMPLE_SIZE), dtype=bool
        )

    row_lats = np.linspace(lat_top, lat_bottom, SAMPLE_SIZE, endpoint=False) + (
        lat_bottom - lat_top
    ) / (2.0 * SAMPLE_SIZE)
    col_lons = np.linspace(lon_left, lon_right, SAMPLE_SIZE, endpoint=False) + (
        lon_right - lon_left
    ) / (2.0 * SAMPLE_SIZE)

    valid_rows = (row_lats >= lat_min) & (row_lats <= lat_max)
    valid_cols = (col_lons >= lon_min) & (col_lons <= lon_max)
    valid_mask = np.outer(valid_rows, valid_cols)

    row_lats_clamped = np.clip(row_lats, lat_min, lat_max)
    col_lons_clamped = np.clip(col_lons, lon_min, lon_max)

    h, w = layer_grid.shape
    iy = ((lat_max - row_lats_clamped) / (lat_max - lat_min) * (h - 1)).astype(np.int32)
    ix = ((col_lons_clamped - lon_min) / (lon_max - lon_min) * (w - 1)).astype(np.int32)

    sampled = layer_grid[iy[:, None], ix[None, :]]
    sampled = sampled.astype(np.float32)
    sampled[~valid_mask] = 0.0
    return sampled, valid_mask


def _colorize(
    sampled_risk: np.ndarray, sampled_conf: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    rgba = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE, 4), dtype=np.uint8)
    low = sampled_risk < 0.33
    mid = (sampled_risk >= 0.33) & (sampled_risk < 0.66)
    high = sampled_risk >= 0.66

    rgba[low, 0:3] = np.array([46, 204, 113], dtype=np.uint8)
    rgba[mid, 0:3] = np.array([241, 196, 15], dtype=np.uint8)
    rgba[high, 0:3] = np.array([231, 76, 60], dtype=np.uint8)

    conf = np.clip(sampled_conf, 0.0, 1.0)
    # Make no-impact areas fully transparent; opacity grows with both confidence and risk intensity.
    impact = np.clip((sampled_risk - 0.08) / 0.92, 0.0, 1.0)
    alpha = conf * impact
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

    rgba[~valid_mask, 3] = 0
    return rgba


@lru_cache(maxsize=80000)
def _render_tile_png(hazard: str, frame_idx: int, z: int, x: int, y: int) -> bytes:
    if hazard not in OVERLAYS:
        raise ValueError("unknown hazard")

    overlay = OVERLAYS[hazard]
    frames = overlay["frames"]
    if frame_idx < 0 or frame_idx >= len(frames):
        raise ValueError("frame index out of range")

    risk_grid = overlay["risk"][frame_idx]
    conf_grid = np.clip(overlay["confidence"][frame_idx], 0.0, 1.0)

    sampled_risk, valid_mask = _sample_layer(
        risk_grid, config=overlay["config"], z=z, x=x, y=y
    )
    sampled_conf, _ = _sample_layer(conf_grid, config=overlay["config"], z=z, x=x, y=y)
    rgba_small = _colorize(
        sampled_risk=sampled_risk, sampled_conf=sampled_conf, valid_mask=valid_mask
    )
    image = Image.fromarray(rgba_small, mode="RGBA").resize(
        (TILE_SIZE, TILE_SIZE), resample=Image.Resampling.BILINEAR
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def predict_wildfires(
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
        [
            [
                float(temp_c),
                float(humidity_pct),
                float(wind_kph),
                float(ffmc),
                float(dmc),
                float(drought_code),
                float(isi),
            ]
        ],
        dtype=torch.float32,
    )
    x = (x - wildfires_mean) / wildfires_std

    with torch.no_grad():
        probability = float(torch.sigmoid(wildfires_model(x))[0, 0].item())
        probability = max(0.0, min(1.0, probability))

    return {
        "region_id": region_id,
        "model_version": WILDFIRES_MODEL_VERSION,
        "ignition_probability_24h": probability,
        "risk_level": _risk_level_wildfires(probability),
    }


def predict_huricaines(
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
    x = (x - huricaines_mean) / huricaines_std

    with torch.no_grad():
        probability = float(torch.sigmoid(huricaines_model(x))[0, 0].item())
        probability = max(0.0, min(1.0, probability))

    return {
        "storm_id": storm_id,
        "model_version": HURICAINES_MODEL_VERSION,
        "ri_probability_24h": probability,
        "risk_level": _risk_level_huricaines(probability),
    }


def _map_html() -> str:
    frame_count = len(FRAMES)
    train_frames = sum(1 for f in FRAMES if _phase_for_frame(f) == "training")
    eval_frames = sum(1 for f in FRAMES if _phase_for_frame(f) == "eval")
    infer_frames = max(1, frame_count - train_frames - eval_frames)
    train_end_year = min(
        int(WILDFIRES_OVERLAY["config"]["training_end_year"]),
        int(HURICAINES_OVERLAY["config"]["training_end_year"]),
    )
    eval_end_year = min(
        int(WILDFIRES_OVERLAY["config"]["eval_end_year"]),
        int(HURICAINES_OVERLAY["config"]["eval_end_year"]),
    )
    default_zoom = max(
        int(WILDFIRES_OVERLAY["config"]["zoom_min"]),
        int(HURICAINES_OVERLAY["config"]["zoom_min"]),
    )

    year_ticks: list[tuple[int, int]] = []
    for idx, frame in enumerate(FRAMES):
        if frame.endswith("-01"):
            year_ticks.append((int(frame[:4]), idx))
    frame_denom = max(1, frame_count - 1)
    year_ticks_html = "".join(
        (
            f'<div class="year-tick" data-frame-index="{idx}" '
            f'style="left:{(idx / frame_denom) * 100.0:.6f}%"><span>{year}</span></div>'
        )
        for year, idx in year_ticks
    )
    month_step_pct = 100.0 / frame_denom

    def _first_metric(bundle: dict[str, object], keys: list[str]) -> float | None:
        for key in keys:
            value = bundle.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    js_config = {
        "api_key": GMAPS_API_KEY,
        "frames": FRAMES,
        "service_id": "disasters",
        "center_lat": 36.0,
        "center_lon": -95.0,
        "default_zoom": default_zoom,
        "zoom_min": 2,
        "zoom_max": 10,
        "training_end_year": train_end_year,
        "eval_end_year": eval_end_year,
        "hazards": {
            "wildfires": {
                "lat_min": float(WILDFIRES_OVERLAY["config"]["lat_min"]),
                "lat_max": float(WILDFIRES_OVERLAY["config"]["lat_max"]),
                "lon_min": float(WILDFIRES_OVERLAY["config"]["lon_min"]),
                "lon_max": float(WILDFIRES_OVERLAY["config"]["lon_max"]),
            },
            "huricaines": {
                "lat_min": float(HURICAINES_OVERLAY["config"]["lat_min"]),
                "lat_max": float(HURICAINES_OVERLAY["config"]["lat_max"]),
                "lon_min": float(HURICAINES_OVERLAY["config"]["lon_min"]),
                "lon_max": float(HURICAINES_OVERLAY["config"]["lon_max"]),
            },
        },
        "model_metrics": {
            "wildfires": {
                "val_accuracy": _first_metric(
                    WILDFIRES_MODEL_BUNDLE, ["val_accuracy", "accuracy"]
                ),
                "val_auc": _first_metric(WILDFIRES_MODEL_BUNDLE, ["val_auc", "auc"]),
            },
            "huricaines": {
                "val_accuracy": _first_metric(
                    HURICAINES_MODEL_BUNDLE, ["val_accuracy", "accuracy"]
                ),
                "val_auc": _first_metric(HURICAINES_MODEL_BUNDLE, ["val_auc", "auc"]),
            },
        },
    }
    config_blob = html.escape(json.dumps(js_config), quote=True)

    return f"""
<div id="risk-map-shell" class="risk-map-shell" data-config="{config_blob}">
  <section class="risk-map-pane">
    <div class="risk-map-header">
      <div class="timeline-row">
        <div class="timeline-wrap">
          <div class="timeline-years">
            {year_ticks_html}
          </div>
          <div class="timeline-track" style="--month-step:{month_step_pct:.6f}%;">
            <div class="timeline-phases">
              <div class="phase-seg train" style="flex:{max(1, train_frames)};"></div>
              <div class="phase-seg eval" style="flex:{max(1, eval_frames)};"></div>
              <div class="phase-seg infer" style="flex:{max(1, infer_frames)};"></div>
            </div>
            <div id="risk-time-progress" class="timeline-progress"></div>
            <div id="risk-now-marker" class="timeline-marker"></div>
            <input id="risk-time-slider" type="range" min="0" max="{frame_count - 1}" value="0" step="1" />
          </div>
        </div>
        <button id="risk-play" type="button" aria-label="Play timeline">
          <span class="play-icon" aria-hidden="true">&#9658;</span>
          <span class="pause-icon" aria-hidden="true">&#10074;&#10074;</span>
        </button>
      </div>
    </div>

    <div class="risk-map-stage">
      <div class="map-overlay-panel">
        <div class="panel-title">Layers</div>
        <div class="overlay-row">
          <select id="hazard-select" class="layer-select">
            <option value="wildfires" selected>Wildfires</option>
            <option value="huricaines">{HURRICANES_LABEL}</option>
          </select>
        </div>
        <div class="overlay-row metrics-row">
          <div class="metric-inline"><span>Accuracy</span><strong id="model-metric-acc">-</strong></div>
          <div class="metric-inline"><span>AUC</span><strong id="model-metric-auc">-</strong></div>
        </div>
      </div>
      <div id="risk-map" class="risk-map"></div>
    </div>
    <div id="risk-map-status" class="risk-map-status"></div>
  </section>
</div>
"""


def _map_bootstrap_js() -> str:
    return """
async () => {
  if (window.bootstrapDisastersMap) {
    return window.bootstrapDisastersMap();
  }
  for (let attempt = 0; attempt < 20; attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 50));
    if (window.bootstrapDisastersMap) {
      return window.bootstrapDisastersMap();
    }
  }
  console.error("Disasters map bootstrap script did not load.");
  return [];
}
"""


def _toggle_model_panel(selection: str) -> tuple[dict[str, bool], dict[str, bool]]:
    show_huricaines = selection == HURRICANES_LABEL
    return gr.update(visible=show_huricaines), gr.update(visible=not show_huricaines)


with gr.Blocks(title=SERVICE_NAME, head=MAP_HEAD) as demo:
    gr.Markdown(f"# {SERVICE_NAME}")

    with gr.Tabs():
        with gr.Tab("Map"):
            gr.HTML(_map_html())

        with gr.Tab("Models"):
            with gr.Row():
                with gr.Column(scale=1, min_width=170):
                    model_selector = gr.Radio(
                        choices=[HURRICANES_LABEL, "Wildfires"],
                        value=HURRICANES_LABEL,
                        show_label=False,
                        container=False,
                    )

                with gr.Column(scale=4):
                    with gr.Group(visible=True) as huricaines_panel:
                        with gr.Row():
                            storm_id = gr.Textbox(label="Storm ID", value="AL09")
                            vmax_kt = gr.Number(label="Max Wind (kt)", value=70)
                            min_pressure_mb = gr.Number(
                                label="Min Pressure (mb)", value=980
                            )
                            lat = gr.Number(label="Latitude", value=22.5)
                            lon = gr.Number(label="Longitude", value=-65.0)
                            month = gr.Number(label="Month", value=9)
                            dvmax_6h = gr.Number(
                                label="Recent Wind Change (kt per 6h)", value=5.0
                            )
                            dpres_6h = gr.Number(
                                label="Recent Pressure Change (mb per 6h)", value=-3.0
                            )

                        huricaines_output = gr.JSON(
                            label=f"{HURRICANES_LABEL} Prediction"
                        )
                        huricaines_run = gr.Button(f"Predict {HURRICANES_LABEL}")
                        huricaines_run.click(
                            fn=predict_huricaines,
                            inputs=[
                                storm_id,
                                vmax_kt,
                                min_pressure_mb,
                                lat,
                                lon,
                                month,
                                dvmax_6h,
                                dpres_6h,
                            ],
                            outputs=huricaines_output,
                        )

                    with gr.Group(visible=False) as wildfires_panel:
                        with gr.Row():
                            region_id = gr.Textbox(label="Region ID", value="norcal")
                            temp_c = gr.Number(label="Temperature (C)", value=34)
                            humidity_pct = gr.Number(label="Humidity (%)", value=22)
                            wind_kph = gr.Number(label="Wind (kph)", value=28)
                            ffmc = gr.Number(label="FFMC", value=92.0)
                            dmc = gr.Number(label="DMC", value=180.0)
                            drought_code = gr.Number(label="DC", value=640.0)
                            isi = gr.Number(label="ISI", value=12.0)

                        wildfires_output = gr.JSON(label="Wildfires Prediction")
                        wildfires_run = gr.Button("Predict Wildfires")
                        wildfires_run.click(
                            fn=predict_wildfires,
                            inputs=[
                                region_id,
                                temp_c,
                                humidity_pct,
                                wind_kph,
                                ffmc,
                                dmc,
                                drought_code,
                                isi,
                            ],
                            outputs=wildfires_output,
                        )

            model_selector.change(
                fn=_toggle_model_panel,
                inputs=model_selector,
                outputs=[huricaines_panel, wildfires_panel],
                queue=False,
                show_progress="hidden",
            )

    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js=_map_bootstrap_js(),
        queue=False,
        show_progress="hidden",
    )


api = FastAPI(title=SERVICE_NAME)
api.mount(
    DISASTERS_STATIC_URL,
    StaticFiles(directory=str(DISASTERS_STATIC_DIR)),
    name="disasters-static",
)


@api.get("/health")
def health() -> dict[str, object]:
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "frames": len(FRAMES),
        "wildfires_model_version": WILDFIRES_MODEL_VERSION,
        "huricaines_model_version": HURICAINES_MODEL_VERSION,
    }


def _serve_tile(hazard: str, frame_idx: int, z: int, x: int, y: int) -> Response:
    if hazard not in OVERLAYS:
        raise HTTPException(status_code=404, detail="unknown hazard")

    if z < 0 or z > 12:
        blank = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
        buffer = io.BytesIO()
        blank.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    try:
        png = _render_tile_png(hazard=hazard, frame_idx=frame_idx, z=z, x=x, y=y)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@api.get("/tiles/{hazard}/{frame_idx}/{z}/{x}/{y}.png")
def tile(hazard: str, frame_idx: int, z: int, x: int, y: int) -> Response:
    return _serve_tile(hazard=hazard, frame_idx=frame_idx, z=z, x=x, y=y)


@api.get("/tiles/{hazard}/{layer}/{frame_idx}/{z}/{x}/{y}.png")
def tile_legacy(
    hazard: str, layer: str, frame_idx: int, z: int, x: int, y: int
) -> Response:
    # Backward-compatible path; the service now exposes one overlay per hazard.
    if layer not in {"risk", "activity", "confidence"}:
        raise HTTPException(status_code=404, detail="unknown layer")
    return _serve_tile(hazard=hazard, frame_idx=frame_idx, z=z, x=x, y=y)


app = gr.mount_gradio_app(api, demo, path="/")


def main() -> None:
    print_http_startup(SERVICE_NAME, HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
