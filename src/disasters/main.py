"""Unified climate-risk service with wildfire + hurricane overlays and inference."""

from __future__ import annotations

from functools import lru_cache
import io
import json
import math
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
import gradio as gr
import numpy as np
from PIL import Image
import torch
import uvicorn

from src.disasters.models.hurricane_artifact import load_model_bundle as load_hurricane_model_bundle
from src.disasters.models.wildfire_artifact import load_model_bundle as load_wildfire_model_bundle


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", os.getenv("PORT", "8080")))
GMAPS_API_KEY = os.getenv("GMAPS_API_KEY", "")
SERVICE_NAME = os.getenv("SERVICE_NAME", "Climate Risk Map Service")

WILDFIRE_MODEL_PATH = Path(
    os.getenv("WILDFIRE_MODEL_PATH", str(Path(__file__).resolve().parent / "models" / "wildfire_model.pt"))
)
HURRICANE_MODEL_PATH = Path(
    os.getenv("HURRICANE_MODEL_PATH", str(Path(__file__).resolve().parent / "models" / "hurricane_model.pt"))
)

WILDFIRE_TILES_DIR = Path(__file__).resolve().parent / "tiles" / "wildfire"
HURRICANE_TILES_DIR = Path(__file__).resolve().parent / "tiles" / "hurricane"

TILE_SIZE = 256
SAMPLE_SIZE = 64


wildfire_model, wildfire_mean, wildfire_std, WILDFIRE_MODEL_VERSION = load_wildfire_model_bundle(WILDFIRE_MODEL_PATH)
hurricane_model, hurricane_mean, hurricane_std, HURRICANE_MODEL_VERSION = load_hurricane_model_bundle(HURRICANE_MODEL_PATH)

WILDFIRE_MODEL_BUNDLE = torch.load(WILDFIRE_MODEL_PATH, map_location="cpu", weights_only=True)
HURRICANE_MODEL_BUNDLE = torch.load(HURRICANE_MODEL_PATH, map_location="cpu", weights_only=True)


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
        activity = cube["activity"].astype(np.float32)
        frames = [str(x) for x in cube["frames".strip()].tolist()]
    else:
        frames = [
            f"{year:04d}-{month:02d}"
            for year in range(int(config["start_year"]), int(config["end_year"]) + 1)
            for month in range(1, 13)
        ]
        risk = np.zeros((len(frames), default_shape[0], default_shape[1]), dtype=np.float32)
        activity = np.zeros_like(risk)

    return {
        "risk": risk,
        "activity": activity,
        "frames": frames,
        "config": config,
    }


WILDFIRE_OVERLAY = _load_overlay(
    cube_path=WILDFIRE_TILES_DIR / "overlay_cube.npz",
    config_path=WILDFIRE_TILES_DIR / "overlay_config.json",
    default_config={
        "start_year": 2000,
        "end_year": 2030,
        "zoom_min": 4,
        "zoom_max": 8,
        "training_end_year": 2018,
        "eval_end_year": 2023,
        "center_lat": 36.5,
        "center_lon": 0.5,
        "lat_min": 30.0,
        "lat_max": 45.5,
        "lon_min": -12.5,
        "lon_max": 12.5,
    },
    default_shape=(120, 180),
)

HURRICANE_OVERLAY = _load_overlay(
    cube_path=HURRICANE_TILES_DIR / "overlay_cube.npz",
    config_path=HURRICANE_TILES_DIR / "overlay_config.json",
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

if len(WILDFIRE_OVERLAY["frames"]) <= len(HURRICANE_OVERLAY["frames"]):
    FRAMES = list(WILDFIRE_OVERLAY["frames"])
else:
    FRAMES = list(HURRICANE_OVERLAY["frames"])


OVERLAYS: dict[str, dict[str, object]] = {
    "wildfire": WILDFIRE_OVERLAY,
    "hurricane": HURRICANE_OVERLAY,
}


def _as_percent(metric: object) -> str:
    if isinstance(metric, (int, float)):
        return f"{float(metric) * 100.0:.2f}%"
    return "n/a"


def _phase_for_frame(frame: str) -> str:
    year = int(frame[:4])
    train_end = min(
        int(WILDFIRE_OVERLAY["config"]["training_end_year"]),
        int(HURRICANE_OVERLAY["config"]["training_end_year"]),
    )
    eval_end = min(
        int(WILDFIRE_OVERLAY["config"]["eval_end_year"]),
        int(HURRICANE_OVERLAY["config"]["eval_end_year"]),
    )
    if year <= train_end:
        return "training"
    if year <= eval_end:
        return "eval"
    return "inference"


def _risk_level_wildfire(probability: float) -> str:
    if probability < 0.20:
        return "low"
    if probability < 0.40:
        return "moderate"
    if probability < 0.65:
        return "high"
    return "extreme"


def _risk_level_hurricane(probability: float) -> str:
    if probability < 0.25:
        return "low"
    if probability < 0.50:
        return "moderate"
    if probability < 0.75:
        return "high"
    return "extreme"


def _tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 2 ** z
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

    if lat_top < lat_min or lat_bottom > lat_max or lon_right < lon_min or lon_left > lon_max:
        return np.zeros((SAMPLE_SIZE, SAMPLE_SIZE), dtype=np.float32), np.zeros((SAMPLE_SIZE, SAMPLE_SIZE), dtype=bool)

    row_lats = np.linspace(lat_top, lat_bottom, SAMPLE_SIZE, endpoint=False) + (lat_bottom - lat_top) / (2.0 * SAMPLE_SIZE)
    col_lons = np.linspace(lon_left, lon_right, SAMPLE_SIZE, endpoint=False) + (lon_right - lon_left) / (2.0 * SAMPLE_SIZE)

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


def _colorize(layer: str, sampled: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rgba = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE, 4), dtype=np.uint8)

    if layer == "risk":
        low = sampled < 0.33
        mid = (sampled >= 0.33) & (sampled < 0.66)
        high = sampled >= 0.66

        rgba[low, 0:3] = np.array([46, 204, 113], dtype=np.uint8)
        rgba[mid, 0:3] = np.array([241, 196, 15], dtype=np.uint8)
        rgba[high, 0:3] = np.array([231, 76, 60], dtype=np.uint8)
        rgba[..., 3] = np.clip((70.0 + sampled * 165.0), 0, 255).astype(np.uint8)
    elif layer == "activity":
        rgba[..., 0] = np.clip(30 + sampled * 110, 0, 255).astype(np.uint8)
        rgba[..., 1] = np.clip(90 + sampled * 80, 0, 255).astype(np.uint8)
        rgba[..., 2] = np.clip(210 + sampled * 45, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(60 + sampled * 170, 0, 255).astype(np.uint8)
    elif layer == "confidence":
        confidence = np.sqrt(np.clip(sampled, 0.0, 1.0))
        rgba[..., 0] = np.clip(215 - confidence * 95, 0, 255).astype(np.uint8)
        rgba[..., 1] = np.clip(215 - confidence * 95, 0, 255).astype(np.uint8)
        rgba[..., 2] = np.clip(215, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(55 + confidence * 180, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported layer: {layer}")

    rgba[~valid_mask, 3] = 0
    return rgba


@lru_cache(maxsize=80000)
def _render_tile_png(hazard: str, layer: str, frame_idx: int, z: int, x: int, y: int) -> bytes:
    if hazard not in OVERLAYS:
        raise ValueError("unknown hazard")

    overlay = OVERLAYS[hazard]
    frames = overlay["frames"]
    if frame_idx < 0 or frame_idx >= len(frames):
        raise ValueError("frame index out of range")

    if layer == "risk":
        grid = overlay["risk"][frame_idx]
    elif layer == "activity":
        grid = overlay["activity"][frame_idx]
    elif layer == "confidence":
        grid = np.clip(overlay["activity"][frame_idx], 0.0, 1.0)
    else:
        raise ValueError("unknown layer")

    sampled, valid_mask = _sample_layer(grid, config=overlay["config"], z=z, x=x, y=y)
    rgba_small = _colorize(layer=layer, sampled=sampled, valid_mask=valid_mask)
    image = Image.fromarray(rgba_small, mode="RGBA").resize((TILE_SIZE, TILE_SIZE), resample=Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def predict_wildfire(
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
    x = (x - wildfire_mean) / wildfire_std

    with torch.no_grad():
        probability = float(torch.sigmoid(wildfire_model(x))[0, 0].item())
        probability = max(0.0, min(1.0, probability))

    return {
        "region_id": region_id,
        "model_version": WILDFIRE_MODEL_VERSION,
        "ignition_probability_24h": probability,
        "risk_level": _risk_level_wildfire(probability),
    }


def predict_hurricane(
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
    x = (x - hurricane_mean) / hurricane_std

    with torch.no_grad():
        probability = float(torch.sigmoid(hurricane_model(x))[0, 0].item())
        probability = max(0.0, min(1.0, probability))

    return {
        "storm_id": storm_id,
        "model_version": HURRICANE_MODEL_VERSION,
        "ri_probability_24h": probability,
        "risk_level": _risk_level_hurricane(probability),
    }


def _map_html() -> str:
    frame_count = len(FRAMES)
    train_frames = sum(1 for f in FRAMES if _phase_for_frame(f) == "training")
    eval_frames = sum(1 for f in FRAMES if _phase_for_frame(f) == "eval")
    infer_frames = max(1, frame_count - train_frames - eval_frames)

    js_config = {
        "api_key": GMAPS_API_KEY,
        "frames": FRAMES,
        "service_id": "disasters",
        "center_lat": 28.0,
        "center_lon": -35.0,
        "zoom_min": 3,
        "zoom_max": 8,
        "training_end_year": min(
            int(WILDFIRE_OVERLAY["config"]["training_end_year"]),
            int(HURRICANE_OVERLAY["config"]["training_end_year"]),
        ),
        "eval_end_year": min(
            int(WILDFIRE_OVERLAY["config"]["eval_end_year"]),
            int(HURRICANE_OVERLAY["config"]["eval_end_year"]),
        ),
        "hazards": {
            "wildfire": {
                "zoom_min": int(WILDFIRE_OVERLAY["config"]["zoom_min"]),
                "zoom_max": int(WILDFIRE_OVERLAY["config"]["zoom_max"]),
            },
            "hurricane": {
                "zoom_min": int(HURRICANE_OVERLAY["config"]["zoom_min"]),
                "zoom_max": int(HURRICANE_OVERLAY["config"]["zoom_max"]),
            },
        },
    }

    return f"""
<div id="risk-map-shell" class="risk-map-shell">
  <div class="risk-map-header">
    <div class="control-row">
      <label for="risk-time-slider"><strong>Time:</strong> <span id="risk-time-value">{FRAMES[0]}</span></label>
      <button id="risk-play" type="button">Play</button>
      <span class="risk-phase">Section: <span id="risk-phase">training</span></span>
    </div>
    <input id="risk-time-slider" type="range" min="0" max="{frame_count - 1}" value="0" step="1" />
  </div>

  <div class="overlay-controls">
    <div class="overlay-card">
      <label><input id="wf-enabled" type="checkbox" checked /> Wildfire overlay</label>
      <select id="wf-layer">
        <option value="risk">Risk Probability</option>
        <option value="activity">Activity Intensity</option>
        <option value="confidence">Confidence</option>
      </select>
    </div>
    <div class="overlay-card">
      <label><input id="hu-enabled" type="checkbox" checked /> Hurricane overlay</label>
      <select id="hu-layer">
        <option value="risk">Risk Probability</option>
        <option value="activity">Activity Intensity</option>
        <option value="confidence">Confidence</option>
      </select>
    </div>
  </div>

  <div class="risk-timeline">
    <div class="seg train" style="flex:{max(1, train_frames)};">Training</div>
    <div class="seg eval" style="flex:{max(1, eval_frames)};">Eval</div>
    <div class="seg infer" style="flex:{max(1, infer_frames)};">Inference</div>
  </div>

  <div id="risk-map" class="risk-map"></div>
  <div id="risk-map-status" class="risk-map-status"></div>

  <div class="risk-legend">
    <span class="legend-item"><i class="dot green"></i> Low</span>
    <span class="legend-item"><i class="dot yellow"></i> Medium</span>
    <span class="legend-item"><i class="dot red"></i> High</span>
  </div>
</div>
<style>
  .risk-map-shell {{ border: 1px solid #d7dde4; border-radius: 10px; padding: 12px; }}
  .risk-map-header {{ display: grid; gap: 8px; margin-bottom: 10px; }}
  .control-row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
  .risk-map-header input[type="range"] {{ width: 100%; }}
  .risk-phase {{ font-size: 13px; color: #334155; }}
  .overlay-controls {{ display: flex; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }}
  .overlay-card {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px; display: flex; gap: 10px; align-items: center; }}
  .risk-timeline {{ display: flex; height: 22px; overflow: hidden; border-radius: 999px; margin-bottom: 10px; font-size: 12px; }}
  .risk-timeline .seg {{ display: flex; align-items: center; justify-content: center; color: #0f172a; }}
  .risk-timeline .train {{ background: #c9f7d7; }}
  .risk-timeline .eval {{ background: #fde68a; }}
  .risk-timeline .infer {{ background: #fecaca; }}
  .risk-map {{ width: 100%; height: 560px; border-radius: 8px; }}
  .risk-map-status {{ margin-top: 8px; font-size: 13px; color: #0f172a; }}
  .risk-legend {{ margin-top: 8px; display: flex; gap: 14px; align-items: center; font-size: 13px; }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; border: 1px solid rgba(15, 23, 42, 0.25); }}
  .dot.green {{ background: rgba(46, 204, 113, 0.65); }}
  .dot.yellow {{ background: rgba(241, 196, 15, 0.7); }}
  .dot.red {{ background: rgba(231, 76, 60, 0.75); }}
</style>
<script>
(() => {{
  const root = document.getElementById("risk-map-shell");
  if (!root || root.dataset.ready === "1") return;
  root.dataset.ready = "1";

  const cfg = {json.dumps(js_config)};
  const slider = root.querySelector("#risk-time-slider");
  const valueNode = root.querySelector("#risk-time-value");
  const phaseNode = root.querySelector("#risk-phase");
  const playBtn = root.querySelector("#risk-play");
  const mapNode = root.querySelector("#risk-map");
  const statusNode = root.querySelector("#risk-map-status");

  const wfEnabled = root.querySelector("#wf-enabled");
  const wfLayer = root.querySelector("#wf-layer");
  const huEnabled = root.querySelector("#hu-enabled");
  const huLayer = root.querySelector("#hu-layer");

  let timer = null;
  let map = null;

  const phaseForFrame = (frame) => {{
    const year = Number(frame.slice(0, 4));
    if (year <= cfg.training_end_year) return "training";
    if (year <= cfg.eval_end_year) return "eval";
    return "inference";
  }};

  const currentFrame = () => cfg.frames[Number(slider.value)] ?? cfg.frames[0];

  const updateLabels = () => {{
    const frame = currentFrame();
    valueNode.textContent = frame;
    phaseNode.textContent = phaseForFrame(frame);
  }};

  const setPlaying = (on) => {{
    if (on && !timer) {{
      playBtn.textContent = "Pause";
      timer = setInterval(() => {{
        const next = (Number(slider.value) + 1) % cfg.frames.length;
        slider.value = String(next);
        installOverlay();
      }}, 900);
      return;
    }}
    if (!on && timer) {{
      clearInterval(timer);
      timer = null;
      playBtn.textContent = "Play";
    }}
  }};

  const tileUrl = (hazard, layer, frameIdx, z, x, y) => `/tiles/${{hazard}}/${{layer}}/${{frameIdx}}/${{z}}/${{x}}/${{y}}.png`;

  const pushOverlay = (hazard, layer, frameIdx, opacity) => {{
    const overlay = new google.maps.ImageMapType({{
      tileSize: new google.maps.Size(256, 256),
      opacity,
      getTileUrl: (coord, zoom) => {{
        const bound = 1 << zoom;
        if (coord.y < 0 || coord.y >= bound) return "";
        const limits = cfg.hazards[hazard];
        if (zoom < limits.zoom_min || zoom > limits.zoom_max) return "";
        const wrappedX = ((coord.x % bound) + bound) % bound;
        return tileUrl(hazard, layer, frameIdx, zoom, wrappedX, coord.y);
      }},
    }});
    map.overlayMapTypes.push(overlay);
  }};

  const installOverlay = () => {{
    const frameIdx = Number(slider.value);
    updateLabels();
    if (!map) return;

    map.overlayMapTypes.clear();
    if (wfEnabled.checked) {{
      pushOverlay("wildfire", wfLayer.value, frameIdx, wfLayer.value === "risk" ? 0.58 : 0.46);
    }}
    if (huEnabled.checked) {{
      pushOverlay("hurricane", huLayer.value, frameIdx, huLayer.value === "risk" ? 0.62 : 0.50);
    }}
  }};

  const initMap = () => {{
    map = new google.maps.Map(mapNode, {{
      center: {{ lat: cfg.center_lat, lng: cfg.center_lon }},
      zoom: cfg.zoom_min,
      minZoom: cfg.zoom_min,
      maxZoom: cfg.zoom_max,
      mapTypeControl: false,
      streetViewControl: false,
      fullscreenControl: false,
      clickableIcons: false,
    }});
    statusNode.textContent = `Precomputed monthly overlays loaded (${{cfg.frames.length}} frames).`;
    installOverlay();
  }};

  const loadGoogleMaps = () => {{
    if (!cfg.api_key) {{
      statusNode.textContent = "Set GMAPS_API_KEY to render Google Maps overlays.";
      updateLabels();
      return;
    }}
    if (window.google && window.google.maps) {{
      initMap();
      return;
    }}
    const callbackName = `gmapsInit_${{cfg.service_id}}`;
    window[callbackName] = () => {{
      delete window[callbackName];
      initMap();
    }};
    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${{cfg.api_key}}&callback=${{callbackName}}`;
    script.async = true;
    script.defer = true;
    script.onerror = () => {{
      statusNode.textContent = "Failed to load Google Maps JavaScript API.";
    }};
    document.head.appendChild(script);
  }};

  slider.addEventListener("input", installOverlay);
  wfEnabled.addEventListener("change", installOverlay);
  huEnabled.addEventListener("change", installOverlay);
  wfLayer.addEventListener("change", installOverlay);
  huLayer.addEventListener("change", installOverlay);
  playBtn.addEventListener("click", () => setPlaying(!timer));

  updateLabels();
  loadGoogleMaps();
}})();
</script>
"""


with gr.Blocks(title=SERVICE_NAME) as demo:
    gr.Markdown(f"# {SERVICE_NAME}")

    gr.Markdown("## Training")
    gr.Markdown(
        f"- Wildfire model version: **{WILDFIRE_MODEL_VERSION}**\n"
        f"- Hurricane model version: **{HURRICANE_MODEL_VERSION}**\n"
        f"- Wildfire training rows: **{int(WILDFIRE_MODEL_BUNDLE.get('dataset_rows', 0))}**\n"
        f"- Hurricane training rows: **{int(HURRICANE_MODEL_BUNDLE.get('dataset_rows', 0))}**"
    )

    gr.Markdown("## Eval")
    gr.Markdown(
        f"- Wildfire val accuracy: **{_as_percent(WILDFIRE_MODEL_BUNDLE.get('val_accuracy'))}**\n"
        f"- Wildfire val balanced accuracy: **{_as_percent(WILDFIRE_MODEL_BUNDLE.get('val_balanced_accuracy'))}**\n"
        f"- Wildfire val AUROC: **{_as_percent(WILDFIRE_MODEL_BUNDLE.get('val_auc'))}**\n"
        f"- Hurricane val accuracy: **{_as_percent(HURRICANE_MODEL_BUNDLE.get('val_accuracy'))}**\n"
        f"- Hurricane val balanced accuracy: **{_as_percent(HURRICANE_MODEL_BUNDLE.get('val_balanced_accuracy'))}**\n"
        f"- Hurricane val AUROC: **{_as_percent(HURRICANE_MODEL_BUNDLE.get('val_auc'))}**"
    )

    gr.Markdown("## Inference")
    gr.HTML(_map_html())

    with gr.Tab("Wildfire"):
        with gr.Row():
            region_id = gr.Textbox(label="Region ID", value="norcal")
            temp_c = gr.Number(label="Temperature (C)", value=34)
            humidity_pct = gr.Number(label="Humidity (%)", value=22)
            wind_kph = gr.Number(label="Wind (kph)", value=28)
            ffmc = gr.Number(label="FFMC", value=92.0)
            dmc = gr.Number(label="DMC", value=180.0)
            drought_code = gr.Number(label="DC", value=640.0)
            isi = gr.Number(label="ISI", value=12.0)

        wildfire_output = gr.JSON(label="Wildfire Prediction")
        wildfire_run = gr.Button("Predict Wildfire")
        wildfire_run.click(
            fn=predict_wildfire,
            inputs=[region_id, temp_c, humidity_pct, wind_kph, ffmc, dmc, drought_code, isi],
            outputs=wildfire_output,
        )

    with gr.Tab("Hurricane"):
        with gr.Row():
            storm_id = gr.Textbox(label="Storm ID", value="AL09")
            vmax_kt = gr.Number(label="Max Wind (kt)", value=70)
            min_pressure_mb = gr.Number(label="Min Pressure (mb)", value=980)
            lat = gr.Number(label="Latitude", value=22.5)
            lon = gr.Number(label="Longitude", value=-65.0)
            month = gr.Number(label="Month", value=9)
            dvmax_6h = gr.Number(label="Recent Wind Change (kt per 6h)", value=5.0)
            dpres_6h = gr.Number(label="Recent Pressure Change (mb per 6h)", value=-3.0)

        hurricane_output = gr.JSON(label="Hurricane Prediction")
        hurricane_run = gr.Button("Predict Hurricane")
        hurricane_run.click(
            fn=predict_hurricane,
            inputs=[storm_id, vmax_kt, min_pressure_mb, lat, lon, month, dvmax_6h, dpres_6h],
            outputs=hurricane_output,
        )


api = FastAPI(title=SERVICE_NAME)


@api.get("/health")
def health() -> dict[str, object]:
    return {
        "service": "disasters",
        "status": "ok",
        "frames": len(FRAMES),
        "wildfire_model_version": WILDFIRE_MODEL_VERSION,
        "hurricane_model_version": HURRICANE_MODEL_VERSION,
    }


@api.get("/tiles/{hazard}/{layer}/{frame_idx}/{z}/{x}/{y}.png")
def tile(hazard: str, layer: str, frame_idx: int, z: int, x: int, y: int) -> Response:
    if hazard not in OVERLAYS:
        raise HTTPException(status_code=404, detail="unknown hazard")
    if layer not in {"risk", "activity", "confidence"}:
        raise HTTPException(status_code=404, detail="unknown layer")

    config = OVERLAYS[hazard]["config"]
    z_min = int(config["zoom_min"])
    z_max = int(config["zoom_max"])
    if z < z_min or z > z_max:
        blank = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
        buffer = io.BytesIO()
        blank.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    try:
        png = _render_tile_png(hazard=hazard, layer=layer, frame_idx=frame_idx, z=z, x=x, y=y)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


app = gr.mount_gradio_app(api, demo, path="/")


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
