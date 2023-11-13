"""Wildfire service with Gradio UI + Google Maps monthly tile overlays."""

from __future__ import annotations

from functools import lru_cache
import io
import json
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException, Response
import gradio as gr
import numpy as np
from PIL import Image
import torch
import uvicorn

from wildfire.model import load_model_bundle


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", os.getenv("PORT", "8010")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent / "model" / "wildfire_model.pt")))
TILES_DIR = Path(__file__).resolve().parent / "tiles"
CUBE_PATH = TILES_DIR / "overlay_cube.npz"
OVERLAY_CONFIG_PATH = TILES_DIR / "overlay_config.json"
GMAPS_API_KEY = os.getenv("GMAPS_API_KEY", "")
SERVICE_NAME = os.getenv("SERVICE_NAME", "Wildfire Ignition-Risk Service")

TILE_SIZE = 256
SAMPLE_SIZE = 64


model, feature_mean, feature_std, MODEL_VERSION = load_model_bundle(MODEL_PATH)
MODEL_BUNDLE = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)


def _load_overlay() -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float | int | str]]:
    config: dict[str, float | int | str] = {
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
    }
    if OVERLAY_CONFIG_PATH.exists():
        loaded = json.loads(OVERLAY_CONFIG_PATH.read_text(encoding="utf-8"))
        config.update(loaded)

    if CUBE_PATH.exists():
        cube = np.load(CUBE_PATH)
        risk = cube["risk"].astype(np.float32)
        activity = cube["activity"].astype(np.float32)
        frames = [str(x) for x in cube["frames"].tolist()]
    else:
        frames = [f"{year:04d}-{month:02d}" for year in range(int(config["start_year"]), int(config["end_year"]) + 1) for month in range(1, 13)]
        risk = np.zeros((len(frames), 120, 180), dtype=np.float32)
        activity = np.zeros_like(risk)

    return risk, activity, frames, config


RISK_CUBE, ACTIVITY_CUBE, FRAMES, OVERLAY_CONFIG = _load_overlay()


def _risk_level(probability: float) -> str:
    if probability < 0.20:
        return "low"
    if probability < 0.40:
        return "moderate"
    if probability < 0.65:
        return "high"
    return "extreme"


def _as_percent(metric: object) -> str:
    if isinstance(metric, (int, float)):
        return f"{float(metric) * 100.0:.2f}%"
    return "n/a"


def _phase_for_frame(frame: str) -> str:
    year = int(frame[:4])
    if year <= int(OVERLAY_CONFIG["training_end_year"]):
        return "training"
    if year <= int(OVERLAY_CONFIG["eval_end_year"]):
        return "eval"
    return "inference"


def _tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 2 ** z
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * y / n))))
    lat_bottom = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + 1) / n))))
    return float(lat_top), float(lat_bottom), float(lon_left), float(lon_right)


def _sample_layer(layer_grid: np.ndarray, z: int, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    lat_top, lat_bottom, lon_left, lon_right = _tile_bounds(z, x, y)
    lat_min = float(OVERLAY_CONFIG["lat_min"])
    lat_max = float(OVERLAY_CONFIG["lat_max"])
    lon_min = float(OVERLAY_CONFIG["lon_min"])
    lon_max = float(OVERLAY_CONFIG["lon_max"])

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


@lru_cache(maxsize=50000)
def _render_tile_png(layer: str, frame_idx: int, z: int, x: int, y: int) -> bytes:
    if frame_idx < 0 or frame_idx >= len(FRAMES):
        raise ValueError("frame index out of range")

    if layer == "risk":
        grid = RISK_CUBE[frame_idx]
    elif layer == "activity":
        grid = ACTIVITY_CUBE[frame_idx]
    elif layer == "confidence":
        grid = np.clip(ACTIVITY_CUBE[frame_idx], 0.0, 1.0)
    else:
        raise ValueError("unknown layer")

    sampled, valid_mask = _sample_layer(grid, z=z, x=x, y=y)
    rgba_small = _colorize(layer=layer, sampled=sampled, valid_mask=valid_mask)
    image = Image.fromarray(rgba_small, mode="RGBA").resize((TILE_SIZE, TILE_SIZE), resample=Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


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


def _map_html() -> str:
    frame_count = len(FRAMES)
    train_frames = sum(1 for f in FRAMES if _phase_for_frame(f) == "training")
    eval_frames = sum(1 for f in FRAMES if _phase_for_frame(f) == "eval")
    infer_frames = max(1, frame_count - train_frames - eval_frames)

    js_config = {
        "service_id": "wildfire",
        "api_key": GMAPS_API_KEY,
        "frames": FRAMES,
        "zoom_min": int(OVERLAY_CONFIG["zoom_min"]),
        "zoom_max": int(OVERLAY_CONFIG["zoom_max"]),
        "center_lat": float(OVERLAY_CONFIG["center_lat"]),
        "center_lon": float(OVERLAY_CONFIG["center_lon"]),
        "training_end_year": int(OVERLAY_CONFIG["training_end_year"]),
        "eval_end_year": int(OVERLAY_CONFIG["eval_end_year"]),
    }

    return f"""
<div id="wildfire-map-shell" class="risk-map-shell">
  <div class="risk-map-header">
    <div class="control-row">
      <label for="wildfire-time-slider"><strong>Time:</strong> <span id="wildfire-time-value">{FRAMES[0]}</span></label>
      <button id="wildfire-play" type="button">Play</button>
      <select id="wildfire-layer">
        <option value="risk">Risk Probability</option>
        <option value="activity">Activity Intensity</option>
        <option value="confidence">Confidence</option>
      </select>
      <span class="risk-phase">Section: <span id="wildfire-phase">training</span></span>
    </div>
    <input id="wildfire-time-slider" type="range" min="0" max="{frame_count - 1}" value="0" step="1" />
  </div>
  <div class="risk-timeline">
    <div class="seg train" style="flex:{max(1, train_frames)};">Training</div>
    <div class="seg eval" style="flex:{max(1, eval_frames)};">Eval</div>
    <div class="seg infer" style="flex:{max(1, infer_frames)};">Inference</div>
  </div>
  <div id="wildfire-map" class="risk-map"></div>
  <div id="wildfire-map-status" class="risk-map-status"></div>
  <div class="risk-legend">
    <span class="legend-item"><i class="dot green"></i> Low</span>
    <span class="legend-item"><i class="dot yellow"></i> Medium</span>
    <span class="legend-item"><i class="dot red"></i> High</span>
  </div>
</div>
<style>
  .risk-map-shell {{ border: 1px solid #d7dde4; border-radius: 10px; padding: 12px; }}
  .risk-map-header {{ display: grid; gap: 8px; margin-bottom: 8px; }}
  .control-row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
  .risk-map-header input[type="range"] {{ width: 100%; }}
  .risk-phase {{ font-size: 13px; color: #334155; }}
  .risk-timeline {{ display: flex; height: 22px; overflow: hidden; border-radius: 999px; margin-bottom: 10px; font-size: 12px; }}
  .risk-timeline .seg {{ display: flex; align-items: center; justify-content: center; color: #0f172a; }}
  .risk-timeline .train {{ background: #c9f7d7; }}
  .risk-timeline .eval {{ background: #fde68a; }}
  .risk-timeline .infer {{ background: #fecaca; }}
  .risk-map {{ width: 100%; height: 520px; border-radius: 8px; }}
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
  const root = document.getElementById("wildfire-map-shell");
  if (!root || root.dataset.ready === "1") return;
  root.dataset.ready = "1";

  const cfg = {json.dumps(js_config)};
  const slider = root.querySelector("#wildfire-time-slider");
  const valueNode = root.querySelector("#wildfire-time-value");
  const phaseNode = root.querySelector("#wildfire-phase");
  const layerSelect = root.querySelector("#wildfire-layer");
  const playBtn = root.querySelector("#wildfire-play");
  const mapNode = root.querySelector("#wildfire-map");
  const statusNode = root.querySelector("#wildfire-map-status");

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

  const tileUrl = (layer, frameIdx, z, x, y) => `/tiles/${{layer}}/${{frameIdx}}/${{z}}/${{x}}/${{y}}.png`;

  const installOverlay = () => {{
    const frameIdx = Number(slider.value);
    updateLabels();
    if (!map) return;
    map.overlayMapTypes.clear();
    const layer = layerSelect.value;
    const overlay = new google.maps.ImageMapType({{
      tileSize: new google.maps.Size(256, 256),
      opacity: layer === "risk" ? 0.66 : 0.55,
      getTileUrl: (coord, zoom) => {{
        const bound = 1 << zoom;
        if (coord.y < 0 || coord.y >= bound) return "";
        const wrappedX = ((coord.x % bound) + bound) % bound;
        return tileUrl(layer, frameIdx, zoom, wrappedX, coord.y);
      }},
    }});
    map.overlayMapTypes.push(overlay);
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
  layerSelect.addEventListener("change", installOverlay);
  playBtn.addEventListener("click", () => setPlaying(!timer));

  updateLabels();
  loadGoogleMaps();
}})();
</script>
"""


with gr.Blocks(title=SERVICE_NAME) as demo:
    gr.Markdown(f"# {SERVICE_NAME}")
    gr.Markdown(f"Model version: `{MODEL_VERSION}`")

    gr.Markdown("## Training")
    gr.Markdown(
        f"- Training window ends: **{int(OVERLAY_CONFIG['training_end_year'])}**\n"
        f"- Training rows: **{int(MODEL_BUNDLE.get('dataset_rows', 0))}**"
    )

    gr.Markdown("## Eval")
    gr.Markdown(
        f"- Validation accuracy: **{_as_percent(MODEL_BUNDLE.get('val_accuracy'))}**\n"
        f"- Validation balanced accuracy: **{_as_percent(MODEL_BUNDLE.get('val_balanced_accuracy'))}**\n"
        f"- Validation AUROC: **{_as_percent(MODEL_BUNDLE.get('val_auc'))}**"
    )

    gr.Markdown("## Inference")
    gr.HTML(_map_html())

    with gr.Row():
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


api = FastAPI(title=SERVICE_NAME)


@api.get("/health")
def health() -> dict[str, object]:
    return {
        "service": "wildfire",
        "model_version": MODEL_VERSION,
        "status": "ok",
        "frames": len(FRAMES),
    }


@api.get("/tiles/{layer}/{frame_idx}/{z}/{x}/{y}.png")
def tile(layer: str, frame_idx: int, z: int, x: int, y: int) -> Response:
    if layer not in {"risk", "activity", "confidence"}:
        raise HTTPException(status_code=404, detail="unknown layer")

    z_min = int(OVERLAY_CONFIG["zoom_min"])
    z_max = int(OVERLAY_CONFIG["zoom_max"])
    if z < z_min or z > z_max:
        blank = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
        buffer = io.BytesIO()
        blank.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    try:
        png = _render_tile_png(layer=layer, frame_idx=frame_idx, z=z, x=x, y=y)
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
