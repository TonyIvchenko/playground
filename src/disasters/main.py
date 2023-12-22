"""Unified climate-risk service with wildfires + huricaines overlays and inference."""

from __future__ import annotations

from functools import lru_cache
import html
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

try:
    from models.huricaines import load_model_bundle as load_huricaines_model_bundle
    from models.wildfires import load_model_bundle as load_wildfires_model_bundle
except ModuleNotFoundError:
    from src.disasters.models.huricaines import load_model_bundle as load_huricaines_model_bundle
    from src.disasters.models.wildfires import load_model_bundle as load_wildfires_model_bundle


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
GMAPS_API_KEY = os.getenv("GMAPS_API_KEY", "")
SERVICE_NAME = os.getenv("SERVICE_NAME", "Climate Risk Map Service")

WILDFIRES_MODEL_PATH = Path(
    os.getenv("WILDFIRES_MODEL_PATH", str(Path(__file__).resolve().parent / "models" / "wildfires.pt"))
)
HURICAINES_MODEL_PATH = Path(
    os.getenv("HURICAINES_MODEL_PATH", str(Path(__file__).resolve().parent / "models" / "huricaines.pt"))
)

WILDFIRES_TILES_DIR = Path(__file__).resolve().parent / "tiles" / "wildfires"
HURICAINES_TILES_DIR = Path(__file__).resolve().parent / "tiles" / "huricaines"

TILE_SIZE = 256
SAMPLE_SIZE = 64


wildfires_model, wildfires_mean, wildfires_std, WILDFIRES_MODEL_VERSION = load_wildfires_model_bundle(WILDFIRES_MODEL_PATH)
huricaines_model, huricaines_mean, huricaines_std, HURICAINES_MODEL_VERSION = load_huricaines_model_bundle(HURICAINES_MODEL_PATH)

WILDFIRES_MODEL_BUNDLE = torch.load(WILDFIRES_MODEL_PATH, map_location="cpu", weights_only=True)
HURICAINES_MODEL_BUNDLE = torch.load(HURICAINES_MODEL_PATH, map_location="cpu", weights_only=True)


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
        risk = np.zeros((len(frames), default_shape[0], default_shape[1]), dtype=np.float32)
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


def _as_percent(metric: object) -> str:
    if isinstance(metric, (int, float)):
        return f"{float(metric) * 100.0:.2f}%"
    return "n/a"


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


def _colorize(sampled_risk: np.ndarray, sampled_conf: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rgba = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE, 4), dtype=np.uint8)
    low = sampled_risk < 0.33
    mid = (sampled_risk >= 0.33) & (sampled_risk < 0.66)
    high = sampled_risk >= 0.66

    rgba[low, 0:3] = np.array([46, 204, 113], dtype=np.uint8)
    rgba[mid, 0:3] = np.array([241, 196, 15], dtype=np.uint8)
    rgba[high, 0:3] = np.array([231, 76, 60], dtype=np.uint8)

    conf = np.clip(sampled_conf, 0.0, 1.0)
    rgba[..., 3] = np.clip(25.0 + conf * 225.0, 0, 255).astype(np.uint8)

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

    sampled_risk, valid_mask = _sample_layer(risk_grid, config=overlay["config"], z=z, x=x, y=y)
    sampled_conf, _ = _sample_layer(conf_grid, config=overlay["config"], z=z, x=x, y=y)
    rgba_small = _colorize(sampled_risk=sampled_risk, sampled_conf=sampled_conf, valid_mask=valid_mask)
    image = Image.fromarray(rgba_small, mode="RGBA").resize((TILE_SIZE, TILE_SIZE), resample=Image.Resampling.BILINEAR)

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
        [[float(temp_c), float(humidity_pct), float(wind_kph), float(ffmc), float(dmc), float(drought_code), float(isi)]],
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
    start_year = int(FRAMES[0][:4])
    end_year = int(FRAMES[-1][:4])
    default_zoom = max(
        int(WILDFIRES_OVERLAY["config"]["zoom_min"]),
        int(HURICAINES_OVERLAY["config"]["zoom_min"]),
    )

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
    }
    config_blob = html.escape(json.dumps(js_config), quote=True)

    return f"""
<div id="risk-map-shell" class="risk-map-shell" data-config="{config_blob}">
  <div class="risk-layout">
    <section class="risk-map-pane">
      <div class="risk-map-header">
        <div class="control-row">
          <label for="risk-time-slider"><strong>Time:</strong> <span id="risk-time-value">{FRAMES[0]}</span></label>
          <button id="risk-play" type="button">Play</button>
          <span class="risk-phase">Section: <span id="risk-phase">training</span></span>
        </div>
        <input id="risk-time-slider" type="range" min="0" max="{frame_count - 1}" value="0" step="1" />
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
    </section>

    <aside class="risk-side-panel">
      <h3>Layers</h3>
      <div class="layer-card">
        <div class="layer-head">
          <label for="wf-visibility">Wildfires</label>
          <select id="wf-visibility">
            <option value="on" selected>Show</option>
            <option value="off">Hide</option>
          </select>
        </div>
        <div>Overlay enabled: color = probability, opacity = confidence/intensity.</div>
      </div>

      <div class="layer-card">
        <div class="layer-head">
          <label for="hu-visibility">Huricaines</label>
          <select id="hu-visibility">
            <option value="on" selected>Show</option>
            <option value="off">Hide</option>
          </select>
        </div>
        <div>Overlay enabled: color = probability, opacity = confidence/intensity.</div>
      </div>

      <h3>Sections</h3>
      <div class="section-card">
        <div><strong>Training:</strong> {start_year}-{train_end_year}</div>
        <div><strong>Eval:</strong> {train_end_year + 1}-{eval_end_year}</div>
        <div><strong>Inference:</strong> {eval_end_year + 1}-{end_year}</div>
      </div>

      <h3>Model Summary</h3>
      <div class="section-card">
        <div><strong>Wildfires:</strong> v{WILDFIRES_MODEL_VERSION}</div>
        <div>Rows: {int(WILDFIRES_MODEL_BUNDLE.get("dataset_rows", 0))}</div>
        <div>AUROC: {_as_percent(WILDFIRES_MODEL_BUNDLE.get("val_auc"))}</div>
        <hr />
        <div><strong>Huricaines:</strong> v{HURICAINES_MODEL_VERSION}</div>
        <div>Rows: {int(HURICAINES_MODEL_BUNDLE.get("dataset_rows", 0))}</div>
        <div>AUROC: {_as_percent(HURICAINES_MODEL_BUNDLE.get("val_auc"))}</div>
      </div>
    </aside>
  </div>
</div>
<style>
  .risk-map-shell {{ border: 1px solid #d7dde4; border-radius: 10px; padding: 10px; background: #ffffff; }}
  .risk-layout {{ display: grid; gap: 10px; grid-template-columns: minmax(0, 1fr) 320px; }}
  .risk-map-pane {{ min-width: 0; }}
  .risk-side-panel {{ border: 1px solid #d7dde4; border-radius: 8px; padding: 10px; background: #f8fafc; height: fit-content; }}
  .risk-side-panel h3 {{ margin: 8px 0 8px; font-size: 16px; }}
  .risk-map-header {{ display: grid; gap: 8px; margin-bottom: 10px; }}
  .control-row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
  .risk-map-header input[type="range"] {{ width: 100%; }}
  .risk-phase {{ font-size: 13px; color: #334155; }}
  .risk-timeline {{ display: flex; height: 24px; overflow: hidden; border-radius: 999px; margin-bottom: 10px; font-size: 12px; border: 1px solid #e2e8f0; }}
  .risk-timeline .seg {{ display: flex; align-items: center; justify-content: center; color: #0f172a; }}
  .risk-timeline .train {{ background: #c9f7d7; }}
  .risk-timeline .eval {{ background: #fde68a; }}
  .risk-timeline .infer {{ background: #fecaca; }}
  .risk-map {{ width: 100%; height: 640px; border-radius: 8px; border: 1px solid #d7dde4; }}
  .risk-map-status {{ margin-top: 8px; font-size: 13px; color: #0f172a; min-height: 18px; }}
  .risk-map-status.error {{ color: #b91c1c; font-weight: 600; }}
  .risk-legend {{ margin-top: 8px; display: flex; gap: 14px; align-items: center; font-size: 13px; }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; border: 1px solid rgba(15, 23, 42, 0.25); }}
  .dot.green {{ background: rgba(46, 204, 113, 0.85); }}
  .dot.yellow {{ background: rgba(241, 196, 15, 0.85); }}
  .dot.red {{ background: rgba(231, 76, 60, 0.85); }}
  .layer-card, .section-card {{ border: 1px solid #d7dde4; border-radius: 8px; padding: 8px; background: #ffffff; display: grid; gap: 8px; margin-bottom: 8px; }}
  .layer-head {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
  @media (max-width: 1100px) {{
    .risk-layout {{ grid-template-columns: 1fr; }}
    .risk-map {{ height: 500px; }}
  }}
</style>
"""


def _map_bootstrap_js() -> str:
    return """
() => {
  const root = document.getElementById("risk-map-shell");
  if (!root || root.dataset.ready === "1") return [];
  root.dataset.ready = "1";

  let cfg = {};
  try {
    cfg = JSON.parse(root.dataset.config || "{}");
  } catch (err) {
    console.error("Failed to parse map config", err);
  }

  const slider = root.querySelector("#risk-time-slider");
  const valueNode = root.querySelector("#risk-time-value");
  const phaseNode = root.querySelector("#risk-phase");
  const playBtn = root.querySelector("#risk-play");
  const mapNode = root.querySelector("#risk-map");
  const statusNode = root.querySelector("#risk-map-status");

  const wfVisibility = root.querySelector("#wf-visibility");
  const huVisibility = root.querySelector("#hu-visibility");

  let timer = null;
  let map = null;
  let mapEngine = "";
  let leafletOverlays = [];

  const updateStatus = (text, isError = false) => {
    statusNode.textContent = text;
    statusNode.classList.toggle("error", Boolean(isError));
  };

  const phaseForFrame = (frame) => {
    const year = Number((frame || "").slice(0, 4));
    if (year <= Number(cfg.training_end_year)) return "training";
    if (year <= Number(cfg.eval_end_year)) return "eval";
    return "inference";
  };

  const currentFrame = () => (cfg.frames && cfg.frames[Number(slider.value)]) || (cfg.frames ? cfg.frames[0] : "n/a");

  const updateLabels = () => {
    const frame = currentFrame();
    valueNode.textContent = frame;
    phaseNode.textContent = phaseForFrame(frame);
  };

  const setPlaying = (on) => {
    if (on && !timer) {
      playBtn.textContent = "Pause";
      timer = setInterval(() => {
        const next = (Number(slider.value) + 1) % cfg.frames.length;
        slider.value = String(next);
        installOverlay();
      }, 900);
      return;
    }
    if (!on && timer) {
      clearInterval(timer);
      timer = null;
      playBtn.textContent = "Play";
    }
  };

  const tileUrl = (hazard, frameIdx, z, x, y) =>
    `/tiles/${hazard}/${frameIdx}/${z}/${x}/${y}.png`;

  const clearOverlays = () => {
    if (!map) return;
    if (mapEngine === "google") {
      map.overlayMapTypes.clear();
      return;
    }
    if (mapEngine === "leaflet") {
      leafletOverlays.forEach((layer) => map.removeLayer(layer));
      leafletOverlays = [];
    }
  };

  const pushOverlay = (hazard, frameIdx) => {
    if (!map) return;
    if (mapEngine === "google") {
      const overlay = new google.maps.ImageMapType({
        tileSize: new google.maps.Size(256, 256),
        opacity: 1.0,
        getTileUrl: (coord, zoom) => {
          if (coord.y < 0 || coord.y >= (1 << zoom)) return "";
          const wrappedX = ((coord.x % (1 << zoom)) + (1 << zoom)) % (1 << zoom);
          return tileUrl(hazard, frameIdx, zoom, wrappedX, coord.y);
        },
      });
      map.overlayMapTypes.push(overlay);
      return;
    }
    if (mapEngine === "leaflet") {
      const layer = window.L.tileLayer(`/tiles/${hazard}/${frameIdx}/{z}/{x}/{y}.png`, {
        opacity: 1.0,
        maxZoom: Number(cfg.zoom_max || 10),
        minZoom: Number(cfg.zoom_min || 2),
      });
      layer.addTo(map);
      leafletOverlays.push(layer);
    }
  };

  const installOverlay = () => {
    updateLabels();
    if (!map) return;

    const frameIdx = Number(slider.value);
    clearOverlays();

    let active = 0;
    if (wfVisibility.value === "on") {
      active += 1;
      pushOverlay("wildfires", frameIdx);
    }
    if (huVisibility.value === "on") {
      active += 1;
      pushOverlay("huricaines", frameIdx);
    }
    if (active === 0) {
      updateStatus("No layers enabled. Turn on at least one layer.");
    } else {
      updateStatus(`Showing ${active} layer(s) for ${currentFrame()}.`);
    }
  };

  const initGoogleMap = () => {
    mapEngine = "google";
    map = new google.maps.Map(mapNode, {
      center: { lat: Number(cfg.center_lat || 36.0), lng: Number(cfg.center_lon || -95.0) },
      zoom: Number(cfg.default_zoom || 4),
      minZoom: Number(cfg.zoom_min || 2),
      maxZoom: Number(cfg.zoom_max || 10),
      mapTypeControl: true,
      streetViewControl: false,
      fullscreenControl: true,
      clickableIcons: false,
    });
    setTimeout(() => {
      if (window.google && window.google.maps) {
        google.maps.event.trigger(map, "resize");
      }
    }, 150);
    installOverlay();
  };

  const initLeafletMap = () => {
    mapEngine = "leaflet";
    map = window.L.map(mapNode, {
      center: [Number(cfg.center_lat || 36.0), Number(cfg.center_lon || -95.0)],
      zoom: Number(cfg.default_zoom || 4),
      minZoom: Number(cfg.zoom_min || 2),
      maxZoom: Number(cfg.zoom_max || 10),
      zoomControl: true,
    });
    window.L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);
    setTimeout(() => map.invalidateSize(), 150);
    updateStatus("Using OpenStreetMap fallback. Set GMAPS_API_KEY to use Google base map.");
    installOverlay();
  };

  const ensureLeaflet = () =>
    new Promise((resolve, reject) => {
      if (window.L && window.L.map) {
        resolve();
        return;
      }

      if (!document.getElementById("leaflet-css")) {
        const link = document.createElement("link");
        link.id = "leaflet-css";
        link.rel = "stylesheet";
        link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
        document.head.appendChild(link);
      }

      const existing = document.getElementById("leaflet-js");
      if (existing) {
        existing.addEventListener("load", () => resolve(), { once: true });
        existing.addEventListener("error", () => reject(new Error("Leaflet JS failed to load")), { once: true });
        return;
      }

      const script = document.createElement("script");
      script.id = "leaflet-js";
      script.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
      script.async = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("Leaflet JS failed to load"));
      document.head.appendChild(script);
    });

  const loadGoogleMaps = () => {
    if (!cfg.api_key) {
      ensureLeaflet()
        .then(initLeafletMap)
        .catch(() => updateStatus("Failed to load map libraries.", true));
      return;
    }
    if (window.google && window.google.maps) {
      initGoogleMap();
      return;
    }
    const callbackName = `gmapsInit_${cfg.service_id}_${Date.now()}`;
    const fallbackTimer = setTimeout(() => {
      if (!map) {
        ensureLeaflet()
          .then(initLeafletMap)
          .catch(() => updateStatus("Failed to initialize Google Maps and fallback map.", true));
      }
    }, 5000);
    window[callbackName] = () => {
      clearTimeout(fallbackTimer);
      delete window[callbackName];
      initGoogleMap();
    };
    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${cfg.api_key}&callback=${callbackName}&v=weekly`;
    script.async = true;
    script.defer = true;
    script.onerror = () => {
      clearTimeout(fallbackTimer);
      ensureLeaflet()
        .then(initLeafletMap)
        .catch(() => updateStatus("Failed to load Google Maps and fallback map.", true));
    };
    document.head.appendChild(script);
  };

  slider.addEventListener("input", installOverlay);
  wfVisibility.addEventListener("change", installOverlay);
  huVisibility.addEventListener("change", installOverlay);
  playBtn.addEventListener("click", () => setPlaying(!timer));

  updateLabels();
  loadGoogleMaps();
  return [];
}
"""


with gr.Blocks(title=SERVICE_NAME) as demo:
    gr.Markdown(f"# {SERVICE_NAME}")
    gr.HTML(_map_html())
    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js=_map_bootstrap_js(),
        queue=False,
        show_progress="hidden",
    )

    with gr.Tab("Wildfires"):
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
            inputs=[region_id, temp_c, humidity_pct, wind_kph, ffmc, dmc, drought_code, isi],
            outputs=wildfires_output,
        )

    with gr.Tab("Huricaines"):
        with gr.Row():
            storm_id = gr.Textbox(label="Storm ID", value="AL09")
            vmax_kt = gr.Number(label="Max Wind (kt)", value=70)
            min_pressure_mb = gr.Number(label="Min Pressure (mb)", value=980)
            lat = gr.Number(label="Latitude", value=22.5)
            lon = gr.Number(label="Longitude", value=-65.0)
            month = gr.Number(label="Month", value=9)
            dvmax_6h = gr.Number(label="Recent Wind Change (kt per 6h)", value=5.0)
            dpres_6h = gr.Number(label="Recent Pressure Change (mb per 6h)", value=-3.0)

        huricaines_output = gr.JSON(label="Huricaines Prediction")
        huricaines_run = gr.Button("Predict Huricaines")
        huricaines_run.click(
            fn=predict_huricaines,
            inputs=[storm_id, vmax_kt, min_pressure_mb, lat, lon, month, dvmax_6h, dpres_6h],
            outputs=huricaines_output,
        )


api = FastAPI(title=SERVICE_NAME)


@api.get("/health")
def health() -> dict[str, object]:
    return {
        "service": "disasters",
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
def tile_legacy(hazard: str, layer: str, frame_idx: int, z: int, x: int, y: int) -> Response:
    # Backward-compatible path; the service now exposes one overlay per hazard.
    if layer not in {"risk", "activity", "confidence"}:
        raise HTTPException(status_code=404, detail="unknown layer")
    return _serve_tile(hazard=hazard, frame_idx=frame_idx, z=z, x=x, y=y)


app = gr.mount_gradio_app(api, demo, path="/")


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
