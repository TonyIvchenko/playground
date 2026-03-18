"""Standalone traffic-safety app with nationwide weekly risk overlay and predictor."""

from __future__ import annotations

from functools import lru_cache
import html
import io
import json
import joblib
import math
import os
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
import gradio as gr
import h3
import numpy as np
from PIL import Image
import uvicorn


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
GMAPS_API_KEY = os.getenv("GMAPS_API_KEY", "")
SERVICE_NAME = os.getenv("SERVICE_NAME", "Traffic Safety")

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
STATIC_URL = "/traffic-safety-static"
MODEL_PATH = APP_DIR / "models" / "traffic_safety.joblib"
TILES_DIR = APP_DIR / "tiles"

MAP_HEAD = """
<link rel="stylesheet" href="/traffic-safety-static/map.css">
<script src="/traffic-safety-static/map.js" defer></script>
""".strip()

TILE_SIZE = 256
SAMPLE_SIZE = 64


def print_http_startup(service_name: str, host: str, port: int) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.service_startup import print_http_service_startup

    print_http_service_startup(service_name, host, port)


def _weekly_frame_labels() -> list[str]:
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return [f"{weekdays[idx // 24]} {idx % 24:02d}:00" for idx in range(24 * 7)]


def _load_joblib_bundle(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    loaded = joblib.load(path)
    return loaded if isinstance(loaded, dict) else {}


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
        frames = [str(value) for value in cube["frames"].tolist()]
    else:
        frames = _weekly_frame_labels()
        risk = np.zeros((len(frames), default_shape[0], default_shape[1]), dtype=np.float32)
        confidence = np.zeros_like(risk)

    return {
        "risk": risk,
        "confidence": confidence,
        "frames": frames,
        "config": config,
    }


OVERLAY = _load_overlay(
    cube_path=TILES_DIR / "overlay.npz",
    config_path=TILES_DIR / "overlay.json",
    default_config={
        "timeline_type": "weekly_cycle",
        "month": 1,
        "zoom_min": 3,
        "zoom_max": 9,
        "center_lat": 39.5,
        "center_lon": -98.35,
        "lat_min": 18.0,
        "lat_max": 72.0,
        "lon_min": -179.0,
        "lon_max": -66.0,
        "model_version": "missing",
    },
    default_shape=(360, 760),
)
MODEL_BUNDLE = _load_joblib_bundle(MODEL_PATH)
MODEL_VERSION = str(
    MODEL_BUNDLE.get("model_version", OVERLAY["config"].get("model_version", "missing"))
)
CELL_INDEX = {
    str(cell): idx for idx, cell in enumerate(MODEL_BUNDLE.get("candidate_cells", []))
}


def _first_metric(bundle: dict[str, object], keys: list[str]) -> float | None:
    for key in keys:
        value = bundle.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    nested_metrics = bundle.get("metrics")
    if isinstance(nested_metrics, dict):
        for key in keys:
            value = nested_metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _timeline() -> dict[str, object]:
    frame_count = len(OVERLAY["frames"])
    month_value = int(OVERLAY["config"].get("month", 1))
    month_labels = [
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    month_label = month_labels[month_value] if 1 <= month_value <= 12 else "Current"
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "type": "weekly_cycle",
        "step_pct": 100.0 / max(1, frame_count - 1),
        "ticks": [
            {"label": weekday, "frame_idx": idx * 24}
            for idx, weekday in enumerate(weekdays)
        ],
        "phases": [
            {
                "kind": "live",
                "label": f"{month_label} weekly pattern",
                "count": max(1, frame_count),
            }
        ],
    }


def _risk_level(probability: float) -> str:
    quantiles = MODEL_BUNDLE.get("risk_quantiles")
    if (
        isinstance(quantiles, list)
        and len(quantiles) >= 3
        and all(isinstance(value, (int, float)) for value in quantiles[:3])
    ):
        low_cut, mid_cut, high_cut = (float(value) for value in quantiles[:3])
        if probability < low_cut:
            return "low"
        if probability < mid_cut:
            return "moderate"
        if probability < high_cut:
            return "high"
        return "extreme"
    if probability < 0.10:
        return "low"
    if probability < 0.25:
        return "moderate"
    if probability < 0.45:
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

    height, width = layer_grid.shape
    iy = ((lat_max - row_lats_clamped) / (lat_max - lat_min) * (height - 1)).astype(
        np.int32
    )
    ix = ((col_lons_clamped - lon_min) / (lon_max - lon_min) * (width - 1)).astype(
        np.int32
    )

    sampled = layer_grid[iy[:, None], ix[None, :]].astype(np.float32)
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
    impact = np.clip((sampled_risk - 0.08) / 0.92, 0.0, 1.0)
    rgba[..., 3] = np.clip(conf * impact * 255.0, 0, 255).astype(np.uint8)
    rgba[~valid_mask, 3] = 0
    return rgba


@lru_cache(maxsize=40000)
def _render_tile_png(frame_idx: int, z: int, x: int, y: int) -> bytes:
    frames = OVERLAY["frames"]
    if frame_idx < 0 or frame_idx >= len(frames):
        raise ValueError("frame index out of range")

    risk_grid = OVERLAY["risk"][frame_idx]
    conf_grid = np.clip(OVERLAY["confidence"][frame_idx], 0.0, 1.0)
    sampled_risk, valid_mask = _sample_layer(
        risk_grid,
        config=OVERLAY["config"],
        z=z,
        x=x,
        y=y,
    )
    sampled_conf, _ = _sample_layer(
        conf_grid,
        config=OVERLAY["config"],
        z=z,
        x=x,
        y=y,
    )
    rgba_small = _colorize(sampled_risk, sampled_conf, valid_mask)
    image = Image.fromarray(rgba_small, mode="RGBA").resize(
        (TILE_SIZE, TILE_SIZE),
        resample=Image.Resampling.BILINEAR,
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def predict_traffic_safety(
    lat: float,
    lon: float,
    day_of_week: int,
    hour: int,
    month: int,
) -> dict[str, object]:
    if not MODEL_BUNDLE:
        raise RuntimeError(f"traffic safety model is unavailable; expected {MODEL_PATH}")

    resolution = int(MODEL_BUNDLE.get("resolution", 5))
    cell_id = h3.latlng_to_cell(float(lat), float(lon), resolution)
    idx = CELL_INDEX.get(cell_id)

    day_of_week = max(1, min(7, int(day_of_week)))
    hour = max(0, min(23, int(hour)))
    month = max(1, min(12, int(month)))

    if idx is None:
        return {
            "model_version": MODEL_VERSION,
            "cell_id": cell_id,
            "lat": float(lat),
            "lon": float(lon),
            "local_day_of_week": day_of_week,
            "local_hour": hour,
            "month": month,
            "historical_cell_events": 0,
            "historical_same_hour_events": 0,
            "risk_score": 0.0,
            "risk_level": "low",
        }

    model = MODEL_BUNDLE["model"]
    candidate_lats = np.asarray(MODEL_BUNDLE["candidate_lats"], dtype=np.float32)
    candidate_lons = np.asarray(MODEL_BUNDLE["candidate_lons"], dtype=np.float32)
    cell_total_counts = np.asarray(MODEL_BUNDLE["cell_total_counts"], dtype=np.float32)
    cell_hour_counts = np.asarray(MODEL_BUNDLE["cell_hour_counts"], dtype=np.float32)

    hour_of_week = (day_of_week - 1) * 24 + hour
    hour_angle = 2.0 * math.pi * float(hour) / 24.0
    dow_angle = 2.0 * math.pi * float(day_of_week - 1) / 7.0
    month_angle = 2.0 * math.pi * float(month) / 12.0
    prior_total = float(cell_total_counts[idx])
    prior_same_hour = float(cell_hour_counts[idx, hour_of_week])

    features = np.array(
        [
            [
                float(candidate_lats[idx]),
                float(candidate_lons[idx]),
                math.sin(hour_angle),
                math.cos(hour_angle),
                math.sin(dow_angle),
                math.cos(dow_angle),
                math.sin(month_angle),
                math.cos(month_angle),
                math.log1p(prior_total),
                math.log1p(prior_same_hour),
                prior_same_hour / max(prior_total, 1.0),
            ]
        ],
        dtype=np.float32,
    )
    probability = float(model.predict_proba(features)[0, 1])
    probability = max(0.0, min(1.0, probability))
    return {
        "model_version": MODEL_VERSION,
        "cell_id": cell_id,
        "lat": float(lat),
        "lon": float(lon),
        "local_day_of_week": day_of_week,
        "local_hour": hour,
        "month": month,
        "historical_cell_events": int(prior_total),
        "historical_same_hour_events": int(prior_same_hour),
        "risk_score": probability,
        "risk_level": _risk_level(probability),
    }


def _map_html() -> str:
    config = OVERLAY["config"]
    js_config = {
        "api_key": GMAPS_API_KEY,
        "service_id": "traffic_safety",
        "frames": [str(frame) for frame in OVERLAY["frames"]],
        "center_lat": float(config["center_lat"]),
        "center_lon": float(config["center_lon"]),
        "default_zoom": int(config.get("zoom_min", 4)),
        "zoom_min": int(config.get("zoom_min", 2)),
        "zoom_max": int(config.get("zoom_max", 10)),
        "timeline": _timeline(),
        "metrics": [
            {
                "label": "ROC AUC",
                "value": _first_metric(MODEL_BUNDLE, ["val_roc_auc", "roc_auc"]),
            },
            {
                "label": "Avg Precision",
                "value": _first_metric(
                    MODEL_BUNDLE,
                    ["val_average_precision", "average_precision"],
                ),
            },
        ],
        "default_frame_idx": 0,
    }
    config_blob = html.escape(json.dumps(js_config), quote=True)
    return f"""
<div id="risk-map-shell" class="risk-map-shell" data-config="{config_blob}">
  <section class="risk-map-pane">
    <div class="risk-map-header">
      <div class="timeline-row">
        <div class="timeline-wrap">
          <div id="risk-timeline-ticks" class="timeline-years"></div>
          <div id="risk-timeline-track" class="timeline-track" style="--frame-step:1%;">
            <div id="risk-timeline-phases" class="timeline-phases"></div>
            <div id="risk-time-progress" class="timeline-progress"></div>
            <div id="risk-now-marker" class="timeline-marker"></div>
            <input id="risk-time-slider" type="range" min="0" max="0" value="0" step="1" />
          </div>
          <div id="risk-frame-label" class="timeline-current-label"></div>
        </div>
        <button id="risk-play" type="button" aria-label="Play timeline">
          <span class="play-icon" aria-hidden="true">&#9658;</span>
          <span class="pause-icon" aria-hidden="true">&#10074;&#10074;</span>
        </button>
      </div>
    </div>

    <div class="risk-map-stage">
      <div class="map-overlay-panel">
        <div class="panel-title">Traffic Safety</div>
        <div class="overlay-row metrics-row">
          <div class="metric-inline"><span id="model-metric-1-label">ROC AUC</span><strong id="model-metric-1-value">-</strong></div>
          <div class="metric-inline"><span id="model-metric-2-label">Avg Precision</span><strong id="model-metric-2-value">-</strong></div>
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
  if (window.bootstrapTrafficSafetyMap) {
    return window.bootstrapTrafficSafetyMap();
  }
  for (let attempt = 0; attempt < 20; attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 50));
    if (window.bootstrapTrafficSafetyMap) {
      return window.bootstrapTrafficSafetyMap();
    }
  }
  console.error("Traffic Safety map bootstrap script did not load.");
  return [];
}
"""


with gr.Blocks(title=SERVICE_NAME, head=MAP_HEAD) as demo:
    gr.Markdown(f"# {SERVICE_NAME}")

    with gr.Tabs():
        with gr.Tab("Map"):
            gr.HTML(_map_html())

        with gr.Tab("Model"):
            with gr.Row():
                lat = gr.Number(label="Latitude", value=34.0522)
                lon = gr.Number(label="Longitude", value=-118.2437)
                day_of_week = gr.Number(label="Day Of Week (1=Mon)", value=5)
                hour = gr.Number(label="Hour", value=17)
                month = gr.Number(label="Month", value=9)

            output = gr.JSON(label="Traffic Safety Prediction")
            run = gr.Button("Predict Traffic Safety")
            run.click(
                fn=predict_traffic_safety,
                inputs=[lat, lon, day_of_week, hour, month],
                outputs=output,
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
api.mount(STATIC_URL, StaticFiles(directory=str(STATIC_DIR)), name="traffic-safety-static")


@api.get("/health")
def health() -> dict[str, object]:
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "frames": len(OVERLAY["frames"]),
        "model_version": MODEL_VERSION,
        "model_ready": bool(MODEL_BUNDLE),
        "overlay_ready": bool((TILES_DIR / "overlay.npz").exists()),
    }


@api.get("/tiles/{frame_idx}/{z}/{x}/{y}.png")
def tile(frame_idx: int, z: int, x: int, y: int) -> Response:
    if z < 0 or z > 12:
        blank = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
        buffer = io.BytesIO()
        blank.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    try:
        png = _render_tile_png(frame_idx=frame_idx, z=z, x=x, y=y)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


app = gr.mount_gradio_app(api, demo, path="/")


def main() -> None:
    print_http_startup(SERVICE_NAME, HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
