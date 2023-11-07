"""Wildfire Gradio app with Google Maps risk overlay and inference form."""

from __future__ import annotations

from pathlib import Path
import json
import os

import gradio as gr
import torch

from wildfire.model import load_model_bundle


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", os.getenv("PORT", "8010")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent / "model" / "wildfire_model.pt")))
TILES_DIR = Path(__file__).resolve().parent / "tiles"
OVERLAY_CONFIG_PATH = TILES_DIR / "overlay_config.json"
GMAPS_API_KEY = os.getenv("GMAPS_API_KEY", "")
SERVICE_NAME = os.getenv("SERVICE_NAME", "Wildfire Ignition-Risk Service")

model, feature_mean, feature_std, MODEL_VERSION = load_model_bundle(MODEL_PATH)
MODEL_BUNDLE = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

if TILES_DIR.exists():
    gr.set_static_paths(paths=[TILES_DIR])


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


def _load_overlay_config() -> dict[str, object]:
    default = {
        "start_year": 2000,
        "end_year": 2030,
        "zoom": 2,
        "training_end_year": 2018,
        "eval_end_year": 2023,
        "center_lat": 37.0,
        "center_lon": -98.0,
    }
    if OVERLAY_CONFIG_PATH.exists():
        loaded = json.loads(OVERLAY_CONFIG_PATH.read_text(encoding="utf-8"))
        default.update(loaded)
    return default


def _map_html() -> str:
    cfg = _load_overlay_config()
    start_year = int(cfg["start_year"])
    end_year = int(cfg["end_year"])
    train_end_year = int(cfg["training_end_year"])
    eval_end_year = int(cfg["eval_end_year"])

    train_span = max(0, min(train_end_year, end_year) - start_year + 1)
    eval_span = max(0, min(eval_end_year, end_year) - max(start_year, train_end_year + 1) + 1)
    infer_span = max(0, end_year - max(start_year, eval_end_year + 1) + 1)

    js_config = {
        "service_id": "wildfire",
        "api_key": GMAPS_API_KEY,
        "tile_root": str(TILES_DIR.resolve()),
        "zoom": int(cfg["zoom"]),
        "start_year": start_year,
        "end_year": end_year,
        "training_end_year": train_end_year,
        "eval_end_year": eval_end_year,
        "center_lat": float(cfg["center_lat"]),
        "center_lon": float(cfg["center_lon"]),
    }

    return f"""
<div id="wildfire-map-shell" class="risk-map-shell">
  <div class="risk-map-header">
    <label for="wildfire-year-slider"><strong>Year:</strong> <span id="wildfire-year-value">{start_year}</span></label>
    <input id="wildfire-year-slider" type="range" min="{start_year}" max="{end_year}" value="{start_year}" step="1" />
    <div class="risk-phase">Section: <span id="wildfire-phase">training</span></div>
  </div>
  <div class="risk-timeline">
    <div class="seg train" style="flex:{train_span};">Training</div>
    <div class="seg eval" style="flex:{eval_span};">Eval</div>
    <div class="seg infer" style="flex:{infer_span};">Inference</div>
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
  .risk-map-header input[type="range"] {{ width: 100%; }}
  .risk-phase {{ font-size: 13px; color: #334155; }}
  .risk-timeline {{ display: flex; height: 22px; overflow: hidden; border-radius: 999px; margin-bottom: 10px; font-size: 12px; }}
  .risk-timeline .seg {{ display: flex; align-items: center; justify-content: center; color: #0f172a; }}
  .risk-timeline .train {{ background: #c9f7d7; }}
  .risk-timeline .eval {{ background: #fde68a; }}
  .risk-timeline .infer {{ background: #fecaca; }}
  .risk-map {{ width: 100%; height: 460px; border-radius: 8px; }}
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
  const slider = root.querySelector("#wildfire-year-slider");
  const yearValue = root.querySelector("#wildfire-year-value");
  const phaseValue = root.querySelector("#wildfire-phase");
  const mapNode = root.querySelector("#wildfire-map");
  const statusNode = root.querySelector("#wildfire-map-status");

  const phaseForYear = (year) => {{
    if (year <= cfg.training_end_year) return "training";
    if (year <= cfg.eval_end_year) return "eval";
    return "inference";
  }};

  const updateLabels = (year) => {{
    yearValue.textContent = String(year);
    phaseValue.textContent = phaseForYear(year);
  }};

  const fileUrl = (year, x, y, z) => {{
    const path = `${{cfg.tile_root}}/${{year}}/${{z}}/${{x}}/${{y}}.png`;
    return `/file=${{encodeURIComponent(path)}}`;
  }};

  let map = null;
  const installOverlay = () => {{
    const year = Number(slider.value);
    updateLabels(year);
    if (!map) return;
    map.overlayMapTypes.clear();
    const overlay = new google.maps.ImageMapType({{
      tileSize: new google.maps.Size(256, 256),
      opacity: 0.58,
      getTileUrl: (coord, zoom) => {{
        if (zoom !== cfg.zoom) return "";
        const bound = 1 << zoom;
        if (coord.y < 0 || coord.y >= bound) return "";
        const wrappedX = ((coord.x % bound) + bound) % bound;
        return fileUrl(year, wrappedX, coord.y, zoom);
      }},
    }});
    map.overlayMapTypes.push(overlay);
  }};

  const initMap = () => {{
    map = new google.maps.Map(mapNode, {{
      center: {{ lat: cfg.center_lat, lng: cfg.center_lon }},
      zoom: cfg.zoom,
      minZoom: cfg.zoom,
      maxZoom: cfg.zoom,
      mapTypeControl: false,
      streetViewControl: false,
      fullscreenControl: false,
      clickableIcons: false,
    }});
    statusNode.textContent = "Overlay tiles are pre-generated through 2030.";
    installOverlay();
  }};

  const loadGoogleMaps = () => {{
    if (!cfg.api_key) {{
      statusNode.textContent = "Set GMAPS_API_KEY to render Google Maps overlays.";
      updateLabels(Number(slider.value));
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
  updateLabels(Number(slider.value));
  loadGoogleMaps();
}})();
</script>
"""


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


with gr.Blocks(title=SERVICE_NAME) as demo:
    gr.Markdown(f"# {SERVICE_NAME}")
    gr.Markdown(f"Model version: `{MODEL_VERSION}`")
    gr.Markdown("## Training")
    gr.Markdown(
        f"- Overlay timeline: **2000-2030** (pre-generated tiles)\n"
        f"- Training window ends: **2018**\n"
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


def main() -> None:
    demo.launch(server_name=HOST, server_port=PORT)


if __name__ == "__main__":
    main()
