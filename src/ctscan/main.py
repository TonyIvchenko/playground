"""Chest CT semantic segmentation service."""

from __future__ import annotations

import base64
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr
from PIL import Image
import uvicorn

try:
    from study import (
        blank_viewer_image,
        issue_slice_stats,
        issue_volume_stats,
        load_study_from_zip_bytes,
        model_backend_error,
        model_backend_metadata,
        model_backend_name,
        read_temp_bundle,
        segment_issues,
        segment_lungs,
        segmentation_backend_error,
        segmentation_backend_name,
        supported_issues,
        window_slice,
        write_temp_bundle,
    )
except ModuleNotFoundError:
    from src.ctscan.study import (
        blank_viewer_image,
        issue_slice_stats,
        issue_volume_stats,
        load_study_from_zip_bytes,
        model_backend_error,
        model_backend_metadata,
        model_backend_name,
        read_temp_bundle,
        segment_issues,
        segment_lungs,
        segmentation_backend_error,
        segmentation_backend_name,
        supported_issues,
        window_slice,
        write_temp_bundle,
    )


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
SERVICE_NAME = os.getenv("SERVICE_NAME", "ctscan")
SAMPLES_MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "ctscan" / "samples" / "samples.json"
SAMPLE_CACHE_DIR = Path(__file__).resolve().parent / "data" / "ctscan" / "samples" / "cache"
LOCAL_LEGACY_CT_ZIPS_PATH = Path(__file__).resolve().parent / "data" / "ctscan" / "raw" / "legacy_sources" / "plethora" / "ct_zips"
DEFAULT_SAMPLE = ""
METRICS_TABLE_COLUMNS = ["Issue", "Lung %", "Volume ml", "Current slice %"]
LUNG_COLOR = "#10b981"
DEFAULT_OPACITY = 0.2
VIEWER_CACHE_DIR = Path(__file__).resolve().parent / "data" / "ctscan" / "viewer_cache"
VIEWER_HEAD = """
<style>
.ctscan-viewer-root { display: grid; gap: 16px; }
.ctscan-viewer-root .ctscan-toolbar { display: grid; grid-template-columns: 220px minmax(240px, 1fr) 240px; gap: 16px; align-items: end; }
.ctscan-viewer-root .ctscan-control { display: grid; gap: 6px; }
.ctscan-viewer-root .ctscan-control label { font-size: 14px; font-weight: 600; }
.ctscan-viewer-root .ctscan-checks { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; min-height: 40px; }
.ctscan-viewer-root .ctscan-check { font-size: 14px; display: inline-flex; gap: 6px; align-items: center; }
.ctscan-viewer-root .ctscan-stage { display: grid; gap: 10px; }
.ctscan-viewer-root .ctscan-viewer { position: relative; width: fit-content; max-width: 100%; background: #000; overflow: hidden; }
.ctscan-viewer-root .ctscan-viewer img { display: block; max-width: 100%; height: auto; }
.ctscan-viewer-root .ctscan-overlay { position: absolute; inset: 0; pointer-events: none; }
.ctscan-viewer-root .ctscan-slider-row { display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: center; }
.ctscan-viewer-root .ctscan-table { border-collapse: collapse; width: 100%; font-size: 14px; }
.ctscan-viewer-root .ctscan-table th,
.ctscan-viewer-root .ctscan-table td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #e5e7eb; }
.ctscan-viewer-root .ctscan-table th { font-weight: 600; }
@media (max-width: 900px) {
  .ctscan-viewer-root .ctscan-toolbar { grid-template-columns: 1fr; }
}
</style>
<script>
(() => {
  function initViewer(root) {
    if (!root || root.dataset.ctscanReady === "1") {
      return;
    }
    const stateNode = root.querySelector(".ctscan-state");
    if (!stateNode) {
      return;
    }
    const state = JSON.parse(stateNode.value || stateNode.textContent || "{}");
    root.dataset.ctscanReady = "1";

    const overlay = root.querySelector(".ctscan-overlay-select");
    const findingWrap = root.querySelector(".ctscan-finding-wrap");
    const opacity = root.querySelector(".ctscan-opacity");
    const opacityValue = root.querySelector(".ctscan-opacity-value");
    const slice = root.querySelector(".ctscan-slice");
    const sliceLabel = root.querySelector(".ctscan-slice-label");
    const base = root.querySelector(".ctscan-base");
    const lung = root.querySelector(".ctscan-lung");
    const rows = Array.from(root.querySelectorAll(".ctscan-table tbody tr"));
    const findingImages = Object.fromEntries(
      state.rows.map((row) => [row.key, root.querySelector('.ctscan-finding[data-key="' + row.key + '"]')])
    );
    const findingChecks = Array.from(root.querySelectorAll(".ctscan-finding-wrap input[type='checkbox']"));

    function selectedKeys() {
      return new Set(findingChecks.filter((node) => node.checked).map((node) => node.value));
    }

    function render() {
      const index = Number(slice.value || 0);
      const alpha = Number(opacity.value || 0);
      const mode = overlay.value;
      const selected = selectedKeys();

      base.src = state.base_images[index];
      lung.src = state.lung_images[index];
      lung.style.opacity = mode === "Lungs" ? String(alpha) : "0";
      findingWrap.style.display = mode === "Findings" ? "grid" : "none";
      opacityValue.textContent = alpha.toFixed(2);
      sliceLabel.textContent = `Slice ${index + 1} / ${state.slice_count}`;

      state.rows.forEach((row, rowIndex) => {
        const image = findingImages[row.key];
        image.src = state.finding_images[row.key][index];
        image.style.opacity = mode === "Findings" && selected.has(row.key) ? String(alpha) : "0";
        const cells = rows[rowIndex].querySelectorAll("td");
        cells[3].textContent = Number(row.slice_percents[index] || 0).toFixed(4);
      });
    }

    overlay.addEventListener("change", render);
    opacity.addEventListener("input", render);
    slice.addEventListener("input", render);
    findingChecks.forEach((node) => node.addEventListener("change", render));
    render();
  }

  function scan() {
    document.querySelectorAll(".ctscan-viewer-root").forEach(initViewer);
  }

  window.addEventListener("load", () => {
    scan();
    const observer = new MutationObserver(() => scan());
    observer.observe(document.body, { childList: true, subtree: true });
  });
})();
</script>
"""


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    cleaned: dict[str, dict[str, str]] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        study_zip = str(value.get("study_zip", "")).strip()
        if not study_zip:
            continue
        cleaned[str(key)] = {"study_zip": study_zip}
    return cleaned


def _candidate_lidc_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.getenv("CTSCAN_LIDC_ROOT", "").strip()
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path(__file__).resolve().parent / "data" / "ctscan" / "raw" / "lidc" / "LIDC-IDRI")
    dedup: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        text = str(root.resolve()) if root.exists() else str(root)
        if text in seen:
            continue
        seen.add(text)
        dedup.append(root)
    return dedup


def _candidate_legacy_ct_zip_dirs() -> list[Path]:
    roots: list[Path] = []
    env_root = os.getenv("CTSCAN_DEMO_CT_ZIPS_ROOT", "").strip()
    if env_root:
        roots.append(Path(env_root))
    roots.append(LOCAL_LEGACY_CT_ZIPS_PATH)
    dedup: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        text = str(root.resolve()) if root.exists() else str(root)
        if text in seen:
            continue
        seen.add(text)
        dedup.append(root)
    return dedup


def _find_demo_ct_zips(limit: int = 24) -> list[Path]:
    samples: list[Path] = []
    for root in _candidate_legacy_ct_zip_dirs():
        if not root.exists():
            continue
        for zip_path in sorted(root.glob("*.zip")):
            samples.append(zip_path.resolve())
            if len(samples) >= limit:
                return samples
    return samples


def _find_demo_lidc_series() -> list[Path]:
    for root in _candidate_lidc_roots():
        if not root.exists():
            continue
        for dirpath, _, filenames in os.walk(root):
            dcm_names = [name for name in filenames if name.lower().endswith(".dcm") and not name.startswith("._")]
            if len(dcm_names) < 32:
                continue
            series_dir = Path(dirpath)
            files = [series_dir / name for name in sorted(dcm_names)]
            if files:
                return files
    return []


def _ensure_auto_demo_manifest() -> dict[str, dict[str, str]]:
    samples_dir = SAMPLES_MANIFEST_PATH.parent
    samples_dir.mkdir(parents=True, exist_ok=True)
    demo_zips = _find_demo_ct_zips()
    if demo_zips:
        manifest: dict[str, dict[str, str]] = {}
        for zip_path in demo_zips:
            key = f"demo_{zip_path.stem.lower()}"
            if key in manifest:
                continue
            manifest[key] = {"study_zip": str(zip_path)}
        SAMPLES_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    sample_zip = samples_dir / "auto_demo_lidc.zip"
    if not sample_zip.exists():
        series_files = _find_demo_lidc_series()
        if not series_files:
            return {}
        with zipfile.ZipFile(sample_zip, "w", compression=zipfile.ZIP_STORED) as archive:
            for file_path in series_files:
                archive.write(file_path, arcname=f"dicom/{file_path.name}")
    manifest = {"auto_demo_lidc": {"study_zip": str(sample_zip.resolve())}}
    SAMPLES_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _candidate_sample_manifest_paths() -> list[Path]:
    paths: list[Path] = []
    env_path = os.getenv("CTSCAN_SAMPLES_MANIFEST_PATH", "").strip()
    if env_path:
        paths.append(Path(env_path))
    paths.append(SAMPLES_MANIFEST_PATH)
    dedup: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        text = str(path.resolve()) if path.exists() else str(path)
        if text in seen:
            continue
        seen.add(text)
        dedup.append(path)
    return dedup


@lru_cache(maxsize=1)
def load_samples_manifest() -> dict[str, dict[str, str]]:
    for path in _candidate_sample_manifest_paths():
        if path.exists():
            manifest = _read_manifest(path)
            if manifest:
                return manifest
    return _ensure_auto_demo_manifest()


def _resolve_sample_path(sample_id: str) -> Path:
    manifest = load_samples_manifest()
    sample = manifest.get(sample_id)
    if not sample:
        raise FileNotFoundError(f"Sample `{sample_id}` is unavailable.")
    study_path = Path(sample["study_zip"])
    if not study_path.is_absolute():
        study_path = (Path(__file__).resolve().parent / study_path).resolve()
    if not study_path.exists():
        raise FileNotFoundError(f"Sample `{sample_id}` study zip is missing at {study_path}.")
    return study_path


def _study_bytes_from_inputs(study_file: str | None, sample_id: str | None) -> bytes:
    if study_file:
        return Path(study_file).read_bytes()
    if sample_id:
        return _resolve_sample_path(sample_id).read_bytes()
    raise ValueError("Provide a study zip or a sample id.")


def _slice_damage_percentages(labels, lung_mask) -> list[float]:
    values: list[float] = []
    for index in range(int(labels.shape[0])):
        lung_pixels = max(int(lung_mask[index].sum()), 1)
        damaged_pixels = int((labels[index] > 0).sum())
        values.append(float((damaged_pixels / lung_pixels) * 100.0))
    return values


def analyze_study_bytes(
    study_bytes: bytes,
    age: float | None = None,
    sex: str | None = None,
    smoking_history: str | None = None,
) -> dict[str, Any]:
    study = load_study_from_zip_bytes(study_bytes)

    lung_mask, backend_used = segment_lungs(study.volume_hu)
    labels = segment_issues(study.volume_hu, lung_mask)
    issue_rows = issue_volume_stats(labels, lung_mask, spacing=study.spacing)

    lung_voxels = max(int(lung_mask.sum()), 1)
    damaged_voxels = int((labels > 0).sum())
    voxel_volume_ml = float(study.spacing[0] * study.spacing[1] * study.spacing[2]) / 1000.0
    lung_volume_ml = float(lung_voxels) * voxel_volume_ml
    damaged_volume_ml = float(damaged_voxels) * voxel_volume_ml
    damaged_percent = float((damaged_voxels / lung_voxels) * 100.0)

    detected_rows = [row for row in issue_rows if row["voxels"] > 0]
    top_issue = max(detected_rows, key=lambda item: item["lung_percent"], default=None)

    qc_reasons = list(study.qc_reasons)
    qc_status = "ok" if not qc_reasons else "rejected"
    slice_damage = _slice_damage_percentages(labels, lung_mask)

    return {
        "version": "segmentation-v1",
        "backend": backend_used,
        "issue_backend": model_backend_name(),
        "qc": {
            "status": qc_status,
            "rejection_reasons": qc_reasons,
        },
        "study_metadata": {
            **study.metadata,
            "age": age,
            "sex": sex,
            "smoking_history": smoking_history,
        },
        "issues": issue_rows,
        "summary": {
            "lung_volume_ml": round(lung_volume_ml, 3),
            "damaged_volume_ml": round(damaged_volume_ml, 3),
            "damaged_percent": round(damaged_percent, 4),
            "detected_issue_count": len(detected_rows),
            "top_issue": None
            if top_issue is None
            else {
                "issue": top_issue["issue"],
                "lung_percent": round(float(top_issue["lung_percent"]), 4),
            },
        },
        "slice_damage_percent": [round(value, 4) for value in slice_damage],
        "_viewer": {
            "bundle_path": write_temp_bundle(study.volume_hu, labels, lung_mask),
            "slice_count": int(study.volume_hu.shape[0]),
        },
    }


def _sample_cache_key(sample_id: str, study_path: Path) -> str:
    model_path = str(model_backend_metadata().get("path") or "")
    model_mtime = 0
    model_size = 0
    if model_path and Path(model_path).exists():
        stat = Path(model_path).stat()
        model_mtime = int(stat.st_mtime)
        model_size = int(stat.st_size)
    study_stat = study_path.stat()
    payload = {
        "sample_id": sample_id,
        "study_path": str(study_path),
        "study_mtime": int(study_stat.st_mtime),
        "study_size": int(study_stat.st_size),
        "model_path": model_path,
        "model_mtime": model_mtime,
        "model_size": model_size,
        "segmentation_backend": segmentation_backend_name(),
        "issue_backend": model_backend_name(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _sample_cache_paths(sample_id: str, study_path: Path) -> tuple[Path, Path, Path]:
    SAMPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _sample_cache_key(sample_id, study_path)
    return (
        SAMPLE_CACHE_DIR / f"{sample_id}.{key}.json",
        SAMPLE_CACHE_DIR / f"{sample_id}.{key}.npz",
        SAMPLE_CACHE_DIR / f"{sample_id}.{key}.html",
    )


def _image_to_data_url(image: Image.Image, format_name: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format_name)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{format_name.lower()};base64,{encoded}"


def _overlay_image(mask, color: tuple[int, int, int]) -> Image.Image:
    rgba = Image.new("RGBA", (int(mask.shape[1]), int(mask.shape[0])), (0, 0, 0, 0))
    if bool(mask.any()):
        alpha = (mask.astype("uint8") * 255)
        overlay = Image.merge(
            "RGBA",
            (
                Image.fromarray(alpha * 0 + color[0]),
                Image.fromarray(alpha * 0 + color[1]),
                Image.fromarray(alpha * 0 + color[2]),
                Image.fromarray(alpha),
            ),
        )
        rgba.alpha_composite(overlay)
    return rgba


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def _viewer_slice_name(index: int) -> str:
    return f"{index:04d}.png"


def _upload_cache_key(study_path: Path) -> str:
    model_path = str(model_backend_metadata().get("path") or "")
    model_mtime = 0
    model_size = 0
    if model_path and Path(model_path).exists():
        stat = Path(model_path).stat()
        model_mtime = int(stat.st_mtime)
        model_size = int(stat.st_size)
    study_stat = study_path.stat()
    payload = {
        "study_path": str(study_path.resolve()),
        "study_mtime": int(study_stat.st_mtime),
        "study_size": int(study_stat.st_size),
        "model_path": model_path,
        "model_mtime": model_mtime,
        "model_size": model_size,
        "segmentation_backend": segmentation_backend_name(),
        "issue_backend": model_backend_name(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _viewer_token_for_sample(sample_id: str, study_path: Path) -> str:
    return f"sample-{sample_id}-{_sample_cache_key(sample_id, study_path)}"


def _viewer_token_for_upload(study_path: Path) -> str:
    return f"upload-{_upload_cache_key(study_path)}"


def _viewer_dir(token: str) -> Path:
    return VIEWER_CACHE_DIR / token


def _viewer_state_path(token: str) -> Path:
    return _viewer_dir(token) / "state.json"


def _build_frontend_state(payload: dict[str, Any]) -> dict[str, Any]:
    _, labels, lung_mask = read_temp_bundle(payload["_viewer"]["bundle_path"])
    issue_defs = supported_issues()
    issue_by_key = {str(item["key"]): item for item in issue_defs}

    slice_stats_by_key: dict[str, list[float]] = {str(item["key"]): [] for item in issue_defs}

    for slice_index in range(int(labels.shape[0])):
        slice_rows = issue_slice_stats(labels, lung_mask, slice_index)
        slice_lookup = {str(row["issue_key"]): row for row in slice_rows}
        for issue in issue_defs:
            key = str(issue["key"])
            slice_stats_by_key[key].append(round(float(slice_lookup.get(key, {}).get("slice_percent", 0.0)), 4))

    rows = []
    for row in payload.get("issues", []):
        key = str(row["issue_key"])
        issue_def = issue_by_key[key]
        rows.append(
            {
                "key": key,
                "label": str(row["issue"]),
                "color": str(issue_def["color"]),
                "lung_percent": round(float(row["lung_percent"]), 4),
                "volume_ml": round(float(row["volume_ml"]), 4),
                "slice_percents": slice_stats_by_key[key],
            }
        )

    default_slice = int(max(range(len(payload.get("slice_damage_percent", [])) or [0]), key=lambda idx: payload.get("slice_damage_percent", [0])[idx]))
    return {
        "slice_count": int(labels.shape[0]),
        "default_slice": default_slice,
        "default_opacity": DEFAULT_OPACITY,
        "rows": rows,
    }


def _write_viewer_assets(token: str, payload: dict[str, Any]) -> None:
    state_path = _viewer_state_path(token)
    if state_path.exists():
        return

    volume_hu, labels, lung_mask = read_temp_bundle(payload["_viewer"]["bundle_path"])
    viewer_state = _build_frontend_state(payload)
    asset_dir = _viewer_dir(token)
    base_dir = asset_dir / "base"
    lung_dir = asset_dir / "lung"
    finding_root = asset_dir / "findings"
    base_dir.mkdir(parents=True, exist_ok=True)
    lung_dir.mkdir(parents=True, exist_ok=True)
    finding_root.mkdir(parents=True, exist_ok=True)

    issue_defs = supported_issues()
    for issue in issue_defs:
        (finding_root / str(issue["key"])).mkdir(parents=True, exist_ok=True)

    for slice_index in range(int(volume_hu.shape[0])):
        grayscale = window_slice(volume_hu[slice_index], "lung")
        Image.fromarray(grayscale, mode="L").convert("RGB").save(base_dir / _viewer_slice_name(slice_index), format="PNG")
        _overlay_image(lung_mask[slice_index], _hex_to_rgb(LUNG_COLOR)).save(lung_dir / _viewer_slice_name(slice_index), format="PNG")
        for issue in issue_defs:
            overlay = _overlay_image(labels[slice_index] == int(issue["id"]), _hex_to_rgb(str(issue["color"])))
            overlay.save(finding_root / str(issue["key"]) / _viewer_slice_name(slice_index), format="PNG")

    state = {
        **viewer_state,
        "token": token,
        "asset_root": f"/viewer-cache/{token}",
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_viewer_state(token: str) -> dict[str, Any]:
    state_path = _viewer_state_path(token)
    if not state_path.exists():
        raise FileNotFoundError(f"Viewer token `{token}` is unavailable.")
    return json.loads(state_path.read_text(encoding="utf-8"))


def _viewer_iframe_html(token: str, row_count: int) -> str:
    height = 620 + 44 * row_count
    return f'<iframe src="/viewer/{token}" style="width:100%;height:{height}px;border:0;display:block" loading="eager"></iframe>'


def _viewer_html(viewer_state: dict[str, Any]) -> str:
    default_slice = int(viewer_state["default_slice"])
    asset_root = str(viewer_state["asset_root"])
    finding_items = "".join(
        f'<label class="ctscan-check"><input type="checkbox" value="{row["key"]}" checked> {row["label"]}</label>'
        for row in viewer_state["rows"]
    )
    table_rows = "".join(
        "<tr>"
        f'<td style="color:{row["color"]}">{row["label"]}</td>'
        f'<td class="lung-pct">{row["lung_percent"]:.4f}</td>'
        f'<td class="volume-ml">{row["volume_ml"]:.4f}</td>'
        f'<td class="slice-pct">{float(row["slice_percents"][default_slice] if row["slice_percents"] else 0.0):.4f}</td>'
        "</tr>"
        for row in viewer_state["rows"]
    )
    payload_json = json.dumps(viewer_state).replace("</script>", "<\\/script>")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #1f2937; background: transparent; }}
    .ctscan-viewer-root {{ display: grid; gap: 16px; }}
    .ctscan-toolbar {{ display: grid; grid-template-columns: 220px minmax(240px, 1fr) 240px; gap: 16px; align-items: end; }}
    .ctscan-control {{ display: grid; gap: 6px; }}
    .ctscan-control label {{ font-size: 14px; font-weight: 600; }}
    .ctscan-checks {{ display: flex; gap: 14px; flex-wrap: wrap; align-items: center; min-height: 40px; }}
    .ctscan-check {{ font-size: 14px; display: inline-flex; gap: 6px; align-items: center; }}
    .ctscan-stage {{ display: grid; gap: 10px; }}
    .ctscan-viewer {{ position: relative; width: fit-content; max-width: 100%; background: #000; overflow: hidden; }}
    .ctscan-viewer img {{ display: block; max-width: 100%; height: auto; }}
    .ctscan-overlay {{ position: absolute; inset: 0; pointer-events: none; }}
    .ctscan-slider-row {{ display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: center; }}
    .ctscan-table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    .ctscan-table th, .ctscan-table td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
    .ctscan-table th {{ font-weight: 600; }}
    @media (max-width: 900px) {{
      .ctscan-toolbar {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<div class="ctscan-viewer-root">
  <div class="ctscan-toolbar">
    <div class="ctscan-control">
      <label>Overlay</label>
      <select class="ctscan-overlay-select">
        <option value="Findings">Findings</option>
        <option value="Lungs">Lungs</option>
      </select>
    </div>
    <div class="ctscan-control ctscan-finding-wrap">
      <label>Findings</label>
      <div class="ctscan-checks">{finding_items}</div>
    </div>
    <div class="ctscan-control">
      <label>Opacity</label>
      <div class="ctscan-slider-row">
        <input class="ctscan-opacity" type="range" min="0" max="1" step="0.05" value="{viewer_state['default_opacity']}">
        <span class="ctscan-opacity-value">{viewer_state['default_opacity']:.2f}</span>
      </div>
    </div>
  </div>
  <div class="ctscan-stage">
    <div class="ctscan-viewer">
      <img class="ctscan-base" src="{asset_root}/base/{_viewer_slice_name(default_slice)}" alt="Axial viewer">
      <img class="ctscan-overlay ctscan-lung" src="{asset_root}/lung/{_viewer_slice_name(default_slice)}" alt="Lung overlay" style="opacity:0">
      {''.join(f'<img class="ctscan-overlay ctscan-finding" data-key="{row["key"]}" src="{asset_root}/findings/{row["key"]}/{_viewer_slice_name(default_slice)}" alt="{row["label"]} overlay" style="opacity:{viewer_state["default_opacity"]}">' for row in viewer_state["rows"])}
    </div>
    <div class="ctscan-slider-row">
      <input class="ctscan-slice" type="range" min="0" max="{max(0, viewer_state["slice_count"] - 1)}" step="1" value="{default_slice}">
      <span class="ctscan-slice-label">Slice {default_slice + 1} / {viewer_state["slice_count"]}</span>
    </div>
  </div>
  <table class="ctscan-table">
    <thead>
      <tr>
        <th>Issue</th>
        <th>Lung %</th>
        <th>Volume ml</th>
        <th>Current slice %</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>
<script type="application/json" id="ctscan-state">{payload_json}</script>
<script>
(() => {{
  const root = document.querySelector(".ctscan-viewer-root");
  const state = JSON.parse(document.getElementById("ctscan-state").textContent || "{{}}");
  const overlay = root.querySelector(".ctscan-overlay-select");
  const findingWrap = root.querySelector(".ctscan-finding-wrap");
  const opacity = root.querySelector(".ctscan-opacity");
  const opacityValue = root.querySelector(".ctscan-opacity-value");
  const slice = root.querySelector(".ctscan-slice");
  const sliceLabel = root.querySelector(".ctscan-slice-label");
  const base = root.querySelector(".ctscan-base");
  const lung = root.querySelector(".ctscan-lung");
  const rows = Array.from(root.querySelectorAll(".ctscan-table tbody tr"));
  const findingImages = Object.fromEntries(
    state.rows.map((row) => [row.key, root.querySelector('.ctscan-finding[data-key="' + row.key + '"]')])
  );
  const findingChecks = Array.from(root.querySelectorAll(".ctscan-finding-wrap input[type='checkbox']"));

  function selectedKeys() {{
    return new Set(findingChecks.filter((node) => node.checked).map((node) => node.value));
  }}

  function render() {{
    const index = Number(slice.value || 0);
    const alpha = Number(opacity.value || 0);
    const mode = overlay.value;
    const selected = selectedKeys();

    base.src = `${{state.asset_root}}/base/${{String(index).padStart(4, "0")}}.png`;
    lung.src = `${{state.asset_root}}/lung/${{String(index).padStart(4, "0")}}.png`;
    lung.style.opacity = mode === "Lungs" ? String(alpha) : "0";
    findingWrap.style.display = mode === "Findings" ? "grid" : "none";
    opacityValue.textContent = alpha.toFixed(2);
    sliceLabel.textContent = `Slice ${{index + 1}} / ${{state.slice_count}}`;

    state.rows.forEach((row, rowIndex) => {{
      const image = findingImages[row.key];
      image.src = `${{state.asset_root}}/findings/${{row.key}}/${{String(index).padStart(4, "0")}}.png`;
      image.style.opacity = mode === "Findings" && selected.has(row.key) ? String(alpha) : "0";
      const cells = rows[rowIndex].querySelectorAll("td");
      cells[3].textContent = Number(row.slice_percents[index] || 0).toFixed(4);
    }});
  }}

  overlay.addEventListener("change", render);
  opacity.addEventListener("input", render);
  slice.addEventListener("input", render);
  findingChecks.forEach((node) => node.addEventListener("change", render));
  render();
}})();
</script>
</body>
</html>
"""


def _blank_viewer_html() -> str:
    blank_path = Path(blank_viewer_image())
    blank_image = Image.open(blank_path)
    data_url = _image_to_data_url(blank_image)
    return f"<div style='display:grid;gap:12px'><img src='{data_url}' alt='Viewer' style='max-width:100%;width:512px;background:#000'><div style='font-size:14px;color:#6b7280'>Upload DICOM file.</div></div>"


def render_sample_cached_html(sample_id: str) -> str:
    study_path = _resolve_sample_path(sample_id)
    payload_path, bundle_path, _html_path = _sample_cache_paths(sample_id, study_path)
    if payload_path.exists() and bundle_path.exists():
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            payload["_viewer"]["bundle_path"] = str(bundle_path)
            token = _viewer_token_for_sample(sample_id, study_path)
            _write_viewer_assets(token, payload)
            return _viewer_iframe_html(token, len(payload.get("issues", [])))
        except Exception:
            pass

    payload = analyze_study_bytes(study_path.read_bytes())
    source_bundle = Path(payload["_viewer"]["bundle_path"])
    bundle_path.write_bytes(source_bundle.read_bytes())
    payload["_viewer"]["bundle_path"] = str(bundle_path)
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    token = _viewer_token_for_sample(sample_id, study_path)
    _write_viewer_assets(token, payload)
    return _viewer_iframe_html(token, len(payload.get("issues", [])))


def render_upload_html(study_file: str | None) -> str:
    if not study_file:
        return _blank_viewer_html()
    study_path = Path(study_file)
    payload = analyze_study_bytes(study_path.read_bytes())
    token = _viewer_token_for_upload(study_path)
    _write_viewer_assets(token, payload)
    return _viewer_iframe_html(token, len(payload.get("issues", [])))


api = FastAPI(title=SERVICE_NAME)
VIEWER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
api.mount("/viewer-cache", StaticFiles(directory=str(VIEWER_CACHE_DIR)), name="viewer-cache")


@api.get("/viewer/{token}", response_class=HTMLResponse)
def viewer(token: str) -> str:
    state = _load_viewer_state(token)
    return _viewer_html(state)


@api.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": "segmentation-v1",
        "segmentation_backend": segmentation_backend_name(),
        "segmentation_backend_error": segmentation_backend_error(),
        "issue_backend": model_backend_name(),
        "issue_backend_error": model_backend_error(),
        "issue_backend_metadata": model_backend_metadata(),
        "issues": supported_issues(),
    }


@api.post("/predict")
async def predict(
    study_zip: UploadFile | None = File(default=None),
    sample_id: str | None = Form(default=None),
    age: float | None = Form(default=None),
    sex: str | None = Form(default=None),
    smoking_history: str | None = Form(default=None),
) -> dict[str, Any]:
    if study_zip is None and not sample_id:
        raise HTTPException(status_code=400, detail="Provide study_zip or sample_id.")
    study_bytes = await study_zip.read() if study_zip is not None else _study_bytes_from_inputs(None, sample_id)
    payload = analyze_study_bytes(
        study_bytes=study_bytes,
        age=age,
        sex=sex,
        smoking_history=smoking_history,
    )
    payload.pop("_viewer", None)
    return payload


def build_demo() -> gr.Blocks:
    sample_keys = sorted(load_samples_manifest().keys())
    initial_sample = DEFAULT_SAMPLE or (sample_keys[0] if sample_keys else "")
    with gr.Blocks(title=SERVICE_NAME, head=VIEWER_HEAD) as demo:
        sample_state = gr.State(value=initial_sample)
        gr.Markdown(
            """
            # CT Scan Semantic Segmentation
            **Research use only.** Upload DICOM file and review semantic issue overlays by slice.
            """
        )
        study_zip = gr.File(label="Upload DICOM file", type="filepath")
        viewer = gr.HTML(value=_blank_viewer_html())

        demo.load(
            fn=render_sample_cached_html if initial_sample else (lambda: _blank_viewer_html()),
            inputs=[sample_state] if initial_sample else None,
            outputs=[viewer],
            show_api=False,
        )

        study_zip.change(
            fn=render_upload_html,
            inputs=[study_zip],
            outputs=[viewer],
            show_api=False,
        )

    return demo


demo = build_demo()
app = gr.mount_gradio_app(api, demo, path="/")


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
