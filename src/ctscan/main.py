"""Chest CT semantic segmentation service."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import zipfile
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import gradio as gr
import numpy as np
import pandas as pd
import uvicorn

try:
    from study import (
        blank_viewer_image,
        issue_rows_for_table,
        issue_slice_stats,
        issue_volume_stats,
        load_study_from_zip_bytes,
        model_backend_error,
        model_backend_metadata,
        model_backend_name,
        read_temp_bundle,
        render_segmentation_slice,
        segment_issues,
        segment_lungs,
        segmentation_backend_error,
        segmentation_backend_name,
        slice_rows_for_table,
        supported_issues,
        write_temp_bundle,
        write_temp_image,
    )
except ModuleNotFoundError:
    from src.ctscan.study import (
        blank_viewer_image,
        issue_rows_for_table,
        issue_slice_stats,
        issue_volume_stats,
        load_study_from_zip_bytes,
        model_backend_error,
        model_backend_metadata,
        model_backend_name,
        read_temp_bundle,
        render_segmentation_slice,
        segment_issues,
        segment_lungs,
        segmentation_backend_error,
        segmentation_backend_name,
        slice_rows_for_table,
        supported_issues,
        write_temp_bundle,
        write_temp_image,
    )


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
SERVICE_NAME = os.getenv("SERVICE_NAME", "ctscan")
SAMPLES_MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "ctscan" / "samples" / "samples.json"
EXTERNAL_SAMPLES_MANIFEST_PATH = Path("/Volumes/Extreme Pro/data/ctscan/samples/samples.json")
DEFAULT_SAMPLE = ""
WINDOW_CHOICES = ["lung", "mediastinal"]
ISSUE_CHOICES = ["all"] + [str(item["key"]) for item in supported_issues()]
ISSUE_TABLE_COLUMNS = ["Issue", "Lung %", "Volume ml", "Voxels"]
SLICE_TABLE_COLUMNS = ["Issue", "Slice % of lung", "Pixels"]


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
    roots.append(Path("/Volumes/Extreme Pro/data/ctscan/raw/lidc/LIDC-IDRI"))
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


@lru_cache(maxsize=1)
def load_samples_manifest() -> dict[str, dict[str, str]]:
    for path in [SAMPLES_MANIFEST_PATH, EXTERNAL_SAMPLES_MANIFEST_PATH]:
        if path.exists():
            manifest = _read_manifest(path)
            if manifest:
                return manifest
    return _ensure_auto_demo_manifest()


def _study_bytes_from_inputs(study_file: str | None, sample_id: str | None) -> bytes:
    if study_file:
        return Path(study_file).read_bytes()
    if sample_id:
        manifest = load_samples_manifest()
        sample = manifest.get(sample_id)
        if sample:
            study_path = Path(__file__).resolve().parent / sample["study_zip"]
            if study_path.exists():
                return study_path.read_bytes()
        raise FileNotFoundError(
            f"Sample `{sample_id}` is unavailable. Run `python scripts/segmentation/download_data.py` from src/ctscan."
        )
    raise ValueError("Provide a study zip or a sample id.")


def format_json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _slice_damage_percentages(labels: np.ndarray, lung_mask: np.ndarray) -> list[float]:
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


def _slice_count_from_payload(payload: Any) -> int:
    return int(payload["_viewer"]["slice_count"])


def render_payload_slice(
    payload: Any,
    slice_index: int,
    preset: str,
    focus_issue: str,
    show_lung_layer: bool,
    show_damage_layer: bool,
    lung_alpha: float,
    damage_alpha: float,
) -> str:
    volume_hu, labels, lung_mask = read_temp_bundle(payload["_viewer"]["bundle_path"])
    image = render_segmentation_slice(
        volume_hu=volume_hu,
        labels=labels,
        lung_mask=lung_mask,
        slice_index=int(slice_index),
        preset=preset,
        focus_issue=focus_issue,
        show_lung_layer=bool(show_lung_layer),
        show_damage_layer=bool(show_damage_layer),
        lung_alpha=float(lung_alpha),
        damage_alpha=float(damage_alpha),
    )
    return write_temp_image(image)


def dataframe_from_payload(payload: Any) -> pd.DataFrame:
    return pd.DataFrame(issue_rows_for_table(payload.get("issues", [])), columns=ISSUE_TABLE_COLUMNS)


def dataframe_for_slice(payload: Any, slice_index: int) -> pd.DataFrame:
    _, labels, lung_mask = read_temp_bundle(payload["_viewer"]["bundle_path"])
    rows = issue_slice_stats(labels, lung_mask, int(slice_index))
    return pd.DataFrame(slice_rows_for_table(rows), columns=SLICE_TABLE_COLUMNS)


def _default_slice_index(payload: Any) -> int:
    values = payload.get("slice_damage_percent", [])
    if not values:
        return 0
    return int(np.argmax(np.asarray(values, dtype=np.float32)))


def analyze_from_inputs(
    sample_id: str,
    study_file: str | None,
    age: float | None,
    sex: str,
    smoking_history: str,
    preset: str,
    focus_issue: str,
    show_lung_layer: bool,
    show_damage_layer: bool,
    lung_alpha: float,
    damage_alpha: float,
) -> tuple[Any, str, pd.DataFrame, pd.DataFrame, str, gr.Slider, str, str, str]:
    study_bytes = _study_bytes_from_inputs(study_file, sample_id or None)
    payload = analyze_study_bytes(
        study_bytes=study_bytes,
        age=age,
        sex=sex or None,
        smoking_history=smoking_history or None,
    )

    default_slice = _default_slice_index(payload)
    image_path = render_payload_slice(
        payload,
        default_slice,
        preset,
        focus_issue,
        show_lung_layer,
        show_damage_layer,
        lung_alpha,
        damage_alpha,
    )
    status_text = "Rejected" if payload["qc"]["status"] != "ok" else "Ready"

    return (
        payload,
        status_text,
        dataframe_from_payload(payload),
        dataframe_for_slice(payload, default_slice),
        image_path,
        gr.Slider(minimum=0, maximum=max(0, _slice_count_from_payload(payload) - 1), value=default_slice, step=1),
        format_json_text(payload["qc"]),
        format_json_text(payload["summary"]),
        format_json_text(payload["study_metadata"]),
    )


def load_default_demo_state(
    sample_id: str,
    preset: str,
    focus_issue: str,
    show_lung_layer: bool,
    show_damage_layer: bool,
    lung_alpha: float,
    damage_alpha: float,
) -> tuple[Any, str, pd.DataFrame, pd.DataFrame, str, gr.Slider, str, str, str]:
    selected = (sample_id or "").strip()
    if not selected:
        return (
            None,
            "No sample loaded",
            pd.DataFrame(columns=ISSUE_TABLE_COLUMNS),
            pd.DataFrame(columns=SLICE_TABLE_COLUMNS),
            blank_viewer_image(),
            gr.Slider(minimum=0, maximum=0, value=0, step=1),
            format_json_text({"status": "empty", "rejection_reasons": []}),
            format_json_text({}),
            format_json_text({}),
        )
    return analyze_from_inputs(
        sample_id=selected,
        study_file=None,
        age=None,
        sex="",
        smoking_history="",
        preset=preset,
        focus_issue=focus_issue,
        show_lung_layer=show_lung_layer,
        show_damage_layer=show_damage_layer,
        lung_alpha=lung_alpha,
        damage_alpha=damage_alpha,
    )


def update_viewer(
    payload: Any,
    slice_index: int,
    preset: str,
    focus_issue: str,
    show_lung_layer: bool,
    show_damage_layer: bool,
    lung_alpha: float,
    damage_alpha: float,
) -> tuple[str, pd.DataFrame]:
    if not payload:
        return blank_viewer_image(), pd.DataFrame(columns=SLICE_TABLE_COLUMNS)
    image_path = render_payload_slice(
        payload,
        int(slice_index),
        preset,
        focus_issue,
        show_lung_layer,
        show_damage_layer,
        lung_alpha,
        damage_alpha,
    )
    slice_df = dataframe_for_slice(payload, int(slice_index))
    return image_path, slice_df


api = FastAPI(title=SERVICE_NAME)


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
    sample_choices = [""] + sample_keys
    initial_sample = DEFAULT_SAMPLE or (sample_keys[0] if sample_keys else "")
    with gr.Blocks(title=SERVICE_NAME) as demo:
        payload_state = gr.State(value=None)
        gr.Markdown(
            """
            # CT Scan Semantic Segmentation
            **Research use only.** Upload chest CT DICOM zip and review semantic issue overlays by slice.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                sample_id = gr.Dropdown(label="Sample case", choices=sample_choices, value=initial_sample)
                study_zip = gr.File(label="Chest CT DICOM zip", type="filepath")
                age = gr.Number(label="Age", value=None, precision=0)
                sex = gr.Dropdown(label="Sex", choices=["", "female", "male"], value="")
                smoking_history = gr.Textbox(label="Smoking history", value="")
                analyze_button = gr.Button("Analyze")
                status_box = gr.Textbox(label="Status", interactive=False)
                summary_json = gr.Code(label="Summary", language="json", interactive=False)
                qc_json = gr.Code(label="QC", language="json", interactive=False)
                metadata_json = gr.Code(label="Study metadata", language="json", interactive=False)

            with gr.Column(scale=2):
                with gr.Row():
                    window_preset = gr.Dropdown(label="Window", choices=WINDOW_CHOICES, value="lung")
                    focus_issue = gr.Dropdown(label="Focus issue", choices=ISSUE_CHOICES, value="all")
                with gr.Row():
                    show_lung_layer = gr.Checkbox(label="Lung mask layer", value=True)
                    show_damage_layer = gr.Checkbox(label="Damage layer", value=True)
                with gr.Row():
                    lung_alpha = gr.Slider(label="Lung opacity", minimum=0.0, maximum=1.0, value=0.32, step=0.05)
                    damage_alpha = gr.Slider(label="Damage opacity", minimum=0.0, maximum=1.0, value=0.45, step=0.05)
                viewer = gr.Image(label="Axial viewer", type="filepath")
                slice_slider = gr.Slider(label="Slice", minimum=0, maximum=0, value=0, step=1)
                issues_df = gr.Dataframe(
                    label="Lung damage by issue type",
                    headers=ISSUE_TABLE_COLUMNS,
                    datatype=["str", "number", "number", "number"],
                    interactive=False,
                )
                slice_df = gr.Dataframe(
                    label="Current slice damage by issue type",
                    headers=SLICE_TABLE_COLUMNS,
                    datatype=["str", "number", "number"],
                    interactive=False,
                )

        analyze_button.click(
            fn=analyze_from_inputs,
            inputs=[
                sample_id,
                study_zip,
                age,
                sex,
                smoking_history,
                window_preset,
                focus_issue,
                show_lung_layer,
                show_damage_layer,
                lung_alpha,
                damage_alpha,
            ],
            outputs=[payload_state, status_box, issues_df, slice_df, viewer, slice_slider, qc_json, summary_json, metadata_json],
                show_api=False,
            )

        demo.load(
            fn=load_default_demo_state,
            inputs=[
                sample_id,
                window_preset,
                focus_issue,
                show_lung_layer,
                show_damage_layer,
                lung_alpha,
                damage_alpha,
            ],
            outputs=[payload_state, status_box, issues_df, slice_df, viewer, slice_slider, qc_json, summary_json, metadata_json],
            show_api=False,
        )

        slice_slider.change(
            fn=update_viewer,
            inputs=[
                payload_state,
                slice_slider,
                window_preset,
                focus_issue,
                show_lung_layer,
                show_damage_layer,
                lung_alpha,
                damage_alpha,
            ],
            outputs=[viewer, slice_df],
            show_api=False,
        )
        window_preset.change(
            fn=update_viewer,
            inputs=[
                payload_state,
                slice_slider,
                window_preset,
                focus_issue,
                show_lung_layer,
                show_damage_layer,
                lung_alpha,
                damage_alpha,
            ],
            outputs=[viewer, slice_df],
            show_api=False,
        )
        focus_issue.change(
            fn=update_viewer,
            inputs=[
                payload_state,
                slice_slider,
                window_preset,
                focus_issue,
                show_lung_layer,
                show_damage_layer,
                lung_alpha,
                damage_alpha,
            ],
            outputs=[viewer, slice_df],
            show_api=False,
        )
        show_lung_layer.change(
            fn=update_viewer,
            inputs=[
                payload_state,
                slice_slider,
                window_preset,
                focus_issue,
                show_lung_layer,
                show_damage_layer,
                lung_alpha,
                damage_alpha,
            ],
            outputs=[viewer, slice_df],
            show_api=False,
        )
        show_damage_layer.change(
            fn=update_viewer,
            inputs=[
                payload_state,
                slice_slider,
                window_preset,
                focus_issue,
                show_lung_layer,
                show_damage_layer,
                lung_alpha,
                damage_alpha,
            ],
            outputs=[viewer, slice_df],
            show_api=False,
        )
        lung_alpha.change(
            fn=update_viewer,
            inputs=[
                payload_state,
                slice_slider,
                window_preset,
                focus_issue,
                show_lung_layer,
                show_damage_layer,
                lung_alpha,
                damage_alpha,
            ],
            outputs=[viewer, slice_df],
            show_api=False,
        )
        damage_alpha.change(
            fn=update_viewer,
            inputs=[
                payload_state,
                slice_slider,
                window_preset,
                focus_issue,
                show_lung_layer,
                show_damage_layer,
                lung_alpha,
                damage_alpha,
            ],
            outputs=[viewer, slice_df],
            show_api=False,
        )

    return demo


demo = build_demo()
app = gr.mount_gradio_app(api, demo, path="/")


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
