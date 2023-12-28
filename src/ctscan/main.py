"""Chest CT pulmonary nodule triage service."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import gradio as gr
import numpy as np
import pandas as pd
import torch
import uvicorn

try:
    from models.nodules import load_model_bundle, predict_logits
    from study import blank_viewer_image, estimate_lung_mask, finding_rows, generate_candidates, load_study_from_zip_bytes, match_prior_findings, read_temp_volume, render_slice_image, write_temp_image, write_temp_volume
except ModuleNotFoundError:
    from src.ctscan.models.nodules import load_model_bundle, predict_logits
    from src.ctscan.study import blank_viewer_image, estimate_lung_mask, finding_rows, generate_candidates, load_study_from_zip_bytes, match_prior_findings, read_temp_volume, render_slice_image, write_temp_image, write_temp_volume


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
SERVICE_NAME = os.getenv("SERVICE_NAME", "ctscan")
MODEL_PATH = Path(os.getenv("NODULES_MODEL_PATH", str(Path(__file__).resolve().parent / "models" / "nodules.pt")))
SAMPLES_MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "ctscan" / "samples" / "samples.json"
DEFAULT_SAMPLE = ""
WINDOW_CHOICES = ["lung", "mediastinal"]
FINDING_COLUMNS = ["Lesion", "Slice", "Diameter mm", "Volume mm3", "Nodule Prob", "Malig Risk", "Growth mm"]


@lru_cache(maxsize=1)
def get_model_bundle() -> tuple[torch.nn.Module, float, float, str, dict[str, float]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {MODEL_PATH}. Run `python scripts/nodules/download_data.py` and `python scripts/nodules/train_model.py`."
        )
    return load_model_bundle(MODEL_PATH)


@lru_cache(maxsize=1)
def load_samples_manifest() -> dict[str, dict[str, str]]:
    if not SAMPLES_MANIFEST_PATH.exists():
        return {}
    return json.loads(SAMPLES_MANIFEST_PATH.read_text(encoding="utf-8"))


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
            f"Sample `{sample_id}` is unavailable. Run `python scripts/nodules/download_data.py` from src/ctscan."
        )
    raise ValueError("Provide a study zip or a sample id.")


def _score_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return candidates
    model, patch_mean, patch_std, _, _ = get_model_bundle()
    patches = torch.from_numpy(np.stack([candidate["patch"] for candidate in candidates], axis=0)).unsqueeze(1)
    logits = predict_logits(model, patches, patch_mean=patch_mean, patch_std=patch_std)
    probs = torch.sigmoid(logits).cpu().numpy()
    for candidate, prob in zip(candidates, probs):
        candidate["nodule_probability"] = float(prob[0])
        candidate["malignancy_risk"] = float(prob[1])
        candidate.pop("patch", None)
    return candidates


def _risk_text(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "No suspicious nodules detected."
    top = max(findings, key=lambda item: item["malignancy_risk"])
    if top["malignancy_risk"] >= 0.7:
        return "High-risk nodule candidate needs radiologist review."
    if top["malignancy_risk"] >= 0.4:
        return "Intermediate-risk nodule candidate found."
    return "Low-risk nodule candidates found."


def analyze_study_bytes(
    study_bytes: bytes,
    prior_study_bytes: bytes | None = None,
    age: float | None = None,
    sex: str | None = None,
    smoking_history: str | None = None,
) -> dict[str, Any]:
    model, _, _, model_version, metrics = get_model_bundle()
    _ = model

    current = load_study_from_zip_bytes(study_bytes)
    prior = load_study_from_zip_bytes(prior_study_bytes) if prior_study_bytes else None

    status = "ok"
    rejection_reasons = list(current.qc_reasons)
    if len(rejection_reasons) > 0:
        status = "rejected"

    current_candidates = _score_candidates(generate_candidates(current.volume_hu, estimate_lung_mask(current.volume_hu)))
    prior_candidates = []
    if prior is not None:
        prior_candidates = _score_candidates(generate_candidates(prior.volume_hu, estimate_lung_mask(prior.volume_hu)))
        current_candidates = match_prior_findings(current_candidates, prior_candidates)

    current_candidates = sorted(
        current_candidates,
        key=lambda item: (float(item["malignancy_risk"]), float(item["nodule_probability"])),
        reverse=True,
    )

    summary = {
        "finding_count": len(current_candidates),
        "highest_nodule_probability": round(max([item["nodule_probability"] for item in current_candidates], default=0.0), 4),
        "highest_malignancy_risk": round(max([item["malignancy_risk"] for item in current_candidates], default=0.0), 4),
        "message": _risk_text(current_candidates),
    }

    for finding in current_candidates:
        finding["center"] = [int(x) for x in finding["center"]]

    return {
        "model_version": model_version,
        "model_metrics": {
            "accuracy": round(float(metrics.get("nodule_accuracy", 0.0)), 4),
            "auc": round(float(metrics.get("malignancy_auc", 0.0)), 4),
        },
        "qc": {
            "status": status,
            "rejection_reasons": rejection_reasons,
        },
        "study_metadata": {
            **current.metadata,
            "age": age,
            "sex": sex,
            "smoking_history": smoking_history,
        },
        "findings": current_candidates,
        "summary": summary,
        "prior_summary": {
            "available": prior is not None,
            "finding_count": len(prior_candidates),
        },
        "_viewer": {
            "volume_shape": list(current.volume_hu.shape),
            "volume_path": write_temp_volume(current.volume_hu),
        },
    }


def _slice_count_from_payload(payload: dict[str, Any]) -> int:
    return int(payload["_viewer"]["volume_shape"][0])


def render_payload_slice(payload: dict[str, Any], slice_index: int, preset: str, lesion_id: str | None) -> str:
    volume_hu = read_temp_volume(payload["_viewer"]["volume_path"])
    image = render_slice_image(volume_hu, payload.get("findings", []), slice_index, preset, lesion_id)
    return write_temp_image(image)


def dataframe_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(finding_rows(payload.get("findings", [])), columns=FINDING_COLUMNS)


def analyze_from_inputs(
    sample_id: str,
    study_file: str | None,
    prior_study_file: str | None,
    age: float | None,
    sex: str,
    smoking_history: str,
    preset: str,
) -> tuple[dict[str, Any], str, pd.DataFrame, str, gr.Slider, gr.Dropdown, dict[str, Any], dict[str, Any], dict[str, Any]]:
    study_bytes = _study_bytes_from_inputs(study_file, sample_id or None)
    prior_bytes = Path(prior_study_file).read_bytes() if prior_study_file else None
    payload = analyze_study_bytes(
        study_bytes=study_bytes,
        prior_study_bytes=prior_bytes,
        age=age,
        sex=sex or None,
        smoking_history=smoking_history or None,
    )
    findings_df = dataframe_from_payload(payload)
    lesion_choices = [str(item["lesion_id"]) for item in payload.get("findings", [])]
    selected_lesion = lesion_choices[0] if lesion_choices else None
    default_slice = int(payload["findings"][0]["slice_index"]) if payload["findings"] else _slice_count_from_payload(payload) // 2
    image_path = render_payload_slice(payload, default_slice, preset, selected_lesion)
    status_text = "Rejected" if payload["qc"]["status"] != "ok" else "Ready"
    return (
        payload,
        status_text,
        findings_df,
        image_path,
        gr.Slider(minimum=0, maximum=max(0, _slice_count_from_payload(payload) - 1), value=default_slice, step=1),
        gr.Dropdown(choices=lesion_choices, value=selected_lesion),
        payload["qc"],
        payload["summary"],
        payload["study_metadata"],
    )


def update_viewer(payload: dict[str, Any], slice_index: int, preset: str, lesion_id: str | None) -> str:
    if not payload:
        return blank_viewer_image()
    return render_payload_slice(payload, int(slice_index), preset, lesion_id)


def select_finding(payload: dict[str, Any], evt: gr.SelectData, preset: str) -> tuple[gr.Slider, gr.Dropdown, str]:
    if not payload or evt.index is None:
        return gr.Slider(), gr.Dropdown(), blank_viewer_image()
    row_index = int(evt.index[0])
    findings = payload.get("findings", [])
    if row_index < 0 or row_index >= len(findings):
        return gr.Slider(), gr.Dropdown(), blank_viewer_image()
    lesion_id = str(findings[row_index]["lesion_id"])
    slice_index = int(findings[row_index]["slice_index"])
    return (
        gr.Slider(value=slice_index),
        gr.Dropdown(value=lesion_id),
        render_payload_slice(payload, slice_index, preset, lesion_id),
    )


api = FastAPI(title=SERVICE_NAME)


@api.get("/health")
def health() -> dict[str, Any]:
    _, _, _, model_version, metrics = get_model_bundle()
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "model_version": model_version,
        "accuracy": round(float(metrics.get("nodule_accuracy", 0.0)), 4),
        "auc": round(float(metrics.get("malignancy_auc", 0.0)), 4),
    }


@api.post("/predict")
async def predict(
    study_zip: UploadFile | None = File(default=None),
    prior_study_zip: UploadFile | None = File(default=None),
    sample_id: str | None = Form(default=None),
    age: float | None = Form(default=None),
    sex: str | None = Form(default=None),
    smoking_history: str | None = Form(default=None),
) -> dict[str, Any]:
    if study_zip is None and not sample_id:
        raise HTTPException(status_code=400, detail="Provide study_zip or sample_id.")
    study_bytes = await study_zip.read() if study_zip is not None else _study_bytes_from_inputs(None, sample_id)
    prior_bytes = await prior_study_zip.read() if prior_study_zip is not None else None
    payload = analyze_study_bytes(
        study_bytes=study_bytes,
        prior_study_bytes=prior_bytes,
        age=age,
        sex=sex,
        smoking_history=smoking_history,
    )
    payload.pop("_viewer", None)
    return payload


def build_demo() -> gr.Blocks:
    sample_choices = [""] + sorted(load_samples_manifest().keys())
    with gr.Blocks(title=SERVICE_NAME) as demo:
        payload_state = gr.State({})
        gr.Markdown(
            """
            # CT Scan Service
            **Research use only.** v1 supports pulmonary nodule review on chest CT only. It is not a diagnosis.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                sample_id = gr.Dropdown(label="Sample case", choices=sample_choices, value=DEFAULT_SAMPLE)
                study_zip = gr.File(label="Chest CT DICOM zip", type="filepath")
                prior_zip = gr.File(label="Prior chest CT zip", type="filepath")
                age = gr.Number(label="Age", value=None, precision=0)
                sex = gr.Dropdown(label="Sex", choices=["", "female", "male"], value="")
                smoking_history = gr.Textbox(label="Smoking history", value="")
                analyze_button = gr.Button("Analyze")
                status_box = gr.Textbox(label="Status", interactive=False)
                qc_json = gr.JSON(label="QC")
                summary_json = gr.JSON(label="Summary")
                metadata_json = gr.JSON(label="Study metadata")
            with gr.Column(scale=2):
                with gr.Row():
                    window_preset = gr.Dropdown(label="Window", choices=WINDOW_CHOICES, value="lung")
                    selected_lesion = gr.Dropdown(label="Finding focus", choices=[], value=None)
                viewer = gr.Image(label="Axial viewer", type="filepath")
                slice_slider = gr.Slider(label="Slice", minimum=0, maximum=0, value=0, step=1)
                findings = gr.Dataframe(
                    label="Findings",
                    headers=FINDING_COLUMNS,
                    datatype=["str", "number", "number", "number", "number", "number", "str"],
                    interactive=False,
                )

        analyze_button.click(
            fn=analyze_from_inputs,
            inputs=[sample_id, study_zip, prior_zip, age, sex, smoking_history, window_preset],
            outputs=[payload_state, status_box, findings, viewer, slice_slider, selected_lesion, qc_json, summary_json, metadata_json],
        )
        slice_slider.change(
            fn=update_viewer,
            inputs=[payload_state, slice_slider, window_preset, selected_lesion],
            outputs=viewer,
        )
        window_preset.change(
            fn=update_viewer,
            inputs=[payload_state, slice_slider, window_preset, selected_lesion],
            outputs=viewer,
        )
        selected_lesion.change(
            fn=update_viewer,
            inputs=[payload_state, slice_slider, window_preset, selected_lesion],
            outputs=viewer,
        )
        findings.select(
            fn=select_finding,
            inputs=[payload_state, window_preset],
            outputs=[slice_slider, selected_lesion, viewer],
        )
    return demo


demo = build_demo()
app = gr.mount_gradio_app(api, demo, path="/")


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
