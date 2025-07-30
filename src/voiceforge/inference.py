from __future__ import annotations

from pathlib import Path

try:
    from model.speecht5 import load_speecht5_bundle, synthesize_to_temp_wav
except ImportError:
    from src.voiceforge.model.speecht5 import (
        load_speecht5_bundle,
        synthesize_to_temp_wav,
    )


def format_initial_status(model_dir: Path) -> str:
    return f"Looking for model in {model_dir}"


def format_missing_reference_status() -> str:
    return "Upload a reference clip first."


def format_missing_text_status() -> str:
    return "Type text to synthesize first."


def format_inference_failure(exc: Exception) -> str:
    return f"Voice synthesis failed: {exc}"


def run_inference(
    reference_audio: str | None, text: str, *, model_dir: Path
) -> tuple[str | None, str]:
    text = (text or "").strip()
    if not reference_audio:
        return None, format_missing_reference_status()
    if not text:
        return None, format_missing_text_status()

    try:
        bundle = load_speecht5_bundle(model_dir=str(model_dir))
        output_path, status = synthesize_to_temp_wav(
            text=text, reference_audio_path=reference_audio, bundle=bundle
        )
        return output_path, status
    except Exception as exc:  # noqa: BLE001
        return None, format_inference_failure(exc)
