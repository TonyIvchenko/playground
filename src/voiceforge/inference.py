from __future__ import annotations

from pathlib import Path

try:
    from model.speecht5 import (
        load_speecht5_bundle,
        pick_device,
        resolve_model_source,
        synthesize_to_temp_wav,
    )
except ImportError:
    from src.voiceforge.model.speecht5 import (
        load_speecht5_bundle,
        pick_device,
        resolve_model_source,
        synthesize_to_temp_wav,
    )


def format_active_checkpoint(model_source: str, *, is_finetuned: bool) -> str:
    if is_finetuned:
        return model_source
    return f"{model_source} (base pretrained fallback)"


def format_status_with_runtime_context(
    message: str, *, resolved_device: str, active_checkpoint: str
) -> str:
    return "\n".join(
        [
            message,
            f"Resolved device: {resolved_device}",
            f"Active checkpoint: {active_checkpoint}",
        ]
    )


def resolve_runtime_context(
    model_dir: Path, *, preferred_device: str = "auto"
) -> tuple[str, str]:
    model_source, is_finetuned = resolve_model_source(model_dir)
    return (
        pick_device(preferred_device),
        format_active_checkpoint(model_source, is_finetuned=is_finetuned),
    )


def format_initial_status(model_dir: Path, *, preferred_device: str = "auto") -> str:
    resolved_device, active_checkpoint = resolve_runtime_context(
        model_dir, preferred_device=preferred_device
    )
    return format_status_with_runtime_context(
        "Ready to synthesize once you upload a reference clip and enter text.",
        resolved_device=resolved_device,
        active_checkpoint=active_checkpoint,
    )


def format_missing_reference_status(model_dir: Path | None = None) -> str:
    message = "Upload a reference clip first."
    if model_dir is None:
        return message
    resolved_device, active_checkpoint = resolve_runtime_context(model_dir)
    return format_status_with_runtime_context(
        message,
        resolved_device=resolved_device,
        active_checkpoint=active_checkpoint,
    )


def format_missing_text_status(model_dir: Path | None = None) -> str:
    message = "Type text to synthesize first."
    if model_dir is None:
        return message
    resolved_device, active_checkpoint = resolve_runtime_context(model_dir)
    return format_status_with_runtime_context(
        message,
        resolved_device=resolved_device,
        active_checkpoint=active_checkpoint,
    )


def format_inference_failure(exc: Exception, model_dir: Path | None = None) -> str:
    message = f"Voice synthesis failed: {exc}"
    if model_dir is None:
        return message
    resolved_device, active_checkpoint = resolve_runtime_context(model_dir)
    return format_status_with_runtime_context(
        message,
        resolved_device=resolved_device,
        active_checkpoint=active_checkpoint,
    )


def run_inference(
    reference_audio: str | None, text: str, *, model_dir: Path
) -> tuple[str | None, str]:
    text = (text or "").strip()
    if not reference_audio:
        return None, format_missing_reference_status(model_dir)
    if not text:
        return None, format_missing_text_status(model_dir)

    try:
        bundle = load_speecht5_bundle(model_dir=str(model_dir))
        output_path, status = synthesize_to_temp_wav(
            text=text, reference_audio_path=reference_audio, bundle=bundle
        )
        return output_path, format_status_with_runtime_context(
            status,
            resolved_device=bundle.device,
            active_checkpoint=format_active_checkpoint(
                bundle.model_source, is_finetuned=bundle.is_finetuned
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return None, format_inference_failure(exc, model_dir)
