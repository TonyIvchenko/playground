from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


SERVICE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = SERVICE_DIR / "models" / "speecht5-finetuned"
DEFAULT_BASE_MODEL = "microsoft/speecht5_tts"
DEFAULT_VOCODER = "microsoft/speecht5_hifigan"
DEFAULT_SPEAKER_ENCODER = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SpeechT5Bundle:
    processor: Any
    model: Any
    vocoder: Any
    speaker_encoder: Any
    device: str
    model_source: str
    is_finetuned: bool
    artifact: dict[str, Any]


def pick_device(preferred: str | None = None) -> str:
    if preferred and preferred != "auto":
        return preferred
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_model_source(model_dir: Path | None = None, base_model: str = DEFAULT_BASE_MODEL) -> tuple[str, bool]:
    target_dir = (model_dir or DEFAULT_MODEL_DIR).resolve()
    if (target_dir / "config.json").exists():
        return str(target_dir), True
    return base_model, False


def read_artifact_metadata(model_dir: Path | None = None) -> dict[str, Any]:
    target_dir = (model_dir or DEFAULT_MODEL_DIR).resolve()
    artifact_path = target_dir / "artifact.json"
    if not artifact_path.exists():
        return {}
    return json.loads(artifact_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def load_speecht5_bundle(
    model_dir: str | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    vocoder_name: str = DEFAULT_VOCODER,
    speaker_encoder_name: str = DEFAULT_SPEAKER_ENCODER,
    preferred_device: str = "auto",
) -> SpeechT5Bundle:
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    model_path = Path(model_dir).resolve() if model_dir else DEFAULT_MODEL_DIR
    source, is_finetuned = resolve_model_source(model_path, base_model)
    artifact = read_artifact_metadata(model_path)
    device = pick_device(preferred_device)

    processor = SpeechT5Processor.from_pretrained(source)
    model = SpeechT5ForTextToSpeech.from_pretrained(source, use_safetensors=True)
    vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name, use_safetensors=True)
    model.to(device)
    vocoder.to(device)
    model.eval()
    vocoder.eval()

    cache_dir = model_path / ".cache" / "speaker_encoder"
    cache_dir.mkdir(parents=True, exist_ok=True)
    speaker_encoder = EncoderClassifier.from_hparams(
        source=speaker_encoder_name,
        savedir=str(cache_dir),
        run_opts={"device": device},
    )

    return SpeechT5Bundle(
        processor=processor,
        model=model,
        vocoder=vocoder,
        speaker_encoder=speaker_encoder,
        device=device,
        model_source=source,
        is_finetuned=is_finetuned,
        artifact=artifact,
    )


def load_audio_mono(audio_path: str | Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    import librosa
    import soundfile as sf

    waveform, sample_rate = sf.read(str(audio_path))
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != target_sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 1.0
    if peak > 1.0:
        waveform = waveform / peak
    return waveform.astype(np.float32)


def coerce_speaker_embedding_dim(embedding: Any, target_dim: int | None) -> Any:
    import torch

    if target_dim is None:
        return embedding
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    current_dim = embedding.shape[-1]
    if current_dim == target_dim:
        return embedding
    if current_dim <= 0:
        raise ValueError("Speaker embedding has no feature dimension.")

    resized = torch.nn.functional.interpolate(
        embedding.unsqueeze(1),
        size=target_dim,
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    return torch.nn.functional.normalize(resized, dim=-1)


def speaker_embedding_from_waveform(waveform: np.ndarray, bundle: SpeechT5Bundle) -> Any:
    target_dim = getattr(bundle.model.config, "speaker_embedding_dim", None)
    return speaker_embedding_from_components(
        waveform,
        bundle.speaker_encoder,
        bundle.device,
        target_dim=target_dim,
    )


def speaker_embedding_from_components(
    waveform: np.ndarray,
    speaker_encoder: Any,
    device: str,
    target_dim: int | None = None,
) -> Any:
    import torch

    tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = speaker_encoder.encode_batch(tensor)
    embedding = embedding.squeeze()
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return coerce_speaker_embedding_dim(embedding, target_dim)


def speaker_embedding_from_audio(audio_path: str | Path, bundle: SpeechT5Bundle) -> Any:
    waveform = load_audio_mono(audio_path)
    return speaker_embedding_from_waveform(waveform, bundle)


def synthesize_speech(
    text: str,
    reference_audio_path: str | Path,
    bundle: SpeechT5Bundle,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[int, np.ndarray]:
    import torch

    speaker_embedding = speaker_embedding_from_audio(reference_audio_path, bundle)
    encoded = bundle.processor(text=text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.device)

    with torch.no_grad():
        audio = bundle.model.generate_speech(
            input_ids,
            speaker_embedding,
            attention_mask=attention_mask,
            vocoder=bundle.vocoder,
        )
    waveform = audio.detach().cpu().numpy().astype(np.float32)
    return sample_rate, waveform


def synthesize_to_temp_wav(text: str, reference_audio_path: str | Path, bundle: SpeechT5Bundle) -> tuple[str, str]:
    import soundfile as sf

    sample_rate, waveform = synthesize_speech(text=text, reference_audio_path=reference_audio_path, bundle=bundle)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
        sf.write(handle.name, waveform, sample_rate)
        output_path = handle.name
    source_label = "fine-tuned" if bundle.is_finetuned else "base pretrained"
    return output_path, f"Generated with {source_label} model on {bundle.device}."
