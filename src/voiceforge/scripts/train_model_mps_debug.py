from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gc
import inspect
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import torch
from transformers import TrainerCallback

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

try:
    from model.speecht5 import (
        DEFAULT_BASE_MODEL,
        DEFAULT_MODEL_DIR,
        DEFAULT_SPEAKER_ENCODER,
        TARGET_SAMPLE_RATE,
        load_speecht5_bundle,
        pick_device,
        speaker_embedding_from_components,
        synthesize_to_temp_wav,
    )
except ImportError:
    from src.voiceforge.model.speecht5 import (
        DEFAULT_BASE_MODEL,
        DEFAULT_MODEL_DIR,
        DEFAULT_SPEAKER_ENCODER,
        TARGET_SAMPLE_RATE,
        load_speecht5_bundle,
        pick_device,
        speaker_embedding_from_components,
        synthesize_to_temp_wav,
    )


def patch_accelerate_compat() -> None:
    try:
        from accelerate import Accelerator
    except ImportError:
        return

    parameters = inspect.signature(Accelerator.unwrap_model).parameters
    if "keep_torch_compile" in parameters:
        return

    original = Accelerator.unwrap_model

    def unwrap_model_compat(self, model, keep_torch_compile=None):  # type: ignore[override]
        return original(self, model)

    Accelerator.unwrap_model = unwrap_model_compat


DEFAULT_DATA_DIR = SERVICE_DIR / "data" / "voiceforge" / "processed"


@dataclass
class BatchStats:
    last_batch_size: int = 0
    last_max_target_length: int = 0
    last_mean_target_length: float = 0.0
    last_max_input_length: int = 0
    last_mean_input_length: float = 0.0
    total_batches_seen: int = 0


@dataclass
class MemorySnapshot:
    rss_mb: float
    mps_current_allocated_mb: float | None = None
    mps_driver_allocated_mb: float | None = None
    mps_recommended_max_mb: float | None = None


@dataclass
class SpeechT5TTSDataCollator:
    processor: Any
    reduction_factor: int
    batch_stats: BatchStats = field(default_factory=BatchStats)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        input_lengths = [len(feature["input_ids"]) for feature in features]
        target_lengths = [len(feature["labels"]) for feature in features]

        self.batch_stats.last_batch_size = len(features)
        self.batch_stats.last_max_target_length = max(target_lengths)
        self.batch_stats.last_mean_target_length = float(sum(target_lengths)) / max(len(target_lengths), 1)
        self.batch_stats.last_max_input_length = max(input_lengths)
        self.batch_stats.last_mean_input_length = float(sum(input_lengths)) / max(len(input_lengths), 1)
        self.batch_stats.total_batches_seen += 1

        input_batch = self.processor.tokenizer.pad(
            [{"input_ids": feature["input_ids"]} for feature in features],
            return_tensors="pt",
        )

        labels = [torch.tensor(feature["labels"], dtype=torch.float32) for feature in features]
        max_len = max(label.shape[0] for label in labels)
        if self.reduction_factor > 1 and max_len % self.reduction_factor:
            max_len += self.reduction_factor - (max_len % self.reduction_factor)

        feature_size = labels[0].shape[1]
        padded_labels = torch.full((len(labels), max_len, feature_size), -100.0, dtype=torch.float32)
        decoder_attention_mask = torch.zeros((len(labels), max_len), dtype=torch.long)

        for index, label in enumerate(labels):
            target_len = label.shape[0]
            if self.reduction_factor > 1:
                target_len -= target_len % self.reduction_factor
            if target_len <= 0:
                target_len = min(label.shape[0], max_len)
            padded_labels[index, :target_len] = label[:target_len]
            decoder_attention_mask[index, :target_len] = 1

        speaker_embeddings = torch.tensor(np.stack([feature["speaker_embeddings"] for feature in features]), dtype=torch.float32)
        batch = dict(input_batch)
        batch["labels"] = padded_labels
        batch["decoder_attention_mask"] = decoder_attention_mask
        batch["speaker_embeddings"] = speaker_embeddings
        return batch


class MPSCacheCallback(TrainerCallback):
    def __init__(self, interval_steps: int) -> None:
        self.interval_steps = interval_steps

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if self.interval_steps <= 0 or state.global_step <= 0:
            return control
        if state.global_step % self.interval_steps == 0 and torch.backends.mps.is_available():
            gc.collect()
            torch.mps.empty_cache()
        return control


class StepDiagnosticsCallback(TrainerCallback):
    def __init__(self, batch_stats: BatchStats, print_every_step: bool = True) -> None:
        self.batch_stats = batch_stats
        self.print_every_step = print_every_step
        self.step_start_time: float | None = None

    @staticmethod
    def _bytes_to_mb(value: int | float | None) -> float | None:
        if value is None:
            return None
        return float(value) / (1024.0 * 1024.0)

    def _memory_snapshot(self) -> MemorySnapshot:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = float(rss_kb) / 1024.0

        current_allocated_mb = None
        driver_allocated_mb = None
        recommended_max_mb = None
        if torch.backends.mps.is_available():
            current_fn = getattr(torch.mps, "current_allocated_memory", None)
            driver_fn = getattr(torch.mps, "driver_allocated_memory", None)
            recommended_fn = getattr(torch.mps, "recommended_max_memory", None)
            if callable(current_fn):
                current_allocated_mb = self._bytes_to_mb(current_fn())
            if callable(driver_fn):
                driver_allocated_mb = self._bytes_to_mb(driver_fn())
            if callable(recommended_fn):
                recommended_max_mb = self._bytes_to_mb(recommended_fn())

        return MemorySnapshot(
            rss_mb=rss_mb,
            mps_current_allocated_mb=current_allocated_mb,
            mps_driver_allocated_mb=driver_allocated_mb,
            mps_recommended_max_mb=recommended_max_mb,
        )

    def on_step_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self.step_start_time = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if not self.print_every_step:
            return control

        elapsed_s = None
        if self.step_start_time is not None:
            elapsed_s = time.perf_counter() - self.step_start_time

        snapshot = self._memory_snapshot()
        payload = {
            "type": "step_diagnostics",
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "step_time_s": round(elapsed_s, 3) if elapsed_s is not None else None,
            "batch_size": self.batch_stats.last_batch_size,
            "batch_max_target_length": self.batch_stats.last_max_target_length,
            "batch_mean_target_length": round(self.batch_stats.last_mean_target_length, 2),
            "batch_max_input_length": self.batch_stats.last_max_input_length,
            "batch_mean_input_length": round(self.batch_stats.last_mean_input_length, 2),
            "rss_mb": round(snapshot.rss_mb, 2),
            "mps_current_allocated_mb": round(snapshot.mps_current_allocated_mb, 2) if snapshot.mps_current_allocated_mb is not None else None,
            "mps_driver_allocated_mb": round(snapshot.mps_driver_allocated_mb, 2) if snapshot.mps_driver_allocated_mb is not None else None,
            "mps_recommended_max_mb": round(snapshot.mps_recommended_max_mb, 2) if snapshot.mps_recommended_max_mb is not None else None,
        }
        print(json.dumps(payload), flush=True)
        return control


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audio_seconds(audio_path: str | Path) -> float:
    import soundfile as sf

    info = sf.info(str(audio_path))
    return float(info.frames) / float(info.samplerate)


def enrich_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.setdefault("text_length", len(item.get("text", "")))
        if "audio_seconds" not in item:
            item["audio_seconds"] = audio_seconds(item["audio_path"])
        enriched.append(item)
    return enriched


def filter_manifest_rows(
    rows: list[dict[str, Any]],
    *,
    max_audio_seconds: float | None = None,
    max_text_chars: int | None = None,
    min_audio_seconds: float | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        text_length = int(row.get("text_length", len(row.get("text", ""))))
        duration = float(row.get("audio_seconds", 0.0))
        if max_audio_seconds is not None and duration > max_audio_seconds:
            continue
        if min_audio_seconds is not None and duration < min_audio_seconds:
            continue
        if max_text_chars is not None and text_length > max_text_chars:
            continue
        filtered.append(row)
    return filtered


def build_device_info(
    *,
    requested_device: str,
    resolved_device: str,
    model: Any,
    training_args: Any,
) -> dict[str, Any]:
    return {
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "model_parameter_device": str(next(model.parameters()).device),
        "use_mps_device": bool(getattr(training_args, "use_mps_device", False)),
    }


def select_preview_rows(
    eval_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    seen_speakers: set[str] = set()
    selected: list[dict[str, Any]] = []
    for pool in (eval_rows, train_rows):
        for row in pool:
            speaker_id = row["speaker_id"]
            if speaker_id in seen_speakers:
                continue
            seen_speakers.add(speaker_id)
            selected.append(row)
            if len(selected) >= limit:
                return selected
    return selected


def write_preview_manifest(preview_dir: Path, rows: list[dict[str, Any]], generated_paths: list[str]) -> Path:
    preview_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = preview_dir / "preview_manifest.json"
    payload = [
        {
            "speaker_id": row["speaker_id"],
            "source": row["source"],
            "utterance_id": row["utterance_id"],
            "reference_audio": row["audio_path"],
            "generated_audio": generated_path,
        }
        for row, generated_path in zip(rows, generated_paths, strict=False)
    ]
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune SpeechT5 for VoiceForge.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--speaker-model", default=DEFAULT_SPEAKER_ENCODER)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--preview-text", default="This is a VoiceForge preview generated after fine-tuning the SpeechT5 model.")
    parser.add_argument("--preview-samples", type=int, default=3)
    parser.add_argument("--max-audio-seconds", type=float, default=None)
    parser.add_argument("--min-audio-seconds", type=float, default=None)
    parser.add_argument("--max-text-chars", type=int, default=None)
    parser.add_argument("--group-by-target-length", action="store_true")
    parser.add_argument("--mps-empty-cache-steps", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--print-memory-every-step", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from datasets import Audio, Dataset
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        SpeechT5ForTextToSpeech,
        SpeechT5Processor,
    )

    patch_accelerate_compat()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(data_dir / "train_manifest.jsonl")
    eval_rows = load_jsonl(data_dir / "eval_manifest.jsonl")
    if not train_rows:
        raise SystemExit(f"No training rows found in {data_dir}. Run scripts/prepare_dataset.py first.")
    raw_train_row_count = len(train_rows)
    raw_eval_row_count = len(eval_rows)

    train_rows = enrich_manifest_rows(train_rows)
    eval_rows = enrich_manifest_rows(eval_rows)

    train_rows = filter_manifest_rows(
        train_rows,
        max_audio_seconds=args.max_audio_seconds,
        min_audio_seconds=args.min_audio_seconds,
        max_text_chars=args.max_text_chars,
    )
    eval_rows = filter_manifest_rows(
        eval_rows,
        max_audio_seconds=args.max_audio_seconds,
        min_audio_seconds=args.min_audio_seconds,
        max_text_chars=args.max_text_chars,
    )

    if args.max_train_samples is not None:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_eval_samples is not None:
        eval_rows = eval_rows[: args.max_eval_samples]

    processor = SpeechT5Processor.from_pretrained(args.base_model)
    model = SpeechT5ForTextToSpeech.from_pretrained(args.base_model, use_safetensors=True)
    speaker_embedding_dim = getattr(model.config, "speaker_embedding_dim", None)
    device = pick_device(args.device)
    model.to(device)

    speaker_encoder = EncoderClassifier.from_hparams(
        source=args.speaker_model,
        savedir=str(output_dir / ".cache" / "speaker_encoder"),
        run_opts={"device": device},
    )

    train_dataset = Dataset.from_list(train_rows).rename_column("audio_path", "audio")
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    eval_dataset = Dataset.from_list(eval_rows).rename_column("audio_path", "audio") if eval_rows else None
    if eval_dataset is not None:
        eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

    speaker_cache: dict[str, np.ndarray] = {}

    def prepare_example(example: dict[str, Any]) -> dict[str, Any]:
        audio = example["audio"]
        encoded = processor(
            text=example["text"],
            audio_target=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=True,
        )
        speaker_id = example["speaker_id"]
        if speaker_id not in speaker_cache:
            embedding = speaker_embedding_from_components(
                np.asarray(audio["array"], dtype=np.float32),
                speaker_encoder=speaker_encoder,
                device=device,
                target_dim=speaker_embedding_dim,
            )
            speaker_cache[speaker_id] = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return {
            "input_ids": encoded["input_ids"],
            "labels": encoded["labels"][0],
            "speaker_embeddings": speaker_cache[speaker_id],
            "target_length": len(encoded["labels"][0]),
        }

    remove_columns = train_dataset.column_names
    train_dataset = train_dataset.map(prepare_example, remove_columns=remove_columns)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(prepare_example, remove_columns=eval_dataset.column_names)

    batch_stats = BatchStats()
    collator = SpeechT5TTSDataCollator(
        processor=processor,
        reduction_factor=model.config.reduction_factor,
        batch_stats=batch_stats,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        num_train_epochs=args.epochs,
        eval_strategy="steps" if eval_dataset is not None and len(eval_dataset) > 0 else "no",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=False,
        load_best_model_at_end=False,
        fp16=False,
        bf16=False,
        group_by_length=args.group_by_target_length,
        length_column_name="target_length" if args.group_by_target_length else None,
        use_mps_device=(device == "mps"),
    )

    callbacks: list[TrainerCallback] = []
    if device == "mps" and args.mps_empty_cache_steps > 0:
        callbacks.append(MPSCacheCallback(args.mps_empty_cache_steps))
    if args.print_memory_every_step:
        callbacks.append(StepDiagnosticsCallback(batch_stats=batch_stats, print_every_step=True))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=callbacks or None,
    )
    device_info = build_device_info(
        requested_device=args.device,
        resolved_device=device,
        model=model,
        training_args=training_args,
    )
    startup_info = {
        "device_info": device_info,
        "raw_train_rows": raw_train_row_count,
        "raw_eval_rows": raw_eval_row_count,
        "filtered_train_rows": len(train_rows),
        "filtered_eval_rows": len(eval_rows),
        "max_audio_seconds": args.max_audio_seconds,
        "min_audio_seconds": args.min_audio_seconds,
        "max_text_chars": args.max_text_chars,
        "group_by_target_length": args.group_by_target_length,
        "mps_empty_cache_steps": args.mps_empty_cache_steps,
        "dataloader_num_workers": args.dataloader_num_workers,
        "print_memory_every_step": args.print_memory_every_step,
    }
    (output_dir / "device_info.json").write_text(json.dumps(startup_info, indent=2), encoding="utf-8")
    print(json.dumps(startup_info, indent=2), flush=True)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    artifact = {
        "base_model": args.base_model,
        "speaker_model": args.speaker_model,
        "device": device,
        "sample_rate": TARGET_SAMPLE_RATE,
        "raw_train_rows": raw_train_row_count,
        "raw_eval_rows": raw_eval_row_count,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "max_audio_seconds": args.max_audio_seconds,
        "min_audio_seconds": args.min_audio_seconds,
        "max_text_chars": args.max_text_chars,
        "group_by_target_length": args.group_by_target_length,
        "device_info": device_info,
    }
    (output_dir / "artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    preview_rows = select_preview_rows(eval_rows, train_rows, args.preview_samples)
    generated_paths: list[str] = []
    if preview_rows:
        bundle = load_speecht5_bundle(model_dir=str(output_dir), preferred_device=device)
        preview_dir = output_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(preview_rows, start=1):
            generated_path, _status = synthesize_to_temp_wav(
                text=args.preview_text,
                reference_audio_path=row["audio_path"],
                bundle=bundle,
            )
            target_path = preview_dir / f"{index:02d}_{row['speaker_id']}.wav"
            Path(generated_path).replace(target_path)
            generated_paths.append(str(target_path))
        manifest_path = write_preview_manifest(preview_dir, preview_rows, generated_paths)
        artifact["preview_manifest"] = str(manifest_path)
        (output_dir / "artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
