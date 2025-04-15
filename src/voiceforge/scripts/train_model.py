from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from model.speecht5 import (
        DEFAULT_BASE_MODEL,
        DEFAULT_MODEL_DIR,
        DEFAULT_SPEAKER_ENCODER,
        TARGET_SAMPLE_RATE,
        pick_device,
        speaker_embedding_from_components,
    )
except ImportError:
    from src.voiceforge.model.speecht5 import (
        DEFAULT_BASE_MODEL,
        DEFAULT_MODEL_DIR,
        DEFAULT_SPEAKER_ENCODER,
        TARGET_SAMPLE_RATE,
        pick_device,
        speaker_embedding_from_components,
    )


SERVICE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = SERVICE_DIR / "data" / "voiceforge" / "processed"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class SpeechT5TTSDataCollator:
    processor: Any
    reduction_factor: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

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

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(data_dir / "train_manifest.jsonl")
    eval_rows = load_jsonl(data_dir / "eval_manifest.jsonl")
    if not train_rows:
        raise SystemExit(f"No training rows found in {data_dir}. Run scripts/prepare_dataset.py first.")

    if args.max_train_samples is not None:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_eval_samples is not None:
        eval_rows = eval_rows[: args.max_eval_samples]

    processor = SpeechT5Processor.from_pretrained(args.base_model)
    model = SpeechT5ForTextToSpeech.from_pretrained(args.base_model)
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
            )
            speaker_cache[speaker_id] = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return {
            "input_ids": encoded["input_ids"],
            "labels": encoded["labels"][0],
            "speaker_embeddings": speaker_cache[speaker_id],
        }

    remove_columns = train_dataset.column_names
    train_dataset = train_dataset.map(prepare_example, remove_columns=remove_columns)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(prepare_example, remove_columns=eval_dataset.column_names)

    collator = SpeechT5TTSDataCollator(processor=processor, reduction_factor=model.config.reduction_factor)

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
        evaluation_strategy="steps" if eval_dataset is not None and len(eval_dataset) > 0 else "no",
        save_strategy="steps",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        load_best_model_at_end=bool(eval_dataset is not None and len(eval_dataset) > 0),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=False,
        use_mps_device=(device == "mps"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    artifact = {
        "base_model": args.base_model,
        "speaker_model": args.speaker_model,
        "device": device,
        "sample_rate": TARGET_SAMPLE_RATE,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
    }
    (output_dir / "artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
