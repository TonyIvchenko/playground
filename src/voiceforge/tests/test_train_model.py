import numpy as np

from src.voiceforge.scripts.run_tiny_smoke_train import (
    SMOKE_TRAIN_PRESET_ARGS,
    TRAIN_MODEL_SCRIPT,
    build_smoke_train_command,
    normalize_passthrough_args,
)
from src.voiceforge.scripts.prune_checkpoints import prune_checkpoints
from src.voiceforge.scripts.train_model import (
    SpeechT5TTSDataCollator,
    filter_manifest_rows,
    select_preview_rows,
    write_preview_manifest,
)


class DummyTokenizer:
    def pad(self, features, return_tensors):
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        for feature in features:
            row = feature["input_ids"]
            pad = [0] * (max_len - len(row))
            input_ids.append(row + pad)
            attention_mask.append([1] * len(row) + [0] * len(pad))
        import torch

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class DummyProcessor:
    tokenizer = DummyTokenizer()


def test_collator_pads_labels_and_embeddings():
    collator = SpeechT5TTSDataCollator(processor=DummyProcessor(), reduction_factor=2)
    batch = collator(
        [
            {
                "input_ids": [1, 2, 3],
                "labels": np.ones((5, 4), dtype=np.float32),
                "speaker_embeddings": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            },
            {
                "input_ids": [1, 2],
                "labels": np.ones((4, 4), dtype=np.float32),
                "speaker_embeddings": np.array([0.4, 0.5, 0.6], dtype=np.float32),
            },
        ]
    )

    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == (2, 6, 4)
    assert batch["speaker_embeddings"].shape == (2, 3)


def test_select_preview_rows_prefers_eval_then_unique_speakers():
    eval_rows = [
        {
            "speaker_id": "eval_a",
            "audio_path": "/eval_a.wav",
            "source": "libritts",
            "utterance_id": "eval_a_1",
        },
        {
            "speaker_id": "eval_a",
            "audio_path": "/eval_a2.wav",
            "source": "libritts",
            "utterance_id": "eval_a_2",
        },
    ]
    train_rows = [
        {
            "speaker_id": "train_b",
            "audio_path": "/train_b.wav",
            "source": "vctk",
            "utterance_id": "train_b_1",
        },
        {
            "speaker_id": "train_c",
            "audio_path": "/train_c.wav",
            "source": "vctk",
            "utterance_id": "train_c_1",
        },
    ]

    selected = select_preview_rows(eval_rows, train_rows, limit=3)
    assert [row["speaker_id"] for row in selected] == ["eval_a", "train_b", "train_c"]


def test_write_preview_manifest(tmp_path):
    rows = [
        {
            "speaker_id": "speaker_1",
            "audio_path": "/ref.wav",
            "source": "libritts",
            "utterance_id": "utt_1",
        },
    ]
    manifest_path = write_preview_manifest(tmp_path, rows, ["/generated.wav"])
    payload = manifest_path.read_text(encoding="utf-8")
    assert "speaker_1" in payload
    assert "/generated.wav" in payload


def test_filter_manifest_rows_applies_duration_and_text_limits():
    rows = [
        {
            "audio_path": "/a.wav",
            "text": "short",
            "audio_seconds": 3.0,
            "text_length": 5,
        },
        {
            "audio_path": "/b.wav",
            "text": "long enough",
            "audio_seconds": 14.0,
            "text_length": 11,
        },
        {
            "audio_path": "/c.wav",
            "text": "x" * 300,
            "audio_seconds": 4.0,
            "text_length": 300,
        },
    ]

    filtered = filter_manifest_rows(
        rows,
        max_audio_seconds=12.0,
        min_audio_seconds=1.0,
        max_text_chars=200,
    )

    assert filtered == [rows[0]]


def test_prune_checkpoints_keeps_newest_steps(tmp_path):
    checkpoint_64 = tmp_path / "checkpoint-64"
    checkpoint_128 = tmp_path / "checkpoint-128"
    checkpoint_256 = tmp_path / "checkpoint-256"
    preview_dir = tmp_path / "previews"
    for path in (checkpoint_64, checkpoint_128, checkpoint_256, preview_dir):
        path.mkdir()

    summary = prune_checkpoints(tmp_path, keep=2)

    assert checkpoint_64.exists() is False
    assert checkpoint_128.exists() is True
    assert checkpoint_256.exists() is True
    assert preview_dir.exists() is True
    assert summary["kept_checkpoints"] == [
        str(checkpoint_128.resolve()),
        str(checkpoint_256.resolve()),
    ]
    assert summary["removed_checkpoints"] == [str(checkpoint_64.resolve())]


def test_prune_checkpoints_dry_run_leaves_directories_in_place(tmp_path):
    checkpoint_8 = tmp_path / "checkpoint-8"
    checkpoint_16 = tmp_path / "checkpoint-16"
    checkpoint_8.mkdir()
    checkpoint_16.mkdir()

    summary = prune_checkpoints(tmp_path, keep=1, dry_run=True)

    assert checkpoint_8.exists() is True
    assert checkpoint_16.exists() is True
    assert summary["kept_checkpoints"] == [str(checkpoint_16.resolve())]
    assert summary["removed_checkpoints"] == [str(checkpoint_8.resolve())]


def test_tiny_smoke_train_command_uses_named_preset_and_passthrough_args():
    command = build_smoke_train_command(
        python_executable="/usr/bin/python3",
        extra_args=["--device", "mps", "--output-dir", "/tmp/voiceforge-smoke"],
    )

    assert command[:2] == ["/usr/bin/python3", str(TRAIN_MODEL_SCRIPT)]
    assert "--max-train-samples" in command
    assert command[command.index("--max-train-samples") + 1] == "32"
    assert "--max-eval-samples" in command
    assert command[command.index("--max-eval-samples") + 1] == "8"
    assert "--save-total-limit" in command
    assert command[command.index("--save-total-limit") + 1] == "1"
    assert command[-4:] == ["--device", "mps", "--output-dir", "/tmp/voiceforge-smoke"]


def test_tiny_smoke_train_passthrough_normalization_strips_separator():
    assert normalize_passthrough_args(["--", "--device", "cpu"]) == ["--device", "cpu"]
    assert normalize_passthrough_args(["--device", "cpu"]) == ["--device", "cpu"]
    assert "--preview-samples" in SMOKE_TRAIN_PRESET_ARGS
