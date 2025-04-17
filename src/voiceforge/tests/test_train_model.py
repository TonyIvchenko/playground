import numpy as np

from src.voiceforge.scripts.train_model import (
    SpeechT5TTSDataCollator,
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
        {"speaker_id": "eval_a", "audio_path": "/eval_a.wav", "source": "libritts", "utterance_id": "eval_a_1"},
        {"speaker_id": "eval_a", "audio_path": "/eval_a2.wav", "source": "libritts", "utterance_id": "eval_a_2"},
    ]
    train_rows = [
        {"speaker_id": "train_b", "audio_path": "/train_b.wav", "source": "vctk", "utterance_id": "train_b_1"},
        {"speaker_id": "train_c", "audio_path": "/train_c.wav", "source": "vctk", "utterance_id": "train_c_1"},
    ]

    selected = select_preview_rows(eval_rows, train_rows, limit=3)
    assert [row["speaker_id"] for row in selected] == ["eval_a", "train_b", "train_c"]


def test_write_preview_manifest(tmp_path):
    rows = [
        {"speaker_id": "speaker_1", "audio_path": "/ref.wav", "source": "libritts", "utterance_id": "utt_1"},
    ]
    manifest_path = write_preview_manifest(tmp_path, rows, ["/generated.wav"])
    payload = manifest_path.read_text(encoding="utf-8")
    assert "speaker_1" in payload
    assert "/generated.wav" in payload
