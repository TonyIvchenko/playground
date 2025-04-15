import numpy as np

from src.voiceforge.scripts.train_model import SpeechT5TTSDataCollator


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
