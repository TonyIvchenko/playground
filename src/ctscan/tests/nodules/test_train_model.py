from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.ctscan.scripts.nodules.train_model import (
    auc_from_scores,
    load_training_dataset,
    split_dataset,
    train_model,
)


def test_load_training_dataset_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_training_dataset(tmp_path / "missing.npz")


def test_load_training_dataset_missing_arrays_raises(tmp_path: Path):
    bad_path = tmp_path / "bad.npz"
    np.savez_compressed(bad_path, patches=np.zeros((1, 1, 16, 16, 16), dtype=np.float32))
    with pytest.raises(ValueError):
        load_training_dataset(bad_path)


def test_split_and_train_model_runs(tmp_path: Path):
    dataset_path = tmp_path / "train.npz"
    patches = np.random.default_rng(7).normal(size=(20, 1, 16, 16, 16)).astype(np.float32)
    nodule_target = np.array([0, 1] * 10, dtype=np.float32)
    malignancy_target = np.array([0, 0, 0, 1] * 5, dtype=np.float32)
    np.savez_compressed(dataset_path, patches=patches, nodule_target=nodule_target, malignancy_target=malignancy_target)

    x, y_nodule, y_malignancy = load_training_dataset(dataset_path)
    split = split_dataset(x, y_nodule, y_malignancy, split_seed=3)
    model, patch_mean, patch_std, accuracy, auc = train_model(*split, epochs=2, batch_size=4, learning_rate=1e-3, seed=5)

    assert patch_std > 0.0
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= auc <= 1.0 or np.isnan(auc)
    with torch.no_grad():
        output = model((split[3] - patch_mean) / patch_std)
    assert output.shape[1] == 2


def test_auc_from_scores():
    y_true = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32)
    y_score = torch.tensor([[0.1], [0.3], [0.7], [0.9]], dtype=torch.float32)
    auc = auc_from_scores(y_true, y_score)
    assert 0.99 <= auc <= 1.0
