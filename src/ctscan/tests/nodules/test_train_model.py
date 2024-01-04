from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.ctscan.models.nodules import PATCH_SHAPE
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
    np.savez_compressed(bad_path, patches=np.zeros((1, 1, *PATCH_SHAPE), dtype=np.int16))
    with pytest.raises(ValueError):
        load_training_dataset(bad_path)


def test_split_and_train_model_runs(tmp_path: Path):
    dataset_path = tmp_path / "train.npz"
    patches = np.random.default_rng(7).normal(size=(20, 1, *PATCH_SHAPE)).astype(np.int16)
    nodule_target = np.array([0, 1] * 10, dtype=np.float32)
    malignancy_target = np.array([0, 0.25, 0, 1] * 5, dtype=np.float32)
    malignancy_mask = np.array([0, 1] * 10, dtype=np.float32)
    series_ids = np.array([f"study-{index // 2}" for index in range(20)])
    np.savez_compressed(
        dataset_path,
        patches=patches,
        nodule_target=nodule_target,
        malignancy_target=malignancy_target,
        malignancy_mask=malignancy_mask,
        series_ids=series_ids,
    )

    x, y_nodule, y_malignancy, y_malignancy_mask, loaded_series_ids = load_training_dataset(dataset_path)
    split = split_dataset(x, y_nodule, y_malignancy, y_malignancy_mask, loaded_series_ids, split_seed=3)
    result = train_model(
        *split,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=5,
        patience=2,
    )

    assert result.patch_std > 0.0
    assert 0.0 <= result.nodule_accuracy <= 1.0
    assert 0.0 <= result.nodule_auc <= 1.0 or np.isnan(result.nodule_auc)
    assert 0.0 <= result.malignancy_auc <= 1.0 or np.isnan(result.malignancy_auc)
    assert 0.0 <= result.nodule_sensitivity <= 1.0
    assert 0.0 <= result.nodule_specificity <= 1.0
    with torch.no_grad():
        output = result.model((split[4] - result.patch_mean) / result.patch_std)
    assert output.shape[1] == 2


def test_auc_from_scores():
    y_true = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32)
    y_score = torch.tensor([[0.1], [0.3], [0.7], [0.9]], dtype=torch.float32)
    auc = auc_from_scores(y_true, y_score)
    assert 0.99 <= auc <= 1.0
