from __future__ import annotations

from pathlib import Path

import torch

from src.ctscan.models.nodules import LegacyPatchNet, PATCH_SHAPE, create_model, load_model_bundle, predict_logits, save_model_bundle


def test_save_load_and_predict_bundle(tmp_path: Path):
    model = create_model()
    path = tmp_path / "nodules.pt"
    save_model_bundle(
        path=path,
        model=model,
        patch_mean=-700.0,
        patch_std=250.0,
        model_version="0.1.0",
        nodule_accuracy=0.75,
        nodule_auc=0.78,
        malignancy_auc=0.81,
        dataset_rows=64,
    )
    loaded_model, patch_mean, patch_std, version, metrics = load_model_bundle(path)
    assert version == "0.1.0"
    assert patch_mean == -700.0
    assert patch_std == 250.0
    assert metrics["nodule_accuracy"] == 0.75
    assert metrics["nodule_auc"] == 0.78
    assert metrics["patch_shape"] == list(PATCH_SHAPE)
    logits = predict_logits(loaded_model, torch.zeros((2, 1, *PATCH_SHAPE), dtype=torch.float32), patch_mean, patch_std)
    assert logits.shape == (2, 2)


def test_load_model_bundle_supports_legacy_encoder(tmp_path: Path):
    path = tmp_path / "legacy.pt"
    legacy = LegacyPatchNet()
    torch.save(
        {
            "state_dict": legacy.state_dict(),
            "patch_mean": -650.0,
            "patch_std": 180.0,
            "model_version": "0.3.0",
            "nodule_accuracy": 0.8,
            "malignancy_auc": 0.82,
            "dataset_rows": 150,
            "patch_shape": [16, 16, 16],
        },
        path,
    )
    loaded_model, patch_mean, patch_std, version, metrics = load_model_bundle(path)
    logits = predict_logits(loaded_model, torch.zeros((1, 1, 16, 16, 16), dtype=torch.float32), patch_mean, patch_std)
    assert version == "0.3.0"
    assert metrics["patch_shape"] == [16, 16, 16]
    assert logits.shape == (1, 2)
