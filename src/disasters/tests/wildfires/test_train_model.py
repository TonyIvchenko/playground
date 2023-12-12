from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.disasters.scripts.wildfires.train_model import (
    auc_from_scores,
    load_raw_dataset,
    prepare_training_dataframe,
    split_dataset,
    train_model,
)


def test_load_raw_dataset_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_raw_dataset(tmp_path / "missing.csv")


def test_load_raw_dataset_missing_columns_raises(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"temp_c": 30.0}]).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_raw_dataset(path)


def test_prepare_training_dataframe_and_split_shapes():
    raw = pd.DataFrame(
        [
            {
                "temp_c": 30.0,
                "humidity_pct": 20.0,
                "wind_kph": 15.0,
                "ffmc": 90.0,
                "dmc": 100.0,
                "drought_code": 300.0,
                "isi": 10.0,
                "target": 1.2,
            },
            {
                "temp_c": 25.0,
                "humidity_pct": 55.0,
                "wind_kph": 8.0,
                "ffmc": 70.0,
                "dmc": 35.0,
                "drought_code": 120.0,
                "isi": 2.0,
                "target": -0.5,
            },
            {
                "temp_c": None,
                "humidity_pct": 35.0,
                "wind_kph": 9.0,
                "ffmc": 75.0,
                "dmc": 45.0,
                "drought_code": 150.0,
                "isi": 3.0,
                "target": 0.0,
            },
        ]
    )

    prepared = prepare_training_dataframe(raw)
    assert len(prepared) == 2
    assert prepared["target"].min() >= 0.0
    assert prepared["target"].max() <= 1.0

    # Duplicate rows to guarantee non-empty val split.
    repeated = pd.concat([prepared] * 10, ignore_index=True)
    x_train, y_train, x_val, y_val = split_dataset(repeated, seed=7)

    assert x_train.shape[1] == 7
    assert y_train.shape[1] == 1
    assert x_val.shape[1] == 7
    assert y_val.shape[1] == 1


def test_auc_and_train_model_runs():
    y_true = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32)
    y_score = torch.tensor([[0.1], [0.2], [0.8], [0.9]], dtype=torch.float32)
    auc = auc_from_scores(y_true, y_score)
    assert 0.99 <= auc <= 1.0

    x_train = torch.tensor(
        [
            [30.0, 20.0, 15.0, 90.0, 100.0, 300.0, 10.0],
            [25.0, 55.0, 8.0, 70.0, 35.0, 120.0, 2.0],
            [32.0, 18.0, 20.0, 92.0, 150.0, 450.0, 14.0],
            [24.0, 60.0, 7.0, 68.0, 30.0, 100.0, 1.5],
        ],
        dtype=torch.float32,
    )
    y_train = torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32)
    x_val = x_train.clone()
    y_val = y_train.clone()

    model, mean, std, acc, bal_acc, val_auc = train_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=42,
    )

    assert mean.shape[1] == 7
    assert std.shape[1] == 7
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= bal_acc <= 1.0
    assert 0.0 <= val_auc <= 1.0
    with torch.no_grad():
        logits = model((x_val - mean) / std)
    assert logits.shape == (4, 1)
