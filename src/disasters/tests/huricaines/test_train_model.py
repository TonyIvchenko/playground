from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.disasters.scripts.huricaines.train_model import (
    auc_from_scores,
    load_raw_dataset,
    prepare_training_dataframe,
    split_dataset,
    train_model,
)


def test_load_raw_dataset_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_raw_dataset(tmp_path / "missing.csv")


def test_prepare_training_dataframe_split_and_auc():
    raw = pd.DataFrame(
        [
            {
                "storm_id": "AL012000",
                "iso_time": "2000-08-01 00:00:00",
                "lat": 20.0,
                "lon": -60.0,
                "vmax_kt": 50.0,
                "min_pressure_mb": 1005.0,
                "source": "merged",
            },
            {
                "storm_id": "AL012000",
                "iso_time": "2000-08-01 06:00:00",
                "lat": 20.4,
                "lon": -60.2,
                "vmax_kt": 56.0,
                "min_pressure_mb": 1000.0,
                "source": "merged",
            },
            {
                "storm_id": "AL012000",
                "iso_time": "2000-08-01 12:00:00",
                "lat": 20.7,
                "lon": -60.5,
                "vmax_kt": 62.0,
                "min_pressure_mb": 996.0,
                "source": "merged",
            },
            {
                "storm_id": "AL012000",
                "iso_time": "2000-08-01 18:00:00",
                "lat": 21.0,
                "lon": -60.8,
                "vmax_kt": 70.0,
                "min_pressure_mb": 990.0,
                "source": "merged",
            },
            {
                "storm_id": "AL012000",
                "iso_time": "2000-08-02 00:00:00",
                "lat": 21.3,
                "lon": -61.2,
                "vmax_kt": 95.0,
                "min_pressure_mb": 975.0,
                "source": "merged",
            },
        ]
    )

    prepared = prepare_training_dataframe(raw)
    assert not prepared.empty
    assert "target" in prepared.columns

    repeated = pd.concat([prepared] * 6, ignore_index=True)
    x_train, y_train, x_val, y_val = split_dataset(repeated, seed=11)
    assert x_train.shape[1] == 11
    assert y_train.shape[1] == 1
    assert x_val.shape[1] == 11
    assert y_val.shape[1] == 1

    y_true = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float32)
    y_score = torch.tensor([[0.1], [0.9], [0.2], [0.8]], dtype=torch.float32)
    auc = auc_from_scores(y_true, y_score)
    assert 0.99 <= auc <= 1.0


def test_train_model_runs_small():
    x_train = torch.tensor(
        [
            [60.0, 990.0, 20.0, -60.0, 8.0, 0.87, -0.50, 20.0, 20.0, 5.0, -2.0],
            [40.0, 1005.0, 15.0, -55.0, 8.0, 0.87, -0.50, 15.0, 5.0, 1.0, 0.0],
            [65.0, 985.0, 22.0, -62.0, 8.0, 0.87, -0.50, 22.0, 25.0, 6.0, -3.0],
            [35.0, 1008.0, 12.0, -50.0, 8.0, 0.87, -0.50, 12.0, 2.0, 0.0, 1.0],
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
        seed=41,
    )

    assert mean.shape[1] == 11
    assert std.shape[1] == 11
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= bal_acc <= 1.0
    assert 0.0 <= val_auc <= 1.0
    with torch.no_grad():
        logits = model((x_val - mean) / std)
    assert logits.shape == (4, 1)
