"""Train wildfire ignition PyTorch model from real wildfire records."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wildfire.model import FEATURE_NAMES, WildfireMLP, create_model, save_model_bundle


TARGET_NAME = "target"


def load_raw_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: `python src/wildfire/scripts/download_data.py`"
        )

    df = pd.read_csv(path)
    required = {"temp_c", "humidity_pct", "wind_kph", "ffmc", "dmc", "drought_code", "isi", "target"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Raw dataset is missing expected columns: {missing}")
    return df


def prepare_training_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    training_df = pd.DataFrame(
        {
            "temp_c": pd.to_numeric(raw_df["temp_c"], errors="coerce"),
            "humidity_pct": pd.to_numeric(raw_df["humidity_pct"], errors="coerce"),
            "wind_kph": pd.to_numeric(raw_df["wind_kph"], errors="coerce"),
            "ffmc": pd.to_numeric(raw_df["ffmc"], errors="coerce"),
            "dmc": pd.to_numeric(raw_df["dmc"], errors="coerce"),
            "drought_code": pd.to_numeric(raw_df["drought_code"], errors="coerce"),
            "isi": pd.to_numeric(raw_df["isi"], errors="coerce"),
            TARGET_NAME: pd.to_numeric(raw_df["target"], errors="coerce").clip(0.0, 1.0),
        }
    )
    training_df = training_df.dropna().reset_index(drop=True)
    return training_df


def split_dataset(
    training_df: pd.DataFrame,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = training_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split = int(len(data) * 0.8)
    train_df = data.iloc[:split]
    val_df = data.iloc[split:]

    x_train = torch.tensor(train_df[FEATURE_NAMES].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[TARGET_NAME].values, dtype=torch.float32).unsqueeze(1)
    x_val = torch.tensor(val_df[FEATURE_NAMES].values, dtype=torch.float32)
    y_val = torch.tensor(val_df[TARGET_NAME].values, dtype=torch.float32).unsqueeze(1)
    return x_train, y_train, x_val, y_val


def auc_from_scores(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    target = y_true.reshape(-1).detach().cpu()
    score = y_score.reshape(-1).detach().cpu()

    pos_mask = target >= 0.5
    n_pos = int(pos_mask.sum().item())
    n_neg = int((~pos_mask).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    score_series = pd.Series(score.numpy())
    ranks = score_series.rank(method="average")
    sum_ranks_pos = float(ranks[pos_mask.numpy()].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def train_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[WildfireMLP, torch.Tensor, torch.Tensor, float, float, float]:
    torch.manual_seed(seed)

    feature_mean = x_train.mean(dim=0, keepdim=True)
    feature_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    x_train_norm = (x_train - feature_mean) / feature_std
    x_val_norm = (x_val - feature_mean) / feature_std

    model = create_model()
    pos_count = max(float(y_train.sum().item()), 1.0)
    neg_count = max(float((1.0 - y_train).sum().item()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(x_train_norm, y_train), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(x_val_norm)
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs >= 0.5).float()
        val_accuracy = float((val_preds == y_val).float().mean().item())
        tp = float(((val_preds == 1.0) & (y_val == 1.0)).sum().item())
        tn = float(((val_preds == 0.0) & (y_val == 0.0)).sum().item())
        fp = float(((val_preds == 1.0) & (y_val == 0.0)).sum().item())
        fn = float(((val_preds == 0.0) & (y_val == 1.0)).sum().item())
        tpr = tp / max(tp + fn, 1.0)
        tnr = tn / max(tn + fp, 1.0)
        val_balanced_accuracy = 0.5 * (tpr + tnr)
        val_auc = float(auc_from_scores(y_true=y_val, y_score=val_probs))

    return model, feature_mean, feature_std, val_accuracy, val_balanced_accuracy, val_auc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wildfire PyTorch model from real data.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("src/wildfire/data/raw/wildfire_training_merged.csv"),
        help="Path to merged canonical wildfire CSV produced by download_data.py.",
    )
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=Path("src/wildfire/data/processed/wildfire_training.csv"),
        help="Where to write processed training rows for inspection.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("src/wildfire/model/wildfire_model.pt"),
        help="Where to store trained model artifact.",
    )
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=41, help="Random seed used to initialize model training.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used to shuffle rows before the train/validation split.",
    )
    parser.add_argument("--model-version", type=str, default="0.5.2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = load_raw_dataset(args.input_csv)
    training_df = prepare_training_dataframe(raw_df)

    args.processed_csv.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(args.processed_csv, index=False)

    x_train, y_train, x_val, y_val = split_dataset(training_df, seed=args.split_seed)
    model, feature_mean, feature_std, val_accuracy, val_balanced_accuracy, val_auc = train_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model_bundle(
        path=args.output_path,
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        model_version=args.model_version,
        val_accuracy=val_accuracy,
        val_balanced_accuracy=val_balanced_accuracy,
        val_auc=val_auc,
        dataset_rows=int(len(training_df)),
    )

    print(f"Saved processed data to: {args.processed_csv}")
    print(f"Saved model to: {args.output_path}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation balanced accuracy: {val_balanced_accuracy:.4f}")
    print(f"Validation AUROC: {val_auc:.4f}")


if __name__ == "__main__":
    main()
