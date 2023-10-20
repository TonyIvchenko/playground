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
    required = {"temp_c", "humidity_pct", "wind_kph", "drought_code", "target"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Raw dataset is missing expected columns: {missing}")
    return df


def prepare_training_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    drought = pd.to_numeric(raw_df["drought_code"], errors="coerce")
    drought_min = drought.min()
    drought_max = drought.max()
    denom = max(drought_max - drought_min, 1e-6)
    drought_index = (drought - drought_min) / denom

    training_df = pd.DataFrame(
        {
            "temp_c": pd.to_numeric(raw_df["temp_c"], errors="coerce"),
            "humidity_pct": pd.to_numeric(raw_df["humidity_pct"], errors="coerce"),
            "wind_kph": pd.to_numeric(raw_df["wind_kph"], errors="coerce"),
            "drought_index": drought_index.astype(float).clip(0.0, 1.0),
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


def train_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[WildfireMLP, torch.Tensor, torch.Tensor, float]:
    torch.manual_seed(seed)

    feature_mean = x_train.mean(dim=0, keepdim=True)
    feature_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    x_train_norm = (x_train - feature_mean) / feature_std
    x_val_norm = (x_val - feature_mean) / feature_std

    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
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

    return model, feature_mean, feature_std, val_accuracy


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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-version", type=str, default="0.3.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = load_raw_dataset(args.input_csv)
    training_df = prepare_training_dataframe(raw_df)

    args.processed_csv.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(args.processed_csv, index=False)

    x_train, y_train, x_val, y_val = split_dataset(training_df, seed=args.seed)
    model, feature_mean, feature_std, val_accuracy = train_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
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
        dataset_rows=int(len(training_df)),
    )

    print(f"Saved processed data to: {args.processed_csv}")
    print(f"Saved model to: {args.output_path}")
    print(f"Validation accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()
