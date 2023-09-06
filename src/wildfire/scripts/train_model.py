"""Train a small PyTorch model for wildfire ignition risk."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_NAMES = ["temp_c", "humidity_pct", "wind_kph", "drought_index"]


class WildfireMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _synthetic_probability(temp_c: float, humidity_pct: float, wind_kph: float, drought_index: float) -> float:
    signal = (
        0.10 * (temp_c - 25.0)
        - 0.06 * (humidity_pct - 30.0)
        + 0.09 * (wind_kph - 20.0)
        + 2.2 * (drought_index - 0.5)
        - 1.6
    )
    return float(torch.sigmoid(torch.tensor(signal)).item())


def generate_synthetic_dataset(samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    features: list[list[float]] = []
    targets: list[float] = []

    for _ in range(samples):
        temp_c = rng.uniform(10.0, 45.0)
        humidity_pct = rng.uniform(5.0, 95.0)
        wind_kph = rng.uniform(0.0, 80.0)
        drought_index = rng.uniform(0.0, 1.0)

        base_prob = _synthetic_probability(temp_c, humidity_pct, wind_kph, drought_index)
        noisy_prob = min(0.99, max(0.01, base_prob + rng.uniform(-0.06, 0.06)))
        target = 1.0 if rng.random() < noisy_prob else 0.0

        features.append([temp_c, humidity_pct, wind_kph, drought_index])
        targets.append(target)

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    return x, y


def load_csv_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[float]] = []
    labels: list[float] = []

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(FEATURE_NAMES + ["target"])
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must contain headers: {sorted(required)}")

        for row in reader:
            rows.append([float(row[name]) for name in FEATURE_NAMES])
            labels.append(float(row["target"]))

    if not rows:
        raise ValueError("CSV dataset is empty.")

    x = torch.tensor(rows, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return x, y


def split_dataset(x: torch.Tensor, y: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=generator)
    split = int(x.shape[0] * 0.8)
    train_idx, val_idx = perm[:split], perm[split:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


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

    model = WildfireMLP()
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
    parser = argparse.ArgumentParser(description="Train wildfire PyTorch model.")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=Path("src/wildfire/model/wildfire_model.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-version", type=str, default="0.1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_csv is not None:
        x, y = load_csv_dataset(args.input_csv)
    else:
        x, y = generate_synthetic_dataset(samples=args.samples, seed=args.seed)

    x_train, y_train, x_val, y_val = split_dataset(x, y, seed=args.seed)
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
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": feature_mean.squeeze(0).tolist(),
            "feature_std": feature_std.squeeze(0).tolist(),
            "model_version": args.model_version,
            "feature_names": FEATURE_NAMES,
            "val_accuracy": val_accuracy,
        },
        args.output_path,
    )

    print(f"Saved model to: {args.output_path}")
    print(f"Validation accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()
