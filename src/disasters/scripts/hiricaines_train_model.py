"""Train hiricaines rapid-intensification PyTorch model from real tracks."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.disasters.models.hiricaines import save_model_bundle
from src.disasters.models.hiricaines import FEATURE_NAMES, HiricainesMLP, create_model


TARGET_NAME = "target"


def load_raw_dataset(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: `python -m src.disasters.scripts.hiricaines_download_data`"
        )

    usecols = ["storm_id", "iso_time", "lat", "lon", "vmax_kt", "min_pressure_mb", "source"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False, nrows=max_rows)
    return df


def prepare_training_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["storm_id"] = df["storm_id"].astype(str).str.strip()
    df["iso_time"] = pd.to_datetime(df["iso_time"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["vmax_kt"] = pd.to_numeric(df["vmax_kt"], errors="coerce")
    df["min_pressure_mb"] = pd.to_numeric(df["min_pressure_mb"], errors="coerce")

    df = df.dropna(subset=["storm_id", "iso_time", "vmax_kt", "lat", "lon"]).copy()
    df = df[df["vmax_kt"].between(0.0, 200.0)].copy()
    df = df[df["lat"].between(-5.0, 70.0)].copy()
    df = df[df["lon"].between(-120.0, 20.0)].copy()
    df = df.sort_values(["storm_id", "iso_time"]).reset_index(drop=True)
    df["month"] = df["iso_time"].dt.month.astype(float)
    angle = 2.0 * math.pi * df["month"] / 12.0
    df["month_sin"] = angle.map(math.sin)
    df["month_cos"] = angle.map(math.cos)
    df["abs_lat"] = df["lat"].abs()
    df["pressure_deficit"] = 1010.0 - df["min_pressure_mb"]
    df["target_time"] = df["iso_time"] + pd.Timedelta(hours=24)

    merged_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("storm_id", sort=False):
        group = group.sort_values("iso_time").copy()
        group["prev_time"] = group["iso_time"].shift(1)
        group["prev_vmax"] = group["vmax_kt"].shift(1)
        group["prev_pressure"] = group["min_pressure_mb"].shift(1)
        delta_hours = (group["iso_time"] - group["prev_time"]).dt.total_seconds() / 3600.0
        step = (delta_hours / 6.0).where(delta_hours > 0.0)
        group["dvmax_6h"] = (group["vmax_kt"] - group["prev_vmax"]) / step
        group["dpres_6h"] = (group["min_pressure_mb"] - group["prev_pressure"]) / step

        future = group[["iso_time", "vmax_kt"]].rename(
            columns={"iso_time": "future_time", "vmax_kt": "future_vmax_kt"}
        )
        merged_group = pd.merge_asof(
            group.sort_values("target_time"),
            future.sort_values("future_time"),
            left_on="target_time",
            right_on="future_time",
            tolerance=pd.Timedelta(hours=3),
            direction="nearest",
        )
        merged_parts.append(merged_group)

    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.dropna(subset=["future_vmax_kt"]).copy()
    merged["target"] = (merged["future_vmax_kt"] - merged["vmax_kt"] >= 30.0).astype(float)
    pressure_median = float(merged["min_pressure_mb"].median())
    if pd.isna(pressure_median):
        pressure_median = 1000.0
    merged["min_pressure_mb"] = merged["min_pressure_mb"].fillna(pressure_median)
    merged["pressure_deficit"] = merged["pressure_deficit"].fillna(1010.0 - pressure_median)
    merged["dvmax_6h"] = pd.to_numeric(merged["dvmax_6h"], errors="coerce").fillna(0.0).clip(-60.0, 60.0)
    merged["dpres_6h"] = pd.to_numeric(merged["dpres_6h"], errors="coerce").fillna(0.0).clip(-60.0, 60.0)

    training_df = merged[FEATURE_NAMES + [TARGET_NAME]].dropna().reset_index(drop=True)
    if training_df.empty:
        raise ValueError("No usable training rows after preprocessing.")
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
) -> tuple[HiricainesMLP, torch.Tensor, torch.Tensor, float, float, float]:
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
    parser = argparse.ArgumentParser(description="Train hiricaines PyTorch model from real data.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("src/disasters/data/hiricaines/raw/hiricaines_tracks_merged.csv"),
        help="Path to merged canonical tracks CSV produced by download_data.py.",
    )
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=Path("src/disasters/data/hiricaines/processed/hiricaines_training.csv"),
        help="Where to write processed training rows for inspection.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("src/disasters/models/hiricaines.pt"),
        help="Where to write trained model artifact.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional number of top raw rows to read. Default uses the full file.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200000,
        help="Max processed rows used for training (sampled after preprocessing).",
    )
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=9e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to initialize model training.")
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
    raw_df = load_raw_dataset(args.input_csv, max_rows=args.max_rows)
    training_df = prepare_training_dataframe(raw_df)
    if args.max_samples is not None and len(training_df) > args.max_samples:
        training_df = training_df.sample(n=args.max_samples, random_state=args.split_seed).reset_index(drop=True)

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
