"""Run a ranked 1-epoch architecture/backbone sweep on CT PNG slices."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

try:
    from .train_unet_backbone import TrainConfig, train
except ImportError:  # pragma: no cover - script execution path
    from train_unet_backbone import TrainConfig, train


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SLICE_DIR = CTSCAN_ROOT / "data" / "legacy_compatible_png"
DEFAULT_OUTPUT_DIR = CTSCAN_ROOT / "model" / "backbone_search"


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def result_row_from_metrics(
    slug: str,
    metrics: dict[str, Any],
    architecture: str,
    encoder: str,
    loss_name: str,
    optimizer_name: str,
    image_size: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    scheduler_name: str,
    sampler_name: str,
    augmentation_name: str,
    gradient_accumulation_steps: int,
    tversky_alpha: float,
    tversky_beta: float,
    ce_label_smoothing: float,
    fpn_decoder_dropout: float,
    fpn_decoder_merge_policy: str,
) -> dict[str, Any]:
    history = metrics.get("history", [])
    final_row = history[-1] if history else {}
    return {
        "trial": slug,
        "architecture": architecture,
        "encoder": encoder,
        "loss": loss_name,
        "optimizer": optimizer_name,
        "image_size": image_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": scheduler_name,
        "sampler": sampler_name,
        "augmentation": augmentation_name,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "tversky_alpha": tversky_alpha,
        "tversky_beta": tversky_beta,
        "ce_label_smoothing": ce_label_smoothing,
        "fpn_decoder_dropout": fpn_decoder_dropout,
        "fpn_decoder_merge_policy": fpn_decoder_merge_policy,
        "val_loss": float(final_row.get("val_loss", 0.0)),
        "val_mean_iou_fg": float(final_row.get("val_mean_iou_fg", 0.0)),
        "val_mean_dice_fg": float(final_row.get("val_mean_dice_fg", 0.0)),
        "test_mean_iou_fg": float(metrics.get("test", {}).get("mean_iou_fg", 0.0)),
        "test_mean_dice_fg": float(metrics.get("test", {}).get("mean_dice_fg", 0.0)),
    }


def write_leaderboard(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    leaderboard_json = output_dir / "leaderboard.json"
    leaderboard_csv = output_dir / "leaderboard.csv"
    leaderboard_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        leaderboard_csv.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with leaderboard_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search 2D CT backbone models with 1-epoch trials.")
    parser.add_argument("--slice-dir", type=Path, default=DEFAULT_SLICE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--architectures", type=str, default="unet,unetplusplus,fpn")
    parser.add_argument("--encoders", type=str, default="resnet18,resnet34,efficientnet-b0,mobilenet_v2")
    parser.add_argument("--losses", type=str, default="dice_ce,ce,focal")
    parser.add_argument("--optimizers", type=str, default="adamw")
    parser.add_argument("--image-sizes", type=str, default="256,384")
    parser.add_argument("--batch-sizes", type=str, default="4,8")
    parser.add_argument("--learning-rates", type=str, default="0.0003,0.001")
    parser.add_argument("--weight-decays", type=str, default="0.0,0.0001")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--class-weight-mode", type=str, default="none")
    parser.add_argument("--scheduler", type=str, default="none")
    parser.add_argument("--sampler", type=str, default="none")
    parser.add_argument("--augmentation", type=str, default="none")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--ce-label-smoothing", type=float, default=0.0)
    parser.add_argument("--fpn-decoder-dropout", type=float, default=0.2)
    parser.add_argument("--fpn-decoder-merge-policy", type=str, default="add")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--selection-metric", type=str, default="val_mean_dice_fg")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--sort-metric", type=str, default="val_mean_dice_fg")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    architectures = parse_list(args.architectures)
    encoders = parse_list(args.encoders)
    losses = parse_list(args.losses)
    optimizers = parse_list(args.optimizers)
    image_sizes = parse_int_list(args.image_sizes)
    batch_sizes = parse_int_list(args.batch_sizes)
    learning_rates = parse_float_list(args.learning_rates)
    weight_decays = parse_float_list(args.weight_decays)
    encoder_weights = None if str(args.encoder_weights).strip().lower() in {"", "none"} else str(args.encoder_weights)

    trials = list(
        itertools.product(
            architectures,
            encoders,
            losses,
            optimizers,
            image_sizes,
            batch_sizes,
            learning_rates,
            weight_decays,
        )
    )
    if args.max_trials > 0:
        trials = trials[: args.max_trials]

    results: list[dict[str, Any]] = []
    total = len(trials)
    for index, (architecture, encoder, loss_name, optimizer_name, image_size, batch_size, learning_rate, weight_decay) in enumerate(trials, start=1):
        slug = (
            f"{index:03d}_{architecture}_{encoder}_{loss_name}_{optimizer_name}"
            f"_img{image_size}_bs{batch_size}_lr{learning_rate:g}_wd{weight_decay:g}"
        ).replace("/", "-")
        print(f"[{index}/{total}] {slug}")

        config = TrainConfig(
            slice_dir=args.slice_dir.resolve(),
            output_path=(output_dir / f"{slug}.pt").resolve(),
            metrics_path=(output_dir / f"{slug}.metrics.json").resolve(),
            model_version=slug,
            architecture=architecture,
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=max(int(args.classes), 2),
            in_channels=max(int(args.in_channels), 1),
            image_size=max(int(image_size), 64),
            batch_size=max(int(batch_size), 1),
            epochs=max(int(args.epochs), 1),
            learning_rate=float(learning_rate),
            weight_decay=max(float(weight_decay), 0.0),
            optimizer_name=optimizer_name,
            loss_name=loss_name,
            class_weight_mode=str(args.class_weight_mode).strip().lower(),
            scheduler_name=str(args.scheduler).strip().lower(),
            sampler_name=str(args.sampler).strip().lower(),
            augmentation_name=str(args.augmentation).strip().lower(),
            gradient_accumulation_steps=max(int(args.gradient_accumulation_steps), 1),
            tversky_alpha=float(args.tversky_alpha),
            tversky_beta=float(args.tversky_beta),
            ce_label_smoothing=max(float(args.ce_label_smoothing), 0.0),
            fpn_decoder_dropout=max(float(args.fpn_decoder_dropout), 0.0),
            fpn_decoder_merge_policy=str(args.fpn_decoder_merge_policy).strip().lower(),
            selection_metric=str(args.selection_metric).strip().lower(),
            num_workers=max(int(args.num_workers), 0),
            seed=int(args.seed),
            device=str(args.device).strip().lower(),
            max_train_batches=max(int(args.max_train_batches), 0),
            max_val_batches=max(int(args.max_val_batches), 0),
            max_test_batches=max(int(args.max_test_batches), 0),
        )

        try:
            if args.skip_existing and config.metrics_path.exists():
                metrics = json.loads(config.metrics_path.read_text(encoding="utf-8"))
                print("  reused existing metrics")
            else:
                metrics = train(config)
            results.append(
                result_row_from_metrics(
                    slug=slug,
                    metrics=metrics,
                    architecture=architecture,
                    encoder=encoder,
                    loss_name=loss_name,
                    optimizer_name=optimizer_name,
                    image_size=image_size,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    scheduler_name=str(args.scheduler).strip().lower(),
                    sampler_name=str(args.sampler).strip().lower(),
                    augmentation_name=str(args.augmentation).strip().lower(),
                    gradient_accumulation_steps=max(int(args.gradient_accumulation_steps), 1),
                    tversky_alpha=float(args.tversky_alpha),
                    tversky_beta=float(args.tversky_beta),
                    ce_label_smoothing=max(float(args.ce_label_smoothing), 0.0),
                    fpn_decoder_dropout=max(float(args.fpn_decoder_dropout), 0.0),
                    fpn_decoder_merge_policy=str(args.fpn_decoder_merge_policy).strip().lower(),
                )
            )
        except Exception as exc:
            results.append(
                {
                    "trial": slug,
                    "architecture": architecture,
                    "encoder": encoder,
                    "loss": loss_name,
                    "optimizer": optimizer_name,
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "scheduler": str(args.scheduler).strip().lower(),
                    "sampler": str(args.sampler).strip().lower(),
                    "augmentation": str(args.augmentation).strip().lower(),
                    "gradient_accumulation_steps": max(int(args.gradient_accumulation_steps), 1),
                    "tversky_alpha": float(args.tversky_alpha),
                    "tversky_beta": float(args.tversky_beta),
                    "ce_label_smoothing": max(float(args.ce_label_smoothing), 0.0),
                    "fpn_decoder_dropout": max(float(args.fpn_decoder_dropout), 0.0),
                    "fpn_decoder_merge_policy": str(args.fpn_decoder_merge_policy).strip().lower(),
                    "error": str(exc),
                }
            )
            print(f"  failed: {exc}")

        leaderboard = sorted(
            results,
            key=lambda row: float(row.get(args.sort_metric, 0.0)),
            reverse=True,
        )
        write_leaderboard(output_dir, leaderboard)

    print("top_trials:")
    for row in sorted(results, key=lambda item: float(item.get(args.sort_metric, 0.0)), reverse=True)[:5]:
        print(
            f"  {row['trial']} {args.sort_metric}={row.get(args.sort_metric, 0.0):.4f} "
            f"val_loss={row.get('val_loss', 0.0):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
