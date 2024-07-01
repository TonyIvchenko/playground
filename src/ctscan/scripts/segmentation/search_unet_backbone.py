"""Run a ranked 1-epoch architecture/backbone sweep on CT PNG slices."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

try:
    from .train_unet_backbone import (
        SUPPORTED_ARCHITECTURES,
        SUPPORTED_AUGMENTATIONS,
        SUPPORTED_LOSSES,
        SUPPORTED_OPTIMIZERS,
        SUPPORTED_SAMPLERS,
        SUPPORTED_SCHEDULERS,
        SUPPORTED_TRAINER_METRICS,
        TrainConfig,
        metric_direction,
        train,
        validate_choice,
        validate_metric_name,
    )
except ImportError:  # pragma: no cover - script execution path
    from train_unet_backbone import (
        SUPPORTED_ARCHITECTURES,
        SUPPORTED_AUGMENTATIONS,
        SUPPORTED_LOSSES,
        SUPPORTED_OPTIMIZERS,
        SUPPORTED_SAMPLERS,
        SUPPORTED_SCHEDULERS,
        SUPPORTED_TRAINER_METRICS,
        TrainConfig,
        metric_direction,
        train,
        validate_choice,
        validate_metric_name,
    )


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SLICE_DIR = CTSCAN_ROOT / "data" / "legacy_compatible_png"
DEFAULT_OUTPUT_DIR = CTSCAN_ROOT / "model" / "backbone_search"
LEADERBOARD_FIELD_ORDER = [
    "trial",
    "architecture",
    "encoder",
    "loss",
    "optimizer",
    "image_size",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "scheduler",
    "sampler",
    "augmentation",
    "gradient_accumulation_steps",
    "tversky_alpha",
    "tversky_beta",
    "ce_label_smoothing",
    "fpn_decoder_dropout",
    "fpn_decoder_merge_policy",
    "val_mean_dice_fg",
    "val_mean_iou_fg",
    "val_loss",
    "test_mean_dice_fg",
    "test_mean_iou_fg",
    "error",
]
SUPPORTED_SWEEP_METRICS = [
    "val_mean_dice_fg",
    "val_mean_iou_fg",
    "val_loss",
    "test_mean_dice_fg",
    "test_mean_iou_fg",
]


def parse_list(value: str) -> list[str]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return list(dict.fromkeys(items))


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    return list(dict.fromkeys(items))


def parse_float_list(value: str) -> list[float]:
    items = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    return list(dict.fromkeys(items))


def require_nonempty(values: list[Any], argument_name: str) -> list[Any]:
    if values:
        return values
    raise ValueError(f"{argument_name} must include at least one value")


def build_trial_slug(
    index: int,
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
) -> str:
    parts = [
        f"{index:03d}",
        architecture,
        encoder,
        loss_name,
        optimizer_name,
        f"img{image_size}",
        f"bs{batch_size}",
        f"lr{learning_rate:g}",
        f"wd{weight_decay:g}",
    ]
    if scheduler_name not in {"", "none"}:
        parts.append(f"sch-{scheduler_name}")
    if sampler_name not in {"", "none"}:
        parts.append(f"smp-{sampler_name}")
    if augmentation_name not in {"", "none"}:
        parts.append(f"aug-{augmentation_name}")
    if gradient_accumulation_steps > 1:
        parts.append(f"acc{gradient_accumulation_steps}")
    if loss_name == "tversky_ce":
        parts.append(f"tv{tversky_alpha:g}-{tversky_beta:g}")
    if ce_label_smoothing > 0.0:
        parts.append(f"ls{ce_label_smoothing:g}")
    if architecture == "fpn":
        if fpn_decoder_dropout != 0.2:
            parts.append(f"drop{fpn_decoder_dropout:g}")
        if fpn_decoder_merge_policy != "add":
            parts.append(f"merge-{fpn_decoder_merge_policy}")
    return "_".join(parts).replace("/", "-")


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


def trial_config_row(
    slug: str,
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
    }


def write_leaderboard(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    leaderboard_json = output_dir / "leaderboard.json"
    leaderboard_csv = output_dir / "leaderboard.csv"
    leaderboard_md = output_dir / "leaderboard.md"
    best_trial_json = output_dir / "best_trial.json"
    best_trial_md = output_dir / "best_trial.md"
    leaderboard_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        leaderboard_csv.write_text("", encoding="utf-8")
        leaderboard_md.write_text("", encoding="utf-8")
        best_trial_json.write_text("{}", encoding="utf-8")
        best_trial_md.write_text("", encoding="utf-8")
        return
    discovered = {key for row in rows for key in row.keys()}
    fieldnames = [name for name in LEADERBOARD_FIELD_ORDER if name in discovered]
    fieldnames.extend(sorted(discovered - set(fieldnames)))
    with leaderboard_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    markdown_lines = [
        "| trial | val_mean_dice_fg | val_mean_iou_fg | val_loss | test_mean_dice_fg | error |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("trial", "")),
                    f"{float(row.get('val_mean_dice_fg', 0.0)):.4f}" if "val_mean_dice_fg" in row else "",
                    f"{float(row.get('val_mean_iou_fg', 0.0)):.4f}" if "val_mean_iou_fg" in row else "",
                    f"{float(row.get('val_loss', 0.0)):.4f}" if "val_loss" in row else "",
                    f"{float(row.get('test_mean_dice_fg', 0.0)):.4f}" if "test_mean_dice_fg" in row else "",
                    str(row.get("error", "")),
                ]
            )
            + " |"
        )
    leaderboard_md.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    best_trial_json.write_text(json.dumps(rows[0], indent=2), encoding="utf-8")
    best = rows[0]
    best_trial_md.write_text(
        "\n".join(
            [
                f"# {best.get('trial', '')}",
                "",
                f"- architecture: `{best.get('architecture', '')}`",
                f"- encoder: `{best.get('encoder', '')}`",
                f"- loss: `{best.get('loss', '')}`",
                f"- optimizer: `{best.get('optimizer', '')}`",
                f"- val dice fg: `{float(best.get('val_mean_dice_fg', 0.0)):.4f}`" if "val_mean_dice_fg" in best else "- val dice fg: ``",
                f"- val iou fg: `{float(best.get('val_mean_iou_fg', 0.0)):.4f}`" if "val_mean_iou_fg" in best else "- val iou fg: ``",
                f"- val loss: `{float(best.get('val_loss', 0.0)):.4f}`" if "val_loss" in best else "- val loss: ``",
                f"- test dice fg: `{float(best.get('test_mean_dice_fg', 0.0)):.4f}`" if "test_mean_dice_fg" in best else "- test dice fg: ``",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_run_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    total_trials: int,
    sort_metric: str,
    selection_metric: str,
    top_k: int,
) -> None:
    summary = {
        "total_trials": int(total_trials),
        "completed_trials": int(sum(1 for row in rows if "error" not in row)),
        "failed_trials": int(sum(1 for row in rows if "error" in row)),
        "sort_metric": str(sort_metric),
        "selection_metric": str(selection_metric),
        "top_k": int(max(top_k, 0)),
        "best_trial": rows[0]["trial"] if rows else None,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "run_summary.md").write_text(
        "\n".join(
            [
                "# Sweep Summary",
                "",
                f"- total trials: `{summary['total_trials']}`",
                f"- completed trials: `{summary['completed_trials']}`",
                f"- failed trials: `{summary['failed_trials']}`",
                f"- sort metric: `{summary['sort_metric']}`",
                f"- selection metric: `{summary['selection_metric']}`",
                f"- top k: `{summary['top_k']}`",
                f"- best trial: `{summary['best_trial'] or ''}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_trial_config(output_dir: Path, slug: str, payload: dict[str, Any]) -> None:
    (output_dir / f"{slug}.config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_trial_plan(output_dir: Path, trials: list[dict[str, Any]]) -> None:
    (output_dir / "trial_plan.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
    if not trials:
        (output_dir / "trial_plan.md").write_text("", encoding="utf-8")
        return
    lines = [
        "| trial | architecture | encoder | loss | optimizer | image_size | batch_size |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in trials:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("trial", "")),
                    str(row.get("architecture", "")),
                    str(row.get("encoder", "")),
                    str(row.get("loss", "")),
                    str(row.get("optimizer", "")),
                    str(row.get("image_size", "")),
                    str(row.get("batch_size", "")),
                ]
            )
            + " |"
        )
    (output_dir / "trial_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def output_paths_summary(output_dir: Path) -> dict[str, str]:
    return {
        "leaderboard_json": str(output_dir / "leaderboard.json"),
        "leaderboard_csv": str(output_dir / "leaderboard.csv"),
        "leaderboard_md": str(output_dir / "leaderboard.md"),
        "best_trial_json": str(output_dir / "best_trial.json"),
        "best_trial_md": str(output_dir / "best_trial.md"),
        "run_summary_json": str(output_dir / "run_summary.json"),
        "trial_plan_json": str(output_dir / "trial_plan.json"),
        "trial_plan_md": str(output_dir / "trial_plan.md"),
    }


def sort_rows(rows: list[dict[str, Any]], metric_name: str) -> list[dict[str, Any]]:
    direction = metric_direction(metric_name)
    if direction > 0:
        return sorted(
            rows,
            key=lambda row: (1 if metric_name in row else 0, float(row.get(metric_name, float("-inf")))),
            reverse=True,
        )
    return sorted(
        rows,
        key=lambda row: (0 if metric_name in row else 1, float(row.get(metric_name, float("inf")))),
    )


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
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--sort-metric", type=str, default="val_mean_dice_fg")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-architectures", action="store_true")
    parser.add_argument("--list-losses", action="store_true")
    parser.add_argument("--list-optimizers", action="store_true")
    parser.add_argument("--list-schedulers", action="store_true")
    parser.add_argument("--list-samplers", action="store_true")
    parser.add_argument("--list-augmentations", action="store_true")
    parser.add_argument("--show-output-paths", action="store_true")
    parser.add_argument("--list-metrics", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.list_metrics:
        for metric_name in SUPPORTED_SWEEP_METRICS:
            print(metric_name)
        return 0
    if args.list_architectures:
        for architecture_name in SUPPORTED_ARCHITECTURES:
            print(architecture_name)
        return 0
    if args.list_losses:
        for loss_name in SUPPORTED_LOSSES:
            print(loss_name)
        return 0
    if args.list_optimizers:
        for optimizer_name in SUPPORTED_OPTIMIZERS:
            print(optimizer_name)
        return 0
    if args.list_schedulers:
        for scheduler_name in SUPPORTED_SCHEDULERS:
            print(scheduler_name)
        return 0
    if args.list_samplers:
        for sampler_name in SUPPORTED_SAMPLERS:
            print(sampler_name)
        return 0
    if args.list_augmentations:
        for augmentation_name in SUPPORTED_AUGMENTATIONS:
            print(augmentation_name)
        return 0
    if args.show_output_paths:
        print(json.dumps(output_paths_summary(output_dir), indent=2))
        return 0
    args.sort_metric = validate_metric_name(args.sort_metric, SUPPORTED_SWEEP_METRICS, "--sort-metric")
    args.selection_metric = validate_metric_name(
        args.selection_metric,
        SUPPORTED_TRAINER_METRICS,
        "--selection-metric",
    )
    args.scheduler = validate_choice(args.scheduler, SUPPORTED_SCHEDULERS, "--scheduler")
    args.sampler = validate_choice(args.sampler, SUPPORTED_SAMPLERS, "--sampler")
    args.augmentation = validate_choice(args.augmentation, SUPPORTED_AUGMENTATIONS, "--augmentation")

    architectures = require_nonempty(parse_list(args.architectures), "--architectures")
    encoders = require_nonempty(parse_list(args.encoders), "--encoders")
    losses = require_nonempty(parse_list(args.losses), "--losses")
    optimizers = require_nonempty(parse_list(args.optimizers), "--optimizers")
    architectures = [validate_choice(name, SUPPORTED_ARCHITECTURES, "--architectures") for name in architectures]
    losses = [validate_choice(name, SUPPORTED_LOSSES, "--losses") for name in losses]
    optimizers = [validate_choice(name, SUPPORTED_OPTIMIZERS, "--optimizers") for name in optimizers]
    image_sizes = require_nonempty(parse_int_list(args.image_sizes), "--image-sizes")
    batch_sizes = require_nonempty(parse_int_list(args.batch_sizes), "--batch-sizes")
    learning_rates = require_nonempty(parse_float_list(args.learning_rates), "--learning-rates")
    weight_decays = require_nonempty(parse_float_list(args.weight_decays), "--weight-decays")
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
    start_index = max(int(args.start_index), 1)
    end_index = int(args.end_index)
    if end_index > 0:
        trials = trials[start_index - 1 : end_index]
    else:
        trials = trials[start_index - 1 :]
    if args.max_trials > 0:
        trials = trials[: args.max_trials]

    planned_trials = [
        trial_config_row(
            slug=build_trial_slug(
                index=index,
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
            ),
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
        for index, (architecture, encoder, loss_name, optimizer_name, image_size, batch_size, learning_rate, weight_decay) in enumerate(trials, start=1)
    ]
    write_trial_plan(output_dir, planned_trials)

    if args.dry_run:
        print("planned_trials:")
        for row in planned_trials:
            print(row["trial"])
        return 0

    results: list[dict[str, Any]] = []
    total = len(trials)
    for index, (architecture, encoder, loss_name, optimizer_name, image_size, batch_size, learning_rate, weight_decay) in enumerate(trials, start=1):
        slug = build_trial_slug(
            index=index,
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
        print(f"[{index}/{total}] {slug}")
        trial_config = trial_config_row(
            slug=slug,
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
        write_trial_config(output_dir, slug, trial_config)

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
            leaderboard = sort_rows(results, args.sort_metric)
            write_leaderboard(output_dir, leaderboard)
            write_run_summary(
                output_dir,
                leaderboard,
                total_trials=total,
                sort_metric=args.sort_metric,
                selection_metric=args.selection_metric,
                top_k=args.top_k,
            )
            if args.fail_fast:
                return 1

        leaderboard = sort_rows(results, args.sort_metric)
        write_leaderboard(output_dir, leaderboard)
        write_run_summary(
            output_dir,
            leaderboard,
            total_trials=total,
            sort_metric=args.sort_metric,
            selection_metric=args.selection_metric,
            top_k=args.top_k,
        )

    print("top_trials:")
    for row in sort_rows(results, args.sort_metric)[: max(int(args.top_k), 0)]:
        print(
            f"  {row['trial']} {args.sort_metric}={row.get(args.sort_metric, 0.0):.4f} "
            f"val_loss={row.get('val_loss', 0.0):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
