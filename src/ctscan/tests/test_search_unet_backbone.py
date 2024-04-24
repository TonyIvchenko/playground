from __future__ import annotations

import json
from pathlib import Path
import sys

from src.ctscan.scripts.segmentation import search_unet_backbone as sweep


def test_result_row_from_metrics_extracts_final_metrics():
    row = sweep.result_row_from_metrics(
        slug="trial001",
        metrics={
            "history": [
                {"val_loss": 0.4, "val_mean_iou_fg": 0.2, "val_mean_dice_fg": 0.3},
                {"val_loss": 0.3, "val_mean_iou_fg": 0.4, "val_mean_dice_fg": 0.5},
            ],
            "test": {"mean_iou_fg": 0.6, "mean_dice_fg": 0.7},
        },
        architecture="fpn",
        encoder="efficientnet-b1",
        loss_name="lovasz_ce",
        optimizer_name="adamw",
        image_size=320,
        batch_size=6,
        learning_rate=2e-4,
        weight_decay=1e-4,
        scheduler_name="none",
        sampler_name="rare_fg",
        augmentation_name="none",
        gradient_accumulation_steps=1,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        ce_label_smoothing=0.0,
        fpn_decoder_dropout=0.2,
        fpn_decoder_merge_policy="add",
    )

    assert row["trial"] == "trial001"
    assert row["val_loss"] == 0.3
    assert row["val_mean_iou_fg"] == 0.4
    assert row["val_mean_dice_fg"] == 0.5
    assert row["test_mean_iou_fg"] == 0.6
    assert row["test_mean_dice_fg"] == 0.7


def test_write_leaderboard_writes_json_and_csv(tmp_path: Path):
    rows = [
        {"trial": "a", "val_mean_dice_fg": 0.5, "val_loss": 0.2},
        {"trial": "b", "val_mean_dice_fg": 0.4, "val_loss": 0.3},
    ]

    sweep.write_leaderboard(tmp_path, rows)

    leaderboard_json = json.loads((tmp_path / "leaderboard.json").read_text(encoding="utf-8"))
    leaderboard_csv = (tmp_path / "leaderboard.csv").read_text(encoding="utf-8")
    best_trial_json = json.loads((tmp_path / "best_trial.json").read_text(encoding="utf-8"))

    assert leaderboard_json[0]["trial"] == "a"
    assert best_trial_json["trial"] == "a"
    assert leaderboard_csv.startswith("trial,val_mean_dice_fg,val_loss")
    assert "a" in leaderboard_csv
    assert "b" in leaderboard_csv


def test_sort_rows_respects_metric_direction():
    rows = [
        {"trial": "better_loss", "val_loss": 0.1, "val_mean_dice_fg": 0.4},
        {"trial": "worse_loss", "val_loss": 0.2, "val_mean_dice_fg": 0.6},
    ]

    by_loss = sweep.sort_rows(rows, "val_loss")
    by_dice = sweep.sort_rows(rows, "val_mean_dice_fg")

    assert by_loss[0]["trial"] == "better_loss"
    assert by_dice[0]["trial"] == "worse_loss"


def test_sort_rows_pushes_missing_metrics_to_bottom():
    rows = [
        {"trial": "failed", "error": "oom"},
        {"trial": "good_loss", "val_loss": 0.1, "val_mean_dice_fg": 0.4},
        {"trial": "good_dice", "val_loss": 0.3, "val_mean_dice_fg": 0.7},
    ]

    by_loss = sweep.sort_rows(rows, "val_loss")
    by_dice = sweep.sort_rows(rows, "val_mean_dice_fg")

    assert by_loss[-1]["trial"] == "failed"
    assert by_dice[-1]["trial"] == "failed"


def test_build_trial_slug_includes_non_default_knobs():
    slug = sweep.build_trial_slug(
        index=7,
        architecture="fpn",
        encoder="efficientnet-b1",
        loss_name="tversky_ce",
        optimizer_name="adamw",
        image_size=320,
        batch_size=6,
        learning_rate=2e-4,
        weight_decay=1e-4,
        scheduler_name="onecycle",
        sampler_name="rare_fg",
        augmentation_name="light",
        gradient_accumulation_steps=2,
        tversky_alpha=0.2,
        tversky_beta=0.8,
        ce_label_smoothing=0.05,
        fpn_decoder_dropout=0.1,
        fpn_decoder_merge_policy="cat",
    )

    assert slug.startswith("007_fpn_efficientnet-b1_tversky_ce_adamw_img320_bs6_lr0.0002_wd0.0001")
    assert "sch-onecycle" in slug
    assert "smp-rare_fg" in slug
    assert "aug-light" in slug
    assert "acc2" in slug
    assert "tv0.2-0.8" in slug
    assert "ls0.05" in slug
    assert "drop0.1" in slug
    assert "merge-cat" in slug


def test_main_reuses_existing_metrics_when_skip_existing(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "search"
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = "001_fpn_resnet18_dice_ce_adamw_img256_bs4_lr0.0003_wd0"
    (output_dir / f"{slug}.metrics.json").write_text(
        json.dumps(
            {
                "history": [{"val_loss": 0.25, "val_mean_iou_fg": 0.45, "val_mean_dice_fg": 0.55}],
                "test": {"mean_iou_fg": 0.5, "mean_dice_fg": 0.6},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search_unet_backbone.py",
            "--slice-dir",
            str(tmp_path / "slice_dataset"),
            "--output-dir",
            str(output_dir),
            "--architectures",
            "fpn",
            "--encoders",
            "resnet18",
            "--losses",
            "dice_ce",
            "--optimizers",
            "adamw",
            "--image-sizes",
            "256",
            "--batch-sizes",
            "4",
            "--learning-rates",
            "0.0003",
            "--weight-decays",
            "0",
            "--skip-existing",
        ],
    )
    monkeypatch.setattr(sweep, "train", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("train should not run")))

    assert sweep.main() == 0

    leaderboard = json.loads((output_dir / "leaderboard.json").read_text(encoding="utf-8"))
    assert leaderboard[0]["trial"] == slug
    assert leaderboard[0]["val_mean_dice_fg"] == 0.55


def test_parse_args_accepts_top_k(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--top-k", "3"])

    args = sweep.parse_args()

    assert args.top_k == 3
