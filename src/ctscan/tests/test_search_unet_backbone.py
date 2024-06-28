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
    leaderboard_md = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    best_trial_json = json.loads((tmp_path / "best_trial.json").read_text(encoding="utf-8"))
    best_trial_md = (tmp_path / "best_trial.md").read_text(encoding="utf-8")

    assert leaderboard_json[0]["trial"] == "a"
    assert best_trial_json["trial"] == "a"
    assert leaderboard_csv.startswith("trial,val_mean_dice_fg,val_loss")
    assert leaderboard_md.startswith("| trial | val_mean_dice_fg |")
    assert "| a | 0.5000 |" in leaderboard_md
    assert best_trial_md.startswith("# a")
    assert "- val dice fg: `0.5000`" in best_trial_md
    assert "a" in leaderboard_csv
    assert "b" in leaderboard_csv


def test_write_run_summary_counts_success_and_failures(tmp_path: Path):
    rows = [
        {"trial": "winner", "val_mean_dice_fg": 0.7},
        {"trial": "failed", "error": "oom"},
    ]

    sweep.write_run_summary(
        tmp_path,
        rows,
        total_trials=3,
        sort_metric="val_mean_dice_fg",
        selection_metric="val_mean_dice_fg",
        top_k=5,
    )

    payload = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    summary_md = (tmp_path / "run_summary.md").read_text(encoding="utf-8")

    assert payload["total_trials"] == 3
    assert payload["completed_trials"] == 1
    assert payload["failed_trials"] == 1
    assert payload["sort_metric"] == "val_mean_dice_fg"
    assert payload["selection_metric"] == "val_mean_dice_fg"
    assert payload["top_k"] == 5
    assert payload["best_trial"] == "winner"
    assert summary_md.startswith("# Sweep Summary")
    assert "- selection metric: `val_mean_dice_fg`" in summary_md
    assert "- best trial: `winner`" in summary_md


def test_write_trial_config_persists_knobs(tmp_path: Path):
    payload = sweep.trial_config_row(
        slug="trial001",
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

    sweep.write_trial_config(tmp_path, "trial001", payload)

    saved = json.loads((tmp_path / "trial001.config.json").read_text(encoding="utf-8"))
    assert saved["trial"] == "trial001"
    assert saved["architecture"] == "fpn"
    assert saved["sampler"] == "rare_fg"


def test_write_trial_plan_persists_trial_list(tmp_path: Path):
    trials = [{"trial": "trial001"}, {"trial": "trial002"}]

    sweep.write_trial_plan(tmp_path, trials)

    saved = json.loads((tmp_path / "trial_plan.json").read_text(encoding="utf-8"))
    saved_md = (tmp_path / "trial_plan.md").read_text(encoding="utf-8")
    assert saved == trials
    assert saved_md.startswith("| trial | architecture |")


def test_output_paths_summary_lists_sweep_artifacts(tmp_path: Path):
    payload = sweep.output_paths_summary(tmp_path)

    assert payload["leaderboard_json"].endswith("leaderboard.json")
    assert payload["best_trial_md"].endswith("best_trial.md")
    assert payload["trial_plan_md"].endswith("trial_plan.md")


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
    run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    run_summary_md = (output_dir / "run_summary.md").read_text(encoding="utf-8")
    trial_config = json.loads((output_dir / f"{slug}.config.json").read_text(encoding="utf-8"))
    assert leaderboard[0]["trial"] == slug
    assert leaderboard[0]["val_mean_dice_fg"] == 0.55
    assert run_summary["completed_trials"] == 1
    assert run_summary["failed_trials"] == 0
    assert run_summary["best_trial"] == slug
    assert run_summary["selection_metric"] == "val_mean_dice_fg"
    assert slug in run_summary_md
    assert trial_config["trial"] == slug
    assert trial_config["architecture"] == "fpn"


def test_parse_args_accepts_top_k(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--top-k", "3"])

    args = sweep.parse_args()

    assert args.top_k == 3


def test_parse_args_accepts_fail_fast(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--fail-fast"])

    args = sweep.parse_args()

    assert args.fail_fast is True


def test_parse_args_accepts_trial_window(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--start-index", "3", "--end-index", "7"])

    args = sweep.parse_args()

    assert args.start_index == 3
    assert args.end_index == 7


def test_parse_args_accepts_show_output_paths(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--show-output-paths"])

    args = sweep.parse_args()

    assert args.show_output_paths is True


def test_parse_args_accepts_list_architectures(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-architectures"])

    args = sweep.parse_args()

    assert args.list_architectures is True


def test_parse_args_accepts_list_losses(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-losses"])

    args = sweep.parse_args()

    assert args.list_losses is True


def test_parse_args_accepts_list_optimizers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-optimizers"])

    args = sweep.parse_args()

    assert args.list_optimizers is True


def test_parse_args_accepts_list_schedulers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-schedulers"])

    args = sweep.parse_args()

    assert args.list_schedulers is True


def test_parse_args_accepts_list_samplers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-samplers"])

    args = sweep.parse_args()

    assert args.list_samplers is True


def test_parse_args_accepts_list_augmentations(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-augmentations"])

    args = sweep.parse_args()

    assert args.list_augmentations is True


def test_parse_args_accepts_list_metrics(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-metrics"])

    args = sweep.parse_args()

    assert args.list_metrics is True


def test_main_dry_run_prints_planned_trials(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search_unet_backbone.py",
            "--slice-dir",
            str(tmp_path / "slice_dataset"),
            "--output-dir",
            str(tmp_path / "search"),
            "--architectures",
            "fpn",
            "--encoders",
            "efficientnet-b1",
            "--losses",
            "lovasz_ce",
            "--optimizers",
            "adamw",
            "--image-sizes",
            "320",
            "--batch-sizes",
            "6",
            "--learning-rates",
            "0.0002",
            "--weight-decays",
            "0.0001",
            "--dry-run",
        ],
    )
    monkeypatch.setattr(sweep, "train", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("train should not run")))

    assert sweep.main() == 0

    output = capsys.readouterr().out
    plan = json.loads((tmp_path / "search" / "trial_plan.json").read_text(encoding="utf-8"))
    plan_md = (tmp_path / "search" / "trial_plan.md").read_text(encoding="utf-8")
    assert "planned_trials:" in output
    assert "001_fpn_efficientnet-b1_lovasz_ce_adamw_img320_bs6_lr0.0002_wd0.0001" in output
    assert plan[0]["trial"] == "001_fpn_efficientnet-b1_lovasz_ce_adamw_img320_bs6_lr0.0002_wd0.0001"
    assert "001_fpn_efficientnet-b1_lovasz_ce_adamw_img320_bs6_lr0.0002_wd0.0001" in plan_md


def test_main_show_output_paths_prints_artifact_locations(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search_unet_backbone.py",
            "--output-dir",
            str(tmp_path / "search"),
            "--show-output-paths",
        ],
    )

    assert sweep.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["leaderboard_json"].endswith("leaderboard.json")
    assert payload["trial_plan_md"].endswith("trial_plan.md")


def test_main_list_architectures_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-architectures"])

    assert sweep.main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "fpn" in output
    assert "unet" in output


def test_main_list_losses_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-losses"])

    assert sweep.main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "dice_ce" in output
    assert "lovasz_ce" in output


def test_main_list_optimizers_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-optimizers"])

    assert sweep.main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "adam" in output
    assert "adamw" in output


def test_main_list_schedulers_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-schedulers"])

    assert sweep.main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "none" in output
    assert "onecycle" in output


def test_main_list_samplers_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-samplers"])

    assert sweep.main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "none" in output
    assert "rare_fg" in output


def test_main_list_augmentations_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-augmentations"])

    assert sweep.main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "none" in output
    assert "light" in output


def test_main_list_metrics_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["search_unet_backbone.py", "--list-metrics"])

    assert sweep.main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "val_mean_dice_fg" in output
    assert "val_loss" in output


def test_main_rejects_unknown_sort_metric(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search_unet_backbone.py",
            "--slice-dir",
            str(tmp_path / "slice_dataset"),
            "--output-dir",
            str(tmp_path / "search"),
            "--sort-metric",
            "bogus_metric",
            "--dry-run",
        ],
    )

    try:
        sweep.main()
    except ValueError as exc:
        assert "unsupported --sort-metric" in str(exc)
        assert "val_mean_dice_fg" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected invalid sort metric to fail")


def test_main_rejects_unknown_selection_metric(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search_unet_backbone.py",
            "--slice-dir",
            str(tmp_path / "slice_dataset"),
            "--output-dir",
            str(tmp_path / "search"),
            "--selection-metric",
            "bogus_metric",
            "--dry-run",
        ],
    )

    try:
        sweep.main()
    except ValueError as exc:
        assert "unsupported --selection-metric" in str(exc)
        assert "val_mean_dice_fg" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected invalid selection metric to fail")


def test_main_rejects_unknown_architecture_family(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search_unet_backbone.py",
            "--slice-dir",
            str(tmp_path / "slice_dataset"),
            "--output-dir",
            str(tmp_path / "search"),
            "--architectures",
            "bogus_net",
            "--dry-run",
        ],
    )

    try:
        sweep.main()
    except ValueError as exc:
        assert "unsupported --architectures" in str(exc)
        assert "fpn" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected invalid architecture family to fail")


def test_main_rejects_unknown_sampler(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search_unet_backbone.py",
            "--slice-dir",
            str(tmp_path / "slice_dataset"),
            "--output-dir",
            str(tmp_path / "search"),
            "--sampler",
            "bogus_sampler",
            "--dry-run",
        ],
    )

    try:
        sweep.main()
    except ValueError as exc:
        assert "unsupported --sampler" in str(exc)
        assert "rare_fg" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected invalid sampler to fail")


def test_main_dry_run_respects_trial_window(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search_unet_backbone.py",
            "--slice-dir",
            str(tmp_path / "slice_dataset"),
            "--output-dir",
            str(tmp_path / "search"),
            "--architectures",
            "fpn,unet",
            "--encoders",
            "efficientnet-b1",
            "--losses",
            "lovasz_ce",
            "--optimizers",
            "adamw",
            "--image-sizes",
            "320",
            "--batch-sizes",
            "6",
            "--learning-rates",
            "0.0002",
            "--weight-decays",
            "0.0001",
            "--start-index",
            "2",
            "--end-index",
            "2",
            "--dry-run",
        ],
    )
    monkeypatch.setattr(sweep, "train", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("train should not run")))

    assert sweep.main() == 0

    output = capsys.readouterr().out
    assert "001_unet_efficientnet-b1_lovasz_ce_adamw_img320_bs6_lr0.0002_wd0.0001" in output
    assert "fpn_efficientnet-b1" not in output


def test_main_fail_fast_stops_after_first_error(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "search"
    train_calls: list[str] = []

    def fail_train(config):
        train_calls.append(config.model_version)
        raise RuntimeError("boom")

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
            "fpn,unet",
            "--encoders",
            "efficientnet-b1",
            "--losses",
            "lovasz_ce",
            "--optimizers",
            "adamw",
            "--image-sizes",
            "320",
            "--batch-sizes",
            "6",
            "--learning-rates",
            "0.0002",
            "--weight-decays",
            "0.0001",
            "--fail-fast",
        ],
    )
    monkeypatch.setattr(sweep, "train", fail_train)

    assert sweep.main() == 1

    leaderboard = json.loads((output_dir / "leaderboard.json").read_text(encoding="utf-8"))
    run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert len(train_calls) == 1
    assert len(leaderboard) == 1
    assert leaderboard[0]["error"] == "boom"
    assert run_summary["failed_trials"] == 1
