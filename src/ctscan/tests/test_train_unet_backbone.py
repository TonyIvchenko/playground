from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn

from src.ctscan.scripts.segmentation.train_unet_backbone import (
    available_encoder_names,
    best_history_row,
    build_model,
    build_scheduler,
    config_output_path,
    config_to_dict,
    compute_class_weights,
    compute_sample_weights,
    LovaszCELoss,
    MulticlassFocalLoss,
    TverskyCELoss,
    PRESET_DEFAULTS,
    SlicePairDataset,
    TrainConfig,
    load_split_rows,
    metric_direction,
    main,
    metrics_summary_path,
    output_paths_summary,
    parse_args,
    split_source,
    split_summary,
    train,
    write_config_snapshot,
    write_metrics_summary,
)


def _write_pair(images_dir: Path, masks_dir: Path, name: str, size: int = 64) -> None:
    Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L").save(images_dir / f"{name}.png")
    Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L").save(masks_dir / f"{name}.png")


def test_load_split_rows_falls_back_to_legacy_split_json(tmp_path: Path):
    root = tmp_path / "legacy_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "case001")
    _write_pair(images_dir, masks_dir, "case002")
    (root / "splits.json").write_text(
        json.dumps({"train": ["case001"], "val": ["case002"], "test": []}),
        encoding="utf-8",
    )

    train_rows = load_split_rows(root, "train")
    val_rows = load_split_rows(root, "val")

    assert train_rows == [{"image": "images/case001.png", "mask": "masks/case001.png"}]
    assert val_rows == [{"image": "images/case002.png", "mask": "masks/case002.png"}]


def test_split_source_reports_csv_json_and_missing(tmp_path: Path):
    root = tmp_path / "legacy_png"
    (root / "splits").mkdir(parents=True, exist_ok=True)
    (root / "splits" / "train.csv").write_text("image,mask\n", encoding="utf-8")
    (root / "splits.json").write_text(json.dumps({"val": []}), encoding="utf-8")

    train_source = split_source(root, "train")
    val_source = split_source(root, "val")
    test_source = split_source(root, "test")

    assert train_source["kind"] == "csv"
    assert train_source["path"].endswith("train.csv")
    assert val_source["kind"] == "json"
    assert val_source["path"].endswith("splits.json")
    assert test_source["kind"] == "json"


def test_slice_pair_dataset_reads_legacy_rows(tmp_path: Path):
    root = tmp_path / "legacy_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "case001", size=80)
    rows = [{"image": "images/case001.png", "mask": "masks/case001.png"}]

    dataset = SlicePairDataset(root, rows, image_size=64)
    image, mask = dataset[0]

    assert tuple(image.shape) == (1, 64, 64)
    assert tuple(mask.shape) == (64, 64)


def test_slice_pair_dataset_light_augmentation_preserves_shape(tmp_path: Path):
    root = tmp_path / "legacy_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "case001", size=80)
    rows = [{"image": "images/case001.png", "mask": "masks/case001.png"}]

    dataset = SlicePairDataset(root, rows, image_size=64, augmentation_name="light")
    image, mask = dataset[0]

    assert tuple(image.shape) == (1, 64, 64)
    assert tuple(mask.shape) == (64, 64)
    assert image.min().item() >= 0.0
    assert image.max().item() <= 1.0


def test_metric_direction_prefers_higher_for_dice_and_lower_for_loss():
    assert metric_direction("val_mean_dice_fg") == 1
    assert metric_direction("val_loss") == -1


def test_best_history_row_uses_metric_direction():
    history = [
        {"epoch": 1.0, "val_mean_dice_fg": 0.4, "val_loss": 0.3},
        {"epoch": 2.0, "val_mean_dice_fg": 0.7, "val_loss": 0.5},
        {"epoch": 3.0, "val_mean_dice_fg": 0.6, "val_loss": 0.2},
    ]

    assert best_history_row(history, "val_mean_dice_fg")["epoch"] == 2.0
    assert best_history_row(history, "val_loss")["epoch"] == 3.0


def test_multiclass_focal_loss_runs_on_logits_and_integer_targets():
    criterion = MulticlassFocalLoss()
    logits = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))

    loss = criterion(logits, target)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_tversky_ce_loss_runs_on_logits_and_integer_targets():
    criterion = TverskyCELoss()
    logits = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))

    loss = criterion(logits, target)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_lovasz_ce_loss_runs_on_logits_and_integer_targets():
    criterion = LovaszCELoss()
    logits = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))

    loss = criterion(logits, target)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_build_scheduler_returns_onecycle_when_requested():
    model = nn.Conv2d(1, 2, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    scheduler = build_scheduler("onecycle", optimizer, learning_rate=3e-4, steps_per_epoch=4, epochs=2)

    assert scheduler is not None


def test_compute_class_weights_scales_down_background(tmp_path: Path):
    root = tmp_path / "legacy_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "case001")
    sparse = np.zeros((64, 64), dtype=np.uint8)
    sparse[0:4, 0:4] = 1
    sparse[4:6, 4:6] = 2
    sparse[6:10, 6:10] = 3
    Image.fromarray(sparse, mode="L").save(masks_dir / "case001.png")

    weights = compute_class_weights(root, [{"image": "images/case001.png", "mask": "masks/case001.png"}], 4, "inverse_sqrt")

    assert weights is not None
    assert tuple(weights.shape) == (4,)
    assert weights[0].item() < weights[1].item()
    assert weights[0].item() < weights[2].item()
    assert weights[0].item() < weights[3].item()
    assert abs(weights.mean().item() - 1.0) < 1e-6


def test_compute_sample_weights_boosts_rare_foreground_slices(tmp_path: Path):
    root = tmp_path / "legacy_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _write_pair(images_dir, masks_dir, "bg")
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L").save(masks_dir / "bg.png")

    _write_pair(images_dir, masks_dir, "common")
    common = np.zeros((64, 64), dtype=np.uint8)
    common[0:8, 0:8] = 3
    Image.fromarray(common, mode="L").save(masks_dir / "common.png")

    _write_pair(images_dir, masks_dir, "rare")
    rare = np.zeros((64, 64), dtype=np.uint8)
    rare[0:8, 0:8] = 1
    Image.fromarray(rare, mode="L").save(masks_dir / "rare.png")

    rows = [
        {"image": "images/bg.png", "mask": "masks/bg.png"},
        {"image": "images/common.png", "mask": "masks/common.png"},
        {"image": "images/rare.png", "mask": "masks/rare.png"},
    ]
    weights = compute_sample_weights(root, rows, 4, "rare_fg")

    assert weights is not None
    assert weights[0] == 1.0
    assert weights[1] > weights[0]
    assert weights[2] > weights[0]


def test_legacy_png_best_preset_sets_measured_winner(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--preset", "legacy_png_best"])

    config = parse_args()

    assert config.architecture == PRESET_DEFAULTS["legacy_png_best"]["architecture"]
    assert config.encoder_name == PRESET_DEFAULTS["legacy_png_best"]["encoder_name"]
    assert config.classes == PRESET_DEFAULTS["legacy_png_best"]["classes"]
    assert config.image_size == PRESET_DEFAULTS["legacy_png_best"]["image_size"]
    assert config.batch_size == PRESET_DEFAULTS["legacy_png_best"]["batch_size"]
    assert config.class_weight_mode == PRESET_DEFAULTS["legacy_png_best"]["class_weight_mode"]
    assert config.scheduler_name == PRESET_DEFAULTS["legacy_png_best"]["scheduler"]
    assert config.sampler_name == PRESET_DEFAULTS["legacy_png_best"]["sampler"]
    assert config.augmentation_name == PRESET_DEFAULTS["legacy_png_best"]["augmentation"]
    assert config.gradient_accumulation_steps == PRESET_DEFAULTS["legacy_png_best"]["gradient_accumulation_steps"]
    assert config.tversky_alpha == PRESET_DEFAULTS["legacy_png_best"]["tversky_alpha"]
    assert config.tversky_beta == PRESET_DEFAULTS["legacy_png_best"]["tversky_beta"]
    assert config.ce_label_smoothing == PRESET_DEFAULTS["legacy_png_best"]["ce_label_smoothing"]
    assert config.fpn_decoder_dropout == PRESET_DEFAULTS["legacy_png_best"]["fpn_decoder_dropout"]
    assert config.fpn_decoder_merge_policy == PRESET_DEFAULTS["legacy_png_best"]["fpn_decoder_merge_policy"]


def test_parse_args_accepts_dry_run(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--dry-run"])

    config = parse_args()

    assert config.dry_run is True


def test_parse_args_accepts_list_presets(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-presets"])

    config = parse_args()

    assert config.list_presets is True


def test_parse_args_accepts_list_architectures(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-architectures"])

    config = parse_args()

    assert config.list_architectures is True


def test_parse_args_accepts_list_encoders(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-encoders"])

    config = parse_args()

    assert config.list_encoders is True


def test_parse_args_accepts_list_losses(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-losses"])

    config = parse_args()

    assert config.list_losses is True


def test_parse_args_accepts_list_optimizers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-optimizers"])

    config = parse_args()

    assert config.list_optimizers is True


def test_parse_args_accepts_list_class_weight_modes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-class-weight-modes"])

    config = parse_args()

    assert config.list_class_weight_modes is True


def test_parse_args_accepts_list_schedulers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-schedulers"])

    config = parse_args()

    assert config.list_schedulers is True


def test_parse_args_accepts_list_samplers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-samplers"])

    config = parse_args()

    assert config.list_samplers is True


def test_parse_args_accepts_list_augmentations(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-augmentations"])

    config = parse_args()

    assert config.list_augmentations is True


def test_parse_args_accepts_list_metrics(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-metrics"])

    config = parse_args()

    assert config.list_metrics is True


def test_parse_args_accepts_inspect_splits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--inspect-splits"])

    config = parse_args()

    assert config.inspect_splits is True


def test_parse_args_accepts_show_output_paths(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--show-output-paths"])

    config = parse_args()

    assert config.show_output_paths is True


def test_config_to_dict_stringifies_paths():
    config = TrainConfig(
        slice_dir=Path("slice_data"),
        output_path=Path("model.pt"),
        metrics_path=Path("metrics.json"),
        model_version="test",
        architecture="fpn",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet",
        classes=4,
        in_channels=1,
        image_size=320,
        batch_size=6,
        epochs=1,
        learning_rate=2e-4,
        weight_decay=1e-4,
        optimizer_name="adamw",
        loss_name="lovasz_ce",
        class_weight_mode="none",
        scheduler_name="none",
        sampler_name="rare_fg",
        augmentation_name="none",
        gradient_accumulation_steps=1,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        ce_label_smoothing=0.0,
        fpn_decoder_dropout=0.2,
        fpn_decoder_merge_policy="add",
        selection_metric="val_mean_dice_fg",
        num_workers=0,
        seed=17,
        device="cpu",
        max_train_batches=0,
        max_val_batches=0,
        max_test_batches=0,
        dry_run=True,
    )

    payload = config_to_dict(config)

    assert payload["slice_dir"] == "slice_data"
    assert payload["output_path"] == "model.pt"
    assert payload["metrics_path"] == "metrics.json"
    assert payload["dry_run"] is True


def test_split_summary_counts_rows_from_legacy_split_json(tmp_path: Path):
    root = tmp_path / "legacy_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    _write_pair(images_dir, masks_dir, "case001")
    _write_pair(images_dir, masks_dir, "case002")
    _write_pair(images_dir, masks_dir, "case003")
    (root / "splits.json").write_text(
        json.dumps({"train": ["case001"], "val": ["case002"], "test": ["case003"]}),
        encoding="utf-8",
    )

    summary = split_summary(root)

    assert summary["train_rows"] == 1
    assert summary["val_rows"] == 1
    assert summary["test_rows"] == 1
    assert summary["total_rows"] == 3
    assert summary["train_source"] == "json"
    assert summary["val_source"] == "json"
    assert summary["test_source"] == "json"
    assert summary["train_source_path"].endswith("splits.json")
    assert summary["val_source_path"].endswith("splits.json")
    assert summary["test_source_path"].endswith("splits.json")


def test_write_config_snapshot_uses_metrics_stem(tmp_path: Path):
    config = TrainConfig(
        slice_dir=tmp_path / "slice_data",
        output_path=tmp_path / "model.pt",
        metrics_path=tmp_path / "model.metrics.json",
        model_version="test",
        architecture="fpn",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet",
        classes=4,
        in_channels=1,
        image_size=320,
        batch_size=6,
        epochs=1,
        learning_rate=2e-4,
        weight_decay=1e-4,
        optimizer_name="adamw",
        loss_name="lovasz_ce",
        class_weight_mode="none",
        scheduler_name="none",
        sampler_name="rare_fg",
        augmentation_name="none",
        gradient_accumulation_steps=1,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        ce_label_smoothing=0.0,
        fpn_decoder_dropout=0.2,
        fpn_decoder_merge_policy="add",
        selection_metric="val_mean_dice_fg",
        num_workers=0,
        seed=17,
        device="cpu",
        max_train_batches=0,
        max_val_batches=0,
        max_test_batches=0,
        dry_run=False,
    )

    path = write_config_snapshot(config)

    assert path == config_output_path(config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics_path"].endswith("model.metrics.json")


def test_output_paths_summary_includes_all_artifacts(tmp_path: Path):
    config = TrainConfig(
        slice_dir=tmp_path / "slice_data",
        output_path=tmp_path / "model.pt",
        metrics_path=tmp_path / "model.metrics.json",
        model_version="test",
        architecture="fpn",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet",
        classes=4,
        in_channels=1,
        image_size=320,
        batch_size=6,
        epochs=1,
        learning_rate=2e-4,
        weight_decay=1e-4,
        optimizer_name="adamw",
        loss_name="lovasz_ce",
        class_weight_mode="none",
        scheduler_name="none",
        sampler_name="rare_fg",
        augmentation_name="none",
        gradient_accumulation_steps=1,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        ce_label_smoothing=0.0,
        fpn_decoder_dropout=0.2,
        fpn_decoder_merge_policy="add",
        selection_metric="val_mean_dice_fg",
        num_workers=0,
        seed=17,
        device="cpu",
        max_train_batches=0,
        max_val_batches=0,
        max_test_batches=0,
    )

    payload = output_paths_summary(config)

    assert payload["checkpoint_path"].endswith("model.pt")
    assert payload["metrics_path"].endswith("model.metrics.json")
    assert payload["config_path"].endswith("model.metrics.config.json")
    assert payload["summary_path"].endswith("model.metrics.md")


def test_write_metrics_summary_creates_markdown_report(tmp_path: Path):
    config = TrainConfig(
        slice_dir=tmp_path / "slice_data",
        output_path=tmp_path / "model.pt",
        metrics_path=tmp_path / "model.metrics.json",
        model_version="test-model",
        architecture="fpn",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet",
        classes=4,
        in_channels=1,
        image_size=320,
        batch_size=6,
        epochs=1,
        learning_rate=2e-4,
        weight_decay=1e-4,
        optimizer_name="adamw",
        loss_name="lovasz_ce",
        class_weight_mode="none",
        scheduler_name="none",
        sampler_name="rare_fg",
        augmentation_name="none",
        gradient_accumulation_steps=1,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        ce_label_smoothing=0.0,
        fpn_decoder_dropout=0.2,
        fpn_decoder_merge_policy="add",
        selection_metric="val_mean_dice_fg",
        num_workers=0,
        seed=17,
        device="cpu",
        max_train_batches=0,
        max_val_batches=0,
        max_test_batches=0,
    )
    metrics = {
        "best_epoch": 2,
        "best_score": 0.7,
        "best_row": {"val_mean_dice_fg": 0.7, "val_mean_iou_fg": 0.6, "val_loss": 0.15},
        "history": [{"val_mean_dice_fg": 0.6, "val_mean_iou_fg": 0.5, "val_loss": 0.2}],
        "test": {"mean_dice_fg": 0.65, "mean_iou_fg": 0.55},
    }

    path = write_metrics_summary(config, metrics)

    assert path == metrics_summary_path(config)
    content = path.read_text(encoding="utf-8")
    assert content.startswith("# test-model")
    assert "- architecture: `fpn`" in content
    assert "- best val dice fg: `0.7000`" in content
    assert "- test dice fg: `0.6500`" in content


def test_main_dry_run_prints_resolved_config(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_unet_backbone.py",
            "--preset",
            "legacy_png_best",
            "--dry-run",
        ],
    )

    assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["architecture"] == PRESET_DEFAULTS["legacy_png_best"]["architecture"]
    assert payload["encoder_name"] == PRESET_DEFAULTS["legacy_png_best"]["encoder_name"]
    assert payload["dry_run"] is True


def test_main_list_presets_prints_available_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-presets"])

    assert main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "default" in output
    assert "legacy_png_best" in output


def test_main_list_architectures_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-architectures"])

    assert main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "fpn" in output
    assert "unet" in output


def test_available_encoder_names_contains_common_backbones():
    names = available_encoder_names()

    assert "resnet18" in names
    assert "efficientnet-b1" in names


def test_main_list_encoders_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-encoders"])

    assert main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "resnet18" in output
    assert "efficientnet-b1" in output


def test_main_list_losses_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-losses"])

    assert main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "dice_ce" in output
    assert "lovasz_ce" in output


def test_main_list_optimizers_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-optimizers"])

    assert main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "adam" in output
    assert "adamw" in output


def test_main_list_class_weight_modes_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-class-weight-modes"])

    assert main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "none" in output
    assert "inverse_sqrt" in output


def test_main_list_schedulers_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-schedulers"])

    assert main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "none" in output
    assert "onecycle" in output


def test_main_list_samplers_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-samplers"])

    assert main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "none" in output
    assert "rare_fg" in output


def test_main_list_augmentations_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-augmentations"])

    assert main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "none" in output
    assert "light" in output


def test_main_list_metrics_prints_supported_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_unet_backbone.py", "--list-metrics"])

    assert main() == 0

    output = capsys.readouterr().out.splitlines()
    assert "val_mean_dice_fg" in output
    assert "val_loss" in output


def test_main_rejects_unknown_selection_metric(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_unet_backbone.py",
            "--selection-metric",
            "bogus_metric",
        ],
    )

    try:
        main()
    except ValueError as exc:
        assert "unsupported --selection-metric" in str(exc)
        assert "val_mean_dice_fg" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected invalid selection metric to fail")


def test_main_rejects_unknown_architecture(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_unet_backbone.py",
            "--architecture",
            "bogus_net",
            "--dry-run",
        ],
    )

    try:
        main()
    except ValueError as exc:
        assert "unsupported --architecture" in str(exc)
        assert "fpn" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected invalid architecture to fail")


def test_main_rejects_unknown_scheduler(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_unet_backbone.py",
            "--scheduler",
            "bogus_scheduler",
            "--dry-run",
        ],
    )

    try:
        main()
    except ValueError as exc:
        assert "unsupported --scheduler" in str(exc)
        assert "onecycle" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected invalid scheduler to fail")


def test_main_rejects_unknown_class_weight_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_unet_backbone.py",
            "--class-weight-mode",
            "bogus_mode",
            "--dry-run",
        ],
    )

    try:
        main()
    except ValueError as exc:
        assert "unsupported --class-weight-mode" in str(exc)
        assert "inverse_sqrt" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected invalid class weight mode to fail")


def test_main_inspect_splits_prints_dataset_counts(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path / "legacy_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    _write_pair(images_dir, masks_dir, "case001")
    (root / "splits.json").write_text(
        json.dumps({"train": ["case001"], "val": [], "test": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_unet_backbone.py",
            "--slice-dir",
            str(root),
            "--inspect-splits",
        ],
    )

    assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["train_rows"] == 1
    assert payload["total_rows"] == 1


def test_main_show_output_paths_prints_artifact_locations(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_unet_backbone.py",
            "--metrics-path",
            "model/example.metrics.json",
            "--output-path",
            "model/example.pt",
            "--show-output-paths",
        ],
    )

    assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint_path"].endswith("model/example.pt")
    assert payload["config_path"].endswith("model/example.metrics.config.json")


def test_build_model_passes_fpn_decoder_settings(monkeypatch):
    captured = {}

    class DummyModel(nn.Module):
        pass

    def fake_fpn(**kwargs):
        captured.update(kwargs)
        return DummyModel()

    monkeypatch.setattr("src.ctscan.scripts.segmentation.train_unet_backbone.smp.FPN", fake_fpn)

    config = TrainConfig(
        slice_dir=Path("."),
        output_path=Path("model.pt"),
        metrics_path=Path("metrics.json"),
        model_version="test",
        architecture="fpn",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        classes=4,
        in_channels=1,
        image_size=320,
        batch_size=6,
        epochs=1,
        learning_rate=2e-4,
        weight_decay=1e-4,
        optimizer_name="adamw",
        loss_name="lovasz_ce",
        class_weight_mode="none",
        scheduler_name="none",
        sampler_name="rare_fg",
        augmentation_name="none",
        gradient_accumulation_steps=1,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        ce_label_smoothing=0.0,
        fpn_decoder_dropout=0.1,
        fpn_decoder_merge_policy="cat",
        selection_metric="val_mean_dice_fg",
        num_workers=0,
        seed=17,
        device="cpu",
        max_train_batches=0,
        max_val_batches=0,
        max_test_batches=0,
    )

    model = build_model(config)

    assert isinstance(model, DummyModel)
    assert captured["decoder_dropout"] == 0.1
    assert captured["decoder_merge_policy"] == "cat"


def test_train_evaluates_test_metrics_with_best_checkpoint_state(tmp_path: Path, monkeypatch):
    root = tmp_path / "slice_png"
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    _write_pair(images_dir, masks_dir, "case001")
    rows = [{"image": "images/case001.png", "mask": "masks/case001.png"}]

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.0]))

        def forward(self, image):  # pragma: no cover - not used by patched run_epoch
            batch, _, height, width = image.shape
            return torch.zeros((batch, 2, height, width), dtype=image.dtype, device=image.device)

    call_state = {"train_epoch": 0, "val_epoch": 0}

    def fake_run_epoch(model, loader, optimizer, scheduler, gradient_accumulation_steps, criterion, device, classes, max_batches, progress_desc):
        if optimizer is not None:
            call_state["train_epoch"] += 1
            model.weight.data.fill_(float(call_state["train_epoch"]))
            return {"loss": 1.0, "mean_iou": 0.0, "mean_dice": 0.0, "mean_iou_fg": 0.0, "mean_dice_fg": 0.0}
        if progress_desc.startswith("Epoch"):
            call_state["val_epoch"] += 1
            if call_state["val_epoch"] == 1:
                return {"loss": 0.5, "mean_iou": 0.0, "mean_dice": 0.0, "mean_iou_fg": 0.8, "mean_dice_fg": 0.8}
            return {"loss": 0.6, "mean_iou": 0.0, "mean_dice": 0.0, "mean_iou_fg": 0.1, "mean_dice_fg": 0.1}
        assert float(model.weight.item()) == 1.0
        return {"loss": 0.4, "mean_iou": 0.0, "mean_dice": 0.0, "mean_iou_fg": 0.7, "mean_dice_fg": 0.7}

    monkeypatch.setattr("src.ctscan.scripts.segmentation.train_unet_backbone.build_model", lambda config: DummyModel())
    monkeypatch.setattr("src.ctscan.scripts.segmentation.train_unet_backbone.build_optimizer", lambda *args, **kwargs: object())
    monkeypatch.setattr("src.ctscan.scripts.segmentation.train_unet_backbone.build_loss", lambda *args, **kwargs: nn.CrossEntropyLoss())
    monkeypatch.setattr("src.ctscan.scripts.segmentation.train_unet_backbone.load_split_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr("src.ctscan.scripts.segmentation.train_unet_backbone.run_epoch", fake_run_epoch)

    config = TrainConfig(
        slice_dir=root,
        output_path=tmp_path / "model.pt",
        metrics_path=tmp_path / "metrics.json",
        model_version="test",
        architecture="fpn",
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        classes=2,
        in_channels=1,
        image_size=64,
        batch_size=1,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_name="adamw",
        loss_name="dice_ce",
        class_weight_mode="none",
        scheduler_name="none",
        sampler_name="none",
        augmentation_name="none",
        gradient_accumulation_steps=1,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        ce_label_smoothing=0.0,
        fpn_decoder_dropout=0.2,
        fpn_decoder_merge_policy="add",
        selection_metric="val_mean_dice_fg",
        num_workers=0,
        seed=17,
        device="cpu",
        max_train_batches=0,
        max_val_batches=0,
        max_test_batches=0,
    )

    metrics = train(config)

    assert metrics["best_epoch"] == 1
    saved_config = config_output_path(config)
    saved_summary = metrics_summary_path(config)
    assert saved_config.exists()
    assert saved_summary.exists()
    saved_payload = json.loads(saved_config.read_text(encoding="utf-8"))
    assert saved_payload["model_version"] == "test"
    assert "# test" in saved_summary.read_text(encoding="utf-8")
    assert metrics["best_row"]["epoch"] == 1.0
