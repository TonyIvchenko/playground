#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CTSCAN_CHANNEL_CLASS_IDS = [5, 6, 3, 4, 1, 2, 7]


def run_command(command: list[str]) -> None:
    print("Running " + " ".join(shlex.quote(part) for part in command), flush=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_ctscan_case(path: Path, channel_index: int) -> None:
    image = np.full((6, 64, 64), 0.2, dtype=np.float32)
    mask = np.zeros((6, 64, 64), dtype=np.uint8)
    mask_multi = np.zeros((len(CTSCAN_CHANNEL_CLASS_IDS), 6, 64, 64), dtype=np.uint8)
    yy, xx = np.ogrid[:64, :64]
    blob = (yy - 32) ** 2 + (xx - 32) ** 2 <= 8**2
    class_id = CTSCAN_CHANNEL_CLASS_IDS[channel_index]
    mask[2:5, blob] = np.uint8(class_id)
    mask_multi[channel_index, 2:5, blob] = np.uint8(1)
    image[2:5, blob] = 0.8
    np.savez_compressed(
        path,
        image=image,
        mask=mask,
        mask_multi=mask_multi,
        spacing=np.asarray([1.5, 1.0, 1.0], dtype=np.float32),
    )


def run_ctscan_smoke(workspace: Path) -> dict[str, object]:
    dataset_dir = workspace / "dataset"
    cases_dir = dataset_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    train_case = cases_dir / "train_case.npz"
    val_case = cases_dir / "val_case.npz"
    _write_ctscan_case(train_case, channel_index=2)
    _write_ctscan_case(val_case, channel_index=3)

    write_csv(
        dataset_dir / "train.csv",
        ["case_id", "source", "path"],
        [{"case_id": "train_case", "source": "fixture", "path": str(train_case)}],
    )
    write_csv(
        dataset_dir / "val.csv",
        ["case_id", "source", "path"],
        [{"case_id": "val_case", "source": "fixture", "path": str(val_case)}],
    )

    manifest = {
        "dataset_name": "fixture",
        "version": "0.0.1",
        "task_type": "multilabel_segmentation",
        "total_spatial_voxels": 6 * 64 * 64 * 2,
        "classes": {
            "0": "background",
            "1": "emphysema",
            "2": "fibrotic_pattern",
            "3": "ground_glass",
            "4": "consolidation",
            "5": "nodule",
            "6": "mass_or_tumor",
            "7": "pleural_effusion",
        },
        "class_channels": [
            {"channel_index": 0, "class_id": 5, "name": "nodule"},
            {"channel_index": 1, "class_id": 6, "name": "mass_or_tumor"},
            {"channel_index": 2, "class_id": 3, "name": "ground_glass"},
            {"channel_index": 3, "class_id": 4, "name": "consolidation"},
            {"channel_index": 4, "class_id": 1, "name": "emphysema"},
            {"channel_index": 5, "class_id": 2, "name": "fibrotic_pattern"},
            {"channel_index": 6, "class_id": 7, "name": "pleural_effusion"},
        ],
        "class_voxels": {
            "0": 10000,
            "1": 100,
            "2": 100,
            "3": 500,
            "4": 500,
            "5": 100,
            "6": 50,
            "7": 50,
        },
    }
    write_json(dataset_dir / "manifest.json", manifest)

    output_path = workspace / "model" / "unet.pt"
    metrics_path = workspace / "model" / "unet.metrics.json"
    run_command(
        [
            sys.executable,
            str(ROOT / "src/ctscan/scripts/segmentation/train_unet.py"),
            "--dataset-dir",
            str(dataset_dir),
            "--output-path",
            str(output_path),
            "--metrics-path",
            str(metrics_path),
            "--model-version",
            "ml-smoke",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "1e-3",
            "--weight-decay",
            "1e-4",
            "--num-workers",
            "0",
            "--negative-stride",
            "2",
            "--base-channels",
            "8",
            "--image-size",
            "64",
            "--device",
            "cpu",
            "--max-train-steps",
            "1",
            "--max-val-steps",
            "1",
        ]
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = {
        "job": "ctscan",
        "dataset_dir": str(dataset_dir),
        "output_path": str(output_path),
        "metrics_path": str(metrics_path),
        "train_slices": metrics.get("train_slices"),
        "val_slices": metrics.get("val_slices"),
    }
    write_json(workspace / "summary.json", summary)
    return summary


def run_disasters_smoke(workspace: Path) -> dict[str, object]:
    wildfires_input = workspace / "wildfires" / "raw" / "wildfires_training_merged.csv"
    wildfires_processed = (
        workspace / "wildfires" / "processed" / "wildfires_training.csv"
    )
    wildfires_model = workspace / "wildfires" / "models" / "wildfires.pt"
    wildfires_rows = [
        {
            "temp_c": 30.0,
            "humidity_pct": 20.0,
            "wind_kph": 15.0,
            "ffmc": 90.0,
            "dmc": 100.0,
            "drought_code": 300.0,
            "isi": 10.0,
            "target": 1.0,
        },
        {
            "temp_c": 25.0,
            "humidity_pct": 55.0,
            "wind_kph": 8.0,
            "ffmc": 70.0,
            "dmc": 35.0,
            "drought_code": 120.0,
            "isi": 2.0,
            "target": 0.0,
        },
        {
            "temp_c": 32.0,
            "humidity_pct": 18.0,
            "wind_kph": 20.0,
            "ffmc": 92.0,
            "dmc": 150.0,
            "drought_code": 450.0,
            "isi": 14.0,
            "target": 1.0,
        },
        {
            "temp_c": 24.0,
            "humidity_pct": 60.0,
            "wind_kph": 7.0,
            "ffmc": 68.0,
            "dmc": 30.0,
            "drought_code": 100.0,
            "isi": 1.5,
            "target": 0.0,
        },
        {
            "temp_c": 29.0,
            "humidity_pct": 28.0,
            "wind_kph": 13.0,
            "ffmc": 88.0,
            "dmc": 90.0,
            "drought_code": 250.0,
            "isi": 8.0,
            "target": 1.0,
        },
        {
            "temp_c": 22.0,
            "humidity_pct": 65.0,
            "wind_kph": 5.0,
            "ffmc": 62.0,
            "dmc": 25.0,
            "drought_code": 80.0,
            "isi": 1.0,
            "target": 0.0,
        },
    ]
    write_csv(
        wildfires_input,
        [
            "temp_c",
            "humidity_pct",
            "wind_kph",
            "ffmc",
            "dmc",
            "drought_code",
            "isi",
            "target",
        ],
        wildfires_rows,
    )
    run_command(
        [
            sys.executable,
            str(ROOT / "src/disasters/scripts/wildfires/train_model.py"),
            "--input-csv",
            str(wildfires_input),
            "--processed-csv",
            str(wildfires_processed),
            "--output-path",
            str(wildfires_model),
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--learning-rate",
            "1e-3",
            "--weight-decay",
            "0.0",
            "--model-version",
            "ml-smoke",
        ]
    )

    hurricane_input = workspace / "huricaines" / "raw" / "huricaines_tracks_merged.csv"
    hurricane_processed = (
        workspace / "huricaines" / "processed" / "huricaines_training.csv"
    )
    hurricane_model = workspace / "huricaines" / "models" / "huricaines.pt"
    base_time = datetime(2000, 8, 1, 0, 0, 0)
    hurricane_rows: list[dict[str, object]] = []
    vmax_offsets = [0.0, 6.0, 12.0, 20.0, 35.0]
    pressure_offsets = [0.0, -5.0, -9.0, -14.0, -22.0]
    for storm_index in range(6):
        storm_id = f"AL{storm_index + 1:02d}2000"
        for step in range(5):
            hurricane_rows.append(
                {
                    "storm_id": storm_id,
                    "iso_time": (base_time + timedelta(hours=6 * step)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "lat": 20.0 + storm_index + (0.3 * step),
                    "lon": -60.0 - storm_index - (0.2 * step),
                    "vmax_kt": 45.0 + (storm_index * 2.0) + vmax_offsets[step],
                    "min_pressure_mb": 1005.0 - storm_index + pressure_offsets[step],
                    "source": "merged",
                }
            )
    write_csv(
        hurricane_input,
        ["storm_id", "iso_time", "lat", "lon", "vmax_kt", "min_pressure_mb", "source"],
        hurricane_rows,
    )
    run_command(
        [
            sys.executable,
            str(ROOT / "src/disasters/scripts/huricaines/train_model.py"),
            "--input-csv",
            str(hurricane_input),
            "--processed-csv",
            str(hurricane_processed),
            "--output-path",
            str(hurricane_model),
            "--max-rows",
            "64",
            "--max-samples",
            "64",
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--learning-rate",
            "1e-3",
            "--weight-decay",
            "0.0",
            "--model-version",
            "ml-smoke",
        ]
    )

    summary = {
        "job": "disasters",
        "wildfires_processed_csv": str(wildfires_processed),
        "wildfires_output_path": str(wildfires_model),
        "wildfires_rows": len(wildfires_rows),
        "huricaines_processed_csv": str(hurricane_processed),
        "huricaines_output_path": str(hurricane_model),
        "huricaines_raw_rows": len(hurricane_rows),
    }
    write_json(workspace / "summary.json", summary)
    return summary


def run_voiceforge_smoke(workspace: Path) -> dict[str, object]:
    raw_dir = workspace / "raw"
    output_dir = workspace / "processed"

    libri_dir = raw_dir / "libritts" / "LibriTTS" / "train-clean-100" / "19" / "198"
    libri_dir.mkdir(parents=True, exist_ok=True)
    for utterance_id, text in (
        ("19_198_000001", "Hello world"),
        ("19_198_000002", "Testing VoiceForge"),
    ):
        (libri_dir / f"{utterance_id}.wav").write_bytes(b"wav")
        (libri_dir / f"{utterance_id}.normalized.txt").write_text(
            text + "\n", encoding="utf-8"
        )

    vctk_txt_dir = raw_dir / "vctk" / "VCTK-Corpus-0.92" / "txt" / "p225"
    vctk_wav_dir = (
        raw_dir / "vctk" / "VCTK-Corpus-0.92" / "wav48_silence_trimmed" / "p225"
    )
    vctk_txt_dir.mkdir(parents=True, exist_ok=True)
    vctk_wav_dir.mkdir(parents=True, exist_ok=True)
    for utterance_id, text in (
        ("p225_001", "One two three"),
        ("p225_002", "Small smoke run"),
    ):
        (vctk_txt_dir / f"{utterance_id}.txt").write_text(text, encoding="utf-8")
        (vctk_wav_dir / f"{utterance_id}_mic1.flac").write_bytes(b"flac")

    run_command(
        [
            sys.executable,
            str(ROOT / "src/voiceforge/scripts/prepare_dataset.py"),
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
            "--max-per-speaker",
            "2",
            "--eval-items-per-speaker",
            "1",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    summary["job"] = "voiceforge"
    write_json(workspace / "summary.json", summary)
    return summary


JOB_RUNNERS: dict[str, Callable[[Path], dict[str, object]]] = {
    "ctscan": run_ctscan_smoke,
    "disasters": run_disasters_smoke,
    "voiceforge": run_voiceforge_smoke,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run capped ML smoke jobs for heavier Playground services."
    )
    parser.add_argument("--job", choices=["all", *sorted(JOB_RUNNERS)], default="all")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Optional directory for smoke-job inputs and outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    job_names = sorted(JOB_RUNNERS) if args.job == "all" else [args.job]

    if args.workspace is None:
        with tempfile.TemporaryDirectory(prefix="playground-ml-smoke-") as temp_dir:
            workspace_root = Path(temp_dir)
            summaries = []
            for job_name in job_names:
                summaries.append(JOB_RUNNERS[job_name](workspace_root / job_name))
            print(json.dumps({"jobs": summaries}, indent=2))
        return 0

    workspace_root = args.workspace.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for job_name in job_names:
        summaries.append(JOB_RUNNERS[job_name](workspace_root / job_name))
    write_json(workspace_root / "summary.json", {"jobs": summaries})
    print(json.dumps({"jobs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
