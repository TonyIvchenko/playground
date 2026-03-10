"""Build a composite chest-CT segmentation dataset for U-Net training."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - optional
    ndi = None

CTSCAN_ROOT = Path(__file__).resolve().parents[2]
if str(CTSCAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CTSCAN_ROOT))

DEFAULT_RAW_DIR = CTSCAN_ROOT / "data" / "ctscan" / "raw"
DEFAULT_SAMPLES_DIR = CTSCAN_ROOT / "data" / "ctscan" / "samples"
DEFAULT_OUTPUT_DIR = CTSCAN_ROOT / "data" / "ctscan" / "processed" / "unet_composite"
DEFAULT_LABELED_MANIFEST = DEFAULT_RAW_DIR / "composite_manifest.csv"
CLASS_NAMES = {
    0: "background",
    1: "emphysema",
    2: "fibrotic_pattern",
    3: "ground_glass",
    4: "consolidation",
    5: "nodule",
}


@dataclass
class CaseSample:
    case_id: str
    source: str
    image_hu: np.ndarray
    mask: np.ndarray
    spacing: tuple[float, float, float]
    metadata: dict[str, Any]


@dataclass
class BuildConfig:
    raw_dir: Path
    samples_dir: Path
    output_dir: Path
    labeled_manifest: Path
    include_samples: bool
    max_samples: int
    target_spacing: tuple[float, float, float]
    val_fraction: float
    seed: int
    min_positive_voxels: int
    overwrite: bool


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build composite U-Net segmentation training dataset.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labeled-manifest", type=Path, default=DEFAULT_LABELED_MANIFEST)
    parser.add_argument("--skip-samples", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all available sample studies.")
    parser.add_argument(
        "--target-spacing",
        type=str,
        default="1.5,1.0,1.0",
        help="Spacing as z,y,x in mm.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-positive-voxels", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return BuildConfig(
        raw_dir=args.raw_dir,
        samples_dir=args.samples_dir,
        output_dir=args.output_dir,
        labeled_manifest=args.labeled_manifest,
        include_samples=not args.skip_samples,
        max_samples=max(int(args.max_samples), 0),
        target_spacing=parse_spacing(args.target_spacing),
        val_fraction=float(np.clip(args.val_fraction, 0.0, 0.5)),
        seed=int(args.seed),
        min_positive_voxels=max(int(args.min_positive_voxels), 1),
        overwrite=bool(args.overwrite),
    )


def parse_spacing(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("target-spacing must be z,y,x")
    values = tuple(float(value) for value in parts)
    if any(value <= 0.0 for value in values):
        raise ValueError("spacing values must be positive")
    return values


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    candidate = (CTSCAN_ROOT / path).resolve()
    if candidate.exists():
        return candidate
    return (base_dir / path).resolve()


def safe_path_text(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(CTSCAN_ROOT))
    except ValueError:
        return str(path.resolve())


def load_array(path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        payload = np.load(path)
        for key in preferred_keys:
            if key in payload:
                return payload[key]
        return payload[payload.files[0]]
    raise ValueError(f"Unsupported array file: {path}")


def parse_label_map(text: str) -> dict[int, int]:
    if not text.strip():
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("label_map must be a JSON object")
    mapping: dict[int, int] = {}
    for key, value in payload.items():
        mapping[int(key)] = int(value)
    return mapping


def remap_mask(mask: np.ndarray, label_map: dict[int, int]) -> np.ndarray:
    if not label_map:
        return mask.astype(np.uint8, copy=False)
    result = np.zeros(mask.shape, dtype=np.uint8)
    for source_label, target_label in label_map.items():
        result[mask == int(source_label)] = np.uint8(target_label)
    return result


def load_labeled_cases(manifest_path: Path) -> list[CaseSample]:
    if not manifest_path.exists():
        return []

    cases: list[CaseSample] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = str((row.get("case_id") or "").strip())
            source = str((row.get("source") or "labeled").strip())
            image_value = str((row.get("image_path") or "").strip())
            mask_value = str((row.get("mask_path") or "").strip())
            if not case_id or not image_value or not mask_value:
                continue

            image_path = resolve_path(manifest_path.parent, image_value)
            mask_path = resolve_path(manifest_path.parent, mask_value)
            if not image_path.exists() or not mask_path.exists():
                continue

            image = load_array(image_path, ("image", "volume_hu", "volume", "arr_0")).astype(np.float32)
            mask = load_array(mask_path, ("mask", "labels", "arr_0")).astype(np.int32)
            if image.shape != mask.shape:
                continue

            label_map = parse_label_map(str(row.get("label_map") or "").strip())
            remapped = remap_mask(mask, label_map)

            spacing = (
                float(row.get("spacing_z") or 1.0),
                float(row.get("spacing_y") or 1.0),
                float(row.get("spacing_x") or 1.0),
            )
            cases.append(
                CaseSample(
                    case_id=case_id,
                    source=source,
                    image_hu=image,
                    mask=remapped,
                    spacing=spacing,
                    metadata={
                        "source_manifest": safe_path_text(manifest_path),
                        "image_path": safe_path_text(image_path),
                        "mask_path": safe_path_text(mask_path),
                    },
                )
            )
    return cases


def load_sample_cases(samples_dir: Path, limit: int) -> list[CaseSample]:
    from study import load_study_from_zip_bytes, segment_issues, segment_lungs

    samples_manifest = samples_dir / "samples.json"
    if not samples_manifest.exists():
        return []

    payload = json.loads(samples_manifest.read_text(encoding="utf-8"))
    items = sorted(payload.items())
    if limit > 0:
        items = items[:limit]

    cases: list[CaseSample] = []
    for sample_id, sample in items:
        study_zip = CTSCAN_ROOT / str(sample["study_zip"])
        if not study_zip.exists():
            continue
        study = load_study_from_zip_bytes(study_zip.read_bytes())
        lung_mask, backend = segment_lungs(study.volume_hu)
        labels = segment_issues(study.volume_hu, lung_mask)
        case_id = f"sample_{sample_id}"
        cases.append(
            CaseSample(
                case_id=case_id,
                source="samples_pseudo",
                image_hu=study.volume_hu.astype(np.float32),
                mask=labels.astype(np.uint8),
                spacing=tuple(float(value) for value in study.spacing),
                metadata={
                    "backend": backend,
                    "sample_id": sample_id,
                    "series_instance_uid": str(sample.get("series_instance_uid", "")),
                    "patient_id": str(sample.get("patient_id", "")),
                },
            )
        )
    return cases


def resample_case(case: CaseSample, target_spacing: tuple[float, float, float]) -> CaseSample:
    if ndi is None:
        return case
    zoom = tuple(float(src / dst) for src, dst in zip(case.spacing, target_spacing))
    if all(abs(factor - 1.0) < 1e-3 for factor in zoom):
        return case

    image = ndi.zoom(case.image_hu, zoom=zoom, order=1)
    mask = ndi.zoom(case.mask, zoom=zoom, order=0)
    return CaseSample(
        case_id=case.case_id,
        source=case.source,
        image_hu=image.astype(np.float32),
        mask=mask.astype(np.uint8),
        spacing=target_spacing,
        metadata={**case.metadata, "resampled": True},
    )


def normalize_image(image_hu: np.ndarray) -> np.ndarray:
    clipped = np.clip(image_hu, -1000.0, 400.0)
    normalized = (clipped + 1000.0) / 1400.0
    return normalized.astype(np.float32)


def write_case(output_dir: Path, case: CaseSample) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case.case_id}.npz"
    image = normalize_image(case.image_hu)
    mask = case.mask.astype(np.uint8)
    np.savez_compressed(
        output_path,
        image=image,
        mask=mask,
        spacing=np.asarray(case.spacing, dtype=np.float32),
    )
    case_path = safe_path_text(output_path)

    return {
        "case_id": case.case_id,
        "source": case.source,
        "path": case_path,
        "shape": [int(dim) for dim in image.shape],
        "spacing": [float(value) for value in case.spacing],
        "positive_voxels": int((mask > 0).sum()),
        "metadata": case.metadata,
    }


def split_rows(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) <= 1 or val_fraction <= 0.0:
        return rows, []

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    val_size = int(round(len(rows) * val_fraction))
    val_size = max(1, min(val_size, len(rows) - 1))
    val_indices = set(int(index) for index in order[:val_size])

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index in val_indices:
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def write_split_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "source", "path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "source": row["source"],
                    "path": row["path"],
                }
            )


def class_voxel_counts(cases_dir: Path) -> dict[int, int]:
    totals = {class_id: 0 for class_id in CLASS_NAMES}
    for file_path in sorted(cases_dir.glob("*.npz")):
        if file_path.name.startswith("._"):
            continue
        try:
            payload = np.load(file_path)
            mask = payload["mask"].astype(np.uint8)
        except Exception:
            continue
        values, counts = np.unique(mask, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            label = int(value)
            if label not in totals:
                totals[label] = 0
            totals[label] += int(count)
    return totals


def build_dataset(config: BuildConfig) -> Path:
    if config.overwrite and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = config.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    all_cases: list[CaseSample] = []
    all_cases.extend(load_labeled_cases(config.labeled_manifest))
    if config.include_samples:
        all_cases.extend(load_sample_cases(config.samples_dir, config.max_samples))

    deduped: list[CaseSample] = []
    seen_ids: set[str] = set()
    for case in all_cases:
        if case.case_id in seen_ids:
            continue
        seen_ids.add(case.case_id)
        deduped.append(case)

    rows: list[dict[str, Any]] = []
    for case in deduped:
        case = resample_case(case, config.target_spacing)
        positives = int((case.mask > 0).sum())
        if positives < config.min_positive_voxels:
            continue
        rows.append(write_case(cases_dir, case))

    train_rows, val_rows = split_rows(rows, config.val_fraction, config.seed)
    write_split_csv(config.output_dir / "train.csv", train_rows)
    write_split_csv(config.output_dir / "val.csv", val_rows)

    voxel_counts = class_voxel_counts(cases_dir)
    manifest = {
        "dataset_name": "ctscan_unet_composite",
        "version": "0.1.0",
        "total_cases": len(rows),
        "train_cases": len(train_rows),
        "val_cases": len(val_rows),
        "target_spacing": [float(value) for value in config.target_spacing],
        "classes": {str(key): value for key, value in CLASS_NAMES.items()},
        "class_voxels": {str(key): int(value) for key, value in voxel_counts.items()},
        "sources": sorted({str(row["source"]) for row in rows}),
        "cases": rows,
    }
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    config = parse_args()
    manifest_path = build_dataset(config)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Wrote: {manifest_path}")
    print(f"total_cases={payload['total_cases']} train={payload['train_cases']} val={payload['val_cases']}")
    print(f"sources={', '.join(payload['sources']) if payload['sources'] else 'none'}")


if __name__ == "__main__":
    main()
