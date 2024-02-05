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
# Class ids kept for compatibility with existing adapters/manifests.
CLASS_NAMES = {
    0: "background",
    1: "emphysema",
    2: "fibrotic_pattern",
    3: "ground_glass",
    4: "consolidation",
    5: "nodule",
    6: "mass_or_tumor",
    7: "pleural_effusion",
}
CHANNEL_CLASS_IDS = [5, 6, 3, 4, 1, 2, 7]
CHANNEL_CLASS_NAMES = [CLASS_NAMES[class_id] for class_id in CHANNEL_CLASS_IDS]
CLASS_ID_TO_CHANNEL = {class_id: index for index, class_id in enumerate(CHANNEL_CLASS_IDS)}


@dataclass
class CaseSample:
    case_id: str
    source: str
    image_hu: np.ndarray
    mask_multi: np.ndarray
    roi_mask: np.ndarray
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
    disable_resample: bool
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
    parser.add_argument("--disable-resample", action="store_true")
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
        min_positive_voxels=max(int(args.min_positive_voxels), 0),
        disable_resample=bool(args.disable_resample),
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


def scalar_to_multi(mask_zyx: np.ndarray) -> np.ndarray:
    channels = np.zeros((len(CHANNEL_CLASS_IDS),) + tuple(mask_zyx.shape), dtype=np.uint8)
    for channel_index, class_id in enumerate(CHANNEL_CLASS_IDS):
        channels[channel_index, mask_zyx == int(class_id)] = np.uint8(1)
    return channels


def normalize_channel_axis(mask: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    if mask.ndim != 4:
        raise ValueError("Expected 4D channel mask")
    if tuple(mask.shape[1:]) == image_shape:
        return mask
    if tuple(mask.shape[:3]) == image_shape:
        return np.moveaxis(mask, -1, 0)
    raise ValueError(f"Channel mask shape {mask.shape} does not match image shape {image_shape}")


def remap_channel_mask(mask_channels: np.ndarray, label_map: dict[int, int]) -> np.ndarray:
    if not label_map:
        if int(mask_channels.shape[0]) != len(CHANNEL_CLASS_IDS):
            raise ValueError(
                f"Channel mask has {mask_channels.shape[0]} channels; expected {len(CHANNEL_CLASS_IDS)} "
                "when no label_map is provided."
            )
        return (mask_channels > 0).astype(np.uint8)

    output = np.zeros((len(CHANNEL_CLASS_IDS),) + tuple(mask_channels.shape[1:]), dtype=np.uint8)
    for source_label, target_label in label_map.items():
        target_channel = CLASS_ID_TO_CHANNEL.get(int(target_label))
        if target_channel is None:
            continue

        source_index: int | None = None
        for candidate in (int(source_label), int(source_label) - 1):
            if 0 <= candidate < int(mask_channels.shape[0]):
                source_index = candidate
                break
        if source_index is None:
            continue
        output[target_channel] = np.maximum(output[target_channel], (mask_channels[source_index] > 0).astype(np.uint8))
    return output


def normalize_roi_mask(mask: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    if mask.ndim == 3:
        roi = mask
    elif mask.ndim == 4:
        channels = normalize_channel_axis(mask, image_shape)
        roi = np.any(channels > 0, axis=0)
    else:
        raise ValueError("ROI mask must be 3D or 4D.")
    if tuple(roi.shape) != image_shape:
        raise ValueError(f"ROI mask shape {tuple(roi.shape)} does not match image shape {image_shape}.")
    return (roi > 0).astype(np.uint8)


def default_thoracic_roi(image_hu: np.ndarray, mask_multi: np.ndarray) -> np.ndarray:
    body_mask = image_hu > -950.0
    thoracic_like = (image_hu > -980.0) & (image_hu < 250.0) & body_mask
    roi = thoracic_like | np.any(mask_multi > 0, axis=0)
    if ndi is not None:
        roi = ndi.binary_fill_holes(roi)
        roi = ndi.binary_dilation(roi, structure=np.ones((1, 5, 5), dtype=bool), iterations=1)
    return roi.astype(np.uint8)


def multi_to_scalar(mask_multi: np.ndarray) -> np.ndarray:
    mask = np.zeros(mask_multi.shape[1:], dtype=np.uint8)
    for channel_index, class_id in enumerate(CHANNEL_CLASS_IDS):
        mask[mask_multi[channel_index] > 0] = np.uint8(class_id)
    return mask


def positive_voxels(mask_multi: np.ndarray, roi_mask: np.ndarray) -> int:
    union = np.any(mask_multi > 0, axis=0) & (roi_mask > 0)
    return int(union.sum())


def iter_labeled_cases(manifest_path: Path):
    if not manifest_path.exists():
        return

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = str((row.get("case_id") or "").strip())
            source = str((row.get("source") or "labeled").strip())
            image_value = str((row.get("image_path") or "").strip())
            mask_value = str((row.get("mask_path") or "").strip())
            roi_value = str((row.get("roi_path") or "").strip())
            if not case_id or not image_value:
                continue

            image_path = resolve_path(manifest_path.parent, image_value)
            if not image_path.exists():
                continue

            image = load_array(image_path, ("image", "volume_hu", "volume", "arr_0")).astype(np.float32)
            label_map = parse_label_map(str(row.get("label_map") or "").strip())
            if mask_value:
                mask_path = resolve_path(manifest_path.parent, mask_value)
                if not mask_path.exists():
                    continue
                raw_mask = load_array(mask_path, ("mask_multi", "mask", "labels", "arr_0"))
                if raw_mask.ndim == 3:
                    if image.shape != raw_mask.shape:
                        continue
                    remapped = remap_mask(raw_mask.astype(np.int32), label_map)
                    mask_multi = scalar_to_multi(remapped)
                elif raw_mask.ndim == 4:
                    channels = normalize_channel_axis(raw_mask, tuple(image.shape))
                    try:
                        mask_multi = remap_channel_mask(channels.astype(np.int32), label_map)
                    except ValueError:
                        continue
                else:
                    continue
            else:
                mask_path = None
                mask_multi = np.zeros((len(CHANNEL_CLASS_IDS),) + tuple(image.shape), dtype=np.uint8)

            if roi_value:
                roi_path = resolve_path(manifest_path.parent, roi_value)
                if not roi_path.exists():
                    continue
                try:
                    roi_mask = normalize_roi_mask(
                        load_array(roi_path, ("roi_mask", "mask", "labels", "arr_0")),
                        tuple(image.shape),
                    )
                except ValueError:
                    continue
            else:
                roi_path = None
                roi_mask = default_thoracic_roi(image, mask_multi)

            spacing = (
                float(row.get("spacing_z") or 1.0),
                float(row.get("spacing_y") or 1.0),
                float(row.get("spacing_x") or 1.0),
            )
            yield CaseSample(
                case_id=case_id,
                source=source,
                image_hu=image,
                mask_multi=mask_multi,
                roi_mask=roi_mask,
                spacing=spacing,
                metadata={
                    "source_manifest": safe_path_text(manifest_path),
                    "image_path": safe_path_text(image_path),
                    "mask_path": safe_path_text(mask_path) if mask_path is not None else "",
                    "roi_path": safe_path_text(roi_path) if roi_path is not None else "",
                },
            )


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
        roi_mask = default_thoracic_roi(study.volume_hu.astype(np.float32), scalar_to_multi(labels.astype(np.uint8)))
        case_id = f"sample_{sample_id}"
        cases.append(
            CaseSample(
                case_id=case_id,
                source="samples_pseudo",
                image_hu=study.volume_hu.astype(np.float32),
                mask_multi=scalar_to_multi(labels.astype(np.uint8)),
                roi_mask=roi_mask,
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
    mask_multi = ndi.zoom(case.mask_multi, zoom=(1.0,) + zoom, order=0)
    roi_mask = ndi.zoom(case.roi_mask.astype(np.uint8), zoom=zoom, order=0)
    return CaseSample(
        case_id=case.case_id,
        source=case.source,
        image_hu=image.astype(np.float32),
        mask_multi=(mask_multi > 0).astype(np.uint8),
        roi_mask=(roi_mask > 0).astype(np.uint8),
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
    mask_multi = (case.mask_multi > 0).astype(np.uint8)
    roi_mask = (case.roi_mask > 0).astype(np.uint8)
    mask = multi_to_scalar(mask_multi)
    np.savez_compressed(
        output_path,
        image=image,
        mask=mask,
        mask_multi=mask_multi,
        roi_mask=roi_mask,
        spacing=np.asarray(case.spacing, dtype=np.float32),
    )
    case_path = safe_path_text(output_path)

    return {
        "case_id": case.case_id,
        "source": case.source,
        "path": case_path,
        "shape": [int(dim) for dim in image.shape],
        "spacing": [float(value) for value in case.spacing],
        "positive_voxels": positive_voxels(mask_multi, roi_mask),
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


def class_voxel_counts(cases_dir: Path) -> tuple[dict[int, int], int]:
    totals = {class_id: 0 for class_id in CLASS_NAMES}
    total_spatial_voxels = 0
    for file_path in sorted(cases_dir.glob("*.npz")):
        if file_path.name.startswith("._"):
            continue
        try:
            payload = np.load(file_path)
            if "mask_multi" in payload:
                mask_multi = payload["mask_multi"].astype(np.uint8)
            else:
                mask_multi = scalar_to_multi(payload["mask"].astype(np.uint8))
            if "roi_mask" in payload:
                roi_mask = payload["roi_mask"].astype(np.uint8) > 0
            else:
                roi_mask = np.ones(mask_multi.shape[1:], dtype=bool)
        except Exception:
            continue
        union = np.any(mask_multi > 0, axis=0) & roi_mask
        total_spatial_voxels += int(roi_mask.sum())
        totals[0] += int((roi_mask & (~union)).sum())
        for channel_index, class_id in enumerate(CHANNEL_CLASS_IDS):
            totals[class_id] += int((mask_multi[channel_index] > 0).astype(np.uint8)[roi_mask].sum())
    return totals, total_spatial_voxels


def _rmtree_onerror(func, path: str, exc_info) -> None:
    exc = exc_info[1]
    if isinstance(exc, FileNotFoundError):
        # External/macOS volumes can race on AppleDouble sidecar files (._*).
        return
    raise exc


def build_dataset(config: BuildConfig) -> Path:
    if config.overwrite and config.output_dir.exists():
        shutil.rmtree(config.output_dir, onerror=_rmtree_onerror)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = config.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for case in iter_labeled_cases(config.labeled_manifest):
        if case.case_id in seen_ids:
            continue
        seen_ids.add(case.case_id)
        if not config.disable_resample:
            case = resample_case(case, config.target_spacing)
        positives = positive_voxels(case.mask_multi, case.roi_mask)
        if positives < config.min_positive_voxels:
            continue
        rows.append(write_case(cases_dir, case))

    if config.include_samples:
        for case in load_sample_cases(config.samples_dir, config.max_samples):
            if case.case_id in seen_ids:
                continue
            seen_ids.add(case.case_id)
            if not config.disable_resample:
                case = resample_case(case, config.target_spacing)
            positives = positive_voxels(case.mask_multi, case.roi_mask)
            if positives < config.min_positive_voxels:
                continue
            rows.append(write_case(cases_dir, case))

    train_rows, val_rows = split_rows(rows, config.val_fraction, config.seed)
    write_split_csv(config.output_dir / "train.csv", train_rows)
    write_split_csv(config.output_dir / "val.csv", val_rows)

    voxel_counts, total_spatial_voxels = class_voxel_counts(cases_dir)
    manifest = {
        "dataset_name": "ctscan_unet_composite",
        "version": "0.2.0",
        "task_type": "multilabel_segmentation",
        "total_cases": len(rows),
        "train_cases": len(train_rows),
        "val_cases": len(val_rows),
        "total_spatial_voxels": int(total_spatial_voxels),
        "target_spacing": [float(value) for value in config.target_spacing],
        "classes": {str(key): value for key, value in CLASS_NAMES.items()},
        "class_channels": [
            {"channel_index": int(index), "class_id": int(class_id), "name": CHANNEL_CLASS_NAMES[index]}
            for index, class_id in enumerate(CHANNEL_CLASS_IDS)
        ],
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
