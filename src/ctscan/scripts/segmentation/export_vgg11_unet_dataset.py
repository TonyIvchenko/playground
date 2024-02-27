"""Export processed CT cases to the legacy VGG11 U-Net notebook dataset layout.

The target layout matches the old training notebook pipeline:
  <output>/dataset/*.nii.gz   (image volumes)
  <output>/mask/*mask.nii     (label volumes)

Mask labels are remapped by default to four classes expected by that notebook:
  0 = background
  1 = focal lesions (nodule + mass_or_tumor)
  2 = opacities (ground_glass + consolidation)
  3 = diffuse/other findings (emphysema + fibrotic_pattern + pleural_effusion)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import SimpleITK as sitk

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional
    _tqdm = None


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = CTSCAN_ROOT / "data" / "ctscan" / "processed" / "unet_composite_full"
DEFAULT_OUTPUT_DIR = CTSCAN_ROOT / "data" / "ctscan" / "exports" / "legacy_vgg11_unet"
DEFAULT_CLASS_MAP = {
    0: 0,
    5: 1,  # nodule
    6: 1,  # mass_or_tumor
    3: 2,  # ground_glass
    4: 2,  # consolidation
    1: 3,  # emphysema
    2: 3,  # fibrotic_pattern
    7: 3,  # pleural_effusion
}


def progress_iter(iterable, total: int | None, desc: str):
    if _tqdm is not None:
        yield from _tqdm(iterable, total=total, desc=desc, unit="case")
        return
    for item in iterable:
        yield item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export processed CT NPZ dataset to legacy notebook NIfTI layout.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--class-map",
        type=str,
        default=json.dumps(DEFAULT_CLASS_MAP),
        help="JSON dict mapping source label id to target label id.",
    )
    parser.add_argument("--default-class", type=int, default=0, help="Fallback target class for unmapped labels.")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output dir before export.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cases with existing exported files.")
    return parser.parse_args()


def parse_class_map(payload: str) -> dict[int, int]:
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("--class-map must be a JSON object")
    result: dict[int, int] = {}
    for key, value in parsed.items():
        result[int(key)] = int(value)
    return result


def safe_resolve_case_path(processed_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (CTSCAN_ROOT / path).resolve()
    if candidate.exists():
        return candidate
    return (processed_dir / value).resolve()


def load_case_rows(processed_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for split_name in ("train.csv", "val.csv", "test.csv"):
        split_path = processed_dir / split_name
        if not split_path.exists():
            continue
        with split_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                case_id = str((row.get("case_id") or "").strip())
                case_path = str((row.get("path") or "").strip())
                if not case_id or not case_path:
                    continue
                if case_id in seen:
                    continue
                seen.add(case_id)
                rows.append({"case_id": case_id, "path": case_path, "source": str(row.get("source") or "")})

    if rows:
        return rows

    # Fallback to manifest rows when split CSVs are missing.
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        return rows
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        return rows
    for row in cases:
        if not isinstance(row, dict):
            continue
        case_id = str((row.get("case_id") or "").strip())
        case_path = str((row.get("path") or "").strip())
        if not case_id or not case_path:
            continue
        if case_id in seen:
            continue
        seen.add(case_id)
        rows.append({"case_id": case_id, "path": case_path, "source": str(row.get("source") or "")})
    return rows


def scalar_mask_from_payload(payload: Any) -> np.ndarray:
    if "mask" in payload:
        return np.asarray(payload["mask"], dtype=np.uint8)
    if "mask_multi" in payload:
        # Channel order in build_dataset: [5, 6, 3, 4, 1, 2, 7]
        class_ids = [5, 6, 3, 4, 1, 2, 7]
        mask_multi = np.asarray(payload["mask_multi"], dtype=np.uint8)
        if mask_multi.ndim != 4:
            raise ValueError("mask_multi must be 4D")
        mask = np.zeros(mask_multi.shape[1:], dtype=np.uint8)
        for channel_index, class_id in enumerate(class_ids):
            if channel_index >= int(mask_multi.shape[0]):
                break
            mask[mask_multi[channel_index] > 0] = np.uint8(class_id)
        return mask
    raise ValueError("Case file has neither 'mask' nor 'mask_multi'")


def remap_mask(mask: np.ndarray, class_map: dict[int, int], default_class: int) -> np.ndarray:
    mapped = np.full(mask.shape, np.uint8(default_class), dtype=np.uint8)
    for source_label, target_label in class_map.items():
        mapped[mask == int(source_label)] = np.uint8(target_label)
    return mapped


def write_nifti(
    image_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    image_path: Path,
    mask_path: Path,
) -> None:
    image_itk = sitk.GetImageFromArray(np.asarray(image_zyx, dtype=np.float32))
    mask_itk = sitk.GetImageFromArray(np.asarray(mask_zyx, dtype=np.uint8))

    spacing_xyz = (float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0]))
    image_itk.SetSpacing(spacing_xyz)
    mask_itk.SetSpacing(spacing_xyz)

    sitk.WriteImage(image_itk, str(image_path), useCompression=True)
    sitk.WriteImage(mask_itk, str(mask_path), useCompression=False)


def export_dataset(
    processed_dir: Path,
    output_dir: Path,
    class_map: dict[int, int],
    default_class: int,
    max_cases: int,
    overwrite: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    dataset_dir = output_dir / "dataset"
    mask_dir = output_dir / "mask"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = load_case_rows(processed_dir)
    if max_cases > 0:
        rows = rows[:max_cases]

    exported = 0
    skipped_existing = 0
    skipped_invalid = 0
    errors: list[str] = []

    for row in progress_iter(rows, total=len(rows), desc="Export cases"):
        case_id = row["case_id"]
        case_path = safe_resolve_case_path(processed_dir, row["path"])
        if not case_path.exists():
            skipped_invalid += 1
            errors.append(f"missing case path: {case_path}")
            continue

        image_path = dataset_dir / f"{case_id}.nii.gz"
        mask_path = mask_dir / f"{case_id}mask.nii"
        if skip_existing and image_path.exists() and mask_path.exists():
            skipped_existing += 1
            continue

        try:
            with np.load(case_path) as payload:
                image = np.asarray(payload["image"], dtype=np.float32)
                mask = scalar_mask_from_payload(payload)
                spacing_arr = np.asarray(payload["spacing"], dtype=np.float32).reshape(-1)
                spacing = (
                    float(spacing_arr[0]) if len(spacing_arr) > 0 else 1.0,
                    float(spacing_arr[1]) if len(spacing_arr) > 1 else 1.0,
                    float(spacing_arr[2]) if len(spacing_arr) > 2 else 1.0,
                )
        except Exception as exc:
            skipped_invalid += 1
            errors.append(f"bad npz for case {case_id}: {exc}")
            continue

        if image.ndim != 3 or mask.ndim != 3 or image.shape != mask.shape:
            skipped_invalid += 1
            errors.append(f"shape mismatch for case {case_id}: image={image.shape} mask={mask.shape}")
            continue

        mapped_mask = remap_mask(mask, class_map=class_map, default_class=default_class)
        write_nifti(image, mapped_mask, spacing, image_path=image_path, mask_path=mask_path)
        exported += 1

    summary = {
        "source_processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "class_map": {str(k): int(v) for k, v in class_map.items()},
        "default_class": int(default_class),
        "total_input_cases": len(rows),
        "exported_cases": int(exported),
        "skipped_existing": int(skipped_existing),
        "skipped_invalid": int(skipped_invalid),
        "max_target_class": int(max(class_map.values()) if class_map else default_class),
        "notebook_expected_classes": int(max(class_map.values()) + 1 if class_map else default_class + 1),
        "errors_preview": errors[:20],
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir.resolve()
    output_dir = args.output_dir.resolve()
    class_map = parse_class_map(args.class_map)

    summary = export_dataset(
        processed_dir=processed_dir,
        output_dir=output_dir,
        class_map=class_map,
        default_class=int(args.default_class),
        max_cases=max(int(args.max_cases), 0),
        overwrite=bool(args.overwrite),
        skip_existing=bool(args.skip_existing),
    )
    print(f"output_dir={output_dir}")
    print(
        f"total_input_cases={summary['total_input_cases']} exported_cases={summary['exported_cases']} "
        f"skipped_existing={summary['skipped_existing']} skipped_invalid={summary['skipped_invalid']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise

