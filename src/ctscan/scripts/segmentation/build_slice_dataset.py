"""Build a 2D slice dataset (PNG image/mask pairs) from processed case NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = CTSCAN_ROOT / "data" / "ctscan" / "processed" / "unet_composite_full"
DEFAULT_OUTPUT_DIR = CTSCAN_ROOT / "data" / "ctscan" / "processed" / "slice_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PNG slice dataset from processed NPZ cases.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--negative-stride", type=int, default=6)
    parser.add_argument("--min-positive-pixels", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all.")
    parser.add_argument("--max-slices-per-case", type=int, default=0, help="0 means all.")
    parser.add_argument("--test-fraction-of-val", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_split(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_dirs(root: Path) -> tuple[Path, Path, Path]:
    images_dir = root / "images"
    masks_dir = root / "masks"
    splits_dir = root / "splits"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, masks_dir, splits_dir


def to_uint8_image(slice_data: np.ndarray) -> np.ndarray:
    arr = np.asarray(slice_data, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    if mn >= 0.0 and mx <= 1.0:
        scaled = arr * 255.0
    else:
        scaled = (arr - mn) / (mx - mn)
        scaled *= 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def write_pairs_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "mask"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def case_rows(
    records: list[dict[str, str]],
    split_name: str,
    images_dir: Path,
    masks_dir: Path,
    output_root: Path,
    negative_stride: int,
    min_positive_pixels: int,
    max_cases: int,
    max_slices_per_case: int,
    overwrite: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    used_cases = 0
    for record in records:
        case_path = Path(record["path"]).expanduser()
        if not case_path.exists():
            continue
        if max_cases > 0 and used_cases >= max_cases:
            break
        used_cases += 1

        payload = np.load(case_path)
        image = np.asarray(payload["image"], dtype=np.float32)
        mask = np.asarray(payload["mask"], dtype=np.uint8)
        if image.ndim != 3 or mask.ndim != 3:
            continue
        if image.shape != mask.shape:
            continue

        exported = 0
        for slice_idx in range(image.shape[0]):
            mask_slice = mask[slice_idx]
            positive = int((mask_slice > 0).sum())
            keep = positive >= min_positive_pixels
            if not keep and negative_stride > 0:
                keep = (slice_idx % negative_stride == 0)
            if not keep:
                continue
            if max_slices_per_case > 0 and exported >= max_slices_per_case:
                break

            stem = f"{split_name}_{record['case_id']}_z{slice_idx:04d}"
            image_path = images_dir / f"{stem}.png"
            mask_path = masks_dir / f"{stem}.png"
            if overwrite or (not image_path.exists()):
                img_uint8 = to_uint8_image(image[slice_idx])
                Image.fromarray(img_uint8, mode="L").save(image_path)
            if overwrite or (not mask_path.exists()):
                Image.fromarray(mask_slice, mode="L").save(mask_path)

            rows.append(
                {
                    "image": str(image_path.resolve().relative_to(output_root.resolve())),
                    "mask": str(mask_path.resolve().relative_to(output_root.resolve())),
                }
            )
            exported += 1
    return rows


def split_val_test(val_rows: list[dict[str, str]], fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not val_rows:
        return [], []
    rng = random.Random(seed)
    shuffled = list(val_rows)
    rng.shuffle(shuffled)
    count_test = int(round(len(shuffled) * max(0.0, min(0.9, fraction))))
    if count_test <= 0:
        return shuffled, []
    if count_test >= len(shuffled):
        return [], shuffled
    test_rows = shuffled[:count_test]
    val_keep = shuffled[count_test:]
    return val_keep, test_rows


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_records = read_split(processed_dir / "train.csv")
    val_records = read_split(processed_dir / "val.csv")

    if not train_records:
        raise FileNotFoundError(f"Missing or empty split: {processed_dir / 'train.csv'}")

    images_dir, masks_dir, splits_dir = ensure_dirs(output_dir)
    negative_stride = max(int(args.negative_stride), 0)
    min_positive_pixels = max(int(args.min_positive_pixels), 0)
    max_cases = max(int(args.max_cases), 0)
    max_slices_per_case = max(int(args.max_slices_per_case), 0)

    train_rows = case_rows(
        records=train_records,
        split_name="train",
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_root=output_dir,
        negative_stride=negative_stride,
        min_positive_pixels=min_positive_pixels,
        max_cases=max_cases,
        max_slices_per_case=max_slices_per_case,
        overwrite=bool(args.overwrite),
    )
    val_raw_rows = case_rows(
        records=val_records,
        split_name="val",
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_root=output_dir,
        negative_stride=negative_stride,
        min_positive_pixels=min_positive_pixels,
        max_cases=max_cases,
        max_slices_per_case=max_slices_per_case,
        overwrite=bool(args.overwrite),
    )
    val_rows, test_rows = split_val_test(
        val_rows=val_raw_rows,
        fraction=float(args.test_fraction_of_val),
        seed=int(args.seed),
    )

    write_pairs_csv(splits_dir / "train.csv", train_rows)
    write_pairs_csv(splits_dir / "val.csv", val_rows)
    write_pairs_csv(splits_dir / "test.csv", test_rows)

    stats = {
        "source_processed_dir": str(processed_dir),
        "negative_stride": negative_stride,
        "min_positive_pixels": min_positive_pixels,
        "max_cases": max_cases,
        "max_slices_per_case": max_slices_per_case,
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "test_pairs": len(test_rows),
    }
    (output_dir / "dataset.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"wrote_slice_dataset={output_dir}")
    print(f"train_pairs={len(train_rows)} val_pairs={len(val_rows)} test_pairs={len(test_rows)}")


if __name__ == "__main__":
    main()
