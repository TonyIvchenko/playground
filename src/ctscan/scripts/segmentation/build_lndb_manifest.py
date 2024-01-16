"""Build LNDb manifest rows and append to composite manifest."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = CTSCAN_ROOT / "data" / "ctscan" / "raw"
DEFAULT_LNDB_ROOT = DEFAULT_RAW_DIR / "lndb"
DEFAULT_CASES_DIR = DEFAULT_LNDB_ROOT / "cases"
DEFAULT_MANIFEST_PATH = DEFAULT_RAW_DIR / "composite_manifest.csv"
SUPPORTED_EXTS = (".nii.gz", ".nii", ".mhd", ".mha", ".nrrd", ".npy", ".npz")
MASK_TOKENS = ("mask", "seg", "segmentation", "label", "labels", "tumor", "nodule", "annot", "annotation", "consensus")
PARENT_MASK_TOKENS = ("mask", "masks", "label", "labels", "annot", "annotation", "consensus")


@dataclass
class PairSpec:
    case_id: str
    image_path: Path
    mask_path: Path
    spacing_zyx: tuple[float, float, float] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LNDb manifest rows for composite segmentation training.")
    parser.add_argument("--lndb-root", type=Path, default=DEFAULT_LNDB_ROOT)
    parser.add_argument("--pairs-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--target-class", type=int, default=5)
    parser.add_argument("--label-map", type=str, default="{}")
    parser.add_argument("--replace-lndb-rows", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(CTSCAN_ROOT))
    except ValueError:
        return str(path.resolve())


def sanitize(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())


def strip_known_extension(name: str) -> str:
    lower = name.lower()
    for ext in SUPPORTED_EXTS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def is_supported_volume(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(ext) for ext in SUPPORTED_EXTS)


def is_mask_name(path: Path) -> bool:
    stem = strip_known_extension(path.name).lower()
    if any(token in stem for token in MASK_TOKENS):
        return True
    parent_parts = [part.lower() for part in path.parent.parts]
    return any(any(token in part for token in PARENT_MASK_TOKENS) for part in parent_parts)


def strip_mask_suffix(stem: str) -> str:
    result = stem
    for token in MASK_TOKENS:
        result = re.sub(rf"([._-])?{re.escape(token)}$", "", result, flags=re.IGNORECASE)
    # LNDb masks are often named like LNDb-0001_rad1 / _rad2 / _rad3
    result = re.sub(r"([._-])?rad[0-9]+$", "", result, flags=re.IGNORECASE)
    return result or stem


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def parse_label_map(text: str) -> dict[int, int]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("label_map must be a JSON object")
    mapping: dict[int, int] = {}
    for key, value in payload.items():
        mapping[int(key)] = int(value)
    return mapping


def parse_spacing(row: dict[str, str]) -> tuple[float, float, float] | None:
    z = str(row.get("spacing_z") or "").strip()
    y = str(row.get("spacing_y") or "").strip()
    x = str(row.get("spacing_x") or "").strip()
    if not z or not y or not x:
        return None
    return (float(z), float(y), float(x))


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


def load_pairs_from_csv(pairs_csv: Path, root_dir: Path) -> list[PairSpec]:
    pairs: list[PairSpec] = []
    with pairs_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            image_value = str((row.get("image_path") or "").strip())
            mask_value = str((row.get("mask_path") or "").strip())
            if not image_value or not mask_value:
                continue
            image_path = resolve_path(root_dir, image_value)
            mask_path = resolve_path(root_dir, mask_value)
            case_id = str((row.get("case_id") or "").strip()) or f"lndb_{index:05d}"
            pairs.append(
                PairSpec(
                    case_id=sanitize(case_id),
                    image_path=image_path,
                    mask_path=mask_path,
                    spacing_zyx=parse_spacing(row),
                )
            )
    return pairs


def discover_pairs(root_dir: Path, max_cases: int) -> list[PairSpec]:
    files = sorted(path for path in root_dir.rglob("*") if path.is_file() and is_supported_volume(path))
    image_by_key: dict[str, list[Path]] = {}
    masks: list[Path] = []
    for path in files:
        stem = strip_known_extension(path.name)
        key = normalize_key(stem)
        if not key:
            continue
        if is_mask_name(path):
            masks.append(path)
            continue
        image_by_key.setdefault(key, []).append(path)

    pairs: list[PairSpec] = []
    for mask_path in masks:
        stem = strip_known_extension(mask_path.name)
        key = normalize_key(strip_mask_suffix(stem))
        candidates = image_by_key.get(key, [])
        if not candidates:
            continue
        image_path = sorted(candidates, key=lambda p: (p.parent != mask_path.parent, str(p)))[0]
        relative = mask_path.relative_to(root_dir)
        case_id = sanitize(f"lndb_{strip_mask_suffix(relative.as_posix())}")
        pairs.append(PairSpec(case_id=case_id, image_path=image_path, mask_path=mask_path, spacing_zyx=None))
        if max_cases > 0 and len(pairs) >= max_cases:
            break
    return pairs


def load_array(path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        payload = np.load(path)
        for key in preferred_keys:
            if key in payload:
                return payload[key]
        return payload[payload.files[0]]
    raise ValueError(f"Unsupported ndarray file: {path}")


def load_volume(path: Path, preferred_keys: tuple[str, ...], is_mask: bool) -> tuple[np.ndarray, tuple[float, float, float]]:
    lower = path.name.lower()
    if lower.endswith(".npy") or lower.endswith(".npz"):
        array = load_array(path, preferred_keys)
        if array.ndim != 3:
            array = np.squeeze(array)
        if array.ndim != 3:
            raise ValueError(f"Expected 3D ndarray: {path}")
        return array, (1.0, 1.0, 1.0)

    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SimpleITK is required for LNDb ingest when using medical volume files. Install with `pip install SimpleITK`."
        ) from exc

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3:
        array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D medical image: {path}")
    spacing_xyz = image.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    if is_mask:
        return array.astype(np.int32), spacing_zyx
    return array.astype(np.float32), spacing_zyx


def remap_mask(mask: np.ndarray, label_map: dict[int, int], target_class: int) -> np.ndarray:
    if label_map:
        result = np.zeros(mask.shape, dtype=np.uint8)
        for source_label, dest_label in label_map.items():
            result[mask == int(source_label)] = np.uint8(dest_label)
        return result
    result = np.zeros(mask.shape, dtype=np.uint8)
    result[mask > 0] = np.uint8(target_class)
    return result


def write_case_npz(
    output_dir: Path,
    case_id: str,
    image_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    overwrite: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_id}.npz"
    if output_path.exists() and not overwrite:
        return output_path
    np.savez_compressed(
        output_path,
        image=image_zyx.astype(np.float32),
        mask=mask_zyx.astype(np.uint8),
        spacing=np.asarray(spacing_zyx, dtype=np.float32),
    )
    return output_path


def read_existing_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "source",
        "image_path",
        "mask_path",
        "label_map",
        "spacing_z",
        "spacing_y",
        "spacing_x",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def merge_rows(existing: list[dict[str, str]], incoming: list[dict[str, str]], replace_lndb_rows: bool) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for row in existing:
        case_id = str(row.get("case_id", "")).strip()
        source = str(row.get("source", "")).strip().lower()
        if not case_id:
            continue
        if replace_lndb_rows and source == "lndb":
            continue
        merged[case_id] = row
    for row in incoming:
        merged[str(row["case_id"])] = row
    return [merged[key] for key in sorted(merged.keys())]


def build_rows(
    pairs: list[PairSpec],
    output_dir: Path,
    label_map: dict[int, int],
    target_class: int,
    overwrite: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, pair in enumerate(pairs, start=1):
        if not pair.image_path.exists() or not pair.mask_path.exists():
            continue

        case_path = output_dir / f"{pair.case_id}.npz"
        if case_path.exists() and not overwrite:
            spacing = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
            try:
                payload = np.load(case_path)
                if "spacing" in payload:
                    spacing = np.asarray(payload["spacing"], dtype=np.float32).reshape(-1)
            except Exception:
                pass
            if spacing.size < 3:
                spacing = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
            rows.append(
                {
                    "case_id": pair.case_id,
                    "source": "lndb",
                    "image_path": safe_rel(case_path),
                    "mask_path": safe_rel(case_path),
                    "label_map": "{}",
                    "spacing_z": str(float(spacing[0])),
                    "spacing_y": str(float(spacing[1])),
                    "spacing_x": str(float(spacing[2])),
                }
            )
            print(f"[{index}] {pair.case_id} reuse existing -> {case_path.name}")
            continue

        try:
            image_zyx, image_spacing = load_volume(
                pair.image_path,
                preferred_keys=("image", "volume_hu", "volume", "arr_0"),
                is_mask=False,
            )
            mask_zyx, _ = load_volume(
                pair.mask_path,
                preferred_keys=("mask", "labels", "segmentation", "arr_0"),
                is_mask=True,
            )
        except Exception as exc:
            print(f"[{index}] skip {pair.case_id}: failed to read pair ({exc})")
            continue
        if image_zyx.shape != mask_zyx.shape:
            print(
                f"[{index}] skip {pair.case_id}: shape mismatch "
                f"image={image_zyx.shape} mask={mask_zyx.shape}"
            )
            continue

        mapped_mask = remap_mask(mask_zyx, label_map=label_map, target_class=target_class)
        positive = int((mapped_mask > 0).sum())
        if positive == 0:
            continue

        spacing_zyx = pair.spacing_zyx or image_spacing
        try:
            case_path = write_case_npz(
                output_dir=output_dir,
                case_id=pair.case_id,
                image_zyx=image_zyx,
                mask_zyx=mapped_mask,
                spacing_zyx=spacing_zyx,
                overwrite=overwrite,
            )
        except Exception as exc:
            print(f"[{index}] skip {pair.case_id}: failed to write case ({exc})")
            continue
        rows.append(
            {
                "case_id": pair.case_id,
                "source": "lndb",
                "image_path": safe_rel(case_path),
                "mask_path": safe_rel(case_path),
                "label_map": "{}",
                "spacing_z": str(float(spacing_zyx[0])),
                "spacing_y": str(float(spacing_zyx[1])),
                "spacing_x": str(float(spacing_zyx[2])),
            }
        )
        print(
            f"[{index}] {pair.case_id} positive_voxels={positive} "
            f"image={pair.image_path.name} mask={pair.mask_path.name} -> {case_path.name}"
        )
    return rows


def main() -> None:
    args = parse_args()
    label_map = parse_label_map(args.label_map)

    if args.pairs_csv is not None:
        pairs = load_pairs_from_csv(args.pairs_csv, args.lndb_root)
    else:
        pairs = discover_pairs(args.lndb_root, max_cases=max(int(args.max_cases), 0))
    if int(args.max_cases) > 0:
        pairs = pairs[: int(args.max_cases)]

    rows = build_rows(
        pairs=pairs,
        output_dir=args.output_dir,
        label_map=label_map,
        target_class=max(int(args.target_class), 1),
        overwrite=bool(args.overwrite),
    )

    existing = read_existing_rows(args.manifest_path)
    merged = merge_rows(existing, rows, replace_lndb_rows=bool(args.replace_lndb_rows))
    write_manifest(args.manifest_path, merged)
    print(f"Wrote manifest: {args.manifest_path}")
    print(f"added_lndb_rows={len(rows)} total_rows={len(merged)}")


if __name__ == "__main__":
    main()
