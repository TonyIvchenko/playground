"""Convert LUNA16 annotations to voxel masks and append manifest rows."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = CTSCAN_ROOT / "data" / "ctscan" / "raw"
DEFAULT_LUNA_ROOT = DEFAULT_RAW_DIR / "luna16"
DEFAULT_CASES_DIR = DEFAULT_LUNA_ROOT / "cases"
DEFAULT_MANIFEST_PATH = DEFAULT_RAW_DIR / "composite_manifest.csv"


@dataclass
class LunaSeriesMeta:
    series_uid: str
    image_path: Path
    origin_xyz: np.ndarray
    spacing_xyz: np.ndarray
    volume_zyx: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LUNA16 manifest rows for composite segmentation training.")
    parser.add_argument("--luna-root", type=Path, default=DEFAULT_LUNA_ROOT)
    parser.add_argument("--annotations-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--max-series", type=int, default=0)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--replace-luna-rows", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(CTSCAN_ROOT))
    except ValueError:
        return str(path.resolve())


def sanitize(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())


def resolve_annotations_csv(luna_root: Path, annotations_csv: Path | None) -> Path:
    if annotations_csv is not None:
        return annotations_csv
    return luna_root / "annotations.csv"


def find_luna_image(luna_root: Path, series_uid: str) -> Path | None:
    for subset_dir in sorted(luna_root.glob("subset*")):
        candidate = subset_dir / f"{series_uid}.mhd"
        if candidate.exists():
            return candidate
    return None


def world_to_voxel_zyx(coord_xyz: np.ndarray, origin_xyz: np.ndarray, spacing_xyz: np.ndarray) -> np.ndarray:
    voxel_xyz = (coord_xyz - origin_xyz) / spacing_xyz
    return np.asarray([voxel_xyz[2], voxel_xyz[1], voxel_xyz[0]], dtype=np.float32)


def draw_ellipsoid_mask(
    mask_zyx: np.ndarray,
    center_zyx: np.ndarray,
    radius_mm: float,
    spacing_zyx: np.ndarray,
) -> None:
    radius_mm = float(max(radius_mm, 1.0))
    rz = max(radius_mm / float(spacing_zyx[0]), 1.0)
    ry = max(radius_mm / float(spacing_zyx[1]), 1.0)
    rx = max(radius_mm / float(spacing_zyx[2]), 1.0)

    cz, cy, cx = [float(value) for value in center_zyx]
    z_min = max(int(np.floor(cz - rz)) - 1, 0)
    z_max = min(int(np.ceil(cz + rz)) + 1, mask_zyx.shape[0] - 1)
    y_min = max(int(np.floor(cy - ry)) - 1, 0)
    y_max = min(int(np.ceil(cy + ry)) + 1, mask_zyx.shape[1] - 1)
    x_min = max(int(np.floor(cx - rx)) - 1, 0)
    x_max = min(int(np.ceil(cx + rx)) + 1, mask_zyx.shape[2] - 1)

    zz, yy, xx = np.ogrid[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
    distance = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    region = mask_zyx[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
    region[distance <= 1.0] = np.uint8(5)


def read_luna_series(image_path: Path) -> LunaSeriesMeta:
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SimpleITK is required for LUNA16 ingest. Install with `pip install SimpleITK`."
        ) from exc

    image = sitk.ReadImage(str(image_path))
    volume_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    origin_xyz = np.asarray(image.GetOrigin(), dtype=np.float32)
    spacing_xyz = np.asarray(image.GetSpacing(), dtype=np.float32)
    series_uid = image_path.stem
    return LunaSeriesMeta(
        series_uid=series_uid,
        image_path=image_path,
        origin_xyz=origin_xyz,
        spacing_xyz=spacing_xyz,
        volume_zyx=volume_zyx,
    )


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


def merge_rows(existing: list[dict[str, str]], luna_rows: list[dict[str, str]], replace_luna_rows: bool) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for row in existing:
        case_id = str(row.get("case_id", "")).strip()
        source = str(row.get("source", "")).strip().lower()
        if not case_id:
            continue
        if replace_luna_rows and source == "luna16":
            continue
        merged[case_id] = row
    for row in luna_rows:
        merged[row["case_id"]] = row
    return [merged[key] for key in sorted(merged.keys())]


def build_luna_rows(
    luna_root: Path,
    annotations_csv: Path,
    output_dir: Path,
    max_series: int,
    radius_scale: float,
    overwrite: bool,
) -> list[dict[str, str]]:
    annotations = pd.read_csv(annotations_csv)
    required = {"seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"}
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"LUNA annotations missing columns: {sorted(missing)}")

    groups = annotations.groupby("seriesuid")
    series_uids = sorted(groups.groups.keys())
    if max_series > 0:
        series_uids = series_uids[:max_series]

    rows: list[dict[str, str]] = []
    for index, series_uid in enumerate(series_uids, start=1):
        image_path = find_luna_image(luna_root, series_uid)
        if image_path is None:
            continue

        case_id = sanitize(f"luna16_{series_uid}")
        case_path = output_dir / f"{case_id}.npz"
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
                    "case_id": case_id,
                    "source": "luna16",
                    "image_path": safe_rel(case_path),
                    "mask_path": safe_rel(case_path),
                    "label_map": "{}",
                    "spacing_z": str(float(spacing[0])),
                    "spacing_y": str(float(spacing[1])),
                    "spacing_x": str(float(spacing[2])),
                }
            )
            print(f"[{index}] {series_uid} reuse existing -> {case_path.name}")
            continue

        try:
            meta = read_luna_series(image_path)
        except Exception as exc:
            print(f"[{index}] skip {series_uid}: failed to read image ({exc})")
            continue
        spacing_zyx = np.asarray([meta.spacing_xyz[2], meta.spacing_xyz[1], meta.spacing_xyz[0]], dtype=np.float32)
        mask_zyx = np.zeros(meta.volume_zyx.shape, dtype=np.uint8)

        series_rows = groups.get_group(series_uid)
        for _, row in series_rows.iterrows():
            center_xyz = np.asarray([row["coordX"], row["coordY"], row["coordZ"]], dtype=np.float32)
            center_zyx = world_to_voxel_zyx(center_xyz, meta.origin_xyz, meta.spacing_xyz)
            radius_mm = float(row["diameter_mm"]) * 0.5 * float(radius_scale)
            draw_ellipsoid_mask(mask_zyx, center_zyx, radius_mm=radius_mm, spacing_zyx=spacing_zyx)

        positive = int((mask_zyx > 0).sum())
        if positive == 0:
            continue

        try:
            case_path = write_case_npz(
                output_dir=output_dir,
                case_id=case_id,
                image_zyx=meta.volume_zyx,
                mask_zyx=mask_zyx,
                spacing_zyx=(float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2])),
                overwrite=overwrite,
            )
        except Exception as exc:
            print(f"[{index}] skip {series_uid}: failed to write case ({exc})")
            continue

        rows.append(
            {
                "case_id": case_id,
                "source": "luna16",
                "image_path": safe_rel(case_path),
                "mask_path": safe_rel(case_path),
                "label_map": "{}",
                "spacing_z": str(float(spacing_zyx[0])),
                "spacing_y": str(float(spacing_zyx[1])),
                "spacing_x": str(float(spacing_zyx[2])),
            }
        )
        print(f"[{index}] {series_uid} nodules={len(series_rows)} positive_voxels={positive} -> {case_path.name}")
    return rows


def main() -> None:
    args = parse_args()
    annotations_csv = resolve_annotations_csv(args.luna_root, args.annotations_csv)
    rows = build_luna_rows(
        luna_root=args.luna_root,
        annotations_csv=annotations_csv,
        output_dir=args.output_dir,
        max_series=max(int(args.max_series), 0),
        radius_scale=float(max(args.radius_scale, 0.1)),
        overwrite=bool(args.overwrite),
    )
    existing = read_existing_rows(args.manifest_path)
    merged = merge_rows(existing, rows, replace_luna_rows=bool(args.replace_luna_rows))
    write_manifest(args.manifest_path, merged)
    print(f"Wrote manifest: {args.manifest_path}")
    print(f"added_luna_rows={len(rows)} total_rows={len(merged)}")


if __name__ == "__main__":
    main()
