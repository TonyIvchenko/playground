"""Generate labeled LIDC rows for the composite U-Net dataset manifest."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = CTSCAN_ROOT / "data" / "ctscan" / "raw"
DEFAULT_DICOM_ROOT = DEFAULT_RAW_DIR / "lidc" / "LIDC-IDRI"
DEFAULT_LIDC_CASES_DIR = DEFAULT_RAW_DIR / "lidc" / "cases"
DEFAULT_COMPOSITE_MANIFEST = DEFAULT_RAW_DIR / "composite_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LIDC nodule masks and write composite manifest rows.")
    parser.add_argument("--dicom-root", type=Path, default=DEFAULT_DICOM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LIDC_CASES_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_COMPOSITE_MANIFEST)
    parser.add_argument("--consensus-level", type=float, default=0.5)
    parser.add_argument("--min-cluster-size", type=int, default=1)
    parser.add_argument("--max-scans", type=int, default=0, help="0 means all scans available in pylidc.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replace-lidc-rows", action="store_true")
    return parser.parse_args()


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(CTSCAN_ROOT))
    except ValueError:
        return str(path.resolve())


def sanitize(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())


def to_zyx(volume_ijk: np.ndarray) -> np.ndarray:
    if volume_ijk.ndim != 3:
        raise ValueError("Expected 3D volume")
    return np.moveaxis(volume_ijk, 2, 0).astype(np.float32, copy=False)


def write_case_npz(
    output_dir: Path,
    case_id: str,
    image_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    overwrite: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_path = output_dir / f"{case_id}.npz"
    if case_path.exists() and not overwrite:
        return case_path
    np.savez_compressed(
        case_path,
        image=image_zyx.astype(np.float32),
        mask=mask_zyx.astype(np.uint8),
        spacing=np.asarray(spacing_zyx, dtype=np.float32),
    )
    return case_path


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


def build_lidc_rows(
    dicom_root: Path,
    output_dir: Path,
    consensus_level: float,
    min_cluster_size: int,
    start_index: int,
    max_scans: int,
    overwrite: bool,
) -> list[dict[str, str]]:
    if not hasattr(np, "int"):  # pragma: no cover - compatibility for pylidc on NumPy >=2
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):  # pragma: no cover
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):  # pragma: no cover
        np.bool = bool  # type: ignore[attr-defined]

    import pylidc as pl
    from pylidc.utils import consensus

    scan_module = importlib.import_module("pylidc.Scan")
    scan_module._get_dicom_file_path_from_config_file = lambda: str(dicom_root.resolve())

    scans = list(pl.query(pl.Scan).all())
    scans = scans[start_index:]
    if max_scans > 0:
        scans = scans[:max_scans]

    rows: list[dict[str, str]] = []
    for index, scan in enumerate(scans, start=start_index + 1):
        case_id = sanitize(f"lidc_{scan.patient_id}_{scan.series_instance_uid[-12:]}")
        spacing_zyx = (float(scan.slice_spacing), float(scan.pixel_spacing), float(scan.pixel_spacing))
        case_path = output_dir / f"{case_id}.npz"
        if case_path.exists() and not overwrite:
            rows.append(
                {
                    "case_id": case_id,
                    "source": "lidc_idri",
                    "image_path": safe_rel(case_path),
                    "mask_path": safe_rel(case_path),
                    "label_map": json.dumps({}),
                    "spacing_z": str(spacing_zyx[0]),
                    "spacing_y": str(spacing_zyx[1]),
                    "spacing_x": str(spacing_zyx[2]),
                }
            )
            print(f"[{index}] {scan.patient_id} reuse existing -> {case_path.name}")
            continue

        try:
            volume_ijk = scan.to_volume(verbose=False)
        except Exception as exc:
            print(f"[{index}] skip {scan.patient_id}: failed to read scan ({exc})")
            continue

        clusters = scan.cluster_annotations()
        if not clusters:
            continue

        mask_ijk = np.zeros_like(volume_ijk, dtype=np.uint8)
        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue
            try:
                cmask, cbbox, _ = consensus(cluster, clevel=consensus_level, pad=[(0, 0), (0, 0), (0, 0)])
            except Exception:
                continue
            region = mask_ijk[cbbox]
            region[cmask] = np.uint8(5)

        positive = int((mask_ijk > 0).sum())
        if positive == 0:
            continue

        image_zyx = to_zyx(volume_ijk)
        mask_zyx = to_zyx(mask_ijk)
        try:
            case_path = write_case_npz(
                output_dir=output_dir,
                case_id=case_id,
                image_zyx=image_zyx,
                mask_zyx=mask_zyx,
                spacing_zyx=spacing_zyx,
                overwrite=overwrite,
            )
        except Exception as exc:
            print(f"[{index}] skip {scan.patient_id}: failed to write case ({exc})")
            continue

        rows.append(
            {
                "case_id": case_id,
                "source": "lidc_idri",
                "image_path": safe_rel(case_path),
                "mask_path": safe_rel(case_path),
                "label_map": json.dumps({}),
                "spacing_z": str(spacing_zyx[0]),
                "spacing_y": str(spacing_zyx[1]),
                "spacing_x": str(spacing_zyx[2]),
            }
        )
        print(
            f"[{index}] {scan.patient_id} clusters={len(clusters)} "
            f"positive_voxels={positive} -> {case_path.name}"
        )
    return rows


def merge_rows(existing: list[dict[str, str]], lidc_rows: list[dict[str, str]], replace_lidc_rows: bool) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for row in existing:
        case_id = str(row.get("case_id", "")).strip()
        source = str(row.get("source", "")).strip().lower()
        if not case_id:
            continue
        if replace_lidc_rows and source == "lidc_idri":
            continue
        merged[case_id] = row
    for row in lidc_rows:
        merged[str(row["case_id"])] = row
    return [merged[key] for key in sorted(merged.keys())]


def main() -> None:
    args = parse_args()
    rows = build_lidc_rows(
        dicom_root=args.dicom_root,
        output_dir=args.output_dir,
        consensus_level=float(np.clip(args.consensus_level, 0.0, 1.0)),
        min_cluster_size=max(int(args.min_cluster_size), 1),
        start_index=max(int(args.start_index), 0),
        max_scans=max(int(args.max_scans), 0),
        overwrite=bool(args.overwrite),
    )
    existing = read_existing_rows(args.manifest_path)
    merged = merge_rows(existing, rows, replace_lidc_rows=bool(args.replace_lidc_rows))
    write_manifest(args.manifest_path, merged)
    print(f"Wrote manifest: {args.manifest_path}")
    print(f"added_lidc_rows={len(rows)} total_rows={len(merged)}")


if __name__ == "__main__":
    main()
