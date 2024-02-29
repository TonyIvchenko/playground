from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np
import SimpleITK as sitk

CTSCAN_ROOT = Path(__file__).resolve().parents[2]
if str(CTSCAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CTSCAN_ROOT))

DEFAULT_RAW_DIR = CTSCAN_ROOT / "data" / "ctscan" / "raw" / "legacy_sources"
DEFAULT_OUTPUT_DIR = CTSCAN_ROOT / "data" / "legacy_compatible"
DEFAULT_MANIFEST_PATH = DEFAULT_RAW_DIR / "legacy_sources_manifest.json"
VALID_LEGACY_LABELS = {0, 1, 2, 3}


try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional
    _tqdm = None


class _SimpleProgress:
    def __init__(self, total: int | None, desc: str, unit: str):
        self.total = total if total and total > 0 else None
        self.desc = desc
        self.unit = unit
        self.count = 0
        self._last_print = 0.0
        self._print(force=True)

    def _print(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_print) < 0.25:
            return
        self._last_print = now
        if self.total:
            pct = 100.0 * (self.count / self.total)
            text = f"\r{self.desc}: {self.count}/{self.total} {self.unit} ({pct:5.1f}%)"
        else:
            text = f"\r{self.desc}: {self.count} {self.unit}"
        print(text, end="", flush=True)

    def update(self, n: int = 1) -> None:
        self.count += int(n)
        self._print(force=False)

    def close(self) -> None:
        self._print(force=True)
        print("", flush=True)


def progress_iter(iterable, total: int | None, desc: str, unit: str):
    if _tqdm is not None:
        yield from _tqdm(iterable, total=total, desc=desc, unit=unit)
        return
    progress = _SimpleProgress(total=total, desc=desc, unit=unit)
    try:
        for item in iterable:
            yield item
            progress.update(1)
    finally:
        progress.close()


@dataclass
class OutputCase:
    case_id: str
    source: str
    image_path: Path
    mask_path: Path
    shape_zyx: tuple[int, int, int]
    labels_present: list[int]


@dataclass
class BuildConfig:
    raw_dir: Path
    output_dir: Path
    manifest_path: Path
    overwrite: bool
    longciu_mask_source: str
    plethora_vote_mode: str


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build a legacy-compatible chest-CT dataset from MedSeg, LongCIU, and PleThora.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--longciu-mask-source",
        type=str,
        default="staple",
        choices=["staple", "annotator1", "annotator2", "annotator3"],
    )
    parser.add_argument(
        "--plethora-vote-mode",
        type=str,
        default="union",
        choices=["union", "majority"],
    )
    args = parser.parse_args()
    raw_dir = args.raw_dir.resolve()
    manifest_path = args.manifest_path.resolve() if args.manifest_path is not None else raw_dir / "legacy_sources_manifest.json"
    return BuildConfig(
        raw_dir=raw_dir,
        output_dir=args.output_dir.resolve(),
        manifest_path=manifest_path,
        overwrite=bool(args.overwrite),
        longciu_mask_source=str(args.longciu_mask_source),
        plethora_vote_mode=str(args.plethora_vote_mode),
    )


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def file_stem(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    if name.endswith(".zip"):
        return name[:-4]
    return path.stem.lower()


def normalize_pair_key(path: Path) -> str:
    key = file_stem(path)
    for token in ("_mask", "-mask", " mask", "_masks", "_labels", "_label", "_seg", "_segmentations", "_segmentation"):
        key = key.replace(token, "")
    key = key.replace("__", "_")
    return key.strip("_-")


def collect_nifti_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        lower = path.name.lower()
        if lower.endswith(".nii") or lower.endswith(".nii.gz"):
            return [path]
        if lower.endswith(".zip"):
            extracted_dir = path.parent / f"{path.stem}_extracted"
            if not extracted_dir.exists() or not any(extracted_dir.rglob("*")):
                with zipfile.ZipFile(path, "r") as archive:
                    archive.extractall(extracted_dir)
            return [p for p in sorted(extracted_dir.rglob("*")) if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))]
        return []
    return [p for p in sorted(path.rglob("*")) if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))]


def load_nifti(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def squeeze_to_3d(arr: np.ndarray) -> np.ndarray:
    while arr.ndim > 3 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    return arr


def remap_labels(mask: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    output = np.zeros(mask.shape, dtype=np.uint8)
    for src, dst in mapping.items():
        output[mask == int(src)] = np.uint8(dst)
    return output


def align_mask_to_image(mask_image: sitk.Image, reference_image: sitk.Image) -> sitk.Image:
    if (
        tuple(mask_image.GetSize()) == tuple(reference_image.GetSize())
        and tuple(round(v, 6) for v in mask_image.GetSpacing()) == tuple(round(v, 6) for v in reference_image.GetSpacing())
        and tuple(round(v, 6) for v in mask_image.GetOrigin()) == tuple(round(v, 6) for v in reference_image.GetOrigin())
        and tuple(round(v, 6) for v in mask_image.GetDirection()) == tuple(round(v, 6) for v in reference_image.GetDirection())
    ):
        return mask_image
    return sitk.Resample(mask_image, reference_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def write_case(output_dir: Path, case_id: str, image_array: np.ndarray, mask_array: np.ndarray, spacing_zyx: tuple[float, float, float]) -> tuple[Path, Path]:
    image_dir = output_dir / "dataset"
    mask_dir = output_dir / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_itk = sitk.GetImageFromArray(np.asarray(image_array, dtype=np.float32))
    mask_itk = sitk.GetImageFromArray(np.asarray(mask_array, dtype=np.uint8))
    spacing_xyz = (float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0]))
    image_itk.SetSpacing(spacing_xyz)
    mask_itk.SetSpacing(spacing_xyz)

    image_path = image_dir / f"{case_id}.nii.gz"
    mask_path = mask_dir / f"{case_id}mask.nii"
    sitk.WriteImage(image_itk, str(image_path), useCompression=True)
    sitk.WriteImage(mask_itk, str(mask_path), useCompression=False)
    return image_path, mask_path


def add_case(rows: list[OutputCase], output_dir: Path, case_id: str, source: str, image_itk: sitk.Image, mask_itk: sitk.Image, label_map: dict[int, int]) -> None:
    image_array = squeeze_to_3d(sitk.GetArrayFromImage(image_itk).astype(np.float32, copy=False))
    aligned_mask = align_mask_to_image(mask_itk, image_itk)
    mask_array = squeeze_to_3d(sitk.GetArrayFromImage(aligned_mask).astype(np.int16, copy=False))
    mapped_mask = remap_labels(mask_array, label_map)
    image_path, mask_path = write_case(
        output_dir=output_dir,
        case_id=case_id,
        image_array=image_array,
        mask_array=mapped_mask,
        spacing_zyx=(float(image_itk.GetSpacing()[2]), float(image_itk.GetSpacing()[1]), float(image_itk.GetSpacing()[0])),
    )
    labels_present = sorted(int(v) for v in np.unique(mapped_mask) if int(v) in VALID_LEGACY_LABELS and int(v) != 0)
    rows.append(
        OutputCase(
            case_id=case_id,
            source=source,
            image_path=image_path,
            mask_path=mask_path,
            shape_zyx=tuple(int(v) for v in image_array.shape),
            labels_present=labels_present,
        )
    )


def build_medseg_cases(raw_dir: Path, source_info: dict[str, object], output_dir: Path, rows: list[OutputCase]) -> None:
    train_images = resolve_path(raw_dir, str(source_info.get("train_images") or ""))
    train_masks = resolve_path(raw_dir, str(source_info.get("train_masks") or ""))
    volume_images = resolve_path(raw_dir, str(source_info.get("volume_images") or ""))
    volume_masks = resolve_path(raw_dir, str(source_info.get("volume_masks") or ""))

    pairs: list[tuple[str, Path, Path]] = []
    if train_images and train_masks and train_images.exists() and train_masks.exists():
        pairs.append(("medseg_train", train_images, train_masks))

    image_candidates = collect_nifti_files(volume_images) if volume_images else []
    mask_candidates = collect_nifti_files(volume_masks) if volume_masks else []
    if image_candidates and mask_candidates:
        if len(image_candidates) == 1 and len(mask_candidates) == 1:
            pairs.append(("medseg_volume", image_candidates[0], mask_candidates[0]))
        else:
            mask_lookup = {normalize_pair_key(path): path for path in mask_candidates}
            for image_path in image_candidates:
                key = normalize_pair_key(image_path)
                mask_path = mask_lookup.get(key)
                if mask_path is None:
                    continue
                pairs.append((f"medseg_{key}", image_path, mask_path))

    for case_id, image_path, mask_path in progress_iter(pairs, total=len(pairs), desc="MedSeg", unit="case"):
        add_case(
            rows=rows,
            output_dir=output_dir,
            case_id=case_id,
            source="medseg_sirm",
            image_itk=load_nifti(image_path),
            mask_itk=load_nifti(mask_path),
            label_map={1: 1, 2: 2, 3: 3},
        )


def build_longciu_cases(raw_dir: Path, source_info: dict[str, object], output_dir: Path, rows: list[OutputCase], mask_source: str) -> None:
    extracted_dir = resolve_path(raw_dir, str(source_info.get("extracted_dir") or ""))
    if extracted_dir is None or not extracted_dir.exists():
        return
    image_path = extracted_dir / "longciu_img.nii.gz"
    mask_map = {
        "staple": extracted_dir / "longciu_STAPLE_tgt.nii.gz",
        "annotator1": extracted_dir / "longciu_1_tgt.nii.gz",
        "annotator2": extracted_dir / "longciu_2_tgt.nii.gz",
        "annotator3": extracted_dir / "longciu_3_tgt.nii.gz",
    }
    mask_path = mask_map[mask_source]
    if not image_path.exists() or not mask_path.exists():
        return

    add_case(
        rows=rows,
        output_dir=output_dir,
        case_id=f"longciu_{mask_source}",
        source="longciu",
        image_itk=load_nifti(image_path),
        mask_itk=load_nifti(mask_path),
        label_map={1: 1, 2: 2},
    )


def reviewer_consensus(mask_paths: list[Path], mode: str) -> sitk.Image:
    masks = [load_nifti(path) for path in mask_paths]
    reference = masks[0]
    arrays = []
    for mask_image in masks:
        arrays.append(squeeze_to_3d(sitk.GetArrayFromImage(align_mask_to_image(mask_image, reference))) > 0)
    stacked = np.stack(arrays, axis=0)
    if mode == "majority":
        threshold = int(math.ceil(stacked.shape[0] / 2.0))
        consensus = stacked.sum(axis=0) >= threshold
    else:
        consensus = np.any(stacked, axis=0)
    output = sitk.GetImageFromArray(consensus.astype(np.uint8))
    output.CopyInformation(reference)
    return output


def load_dicom_series_zip(zip_path: Path) -> sitk.Image:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(tmp_path)
        candidate_dirs = [tmp_path] + [path for path in tmp_path.rglob("*") if path.is_dir()]
        best_files: list[str] = []
        for directory in candidate_dirs:
            dicom_files = [p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")]
            if len(dicom_files) <= len(best_files):
                continue
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or []
            if series_ids:
                files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_ids[0])
            else:
                files = [str(p) for p in sorted(dicom_files)]
            if len(files) > len(best_files):
                best_files = list(files)
        if not best_files:
            raise ValueError(f"No DICOM series found in {zip_path}")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(best_files)
        return reader.Execute()


def build_plethora_cases(raw_dir: Path, source_info: dict[str, object], output_dir: Path, rows: list[OutputCase], vote_mode: str) -> None:
    masks_dir = resolve_path(raw_dir, str(source_info.get("effusion_masks_dir") or ""))
    series_rows = list(source_info.get("ct_series") or [])
    if masks_dir is None or not masks_dir.exists() or not series_rows:
        return

    tasks: list[tuple[str, Path, list[Path]]] = []
    for item in series_rows:
        if not isinstance(item, dict):
            continue
        patient_id = str(item.get("patient_id") or "").strip()
        zip_path = resolve_path(raw_dir, str(item.get("zip_path") or ""))
        if not patient_id or zip_path is None or not zip_path.exists():
            continue
        patient_dir = masks_dir / "Effusions" / patient_id
        if not patient_dir.exists():
            continue
        mask_paths = sorted(patient_dir.glob("*.nii.gz"))
        if not mask_paths:
            continue
        tasks.append((patient_id, zip_path, mask_paths))

    for patient_id, zip_path, mask_paths in progress_iter(tasks, total=len(tasks), desc="PleThora", unit="case"):
        image_itk = load_dicom_series_zip(zip_path)
        mask_itk = reviewer_consensus(mask_paths, mode=vote_mode)
        add_case(
            rows=rows,
            output_dir=output_dir,
            case_id=f"plethora_{patient_id.lower()}",
            source="plethora",
            image_itk=image_itk,
            mask_itk=mask_itk,
            label_map={1: 3},
        )


def write_metadata(output_dir: Path, rows: list[OutputCase]) -> None:
    meta_path = output_dir / "metadata.csv"
    with meta_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "source", "image_path", "mask_path", "shape_zyx", "labels_present"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row.case_id,
                    "source": row.source,
                    "image_path": row.image_path,
                    "mask_path": row.mask_path,
                    "shape_zyx": "x".join(str(v) for v in row.shape_zyx),
                    "labels_present": ",".join(str(v) for v in row.labels_present),
                }
            )

    summary = {
        "total_cases": len(rows),
        "sources": {},
    }
    for row in rows:
        source_summary = summary["sources"].setdefault(row.source, {"cases": 0, "labels": set()})
        source_summary["cases"] += 1
        source_summary["labels"].update(row.labels_present)
    for payload in summary["sources"].values():
        payload["labels"] = sorted(payload["labels"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_dataset(config: BuildConfig) -> dict[str, object]:
    if config.output_dir.exists() and config.overwrite:
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not config.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {config.manifest_path}")
    manifest = read_json(config.manifest_path)
    sources = manifest.get("sources") or {}
    if not isinstance(sources, dict):
        raise ValueError("Manifest sources field is not a dictionary")

    rows: list[OutputCase] = []
    medseg_info = sources.get("medseg_sirm")
    if isinstance(medseg_info, dict):
        build_medseg_cases(config.raw_dir, medseg_info, config.output_dir, rows)

    longciu_info = sources.get("longciu")
    if isinstance(longciu_info, dict):
        build_longciu_cases(config.raw_dir, longciu_info, config.output_dir, rows, config.longciu_mask_source)

    plethora_info = sources.get("plethora")
    if isinstance(plethora_info, dict):
        build_plethora_cases(config.raw_dir, plethora_info, config.output_dir, rows, config.plethora_vote_mode)

    write_metadata(config.output_dir, rows)
    return {
        "total_cases": len(rows),
        "output_dir": str(config.output_dir),
    }


def main() -> None:
    config = parse_args()
    summary = build_dataset(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
