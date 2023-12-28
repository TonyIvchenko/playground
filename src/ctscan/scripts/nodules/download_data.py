"""Prepare public dataset manifests, demo studies, and smoke training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import urllib.parse
import urllib.request

import numpy as np


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
NBIA_GET_SERIES_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"
NBIA_GET_IMAGE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
DATASET_MANIFEST = [
    {
        "name": "LIDC-IDRI",
        "role": "primary detection and malignancy weak labels",
        "source_url": "https://www.cancerimagingarchive.net/collection/lidc-idri/",
        "access": "public",
        "notes": "Use TCIA / NBIA APIs for de-identified chest CT series.",
    },
    {
        "name": "LUNA16",
        "role": "clean benchmark subset for detector training",
        "source_url": "https://luna16.grand-challenge.org/",
        "access": "public",
        "notes": "Cleaner subset derived from LIDC-IDRI for nodule detection benchmarking.",
    },
    {
        "name": "LNDb",
        "role": "external validation",
        "source_url": "https://lndb.grand-challenge.org/",
        "access": "public",
        "notes": "Use as external validation set after detector calibration.",
    },
]
DEMO_PATIENTS = {
    "lidc_idri_0001": "LIDC-IDRI-0001",
    "lidc_idri_0002": "LIDC-IDRI-0002",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare chest CT demo data and manifests.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=CTSCAN_ROOT / "data" / "ctscan" / "raw",
        help="Directory for raw dataset manifests and downloaded studies.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=CTSCAN_ROOT / "data" / "ctscan" / "samples",
        help="Directory for demo chest CT studies.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=CTSCAN_ROOT / "data" / "ctscan" / "processed",
        help="Directory for smoke training artifacts.",
    )
    parser.add_argument(
        "--skip-samples",
        action="store_true",
        help="Skip downloading demo TCIA studies.",
    )
    parser.add_argument(
        "--skip-smoke-dataset",
        action="store_true",
        help="Skip generating the deterministic smoke training dataset.",
    )
    return parser.parse_args()


def _fetch_json(url: str, params: dict[str, str]) -> list[dict[str, Any]]:
    request_url = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(request_url, timeout=120) as response:
        return json.load(response)


def _download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        output_path.write_bytes(response.read())


def write_dataset_manifest(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "public_datasets.json"
    manifest_path.write_text(json.dumps(DATASET_MANIFEST, indent=2), encoding="utf-8")
    return manifest_path


def fetch_series_uid(patient_id: str) -> str:
    series = _fetch_json(
        NBIA_GET_SERIES_URL,
        {
            "Collection": "LIDC-IDRI",
            "PatientID": patient_id,
            "Modality": "CT",
            "format": "json",
        },
    )
    if not series:
        raise ValueError(f"No CT series returned for patient {patient_id}")
    ranked = sorted(series, key=lambda item: int(item.get("ImageCount", 0)), reverse=True)
    series_uid = str(ranked[0]["SeriesInstanceUID"])
    if not series_uid:
        raise ValueError(f"Missing SeriesInstanceUID for patient {patient_id}")
    return series_uid


def download_demo_studies(samples_dir: Path) -> Path:
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}
    for sample_id, patient_id in DEMO_PATIENTS.items():
        series_uid = fetch_series_uid(patient_id)
        output_path = samples_dir / sample_id / "study.zip"
        if not output_path.exists():
            _download_file(
                NBIA_GET_IMAGE_URL + "?" + urllib.parse.urlencode({"SeriesInstanceUID": series_uid}),
                output_path,
            )
        manifest[sample_id] = {
            "patient_id": patient_id,
            "series_instance_uid": series_uid,
            "study_zip": str(output_path.relative_to(CTSCAN_ROOT)),
        }

    manifest_path = samples_dir / "samples.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _make_patch(radius: int, seed: int, positive: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    patch = rng.normal(loc=-820.0, scale=55.0, size=(16, 16, 16)).astype(np.float32)
    zz, yy, xx = np.indices((16, 16, 16))
    center = np.array([8.0, 8.0, 8.0], dtype=np.float32)
    distance = np.sqrt(((zz - center[0]) ** 2) + ((yy - center[1]) ** 2) + ((xx - center[2]) ** 2))
    if positive:
        patch[distance <= radius] = rng.normal(loc=90.0 + radius * 12.0, scale=18.0, size=int((distance <= radius).sum()))
    else:
        patch[distance <= max(1, radius - 1)] = rng.normal(loc=-420.0, scale=40.0, size=int((distance <= max(1, radius - 1)).sum()))
    return patch


def build_smoke_training_dataset(processed_dir: Path, rows: int = 192) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    patches = np.zeros((rows, 1, 16, 16, 16), dtype=np.float32)
    nodule_target = np.zeros((rows,), dtype=np.float32)
    malignancy_target = np.zeros((rows,), dtype=np.float32)

    for index in range(rows):
        positive = index % 2 == 0
        radius = 2 + (index % 4)
        patches[index, 0] = _make_patch(radius=radius, seed=index + 17, positive=positive)
        nodule_target[index] = 1.0 if positive else 0.0
        malignancy_target[index] = 1.0 if positive and radius >= 4 else 0.0

    output_path = processed_dir / "nodules_training.npz"
    np.savez_compressed(
        output_path,
        patches=patches,
        nodule_target=nodule_target,
        malignancy_target=malignancy_target,
    )
    return output_path


def main() -> None:
    args = parse_args()
    manifest_path = write_dataset_manifest(args.raw_dir)
    print(f"Wrote public dataset manifest: {manifest_path}")

    if not args.skip_samples:
        samples_manifest = download_demo_studies(args.samples_dir)
        print(f"Prepared demo studies: {samples_manifest}")

    if not args.skip_smoke_dataset:
        smoke_dataset = build_smoke_training_dataset(args.processed_dir)
        print(f"Prepared smoke training dataset: {smoke_dataset}")


if __name__ == "__main__":
    main()
