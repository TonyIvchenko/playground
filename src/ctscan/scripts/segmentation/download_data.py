"""Prepare manifests and lightweight demo studies for CT segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.parse
import urllib.request

import requests


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
NBIA_GET_SERIES_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"
NBIA_GET_IMAGE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
DATASET_MANIFEST = [
    {
        "name": "LIDC-IDRI",
        "role": "lung nodule contour/annotation source",
        "source_url": "https://www.cancerimagingarchive.net/collection/lidc-idri/",
        "access": "public",
        "notes": "Useful for lung nodule segmentation and review tasks.",
    },
    {
        "name": "LUNA16",
        "role": "benchmark split for nodule detection tasks",
        "source_url": "https://luna16.grand-challenge.org/",
        "access": "public",
        "notes": "Derived from LIDC-IDRI; benchmark-only in this service.",
    },
    {
        "name": "NLSTseg",
        "role": "pixel-level low-dose CT lesion segmentation",
        "source_url": "https://doi.org/10.5281/zenodo.14838349",
        "access": "public",
        "notes": "Use as primary lesion segmentation source when available.",
    },
    {
        "name": "LNDb",
        "role": "external validation benchmark",
        "source_url": "https://lndb.grand-challenge.org/",
        "access": "public",
        "notes": "Recommended for external validation only; check license constraints.",
    },
]
DEMO_PATIENTS = {
    "lidc_idri_0001": "LIDC-IDRI-0001",
    "lidc_idri_0002": "LIDC-IDRI-0002",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare demo DICOM studies and segmentation dataset manifest.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=CTSCAN_ROOT / "data" / "ctscan" / "raw",
        help="Output directory for dataset manifest.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=CTSCAN_ROOT / "data" / "ctscan" / "samples",
        help="Output directory for demo studies.",
    )
    parser.add_argument(
        "--skip-samples",
        action="store_true",
        help="Skip downloading demo studies.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="HTTP timeout for NBIA requests.",
    )
    return parser.parse_args()


def _fetch_json(url: str, params: dict[str, str], timeout_sec: int) -> list[dict[str, object]]:
    request_url = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(request_url, timeout=timeout_sec) as response:
        return json.load(response)


def _download_file(url: str, output_path: Path, timeout_sec: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return

    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    with requests.get(url, timeout=(timeout_sec, timeout_sec * 5), stream=True, headers={"User-Agent": "playground-ctscan/segmentation"}) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    temp_path.replace(output_path)


def write_dataset_manifest(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "public_datasets.json"
    manifest_path.write_text(json.dumps(DATASET_MANIFEST, indent=2), encoding="utf-8")
    return manifest_path


def fetch_series_uid(patient_id: str, timeout_sec: int) -> str:
    series = _fetch_json(
        NBIA_GET_SERIES_URL,
        {
            "Collection": "LIDC-IDRI",
            "PatientID": patient_id,
            "Modality": "CT",
            "format": "json",
        },
        timeout_sec=timeout_sec,
    )
    if not series:
        raise ValueError(f"No CT series returned for patient {patient_id}")
    ranked = sorted(series, key=lambda item: int(item.get("ImageCount", 0)), reverse=True)
    series_uid = str(ranked[0]["SeriesInstanceUID"])
    if not series_uid:
        raise ValueError(f"Missing SeriesInstanceUID for patient {patient_id}")
    return series_uid


def download_demo_studies(samples_dir: Path, timeout_sec: int) -> Path:
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}
    for sample_id, patient_id in DEMO_PATIENTS.items():
        series_uid = fetch_series_uid(patient_id, timeout_sec=timeout_sec)
        output_path = samples_dir / sample_id / "study.zip"
        if not output_path.exists():
            print(f"Downloading sample {sample_id}: {patient_id} ({series_uid})")
            _download_file(
                NBIA_GET_IMAGE_URL + "?" + urllib.parse.urlencode({"SeriesInstanceUID": series_uid}),
                output_path,
                timeout_sec=timeout_sec,
            )
        manifest[sample_id] = {
            "patient_id": patient_id,
            "series_instance_uid": series_uid,
            "study_zip": str(output_path.relative_to(CTSCAN_ROOT)),
        }

    manifest_path = samples_dir / "samples.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    manifest_path = write_dataset_manifest(args.raw_dir)
    print(f"Wrote dataset manifest: {manifest_path}")

    if not args.skip_samples:
        samples_manifest_path = download_demo_studies(args.samples_dir, timeout_sec=args.timeout_sec)
        print(f"Wrote samples manifest: {samples_manifest_path}")


if __name__ == "__main__":
    main()
