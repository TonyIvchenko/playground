"""Download LIDC-IDRI CT series into a local DICOM tree."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
import urllib.parse
import urllib.request
import zipfile

import requests


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
NBIA_GET_SERIES_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"
NBIA_GET_IMAGE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
DEFAULT_RAW_DIR = CTSCAN_ROOT / "data" / "ctscan" / "raw" / "lidc"


@dataclass
class DownloadConfig:
    raw_dir: Path
    dicom_root: Path
    series_csv: Path
    max_series: int
    start_index: int
    timeout_sec: int
    retries: int
    retry_backoff_sec: int
    resume_series_uid: str
    keep_zip: bool
    overwrite: bool
    dry_run: bool
    stop_on_error: bool


def parse_args() -> DownloadConfig:
    parser = argparse.ArgumentParser(description="Download LIDC-IDRI CT series from TCIA NBIA.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--dicom-root", type=Path, default=None)
    parser.add_argument("--series-csv", type=Path, default=None)
    parser.add_argument("--max-series", type=int, default=0, help="0 means all series.")
    parser.add_argument("--start-index", type=int, default=0, help="Start from series index for resume.")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=int, default=8)
    parser.add_argument("--resume-series-uid", type=str, default="")
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    dicom_root = args.dicom_root or (raw_dir / "LIDC-IDRI")
    series_csv = args.series_csv or (raw_dir / "series_manifest.csv")
    return DownloadConfig(
        raw_dir=raw_dir,
        dicom_root=dicom_root,
        series_csv=series_csv,
        max_series=max(int(args.max_series), 0),
        start_index=max(int(args.start_index), 0),
        timeout_sec=max(int(args.timeout_sec), 30),
        retries=max(int(args.retries), 1),
        retry_backoff_sec=max(int(args.retry_backoff_sec), 1),
        resume_series_uid=str(args.resume_series_uid).strip(),
        keep_zip=bool(args.keep_zip),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        stop_on_error=bool(args.stop_on_error),
    )


def fetch_series(timeout_sec: int) -> list[dict[str, Any]]:
    url = NBIA_GET_SERIES_URL + "?" + urllib.parse.urlencode(
        {"Collection": "LIDC-IDRI", "Modality": "CT", "format": "json"}
    )
    with urllib.request.urlopen(url, timeout=timeout_sec) as response:
        payload = response.read().decode("utf-8")
    items = json.loads(payload)
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        series_uid = str(item.get("SeriesInstanceUID", "")).strip()
        if not series_uid:
            continue
        unique[series_uid] = item
    rows = list(unique.values())
    rows.sort(
        key=lambda row: (
            str(row.get("PatientID", "")),
            str(row.get("StudyInstanceUID", "")),
            str(row.get("SeriesInstanceUID", "")),
        )
    )
    return rows


def write_series_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SeriesDescription",
        "StudyDescription",
        "BodyPartExamined",
        "ImageCount",
        "SeriesDate",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output_path


def _safe(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text if text else default


def _series_dir(config: DownloadConfig, row: dict[str, Any]) -> Path:
    patient_id = _safe(row.get("PatientID"), "unknown-patient")
    study_uid = _safe(row.get("StudyInstanceUID"), "unknown-study")
    series_uid = _safe(row.get("SeriesInstanceUID"), "unknown-series")
    return config.dicom_root / patient_id / study_uid / series_uid


def _download_zip_requests(url: str, temp_path: Path, timeout_sec: int) -> None:
    with requests.get(
        url,
        timeout=(timeout_sec, timeout_sec * 5),
        stream=True,
        headers={
            "User-Agent": "playground-ctscan/lidc-download",
            "Connection": "close",
        },
    ) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)


def _download_zip_urllib(url: str, temp_path: Path, timeout_sec: int) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "playground-ctscan/lidc-download",
            "Connection": "close",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_sec * 5) as response, temp_path.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)


def _download_zip(
    series_uid: str,
    output_path: Path,
    timeout_sec: int,
    retries: int,
    retry_backoff_sec: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return

    url = NBIA_GET_IMAGE_URL + "?" + urllib.parse.urlencode({"SeriesInstanceUID": series_uid})
    temp_path = output_path.with_suffix(".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        for backend_name, backend in (
            ("requests", _download_zip_requests),
            ("urllib", _download_zip_urllib),
        ):
            try:
                backend(url, temp_path, timeout_sec)
                temp_path.replace(output_path)
                return
            except Exception as exc:
                last_error = exc
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                print(f"  attempt={attempt}/{retries} backend={backend_name} failed: {exc}")
        if attempt < retries:
            time.sleep(min(retry_backoff_sec * attempt, 90))

    raise RuntimeError(f"Failed to download series {series_uid} after {retries} attempts: {last_error}")


def _extract_dicom(zip_path: Path, series_dir: Path, overwrite: bool) -> int:
    series_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            suffix = Path(member.filename).suffix.lower()
            if suffix not in {".dcm", ""}:
                continue
            target = series_dir / Path(member.filename).name
            if target.exists() and not overwrite:
                count += 1
                continue
            with archive.open(member, "r") as source:
                target.write_bytes(source.read())
            count += 1
    return count


def _index_for_series_uid(rows: list[dict[str, Any]], series_uid: str) -> int:
    for idx, row in enumerate(rows):
        if str(row.get("SeriesInstanceUID", "")).strip() == series_uid:
            return idx
    return -1


def download_series(config: DownloadConfig, rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    zip_dir = config.raw_dir / "zips"
    downloaded = 0
    skipped = 0
    failed = 0

    start_index = config.start_index
    if config.resume_series_uid:
        found_index = _index_for_series_uid(rows, config.resume_series_uid)
        if found_index < 0:
            raise ValueError(f"Could not find series UID in manifest: {config.resume_series_uid}")
        start_index = found_index

    selected = rows[start_index:]
    if config.max_series > 0:
        selected = selected[: config.max_series]

    for index, row in enumerate(selected, start=start_index + 1):
        series_uid = _safe(row.get("SeriesInstanceUID"), "")
        if not series_uid:
            skipped += 1
            continue

        series_dir = _series_dir(config, row)
        existing_dicoms = list(series_dir.glob("*.dcm")) if series_dir.exists() else []
        if existing_dicoms and not config.overwrite:
            print(f"[{index}] skip existing {series_uid} ({len(existing_dicoms)} files)")
            skipped += 1
            continue

        zip_path = zip_dir / f"{series_uid}.zip"
        print(f"[{index}] download {series_uid}")
        if config.dry_run:
            downloaded += 1
            continue

        try:
            _download_zip(
                series_uid,
                output_path=zip_path,
                timeout_sec=config.timeout_sec,
                retries=config.retries,
                retry_backoff_sec=config.retry_backoff_sec,
            )
            file_count = _extract_dicom(zip_path=zip_path, series_dir=series_dir, overwrite=config.overwrite)
            print(f"[{index}] extracted {file_count} files -> {series_dir}")
            downloaded += 1
            if not config.keep_zip and zip_path.exists():
                zip_path.unlink(missing_ok=True)
        except Exception as exc:
            failed += 1
            print(f"[{index}] failed {series_uid}: {exc}")
            if config.stop_on_error:
                raise
            continue
    return downloaded, skipped, failed


def main() -> None:
    config = parse_args()
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.dicom_root.mkdir(parents=True, exist_ok=True)

    rows = fetch_series(timeout_sec=config.timeout_sec)
    csv_path = write_series_csv(rows, config.series_csv)
    print(f"Wrote series manifest: {csv_path}")
    print(f"Total CT series metadata rows: {len(rows)}")

    downloaded, skipped, failed = download_series(config, rows)
    print(f"downloaded={downloaded} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
