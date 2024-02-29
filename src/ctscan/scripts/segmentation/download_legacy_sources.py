from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.parse
import zipfile

import requests


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = CTSCAN_ROOT / "data" / "ctscan" / "raw" / "legacy_sources"
TCIA_SERIES_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"
TCIA_IMAGE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
GDRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}
MEDSEG_FILES = (
    {
        "key": "train_images",
        "url": "https://drive.google.com/file/d/1SJoMelgRqb0EuqlTuq6dxBWf2j9Kno8S/view?usp=sharing",
        "filename": "medseg_train_images.nii.gz",
    },
    {
        "key": "train_masks",
        "url": "https://drive.google.com/open?id=1MEqpbpwXjrLrH42DqDygWeSkDq0bi92f",
        "filename": "medseg_train_masks.nii.gz",
    },
    {
        "key": "train_slice_map",
        "url": "https://drive.google.com/file/d/1bnhpEMyuJv2Pvg5nh1CsyNd0_8Gx9u0i/view?usp=sharing",
        "filename": "medseg_train_slice_map.csv",
    },
    {
        "key": "test_images",
        "url": "https://drive.google.com/open?id=1Tl5PTS2rmajWKJMrYcZ2Na5DURvbbpit",
        "filename": "medseg_test_images.nii.gz",
    },
    {
        "key": "train_lung_masks",
        "url": "https://drive.google.com/file/d/1zj4N_KV0LBko1VSQ7FPZ38eaEGNU0K6-/view?usp=sharing",
        "filename": "medseg_train_lung_masks.nii.gz",
    },
    {
        "key": "volume_images",
        "url": "https://drive.google.com/file/d/1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m/view?usp=sharing",
        "filename": "medseg_volume_images",
    },
    {
        "key": "volume_masks",
        "url": "https://drive.google.com/file/d/1gVuDwFeAGa6jIVX9MeJV5ByIHFpOo5Bp/view?usp=sharing",
        "filename": "medseg_volume_masks",
    },
    {
        "key": "volume_lung_masks",
        "url": "https://drive.google.com/file/d/1MIp89YhuAKh4as2v_5DUoExgt6-y3AnH/view?usp=sharing",
        "filename": "medseg_volume_lung_masks",
    },
)
PLETHORA_FILES = (
    {
        "key": "thoracic_masks_zip",
        "url": "https://www.cancerimagingarchive.net/wp-content/uploads/PleThora-Thoracic_Cavities-June-2020.zip",
        "filename": "PleThora-Thoracic_Cavities-June-2020.zip",
    },
    {
        "key": "effusion_masks_zip",
        "url": "https://www.cancerimagingarchive.net/wp-content/uploads/PleThora-Effusions-June-2020.zip",
        "filename": "PleThora-Effusions-June-2020.zip",
    },
    {
        "key": "ct_manifest",
        "url": "https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC-Radiomics-OriginalCTs.tcia",
        "filename": "NSCLC-Radiomics-OriginalCTs.tcia",
    },
)
LONGCIU_DOI = "https://doi.org/10.25820/data.007301"
LONGCIU_README = "https://raw.githubusercontent.com/MICLab-Unicamp/LongCIU/main/README.md"
FIGSHARE_API_URL = "https://api.figshare.com/v2/articles"
MEDSEG_FIGSHARE_ARTICLES = (
    13521488,
    13521509,
)
MEDSEG_FIGSHARE_FILE_MAP = {
    "tr_im.nii.gz": "train_images",
    "tr_mask.nii.gz": "train_masks",
    "val_im.nii.gz": "test_images",
    "rp_im.zip": "volume_images",
    "rp_msk.zip": "volume_masks",
    "rp_lung_msk.zip": "volume_lung_masks",
}
LONGCIU_NOTE = (
    "LongCIU does not expose a stable direct file URL in a machine-friendly way from the public DOI page. "
    "Download longciu.zip from the DOI landing page and pass --longciu-archive /path/to/longciu.zip, "
    "or place it at <raw-dir>/longciu/downloads/longciu.zip before rerunning this script."
)


class ProgressWriter:
    def __init__(self, total_bytes: int | None, desc: str):
        self.total_bytes = total_bytes if total_bytes and total_bytes > 0 else None
        self.desc = desc
        self.written = 0
        self.last_update = 0.0
        self._print(force=True)

    def _print(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self.last_update) < 0.25:
            return
        self.last_update = now
        if self.total_bytes:
            pct = 100.0 * (self.written / self.total_bytes)
            msg = f"\r{self.desc}: {self.written / (1 << 20):7.1f} / {self.total_bytes / (1 << 20):7.1f} MB ({pct:5.1f}%)"
        else:
            msg = f"\r{self.desc}: {self.written / (1 << 20):7.1f} MB"
        print(msg, end="", flush=True)

    def update(self, n: int) -> None:
        self.written += int(n)
        self._print(force=False)

    def close(self) -> None:
        self._print(force=True)
        print("", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MedSeg/SIRM, LongCIU, and PleThora sources for the legacy CT notebook pipeline.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--datasets", type=str, default="medseg,longciu,plethora")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--longciu-archive", type=Path, default=None)
    parser.add_argument("--longciu-url", type=str, default="")
    parser.add_argument("--plethora-max-patients", type=int, default=0, help="0 means all effusion-positive patients.")
    parser.add_argument("--skip-ct-download", action="store_true", help="Download PleThora masks only; skip the paired CT series zips.")
    return parser.parse_args()


def selected_datasets(value: str) -> set[str]:
    items = {part.strip().lower() for part in value.split(",") if part.strip()}
    supported = {"medseg", "longciu", "plethora"}
    unknown = sorted(items - supported)
    if unknown:
        raise ValueError(f"Unsupported dataset names: {', '.join(unknown)}")
    return items


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_google_drive_file_id(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query_id = urllib.parse.parse_qs(parsed.query).get("id")
    if query_id:
        return query_id[0]
    match = re.search(r"/d/([A-Za-z0-9_-]+)", url)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract Google Drive file ID from {url}")


def extract_google_drive_download_form(html: str) -> tuple[str | None, dict[str, str]]:
    form_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', html, flags=re.IGNORECASE)
    if not form_match:
        return None, {}
    action = form_match.group(1)
    params = {
        match.group(1): match.group(2)
        for match in re.finditer(r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"', html, flags=re.IGNORECASE)
    }
    return action, params


def is_html_response(response: requests.Response) -> bool:
    content_type = str(response.headers.get("Content-Type") or "").lower()
    return "text/html" in content_type or "application/xhtml+xml" in content_type


def stream_to_file(response: requests.Response, output_path: Path, desc: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    total = response.headers.get("Content-Length")
    progress = ProgressWriter(int(total) if total and total.isdigit() else None, desc)
    try:
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))
    finally:
        progress.close()
    temp_path.replace(output_path)


def maybe_fix_extension(path: Path) -> Path:
    suffixes = ''.join(path.suffixes).lower()
    if suffixes in {'.nii', '.nii.gz', '.zip', '.csv', '.tcia'}:
        return path
    with path.open('rb') as handle:
        head = handle.read(1024)
    lowered = head.lstrip().lower()
    if lowered.startswith(b'<!doctype html') or lowered.startswith(b'<html'):
        return path
    inferred = ''
    if head.startswith(b'PK\x03\x04'):
        inferred = '.zip'
    elif head.startswith(b'\x1f\x8b'):
        inferred = '.nii.gz'
    else:
        try:
            first_line = head.decode('utf-8', errors='ignore').splitlines()[0]
        except IndexError:
            first_line = ''
        if ',' in first_line:
            inferred = '.csv'
    if not inferred:
        return path
    renamed = path.with_name(path.name + inferred)
    if renamed.exists():
        return renamed
    path.replace(renamed)
    return renamed


def download_google_drive(url: str, output_path: Path, timeout_sec: int, retries: int, overwrite: bool) -> Path:
    if output_path.exists() and not overwrite:
        return maybe_fix_extension(output_path)
    file_id = extract_google_drive_file_id(url)
    session = requests.Session()
    params = {"export": "download", "id": file_id}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(GDRIVE_DOWNLOAD_URL, params=params, headers=HTTP_HEADERS, timeout=(timeout_sec, timeout_sec * 10), stream=True)
            response.raise_for_status()
            token = None
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    token = value
                    break
            if token:
                response.close()
                params["confirm"] = token
                response = session.get(GDRIVE_DOWNLOAD_URL, params=params, headers=HTTP_HEADERS, timeout=(timeout_sec, timeout_sec * 10), stream=True)
                response.raise_for_status()
            if is_html_response(response):
                html = response.text
                response.close()
                action, form_params = extract_google_drive_download_form(html)
                if action and form_params:
                    response = session.get(action, params=form_params, headers=HTTP_HEADERS, timeout=(timeout_sec, timeout_sec * 10), stream=True)
                    response.raise_for_status()
                else:
                    raise RuntimeError("Google Drive returned HTML instead of file content.")
            if is_html_response(response):
                html = response.text
                response.close()
                raise RuntimeError(f"Google Drive returned HTML instead of file content: {html[:200]!r}")
            stream_to_file(response, output_path, desc=f"Downloading {output_path.name}")
            response.close()
            return maybe_fix_extension(output_path)
        except Exception as exc:  # pragma: no cover - network instability path
            last_error = exc
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def download_http(url: str, output_path: Path, timeout_sec: int, retries: int, overwrite: bool) -> Path:
    if output_path.exists() and not overwrite:
        return maybe_fix_extension(output_path)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HTTP_HEADERS, timeout=(timeout_sec, timeout_sec * 10), stream=True)
            response.raise_for_status()
            stream_to_file(response, output_path, desc=f"Downloading {output_path.name}")
            response.close()
            return maybe_fix_extension(output_path)
        except Exception as exc:  # pragma: no cover - network instability path
            last_error = exc
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def fetch_figshare_file_urls(article_id: int, timeout_sec: int) -> dict[str, str]:
    response = requests.get(
        f"{FIGSHARE_API_URL}/{int(article_id)}",
        headers=HTTP_HEADERS,
        timeout=(timeout_sec, timeout_sec * 2),
    )
    response.raise_for_status()
    payload = response.json()
    files = payload.get("files") or []
    output: dict[str, str] = {}
    for item in files:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        file_id = item.get("id")
        if file_id is not None:
            url = f"{FIGSHARE_API_URL.rsplit('/', 1)[0]}/file/download/{int(file_id)}"
        else:
            url = str(item.get("download_url") or "").strip()
        if name and url:
            output[name] = url
    return output


def ensure_extracted_zip(zip_path: Path, output_dir: Path, overwrite: bool) -> Path:
    if output_dir.exists() and any(output_dir.rglob("*")) and not overwrite:
        return output_dir
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)
    return output_dir


def read_tcia_series_from_manifest(path: Path) -> list[str]:
    series: list[str] = []
    in_block = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped == "ListOfSeriesToDownload=":
            in_block = True
            continue
        if in_block and stripped:
            series.append(stripped)
    return series


def fetch_series_metadata(patient_id: str, timeout_sec: int, retries: int) -> dict[str, object]:
    params = {
        "Collection": "NSCLC-Radiomics",
        "PatientID": patient_id,
        "Modality": "CT",
        "format": "json",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(TCIA_SERIES_URL, params=params, headers=HTTP_HEADERS, timeout=timeout_sec)
            response.raise_for_status()
            payload = response.json()
            if not payload:
                raise ValueError(f"No CT series returned for patient {patient_id}")
            ranked = sorted(payload, key=lambda item: int(item.get("ImageCount", 0)), reverse=True)
            return ranked[0]
        except Exception as exc:  # pragma: no cover - network instability path
            last_error = exc
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"Failed to query TCIA series for {patient_id}: {last_error}")


def list_zip_patient_ids(zip_path: Path, pattern: str) -> list[str]:
    patient_ids: set[str] = set()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            match = re.search(pattern, name)
            if match:
                patient_ids.add(match.group(1))
    return sorted(patient_ids)


def maybe_copy_local_archive(source_path: Path, output_path: Path, overwrite: bool) -> Path:
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    return output_path


def download_medseg(raw_dir: Path, timeout_sec: int, retries: int, overwrite: bool) -> dict[str, str]:
    source_dir = ensure_dir(raw_dir / "medseg_sirm")
    downloads_dir = ensure_dir(source_dir / "downloads")
    manifest: dict[str, str] = {
        "source_url": "https://medicalsegmentation.com/covid19/",
        "figshare_dataset_1": "https://figshare.com/articles/dataset/MedSeg_Covid_Dataset_1/13521488",
        "figshare_dataset_2": "https://figshare.com/articles/dataset/Covid_Dataset_2/13521509",
        "label_space": "0=background, 1=ground_glass, 2=consolidation, 3=pleural_effusion",
    }
    figshare_urls: dict[str, str] = {}
    for article_id in MEDSEG_FIGSHARE_ARTICLES:
        figshare_urls.update(fetch_figshare_file_urls(article_id, timeout_sec=timeout_sec))
    for item in MEDSEG_FILES:
        output_path = downloads_dir / item["filename"]
        figshare_name = next((name for name, key in MEDSEG_FIGSHARE_FILE_MAP.items() if key == item["key"]), None)
        if figshare_name and figshare_name in figshare_urls:
            manifest[item["key"]] = str(download_http(figshare_urls[figshare_name], output_path, timeout_sec, retries, overwrite))
        else:
            manifest[item["key"]] = str(download_google_drive(item["url"], output_path, timeout_sec, retries, overwrite))
    write_json(source_dir / "manifest.json", manifest)
    return manifest


def download_longciu(raw_dir: Path, timeout_sec: int, retries: int, overwrite: bool, archive_path: Path | None, archive_url: str) -> dict[str, str]:
    source_dir = ensure_dir(raw_dir / "longciu")
    downloads_dir = ensure_dir(source_dir / "downloads")
    extracted_dir = source_dir / "extracted"
    manifest: dict[str, str] = {
        "source_url": LONGCIU_DOI,
        "paper_readme": LONGCIU_README,
        "note": LONGCIU_NOTE,
    }

    local_archive = downloads_dir / "longciu.zip"
    if archive_path is not None:
        local_archive = maybe_copy_local_archive(archive_path.resolve(), local_archive, overwrite=overwrite)
    elif archive_url.strip():
        local_archive = download_http(archive_url.strip(), local_archive, timeout_sec, retries, overwrite)
    elif local_archive.exists():
        pass
    else:
        write_json(source_dir / "manifest.json", manifest)
        return manifest

    manifest["archive"] = str(local_archive)
    manifest["extracted_dir"] = str(ensure_extracted_zip(local_archive, extracted_dir, overwrite=overwrite))
    write_json(source_dir / "manifest.json", manifest)
    return manifest


def download_plethora(raw_dir: Path, timeout_sec: int, retries: int, overwrite: bool, max_patients: int, skip_ct_download: bool) -> dict[str, object]:
    source_dir = ensure_dir(raw_dir / "plethora")
    downloads_dir = ensure_dir(source_dir / "downloads")
    masks_dir = ensure_dir(source_dir / "masks")
    ct_zip_dir = ensure_dir(source_dir / "ct_zips")
    manifest: dict[str, object] = {
        "source_url": "https://www.cancerimagingarchive.net/analysis-result/plethora/",
        "collection_url": "https://www.cancerimagingarchive.net/collection/nsclc-radiomics/",
        "label_space": "0=background, 3=pleural_effusion",
    }

    local_files: dict[str, str] = {}
    for item in PLETHORA_FILES:
        path = download_http(item["url"], downloads_dir / item["filename"], timeout_sec, retries, overwrite)
        local_files[item["key"]] = str(path)
    manifest.update(local_files)

    effusion_zip = Path(local_files["effusion_masks_zip"])
    thoracic_zip = Path(local_files["thoracic_masks_zip"])
    effusion_dir = ensure_extracted_zip(effusion_zip, masks_dir / "effusions", overwrite=overwrite)
    thoracic_dir = ensure_extracted_zip(thoracic_zip, masks_dir / "thoracic", overwrite=overwrite)
    manifest["effusion_masks_dir"] = str(effusion_dir)
    manifest["thoracic_masks_dir"] = str(thoracic_dir)

    effusion_patients = list_zip_patient_ids(effusion_zip, r"(LUNG1-\d+)")
    if max_patients > 0:
        effusion_patients = effusion_patients[:max_patients]

    series_rows: list[dict[str, object]] = []
    if not skip_ct_download:
        for patient_id in effusion_patients:
            metadata = fetch_series_metadata(patient_id, timeout_sec=timeout_sec, retries=retries)
            series_uid = str(metadata.get("SeriesInstanceUID", "")).strip()
            if not series_uid:
                raise ValueError(f"Missing SeriesInstanceUID for {patient_id}")
            zip_path = ct_zip_dir / f"{patient_id}.zip"
            download_http(
                TCIA_IMAGE_URL + "?" + urllib.parse.urlencode({"SeriesInstanceUID": series_uid}),
                zip_path,
                timeout_sec,
                retries,
                overwrite,
            )
            series_rows.append(
                {
                    "patient_id": patient_id,
                    "series_instance_uid": series_uid,
                    "image_count": int(metadata.get("ImageCount", 0) or 0),
                    "zip_path": str(zip_path),
                }
            )
    manifest["effusion_patient_ids"] = effusion_patients
    manifest["ct_series"] = series_rows
    manifest["ct_manifest_series_uids"] = read_tcia_series_from_manifest(Path(local_files["ct_manifest"]))
    write_json(source_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    datasets = selected_datasets(args.datasets)

    master_manifest: dict[str, object] = {
        "schema_version": 1,
        "raw_dir": str(raw_dir),
        "created_at_unix": int(time.time()),
        "sources": {},
    }

    if "medseg" in datasets:
        print("Preparing MedSeg/SIRM")
        master_manifest["sources"]["medseg_sirm"] = download_medseg(raw_dir, args.timeout_sec, args.retries, args.overwrite)

    if "longciu" in datasets:
        print("Preparing LongCIU")
        master_manifest["sources"]["longciu"] = download_longciu(
            raw_dir,
            args.timeout_sec,
            args.retries,
            args.overwrite,
            archive_path=args.longciu_archive,
            archive_url=args.longciu_url,
        )

    if "plethora" in datasets:
        print("Preparing PleThora")
        master_manifest["sources"]["plethora"] = download_plethora(
            raw_dir,
            args.timeout_sec,
            args.retries,
            args.overwrite,
            max_patients=max(int(args.plethora_max_patients), 0),
            skip_ct_download=bool(args.skip_ct_download),
        )

    write_json(raw_dir / "legacy_sources_manifest.json", master_manifest)
    print(f"Wrote manifest: {raw_dir / 'legacy_sources_manifest.json'}")
    if "longciu" in datasets and not master_manifest["sources"].get("longciu", {}).get("archive"):
        print(LONGCIU_NOTE)


if __name__ == "__main__":
    main()
