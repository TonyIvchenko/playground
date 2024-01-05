"""Prepare public chest CT manifests, full LIDC training data, and demo studies."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import requests

CTSCAN_ROOT = Path(__file__).resolve().parents[2]
if str(CTSCAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CTSCAN_ROOT))

from models.nodules import PATCH_SHAPE
from study import TARGET_SPACING, estimate_lung_mask, extract_resampled_patch, load_study_from_zip_bytes


NBIA_GET_SERIES_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"
NBIA_GET_IMAGE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
LIDC_XML_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/LIDC-XML-only.zip"
XML_NS = {"nih": "http://www.nih.gov"}
DATASET_MANIFEST = [
    {
        "name": "LIDC-IDRI",
        "role": "primary nodule detection and malignancy labels",
        "source_url": "https://www.cancerimagingarchive.net/collection/lidc-idri/",
        "access": "public",
        "notes": "Use all labeled LIDC-IDRI series plus XML radiologist annotations.",
    },
    {
        "name": "LUNA16",
        "role": "benchmark split only",
        "source_url": "https://luna16.grand-challenge.org/",
        "access": "public",
        "notes": "LUNA16 is derived from LIDC-IDRI, so it is not extra training data once full LIDC is used.",
    },
    {
        "name": "LNDb",
        "role": "external validation",
        "source_url": "https://lndb.grand-challenge.org/",
        "access": "public",
        "notes": "Keep for external validation rather than mixing it into training.",
    },
]
DEMO_PATIENTS = {
    "lidc_idri_0001": "LIDC-IDRI-0001",
    "lidc_idri_0002": "LIDC-IDRI-0002",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare public chest CT demo data and full LIDC patches.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=CTSCAN_ROOT / "data" / "ctscan" / "raw",
        help="Directory for manifests and downloaded raw studies.",
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
        help="Directory for canonical training artifacts.",
    )
    parser.add_argument(
        "--lidc-study-limit",
        type=int,
        default=0,
        help="How many labeled LIDC studies to use. `0` means all available labeled studies.",
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=1,
        help="How many negative lung patches to sample per positive lesion cluster.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=6,
        help="Concurrent workers for TCIA study downloads.",
    )
    parser.add_argument(
        "--process-workers",
        type=int,
        default=max(1, min(8, max(1, (os.cpu_count() or 2) // 2))),
        help="Parallel workers for patch extraction once raw series are cached.",
    )
    parser.add_argument(
        "--skip-samples",
        action="store_true",
        help="Skip downloading demo TCIA studies.",
    )
    parser.add_argument(
        "--skip-real-dataset",
        action="store_true",
        help="Skip building the real LIDC patch training dataset.",
    )
    return parser.parse_args()


def _fetch_json(url: str, params: dict[str, str]) -> list[dict[str, Any]]:
    request_url = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(request_url, timeout=120) as response:
        return json.load(response)


def _download_file(url: str, output_path: Path, retries: int = 3) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return

    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, timeout=(30, 600), stream=True, headers={"User-Agent": "playground-ctscan/1.0"}) as response:
                response.raise_for_status()
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
            temp_path.replace(output_path)
            return
        except Exception as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def write_dataset_manifest(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "public_datasets.json"
    manifest_path.write_text(json.dumps(DATASET_MANIFEST, indent=2), encoding="utf-8")
    return manifest_path


def download_lidc_xml(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "LIDC-XML-only.zip"
    if not output_path.exists():
        print(f"Downloading LIDC XML annotations: {LIDC_XML_URL}")
        _download_file(LIDC_XML_URL, output_path)
    return output_path


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
            print(f"Downloading demo study {sample_id}: {patient_id} ({series_uid})")
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


def _text(node: ET.Element | None, tag: str) -> str:
    if node is None:
        return ""
    child = node.find(f"nih:{tag}", XML_NS)
    return "" if child is None or child.text is None else child.text.strip()


def parse_lidc_xml_bytes(xml_bytes: bytes) -> dict[str, Any] | None:
    root = ET.fromstring(xml_bytes)
    header = root.find("nih:ResponseHeader", XML_NS)
    series_uid = _text(header, "SeriesInstanceUid")
    study_uid = _text(header, "StudyInstanceUID")
    if not series_uid:
        return None

    annotations: list[dict[str, Any]] = []
    non_nodules: list[dict[str, Any]] = []
    for session in root.findall("nih:readingSession", XML_NS):
        reader_id = _text(session, "servicingRadiologistID")
        for nodule in session.findall("nih:unblindedReadNodule", XML_NS):
            characteristics = nodule.find("nih:characteristics", XML_NS)
            rois = nodule.findall("nih:roi", XML_NS)
            if characteristics is None or not rois:
                continue

            xs: list[int] = []
            ys: list[int] = []
            sop_uids: list[str] = []
            for roi in rois:
                if _text(roi, "inclusion").upper() == "FALSE":
                    continue
                sop_uid = _text(roi, "imageSOP_UID")
                if not sop_uid:
                    continue
                edge_maps = roi.findall("nih:edgeMap", XML_NS)
                if not edge_maps:
                    continue
                for edge in edge_maps:
                    x_coord = _text(edge, "xCoord")
                    y_coord = _text(edge, "yCoord")
                    if x_coord and y_coord:
                        xs.append(int(float(x_coord)))
                        ys.append(int(float(y_coord)))
                        sop_uids.append(sop_uid)

            malignancy = int(_text(characteristics, "malignancy") or 0)
            if malignancy <= 0 or not xs or not ys or not sop_uids:
                continue

            annotations.append(
                {
                    "series_instance_uid": series_uid,
                    "study_instance_uid": study_uid,
                    "reader_id": reader_id,
                    "nodule_id": _text(nodule, "noduleID"),
                    "x_coords": xs,
                    "y_coords": ys,
                    "sop_uids": sop_uids,
                    "malignancy": malignancy,
                }
            )

        for non_nodule in session.findall("nih:nonNodule", XML_NS):
            sop_uid = _text(non_nodule, "imageSOP_UID")
            locus = non_nodule.find("nih:locus", XML_NS)
            x_coord = _text(locus, "xCoord")
            y_coord = _text(locus, "yCoord")
            if not sop_uid or not x_coord or not y_coord:
                continue
            non_nodules.append(
                {
                    "series_instance_uid": series_uid,
                    "study_instance_uid": study_uid,
                    "reader_id": reader_id,
                    "non_nodule_id": _text(non_nodule, "nonNoduleID"),
                    "sop_uid": sop_uid,
                    "x_coord": int(float(x_coord)),
                    "y_coord": int(float(y_coord)),
                }
            )

    if not annotations and not non_nodules:
        return None

    return {
        "series_instance_uid": series_uid,
        "study_instance_uid": study_uid,
        "annotations": annotations,
        "non_nodules": non_nodules,
    }


def collect_real_annotation_manifest(xml_zip_path: Path, study_limit: int | None = None) -> list[dict[str, Any]]:
    selected_by_series: dict[str, dict[str, Any]] = {}
    ordered_series: list[str] = []
    limit = None if not study_limit else int(study_limit)
    with zipfile.ZipFile(xml_zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".xml"):
                continue
            parsed = parse_lidc_xml_bytes(archive.read(name))
            if parsed is None:
                continue
            series_uid = str(parsed["series_instance_uid"])
            if series_uid in selected_by_series:
                selected_by_series[series_uid]["annotations"].extend(parsed["annotations"])
                selected_by_series[series_uid]["non_nodules"].extend(parsed["non_nodules"])
                selected_by_series[series_uid]["xml_names"].append(name)
            else:
                parsed["xml_names"] = [name]
                selected_by_series[series_uid] = parsed
                ordered_series.append(series_uid)
            if limit is not None and len(ordered_series) >= limit:
                break
    return [selected_by_series[series_uid] for series_uid in ordered_series]


def _series_zip_path(raw_dir: Path, series_uid: str) -> Path:
    return raw_dir / "lidc" / f"{series_uid}.zip"


def download_lidc_series(raw_dir: Path, series_uid: str) -> Path:
    output_path = _series_zip_path(raw_dir, series_uid)
    if not output_path.exists():
        _download_file(
            NBIA_GET_IMAGE_URL + "?" + urllib.parse.urlencode({"SeriesInstanceUID": series_uid}),
            output_path,
        )
    return output_path


def sync_lidc_series(raw_dir: Path, series_uids: list[str], download_workers: int) -> None:
    missing = [series_uid for series_uid in series_uids if not _series_zip_path(raw_dir, series_uid).exists()]
    if not missing:
        print(f"LIDC cache already contains all {len(series_uids)} requested series.")
        return

    print(f"Downloading {len(missing)} missing LIDC series with {download_workers} workers.")

    def _task(series_uid: str) -> str:
        print(f"  queue download: {series_uid}")
        download_lidc_series(raw_dir, series_uid)
        return series_uid

    completed = 0
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, download_workers)) as executor:
        futures = {executor.submit(_task, series_uid): series_uid for series_uid in missing}
        for future in as_completed(futures):
            series_uid = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((series_uid, str(exc)))
                print(f"  failed download: {series_uid}: {exc}")
            completed += 1
            if completed <= 10 or completed % 25 == 0 or completed == len(missing):
                print(f"  downloaded {completed}/{len(missing)} series")
    if failures:
        print(f"Skipped {len(failures)} failed series downloads.")


def _center_from_annotation(annotation: dict[str, Any], sop_uid_to_index: dict[str, int]) -> tuple[int, int, int] | None:
    z_indices = [sop_uid_to_index[sop_uid] for sop_uid in annotation["sop_uids"] if sop_uid in sop_uid_to_index]
    if not z_indices:
        return None
    center_z = int(round(float(np.median(z_indices))))
    center_y = int(round(float(np.mean(annotation["y_coords"]))))
    center_x = int(round(float(np.mean(annotation["x_coords"]))))
    return center_z, center_y, center_x


def _center_from_non_nodule(annotation: dict[str, Any], sop_uid_to_index: dict[str, int]) -> tuple[int, int, int] | None:
    sop_uid = str(annotation["sop_uid"])
    if sop_uid not in sop_uid_to_index:
        return None
    return (
        int(sop_uid_to_index[sop_uid]),
        int(annotation["y_coord"]),
        int(annotation["x_coord"]),
    )


def cluster_annotations_for_study(
    study_manifest: dict[str, Any],
    sop_uid_to_index: dict[str, int],
    spacing: tuple[float, float, float],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    spacing_vec = np.array(spacing, dtype=np.float32)
    for annotation in study_manifest["annotations"]:
        center = _center_from_annotation(annotation, sop_uid_to_index)
        if center is None:
            continue
        raw_center = np.array(center, dtype=np.float32)
        points.append(
            {
                "raw_center": raw_center,
                "mm_center": raw_center * spacing_vec,
                "malignancy": float(annotation["malignancy"]),
                "reader_id": annotation["reader_id"],
                "nodule_id": annotation["nodule_id"],
            }
        )

    clusters: list[dict[str, Any]] = []
    for point in points:
        assigned = False
        for cluster in clusters:
            if float(np.linalg.norm(point["mm_center"] - cluster["mm_center"])) <= 8.0:
                cluster["members"].append(point)
                member_raw = np.stack([member["raw_center"] for member in cluster["members"]], axis=0)
                member_mm = np.stack([member["mm_center"] for member in cluster["members"]], axis=0)
                cluster["raw_center"] = member_raw.mean(axis=0)
                cluster["mm_center"] = member_mm.mean(axis=0)
                assigned = True
                break
        if not assigned:
            clusters.append(
                {
                    "members": [point],
                    "raw_center": point["raw_center"].copy(),
                    "mm_center": point["mm_center"].copy(),
                }
            )

    outputs: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters):
        malignancy_values = np.array([member["malignancy"] for member in cluster["members"]], dtype=np.float32)
        outputs.append(
            {
                "lesion_id": f"lesion-{index + 1}",
                "center": tuple(int(round(x)) for x in cluster["raw_center"].tolist()),
                "reader_count": int(len(cluster["members"])),
                "malignancy_mean": float(malignancy_values.mean()),
                "malignancy_std": float(malignancy_values.std(ddof=0)),
            }
        )
    return outputs


def cluster_non_nodules_for_study(
    study_manifest: dict[str, Any],
    sop_uid_to_index: dict[str, int],
    spacing: tuple[float, float, float],
) -> list[tuple[int, int, int]]:
    spacing_vec = np.array(spacing, dtype=np.float32)
    clusters: list[dict[str, Any]] = []
    for annotation in study_manifest.get("non_nodules", []):
        center = _center_from_non_nodule(annotation, sop_uid_to_index)
        if center is None:
            continue
        raw_center = np.array(center, dtype=np.float32)
        mm_center = raw_center * spacing_vec
        assigned = False
        for cluster in clusters:
            if float(np.linalg.norm(mm_center - cluster["mm_center"])) <= 6.0:
                cluster["members"].append(raw_center)
                member_raw = np.stack(cluster["members"], axis=0)
                cluster["raw_center"] = member_raw.mean(axis=0)
                cluster["mm_center"] = cluster["raw_center"] * spacing_vec
                assigned = True
                break
        if not assigned:
            clusters.append(
                {
                    "members": [raw_center],
                    "raw_center": raw_center,
                    "mm_center": mm_center,
                }
            )
    return [tuple(int(round(x)) for x in cluster["raw_center"].tolist()) for cluster in clusters]


def _sample_negative_centers(
    volume_hu: np.ndarray,
    spacing: tuple[float, float, float],
    positive_centers: list[tuple[int, int, int]],
    count: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    lung_mask = estimate_lung_mask(volume_hu)
    coords = np.argwhere(lung_mask)
    if len(coords) == 0:
        return []

    rng = np.random.default_rng(seed)
    rng.shuffle(coords)
    spacing_vec = np.array(spacing, dtype=np.float32)
    positive_mm = [np.array(center, dtype=np.float32) * spacing_vec for center in positive_centers]

    negatives: list[tuple[int, int, int]] = []
    for coord in coords:
        point_mm = coord.astype(np.float32) * spacing_vec
        if any(float(np.linalg.norm(point_mm - positive_center)) < 18.0 for positive_center in positive_mm):
            continue
        negatives.append((int(coord[0]), int(coord[1]), int(coord[2])))
        if len(negatives) >= count:
            break
    return negatives


def process_lidc_series(
    raw_dir: Path,
    study_manifest: dict[str, Any],
    negatives_per_positive: int,
) -> tuple[list[np.ndarray], list[float], list[float], list[float], list[str], list[dict[str, Any]]]:
    series_uid = str(study_manifest["series_instance_uid"])
    study_zip_path = _series_zip_path(raw_dir, series_uid)
    study = load_study_from_zip_bytes(study_zip_path.read_bytes(), resample=False)
    lesion_clusters = cluster_annotations_for_study(study_manifest, study.sop_uid_to_index, study.spacing)
    hard_negative_centers = cluster_non_nodules_for_study(study_manifest, study.sop_uid_to_index, study.spacing)
    if not lesion_clusters and not hard_negative_centers:
        return [], [], [], [], [], []

    patches: list[np.ndarray] = []
    nodule_target: list[float] = []
    malignancy_target: list[float] = []
    malignancy_mask: list[float] = []
    series_ids: list[str] = []
    manifest_rows: list[dict[str, Any]] = []

    positive_centers = [cluster["center"] for cluster in lesion_clusters]
    for cluster in lesion_clusters:
        patch = extract_resampled_patch(study.volume_hu, cluster["center"], study.spacing, PATCH_SHAPE)
        patches.append(np.clip(np.rint(patch), -2000, 2000).astype(np.int16))
        nodule_target.append(1.0)
        malignancy_target.append(float(np.clip((cluster["malignancy_mean"] - 1.0) / 4.0, 0.0, 1.0)))
        malignancy_mask.append(1.0)
        series_ids.append(series_uid)
        manifest_rows.append(
            {
                "series_instance_uid": series_uid,
                "lesion_id": cluster["lesion_id"],
                "patch_type": "positive",
                "reader_count": cluster["reader_count"],
                "malignancy_mean": round(cluster["malignancy_mean"], 3),
                "malignancy_std": round(cluster["malignancy_std"], 3),
            }
        )

    for index, center in enumerate(hard_negative_centers):
        patch = extract_resampled_patch(study.volume_hu, center, study.spacing, PATCH_SHAPE)
        patches.append(np.clip(np.rint(patch), -2000, 2000).astype(np.int16))
        nodule_target.append(0.0)
        malignancy_target.append(0.0)
        malignancy_mask.append(0.0)
        series_ids.append(series_uid)
        manifest_rows.append(
            {
                "series_instance_uid": series_uid,
                "lesion_id": f"non-nodule-{index + 1}",
                "patch_type": "hard_negative",
                "reader_count": 0,
                "malignancy_mean": 0.0,
                "malignancy_std": 0.0,
            }
        )

    negative_centers = _sample_negative_centers(
        study.volume_hu,
        spacing=study.spacing,
        positive_centers=positive_centers,
        count=max(0, negatives_per_positive * len(lesion_clusters) - len(hard_negative_centers)),
        seed=sum(ord(ch) for ch in series_uid) % (2**32),
    )
    for index, center in enumerate(negative_centers):
        patch = extract_resampled_patch(study.volume_hu, center, study.spacing, PATCH_SHAPE)
        patches.append(np.clip(np.rint(patch), -2000, 2000).astype(np.int16))
        nodule_target.append(0.0)
        malignancy_target.append(0.0)
        malignancy_mask.append(0.0)
        series_ids.append(series_uid)
        manifest_rows.append(
            {
                "series_instance_uid": series_uid,
                "lesion_id": f"negative-{index + 1}",
                "patch_type": "negative",
                "reader_count": 0,
                "malignancy_mean": 0.0,
                "malignancy_std": 0.0,
            }
        )

    return patches, nodule_target, malignancy_target, malignancy_mask, series_ids, manifest_rows


def _process_lidc_series_task(
    raw_dir_str: str,
    study_manifest: dict[str, Any],
    negatives_per_positive: int,
) -> tuple[list[np.ndarray], list[float], list[float], list[float], list[str], list[dict[str, Any]]]:
    return process_lidc_series(
        raw_dir=Path(raw_dir_str),
        study_manifest=study_manifest,
        negatives_per_positive=negatives_per_positive,
    )


def build_real_training_dataset(
    raw_dir: Path,
    processed_dir: Path,
    study_limit: int | None,
    negatives_per_positive: int,
    download_workers: int = 6,
    process_workers: int = 1,
) -> tuple[Path, Path]:
    xml_zip_path = download_lidc_xml(raw_dir)
    manifest_entries = collect_real_annotation_manifest(xml_zip_path, study_limit=study_limit)
    print(f"Selected {len(manifest_entries)} LIDC studies with radiologist malignancy labels.")

    series_uids = [str(entry["series_instance_uid"]) for entry in manifest_entries]
    sync_lidc_series(raw_dir, series_uids, download_workers=download_workers)

    processed_dir.mkdir(parents=True, exist_ok=True)
    all_patches: list[np.ndarray] = []
    all_nodule_target: list[float] = []
    all_malignancy_target: list[float] = []
    all_malignancy_mask: list[float] = []
    all_series_ids: list[str] = []
    all_manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    if process_workers <= 1:
        iterable: list[tuple[int, dict[str, Any], tuple[list[np.ndarray], list[float], list[float], list[float], list[str], list[dict[str, Any]]]]] = []
        for study_index, study_manifest in enumerate(manifest_entries):
            series_uid = str(study_manifest["series_instance_uid"])
            if study_index < 10 or (study_index + 1) % 25 == 0 or study_index + 1 == len(manifest_entries):
                print(f"[{study_index + 1}/{len(manifest_entries)}] Building patches for series {series_uid}")
            try:
                result = process_lidc_series(
                    raw_dir=raw_dir,
                    study_manifest=study_manifest,
                    negatives_per_positive=negatives_per_positive,
                )
                iterable.append((study_index, study_manifest, result))
            except Exception as exc:
                failures.append({"series_instance_uid": series_uid, "error": str(exc)})
                print(f"  failed to process {series_uid}: {exc}")
    else:
        iterable = []
        print(f"Extracting patches with {process_workers} workers.")
        with ProcessPoolExecutor(max_workers=process_workers) as executor:
            futures = {
                executor.submit(
                    _process_lidc_series_task,
                    str(raw_dir),
                    study_manifest,
                    negatives_per_positive,
                ): (study_index, study_manifest)
                for study_index, study_manifest in enumerate(manifest_entries)
            }
            completed = 0
            for future in as_completed(futures):
                study_index, study_manifest = futures[future]
                series_uid = str(study_manifest["series_instance_uid"])
                completed += 1
                try:
                    result = future.result()
                    iterable.append((study_index, study_manifest, result))
                except Exception as exc:
                    failures.append({"series_instance_uid": series_uid, "error": str(exc)})
                    print(f"  failed to process {series_uid}: {exc}")
                if completed <= 10 or completed % 25 == 0 or completed == len(manifest_entries):
                    print(f"  processed {completed}/{len(manifest_entries)} series")

        iterable.sort(key=lambda item: item[0])

    for _study_index, _study_manifest, result in iterable:
        patches, nodule_target, malignancy_target, malignancy_mask, series_ids, manifest_rows = result
        all_patches.extend(patches)
        all_nodule_target.extend(nodule_target)
        all_malignancy_target.extend(malignancy_target)
        all_malignancy_mask.extend(malignancy_mask)
        all_series_ids.extend(series_ids)
        all_manifest_rows.extend(manifest_rows)

    if not all_patches:
        raise RuntimeError("No real LIDC training patches were extracted.")

    dataset_path = processed_dir / "nodules_training.npz"
    np.savez_compressed(
        dataset_path,
        patches=np.stack(all_patches, axis=0).astype(np.int16)[:, None, :, :, :],
        nodule_target=np.array(all_nodule_target, dtype=np.float32),
        malignancy_target=np.array(all_malignancy_target, dtype=np.float32),
        malignancy_mask=np.array(all_malignancy_mask, dtype=np.float32),
        series_ids=np.array(all_series_ids),
    )
    manifest_path = processed_dir / "nodules_training_manifest.json"
    manifest_path.write_text(json.dumps(all_manifest_rows, indent=2), encoding="utf-8")
    if failures:
        failure_path = processed_dir / "nodules_failed_series.json"
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return dataset_path, manifest_path


def _make_patch(radius: int, seed: int, positive: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    patch = rng.normal(loc=-820.0, scale=55.0, size=PATCH_SHAPE).astype(np.float32)
    zz, yy, xx = np.indices(PATCH_SHAPE)
    center = (np.array(PATCH_SHAPE, dtype=np.float32) - 1.0) / 2.0
    distance = np.sqrt(((zz - center[0]) ** 2) + ((yy - center[1]) ** 2) + ((xx - center[2]) ** 2))
    if positive:
        patch[distance <= radius] = rng.normal(loc=90.0 + radius * 12.0, scale=18.0, size=int((distance <= radius).sum()))
    else:
        patch[distance <= max(1, radius - 1)] = rng.normal(loc=-420.0, scale=40.0, size=int((distance <= max(1, radius - 1)).sum()))
    return patch


def build_smoke_training_dataset(processed_dir: Path, rows: int = 192) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    patches = np.zeros((rows, 1, *PATCH_SHAPE), dtype=np.int16)
    nodule_target = np.zeros((rows,), dtype=np.float32)
    malignancy_target = np.zeros((rows,), dtype=np.float32)
    malignancy_mask = np.zeros((rows,), dtype=np.float32)
    series_ids = np.array([f"smoke-{index // 2}" for index in range(rows)])

    for index in range(rows):
        positive = index % 2 == 0
        radius = 2 + (index % 4)
        patches[index, 0] = np.clip(np.rint(_make_patch(radius=radius, seed=index + 17, positive=positive)), -2000, 2000).astype(np.int16)
        nodule_target[index] = 1.0 if positive else 0.0
        malignancy_target[index] = 1.0 if positive and radius >= 4 else 0.25 if positive else 0.0
        malignancy_mask[index] = 1.0 if positive else 0.0

    output_path = processed_dir / "nodules_training_smoke.npz"
    np.savez_compressed(
        output_path,
        patches=patches,
        nodule_target=nodule_target,
        malignancy_target=malignancy_target,
        malignancy_mask=malignancy_mask,
        series_ids=series_ids,
    )
    return output_path


def main() -> None:
    args = parse_args()
    manifest_path = write_dataset_manifest(args.raw_dir)
    print(f"Wrote public dataset manifest: {manifest_path}")

    if not args.skip_samples:
        samples_manifest = download_demo_studies(args.samples_dir)
        print(f"Prepared demo studies: {samples_manifest}")

    if not args.skip_real_dataset:
        study_limit = None if args.lidc_study_limit == 0 else args.lidc_study_limit
        dataset_path, dataset_manifest = build_real_training_dataset(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            study_limit=study_limit,
            negatives_per_positive=args.negatives_per_positive,
            download_workers=args.download_workers,
            process_workers=args.process_workers,
        )
        print(f"Prepared real LIDC training dataset: {dataset_path}")
        print(f"Prepared real LIDC dataset manifest: {dataset_manifest}")

    smoke_dataset = build_smoke_training_dataset(args.processed_dir)
    print(f"Prepared smoke fallback dataset: {smoke_dataset}")


if __name__ == "__main__":
    main()
