"""Prepare public chest CT manifests, real LIDC training data, and demo studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

CTSCAN_ROOT = Path(__file__).resolve().parents[2]
if str(CTSCAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CTSCAN_ROOT))

from study import TARGET_SPACING, estimate_lung_mask, extract_patch, load_study_from_zip_bytes, resample_volume


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
        "notes": "Use TCIA / NBIA APIs for DICOM series and LIDC-XML-only.zip for radiologist annotations.",
    },
    {
        "name": "LUNA16",
        "role": "clean benchmark subset for later detector evaluation",
        "source_url": "https://luna16.grand-challenge.org/",
        "access": "public",
        "notes": "Not downloaded by default in v1 because the initial trainer now uses real LIDC studies directly.",
    },
    {
        "name": "LNDb",
        "role": "external validation",
        "source_url": "https://lndb.grand-challenge.org/",
        "access": "public",
        "notes": "Reserve for external validation after the LIDC-based trainer stabilizes.",
    },
]
DEMO_PATIENTS = {
    "lidc_idri_0001": "LIDC-IDRI-0001",
    "lidc_idri_0002": "LIDC-IDRI-0002",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare public chest CT demo data and real LIDC patches.")
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
        default=7,
        help="How many real LIDC studies to download and convert into training patches.",
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=2,
        help="How many negative lung patches to sample per positive LIDC nodule patch.",
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


def _download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response:
        output_path.write_bytes(response.read())


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
                inclusion = _text(roi, "inclusion").upper()
                if inclusion == "FALSE":
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

            if not xs or not ys or not sop_uids:
                continue

            malignancy = int(_text(characteristics, "malignancy") or 0)
            if malignancy <= 0:
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

    if not annotations:
        return None

    return {
        "series_instance_uid": series_uid,
        "study_instance_uid": study_uid,
        "annotations": annotations,
    }


def collect_real_annotation_manifest(xml_zip_path: Path, study_limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_series: set[str] = set()
    with zipfile.ZipFile(xml_zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".xml"):
                continue
            parsed = parse_lidc_xml_bytes(archive.read(name))
            if parsed is None:
                continue
            series_uid = str(parsed["series_instance_uid"])
            if series_uid in seen_series:
                continue
            seen_series.add(series_uid)
            parsed["xml_name"] = name
            selected.append(parsed)
            if len(selected) >= study_limit:
                break
    return selected


def download_lidc_series(raw_dir: Path, series_uid: str) -> Path:
    lidc_dir = raw_dir / "lidc"
    lidc_dir.mkdir(parents=True, exist_ok=True)
    output_path = lidc_dir / f"{series_uid}.zip"
    if not output_path.exists():
        print(f"  downloading TCIA series zip: {series_uid}")
        _download_file(
            NBIA_GET_IMAGE_URL + "?" + urllib.parse.urlencode({"SeriesInstanceUID": series_uid}),
            output_path,
        )
    return output_path


def _center_from_annotation(annotation: dict[str, Any], sop_uid_to_index: dict[str, int]) -> tuple[int, int, int] | None:
    z_indices = [sop_uid_to_index[sop_uid] for sop_uid in annotation["sop_uids"] if sop_uid in sop_uid_to_index]
    if not z_indices:
        return None
    center_z = int(round(float(np.median(z_indices))))
    center_y = int(round(float(np.mean(annotation["y_coords"]))))
    center_x = int(round(float(np.mean(annotation["x_coords"]))))
    return center_z, center_y, center_x


def _rescaled_center(center: tuple[int, int, int], raw_spacing: tuple[float, float, float]) -> tuple[int, int, int]:
    scale = np.array(raw_spacing, dtype=np.float32) / np.array(TARGET_SPACING, dtype=np.float32)
    rescaled = np.round(np.array(center, dtype=np.float32) * scale).astype(int)
    return int(rescaled[0]), int(rescaled[1]), int(rescaled[2])


def _sample_negative_centers(
    volume_hu: np.ndarray,
    positive_centers: list[tuple[int, int, int]],
    count: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    lung_mask = estimate_lung_mask(volume_hu)
    coords = np.argwhere(lung_mask)
    if len(coords) == 0:
        return []
    rng.shuffle(coords)
    negatives: list[tuple[int, int, int]] = []
    for coord in coords:
        point = np.array(coord, dtype=np.float32)
        if any(np.linalg.norm(point - np.array(pos, dtype=np.float32)) < 18.0 for pos in positive_centers):
            continue
        negatives.append((int(coord[0]), int(coord[1]), int(coord[2])))
        if len(negatives) >= count:
            break
    return negatives


def build_real_training_dataset(
    raw_dir: Path,
    processed_dir: Path,
    study_limit: int,
    negatives_per_positive: int,
) -> tuple[Path, Path]:
    xml_zip_path = download_lidc_xml(raw_dir)
    manifest_entries = collect_real_annotation_manifest(xml_zip_path, study_limit=study_limit)
    print(f"Selected {len(manifest_entries)} LIDC studies with radiologist malignancy labels.")
    processed_dir.mkdir(parents=True, exist_ok=True)

    patches: list[np.ndarray] = []
    nodule_target: list[float] = []
    malignancy_target: list[float] = []
    manifest_rows: list[dict[str, Any]] = []

    for study_index, study_manifest in enumerate(manifest_entries):
        series_uid = str(study_manifest["series_instance_uid"])
        print(f"[{study_index + 1}/{len(manifest_entries)}] Building patches for series {series_uid}")
        study_zip_path = download_lidc_series(raw_dir, series_uid)
        study = load_study_from_zip_bytes(study_zip_path.read_bytes(), resample=False)
        resampled_volume, _ = resample_volume(study.volume_hu, study.spacing)

        positive_centers: list[tuple[int, int, int]] = []
        for annotation in study_manifest["annotations"]:
            raw_center = _center_from_annotation(annotation, study.sop_uid_to_index)
            if raw_center is None:
                continue
            center = _rescaled_center(raw_center, study.spacing)
            patch = extract_patch(resampled_volume, center, (16, 16, 16))
            patches.append(patch)
            nodule_target.append(1.0)
            malignancy_target.append(1.0 if int(annotation["malignancy"]) >= 4 else 0.0)
            positive_centers.append(center)
            manifest_rows.append(
                {
                    "series_instance_uid": series_uid,
                    "reader_id": annotation["reader_id"],
                    "nodule_id": annotation["nodule_id"],
                    "patch_type": "positive",
                    "malignancy": int(annotation["malignancy"]),
                }
            )

        negative_centers = _sample_negative_centers(
            resampled_volume,
            positive_centers=positive_centers,
            count=max(1, negatives_per_positive * max(1, len(positive_centers))),
            seed=study_index + 13,
        )
        for neg_index, center in enumerate(negative_centers):
            patches.append(extract_patch(resampled_volume, center, (16, 16, 16)))
            nodule_target.append(0.0)
            malignancy_target.append(0.0)
            manifest_rows.append(
                {
                    "series_instance_uid": series_uid,
                    "reader_id": "",
                    "nodule_id": f"negative-{neg_index + 1}",
                    "patch_type": "negative",
                    "malignancy": 0,
                }
            )

        print(
            f"  positives={sum(row['patch_type'] == 'positive' and row['series_instance_uid'] == series_uid for row in manifest_rows)} "
            f"negatives={sum(row['patch_type'] == 'negative' and row['series_instance_uid'] == series_uid for row in manifest_rows)}"
        )

    if not patches:
        raise RuntimeError("No real LIDC training patches were extracted.")

    dataset_path = processed_dir / "nodules_training.npz"
    np.savez_compressed(
        dataset_path,
        patches=np.stack(patches, axis=0).astype(np.float32)[:, None, :, :, :],
        nodule_target=np.array(nodule_target, dtype=np.float32),
        malignancy_target=np.array(malignancy_target, dtype=np.float32),
    )
    manifest_path = processed_dir / "nodules_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    return dataset_path, manifest_path


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

    output_path = processed_dir / "nodules_training_smoke.npz"
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

    if not args.skip_real_dataset:
        dataset_path, dataset_manifest = build_real_training_dataset(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            study_limit=args.lidc_study_limit,
            negatives_per_positive=args.negatives_per_positive,
        )
        print(f"Prepared real LIDC training dataset: {dataset_path}")
        print(f"Prepared real LIDC dataset manifest: {dataset_manifest}")

    smoke_dataset = build_smoke_training_dataset(args.processed_dir)
    print(f"Prepared smoke fallback dataset: {smoke_dataset}")


if __name__ == "__main__":
    main()
