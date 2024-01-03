from __future__ import annotations

from dataclasses import dataclass
import io
import tempfile
from typing import Any
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import pydicom
from pydicom.dataset import FileDataset


TARGET_SPACING = (1.0, 1.0, 1.0)
WINDOW_PRESETS = {
    "lung": (-600.0, 1500.0),
    "mediastinal": (40.0, 400.0),
}
CHEST_TERMS = ("chest", "thorax", "lung", "lungs")


@dataclass
class LoadedStudy:
    volume_hu: np.ndarray
    spacing: tuple[float, float, float]
    metadata: dict[str, Any]
    qc_reasons: list[str]
    sop_uid_to_index: dict[str, int]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _series_sort_key(ds: FileDataset) -> tuple[float, float]:
    position = getattr(ds, "ImagePositionPatient", None)
    if position and len(position) >= 3:
        return (float(position[2]), float(getattr(ds, "InstanceNumber", 0)))
    return (float(getattr(ds, "InstanceNumber", 0)), 0.0)


def _slice_spacing(slices: list[FileDataset]) -> float:
    if len(slices) >= 2:
        z_values = []
        for ds in slices:
            position = getattr(ds, "ImagePositionPatient", None)
            if position and len(position) >= 3:
                z_values.append(float(position[2]))
        if len(z_values) >= 2:
            diffs = np.diff(sorted(z_values))
            diffs = np.abs(diffs[diffs != 0])
            if len(diffs):
                return float(np.median(diffs))
    thickness = getattr(slices[0], "SliceThickness", 1.0)
    return max(float(thickness), 0.5)


def _pixel_spacing(ds: FileDataset) -> tuple[float, float]:
    spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    if len(spacing) >= 2:
        return max(float(spacing[0]), 0.5), max(float(spacing[1]), 0.5)
    return 1.0, 1.0


def _resample_axis(volume: np.ndarray, axis: int, old_spacing: float, new_spacing: float) -> np.ndarray:
    if abs(old_spacing - new_spacing) < 1e-6:
        return volume
    old_size = volume.shape[axis]
    new_size = max(1, int(round(old_size * old_spacing / new_spacing)))
    source = np.linspace(0.0, old_size - 1, old_size)
    target = np.linspace(0.0, old_size - 1, new_size)
    moved = np.moveaxis(volume, axis, 0)
    out = np.empty((new_size,) + moved.shape[1:], dtype=np.float32)
    for index in np.ndindex(moved.shape[1:]):
        out[(slice(None),) + index] = np.interp(target, source, moved[(slice(None),) + index])
    return np.moveaxis(out, 0, axis)


def resample_volume(volume_hu: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, tuple[float, float, float]]:
    resampled = volume_hu.astype(np.float32, copy=False)
    current_spacing = spacing
    for axis, (old_spacing, new_spacing) in enumerate(zip(current_spacing, TARGET_SPACING)):
        resampled = _resample_axis(resampled, axis=axis, old_spacing=float(old_spacing), new_spacing=float(new_spacing))
    return resampled.astype(np.float32), TARGET_SPACING


def _body_bounds(volume_hu: np.ndarray) -> tuple[slice, slice, slice]:
    body_mask = volume_hu > -900.0
    coords = np.argwhere(body_mask)
    if len(coords) == 0:
        shape = volume_hu.shape
        return slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return slice(int(mins[0]), int(maxs[0])), slice(int(mins[1]), int(maxs[1])), slice(int(mins[2]), int(maxs[2]))


def estimate_lung_mask(volume_hu: np.ndarray) -> np.ndarray:
    z_slice, y_slice, x_slice = _body_bounds(volume_hu)
    mask = np.zeros_like(volume_hu, dtype=bool)
    body_crop = volume_hu[z_slice, y_slice, x_slice]
    lung_crop = (body_crop < -320.0) & (body_crop > -980.0)
    mask[z_slice, y_slice, x_slice] = lung_crop
    return mask


def _expand_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    expanded = mask.copy()
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                src_z0 = max(0, -dz)
                src_y0 = max(0, -dy)
                src_x0 = max(0, -dx)
                src_z1 = mask.shape[0] - max(0, dz)
                src_y1 = mask.shape[1] - max(0, dy)
                src_x1 = mask.shape[2] - max(0, dx)

                dst_z0 = max(0, dz)
                dst_y0 = max(0, dy)
                dst_x0 = max(0, dx)
                dst_z1 = dst_z0 + (src_z1 - src_z0)
                dst_y1 = dst_y0 + (src_y1 - src_y0)
                dst_x1 = dst_x0 + (src_x1 - src_x0)

                expanded[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
    return expanded


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int, int, int]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return (0, 0, 0, 0, 0, 0)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return (
        int(mins[0]),
        int(mins[1]),
        int(mins[2]),
        int(maxs[0]),
        int(maxs[1]),
        int(maxs[2]),
    )


def extract_patch(volume_hu: np.ndarray, center: tuple[int, int, int], patch_shape: tuple[int, int, int]) -> np.ndarray:
    dz, dy, dx = patch_shape
    cz, cy, cx = center
    z0 = cz - dz // 2
    y0 = cy - dy // 2
    x0 = cx - dx // 2
    patch = np.full(patch_shape, -1000.0, dtype=np.float32)
    z1 = z0 + dz
    y1 = y0 + dy
    x1 = x0 + dx

    src_z0 = max(0, z0)
    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_z1 = min(volume_hu.shape[0], z1)
    src_y1 = min(volume_hu.shape[1], y1)
    src_x1 = min(volume_hu.shape[2], x1)

    dst_z0 = src_z0 - z0
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0

    patch[
        dst_z0 : dst_z0 + (src_z1 - src_z0),
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] = volume_hu[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
    return patch


def generate_candidates(volume_hu: np.ndarray, lung_mask: np.ndarray, max_candidates: int = 12) -> list[dict[str, Any]]:
    search_mask = _expand_mask(lung_mask, radius=3)
    signal = np.where(search_mask, volume_hu, -2000.0)
    threshold_mask = signal > -250.0
    coords = np.argwhere(threshold_mask)
    if len(coords) == 0:
        return []

    scores = signal[threshold_mask]
    order = np.argsort(scores)[::-1]
    taken: list[np.ndarray] = []
    candidates: list[dict[str, Any]] = []

    for idx in order:
        coord = coords[idx]
        if any(np.linalg.norm(coord - prev) < 8.0 for prev in taken):
            continue
        cz, cy, cx = int(coord[0]), int(coord[1]), int(coord[2])
        local_patch = extract_patch(threshold_mask.astype(np.uint8), (cz, cy, cx), (8, 12, 12))
        bbox = _bbox_from_mask(local_patch > 0)
        voxel_count = int((local_patch > 0).sum())
        diameter_mm = float(max(3.0, round((voxel_count ** (1.0 / 3.0)) * 1.8, 2)))
        volume_mm3 = float(max(1.0, voxel_count))
        patch = extract_patch(volume_hu, (cz, cy, cx), (16, 16, 16))
        candidates.append(
            {
                "lesion_id": f"lesion-{len(candidates) + 1}",
                "center": (cz, cy, cx),
                "slice_index": cz,
                "bbox": {
                    "z0": max(0, cz - 4 + bbox[0]),
                    "y0": max(0, cy - 6 + bbox[1]),
                    "x0": max(0, cx - 6 + bbox[2]),
                    "z1": min(volume_hu.shape[0] - 1, cz - 4 + bbox[3]),
                    "y1": min(volume_hu.shape[1] - 1, cy - 6 + bbox[4]),
                    "x1": min(volume_hu.shape[2] - 1, cx - 6 + bbox[5]),
                },
                "diameter_mm": diameter_mm,
                "volume_mm3": volume_mm3,
                "mean_hu": float(patch.mean()),
                "max_hu": float(patch.max()),
                "patch": patch,
            }
        )
        taken.append(coord)
        if len(candidates) >= max_candidates:
            break

    return sorted(candidates, key=lambda item: item["slice_index"])


def load_study_from_zip_bytes(zip_bytes: bytes, resample: bool = True) -> LoadedStudy:
    qc_reasons: list[str] = []
    datasets: list[FileDataset] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            with archive.open(member, "r") as handle:
                raw = handle.read()
            try:
                ds = pydicom.dcmread(io.BytesIO(raw), force=True)
            except Exception:
                continue
            if getattr(ds, "Modality", "") != "CT":
                continue
            if "PixelData" not in ds:
                continue
            datasets.append(ds)

    if not datasets:
        raise ValueError("Zip did not contain readable CT DICOM slices.")

    series_groups: dict[str, list[FileDataset]] = {}
    for ds in datasets:
        series_uid = _safe_str(getattr(ds, "SeriesInstanceUID", "")) or "default"
        series_groups.setdefault(series_uid, []).append(ds)

    selected = max(series_groups.values(), key=len)
    selected.sort(key=_series_sort_key)

    spacing_y, spacing_x = _pixel_spacing(selected[0])
    spacing_z = _slice_spacing(selected)

    body_part = " ".join(
        filter(
            None,
            [
                _safe_str(getattr(selected[0], "BodyPartExamined", "")),
                _safe_str(getattr(selected[0], "StudyDescription", "")),
                _safe_str(getattr(selected[0], "SeriesDescription", "")),
            ],
        )
    ).lower()
    if body_part and not any(term in body_part for term in CHEST_TERMS):
        qc_reasons.append("Study metadata does not look like a chest CT.")

    if len(selected) < 16:
        qc_reasons.append("Study has too few slices for reliable nodule review.")

    slices = []
    for ds in selected:
        pixels = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(pixels * slope + intercept)

    sop_uid_to_index = {
        _safe_str(getattr(ds, "SOPInstanceUID", "")): index for index, ds in enumerate(selected)
    }
    volume_hu = np.stack(slices, axis=0).astype(np.float32)
    spacing = (spacing_z, spacing_y, spacing_x)
    if resample:
        volume_hu, spacing = resample_volume(volume_hu, spacing)

    metadata = {
        "patient_id": _safe_str(getattr(selected[0], "PatientID", "")),
        "study_instance_uid": _safe_str(getattr(selected[0], "StudyInstanceUID", "")),
        "series_instance_uid": _safe_str(getattr(selected[0], "SeriesInstanceUID", "")),
        "body_part_examined": _safe_str(getattr(selected[0], "BodyPartExamined", "")),
        "series_description": _safe_str(getattr(selected[0], "SeriesDescription", "")),
        "study_description": _safe_str(getattr(selected[0], "StudyDescription", "")),
        "slice_count": int(volume_hu.shape[0]),
        "rows": int(volume_hu.shape[1]),
        "cols": int(volume_hu.shape[2]),
        "spacing": [float(x) for x in spacing],
    }

    return LoadedStudy(
        volume_hu=volume_hu,
        spacing=spacing,
        metadata=metadata,
        qc_reasons=qc_reasons,
        sop_uid_to_index=sop_uid_to_index,
    )


def match_prior_findings(
    current_findings: list[dict[str, Any]],
    prior_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not prior_findings:
        return current_findings

    prior_centers = [np.array(item["center"], dtype=np.float32) for item in prior_findings]
    for finding in current_findings:
        center = np.array(finding["center"], dtype=np.float32)
        distances = [float(np.linalg.norm(center - prior_center)) for prior_center in prior_centers]
        best_index = int(np.argmin(distances))
        best_prior = prior_findings[best_index]
        distance = distances[best_index]
        if distance <= 12.0:
            prior_diameter = float(best_prior["diameter_mm"])
            current_diameter = float(finding["diameter_mm"])
            finding["growth"] = {
                "matched_prior_lesion_id": str(best_prior["lesion_id"]),
                "diameter_delta_mm": round(current_diameter - prior_diameter, 2),
                "volume_delta_mm3": round(float(finding["volume_mm3"]) - float(best_prior["volume_mm3"]), 2),
                "doubling_time_days_estimate": round(max(30.0, 365.0 / max(current_diameter / max(prior_diameter, 1.0), 1.0)), 2),
            }
        else:
            finding["growth"] = None
    return current_findings


def window_slice(slice_hu: np.ndarray, preset: str) -> np.ndarray:
    level, width = WINDOW_PRESETS.get(preset, WINDOW_PRESETS["lung"])
    lower = level - width / 2.0
    upper = level + width / 2.0
    image = np.clip((slice_hu - lower) / max(width, 1.0), 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def render_slice_image(
    volume_hu: np.ndarray,
    findings: list[dict[str, Any]],
    slice_index: int,
    preset: str,
    selected_lesion_id: str | None = None,
) -> Image.Image:
    clamped_index = int(np.clip(slice_index, 0, volume_hu.shape[0] - 1))
    image = Image.fromarray(window_slice(volume_hu[clamped_index], preset), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    for finding in findings:
        bbox = finding["bbox"]
        if not (bbox["z0"] <= clamped_index <= bbox["z1"]):
            continue
        color = "#ff6b57" if finding["lesion_id"] == selected_lesion_id else "#ffd166"
        draw.rectangle(
            [
                (int(bbox["x0"]), int(bbox["y0"])),
                (int(bbox["x1"]), int(bbox["y1"])),
            ],
            outline=color,
            width=2,
        )
        draw.text((int(bbox["x0"]) + 2, int(bbox["y0"]) + 2), str(finding["lesion_id"]), fill=color)
    return image


def write_temp_image(image: Image.Image) -> str:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(handle.name)
    return handle.name


def write_temp_volume(volume_hu: np.ndarray) -> str:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(handle.name, volume_hu.astype(np.float32))
    return handle.name


def read_temp_volume(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def finding_rows(findings: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for finding in findings:
        growth = finding.get("growth") or {}
        rows.append(
            [
                finding["lesion_id"],
                int(finding["slice_index"]),
                round(float(finding["diameter_mm"]), 2),
                round(float(finding["volume_mm3"]), 2),
                round(float(finding["nodule_probability"]), 3),
                round(float(finding["malignancy_risk"]), 3),
                "" if not growth else round(float(growth["diameter_delta_mm"]), 2),
            ]
        )
    return rows


def blank_viewer_image() -> str:
    image = Image.new("RGB", (512, 512), "#111111")
    return write_temp_image(image)
