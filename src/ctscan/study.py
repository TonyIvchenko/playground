from __future__ import annotations

from dataclasses import dataclass
import io
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import FileDataset

try:
    import torch
except Exception:  # pragma: no cover - optional runtime fallback
    torch = None
try:
    import segmentation_models_pytorch as _smp
except Exception:  # pragma: no cover - optional runtime fallback
    _smp = None

try:
    from model.unet import UNet2D
    from model.legacy_vgg11_unet import LegacyVGG11UNet
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from src.ctscan.model.unet import UNet2D
    from src.ctscan.model.legacy_vgg11_unet import LegacyVGG11UNet


WINDOW_PRESETS = {
    "lung": (-600.0, 1500.0),
    "mediastinal": (40.0, 400.0),
}
CHEST_TERMS = ("chest", "thorax", "lung", "lungs")
LUNG_LAYER_COLOR = (16, 185, 129)
DEFAULT_LUNG_ALPHA = 0.32
DEFAULT_DAMAGE_ALPHA = 0.45
RULE_ISSUE_CLASSES: tuple[dict[str, Any], ...] = (
    {"id": 1, "key": "emphysema", "label": "Emphysema", "color": (37, 99, 235), "hu_min": -2048.0, "hu_max": -950.0},
    {"id": 2, "key": "fibrotic_pattern", "label": "Fibrotic Pattern", "color": (124, 58, 237), "hu_min": -900.0, "hu_max": -750.0},
    {"id": 3, "key": "ground_glass", "label": "Ground-Glass Opacity", "color": (245, 158, 11), "hu_min": -750.0, "hu_max": -350.0},
    {"id": 4, "key": "consolidation", "label": "Consolidation", "color": (220, 38, 38), "hu_min": -350.0, "hu_max": 120.0},
    {"id": 5, "key": "nodule", "label": "Nodule", "color": (236, 72, 153), "hu_min": -350.0, "hu_max": 200.0},
)
LEGACY_ISSUE_CLASSES: tuple[dict[str, Any], ...] = (
    {"id": 1, "key": "ground_glass", "label": "Ground-Glass", "color": (245, 158, 11)},
    {"id": 2, "key": "consolidation", "label": "Consolidation", "color": (255, 0, 0)},
    {"id": 3, "key": "pleural_effusion", "label": "Pleural Effusion", "color": (0, 0, 255)},
)
MODEL_PATH = Path(os.getenv("CTSCAN_MODEL_PATH", str(Path(__file__).resolve().parent / "model" / "unet.pt")))
MODEL_PATH_EXPLICIT = bool(os.getenv("CTSCAN_MODEL_PATH", "").strip())
MODEL_BATCH_SIZE = max(int(os.getenv("CTSCAN_INFER_BATCH_SIZE", "8")), 1)
LEGACY_MODEL_INPUT_SIZE = 512


_SIMPLEITK = None
_LMInferer = None
_LUNGMASK_ERROR: str | None = None
_LUNGMASK_INFERER = None
_CHECKPOINT_INFO_ATTEMPTED = False
_CHECKPOINT_INFO: dict[str, Any] = {}
_MODEL_LOAD_ATTEMPTED = False
_MODEL_INFERER = None
_MODEL_DEVICE = None
_MODEL_META: dict[str, Any] = {}
_MODEL_ERROR: str | None = None
try:
    import SimpleITK as _SITK
    from lungmask import LMInferer as _LM

    _SIMPLEITK = _SITK
    _LMInferer = _LM
except Exception as exc:  # pragma: no cover - optional dependency
    _LUNGMASK_ERROR = str(exc)


@dataclass
class LoadedStudy:
    volume_hu: np.ndarray
    spacing: tuple[float, float, float]
    metadata: dict[str, Any]
    qc_reasons: list[str]


def segmentation_backend_name() -> str:
    return "lungmask" if _SIMPLEITK is not None and _LMInferer is not None else "threshold"


def segmentation_backend_error() -> str | None:
    return _LUNGMASK_ERROR


def _is_tensor_mapping(payload: Any) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    if torch is None:
        return False
    return all(isinstance(key, str) and hasattr(value, "shape") for key, value in payload.items())


def _inspect_model_checkpoint() -> dict[str, Any]:
    global _CHECKPOINT_INFO_ATTEMPTED
    global _CHECKPOINT_INFO

    if _CHECKPOINT_INFO_ATTEMPTED:
        return dict(_CHECKPOINT_INFO)
    _CHECKPOINT_INFO_ATTEMPTED = True

    info = {
        "path": str(MODEL_PATH),
        "exists": MODEL_PATH.exists(),
        "model_type": None,
        "format": None,
    }
    if os.getenv("CTSCAN_DISABLE_MODEL", "0").strip() in {"1", "true", "TRUE"}:
        info["format"] = "disabled"
        _CHECKPOINT_INFO = info
        return dict(_CHECKPOINT_INFO)
    if torch is None or not MODEL_PATH.exists():
        _CHECKPOINT_INFO = info
        return dict(_CHECKPOINT_INFO)

    try:
        checkpoint = torch.load(str(MODEL_PATH), map_location="cpu")
        if _is_tensor_mapping(checkpoint):
            legacy_keys = {"init_conv.weight", "conv1.weight", "center.conv1.weight", "dec5.conv1.weight"}
            model_type = "legacy_vgg11_unet" if legacy_keys.issubset(set(checkpoint.keys())) else "state_dict"
            info["model_type"] = model_type
            info["format"] = "raw_state_dict"
        elif isinstance(checkpoint, dict):
            info["model_type"] = str(checkpoint.get("model_type", "unet2d")).strip().lower() or "unet2d"
            info["format"] = "checkpoint_dict"
        else:
            info["format"] = type(checkpoint).__name__
    except Exception as exc:  # pragma: no cover - runtime checkpoint inspection errors
        info["error"] = str(exc)

    _CHECKPOINT_INFO = info
    return dict(_CHECKPOINT_INFO)


def _active_issue_classes() -> tuple[dict[str, Any], ...]:
    model_type = str(_inspect_model_checkpoint().get("model_type") or "").strip().lower()
    if model_type == "legacy_vgg11_unet":
        return LEGACY_ISSUE_CLASSES
    return RULE_ISSUE_CLASSES


def _active_issue_by_key() -> dict[str, dict[str, Any]]:
    return {str(item["key"]): item for item in _active_issue_classes()}


def _active_issue_by_id() -> dict[int, dict[str, Any]]:
    return {int(item["id"]): item for item in _active_issue_classes()}


def supported_issues() -> list[dict[str, str | int]]:
    items: list[dict[str, str | int]] = []
    for issue in _active_issue_classes():
        red, green, blue = issue["color"]
        items.append(
            {
                "id": int(issue["id"]),
                "key": str(issue["key"]),
                "label": str(issue["label"]),
                "color": f"#{red:02x}{green:02x}{blue:02x}",
            }
        )
    return items


def model_backend_name() -> str:
    info = _inspect_model_checkpoint()
    model_type = str(info.get("model_type") or "").strip().lower()
    if not model_type:
        model = _get_model_inferer()
        if model is None:
            return "threshold_rules"
        model_type = str(_MODEL_META.get("model_type", "unet2d")).strip().lower()
    if model_type == "unet_pretrained_backbone":
        return "unet_backbone"
    return model_type or "unet2d"


def model_backend_error() -> str | None:
    _get_model_inferer()
    return _MODEL_ERROR


def model_backend_metadata() -> dict[str, Any]:
    _get_model_inferer()
    metadata = dict(_inspect_model_checkpoint())
    metadata.update(_MODEL_META)
    metadata["issue_schema"] = "legacy" if _active_issue_classes() == LEGACY_ISSUE_CLASSES else "rule_based"
    return metadata


def _resolve_model_device():
    if torch is None:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_volume_for_model(volume_hu: np.ndarray) -> np.ndarray:
    clipped = np.clip(volume_hu, -1000.0, 400.0)
    normalized = (clipped + 1000.0) / 1400.0
    return normalized.astype(np.float32, copy=False)


def _normalize_slice_for_legacy_model(slice_hu: np.ndarray) -> np.ndarray:
    lower = float(np.min(slice_hu))
    upper = float(np.max(slice_hu))
    scale = max(upper - lower, 1e-6)
    scaled = ((slice_hu.astype(np.float32, copy=False) - lower) / scale) * 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _resize_image_array(image: np.ndarray, size_xy: tuple[int, int], resample: int) -> np.ndarray:
    width, height = int(size_xy[0]), int(size_xy[1])
    return np.asarray(Image.fromarray(image).resize((width, height), resample=resample))


def _get_model_inferer():
    global _MODEL_LOAD_ATTEMPTED
    global _MODEL_INFERER
    global _MODEL_DEVICE
    global _MODEL_META
    global _MODEL_ERROR

    if _MODEL_INFERER is not None:
        return _MODEL_INFERER
    if _MODEL_LOAD_ATTEMPTED:
        return None
    _MODEL_LOAD_ATTEMPTED = True

    if os.getenv("CTSCAN_DISABLE_MODEL", "0").strip() in {"1", "true", "TRUE"}:
        _MODEL_ERROR = "Model loading disabled by CTSCAN_DISABLE_MODEL."
        return None
    if torch is None:
        _MODEL_ERROR = "PyTorch is not installed."
        return None
    if not MODEL_PATH.exists():
        _MODEL_ERROR = f"Checkpoint not found at {MODEL_PATH}."
        return None

    try:
        checkpoint = torch.load(str(MODEL_PATH), map_location="cpu")
        checkpoint_info = _inspect_model_checkpoint()
        issue_by_id = _active_issue_by_id()

        if _is_tensor_mapping(checkpoint):
            state_dict = checkpoint
            model_type = "legacy_vgg11_unet"
            model = LegacyVGG11UNet(
                in_channels=1,
                out_channels=max(issue_by_id.keys()) + 1,
                batch_norm=True,
                upscale_mode="bilinear",
                encoder_weights=None,
            )
            base_channels = None
            encoder_name = "vgg11"
            encoder_weights = "legacy"
            model_version = ""
            best_epoch = None
            best_score = None
            input_size = LEGACY_MODEL_INPUT_SIZE
        elif isinstance(checkpoint, dict):
            model_type = str(checkpoint.get("model_type", "unet2d")).strip().lower() or "unet2d"
            state_dict = checkpoint.get("state_dict")
            if not isinstance(state_dict, dict):
                raise ValueError("Checkpoint is missing `state_dict`.")

            if model_type == "unet_pretrained_backbone":
                if _smp is None:
                    raise RuntimeError("segmentation_models_pytorch is required for pretrained-backbone checkpoints.")
                encoder_name = str(checkpoint.get("encoder_name", "resnet34"))
                in_channels = int(checkpoint.get("in_channels", 1))
                num_classes = int(checkpoint.get("classes", max(issue_by_id.keys()) + 1))
                model = _smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=in_channels,
                    classes=num_classes,
                )
                base_channels = None
                encoder_weights = checkpoint.get("encoder_weights")
                input_size = None
            elif model_type == "legacy_vgg11_unet":
                model = LegacyVGG11UNet(
                    in_channels=1,
                    out_channels=max(issue_by_id.keys()) + 1,
                    batch_norm=bool((checkpoint.get("model_config") or {}).get("batch_norm", True)),
                    upscale_mode=str((checkpoint.get("model_config") or {}).get("upscale_mode", "bilinear")),
                    encoder_weights=None,
                )
                base_channels = None
                encoder_name = "vgg11"
                encoder_weights = checkpoint.get("encoder_weights", "imagenet")
                input_size = int((checkpoint.get("model_config") or {}).get("image_size", LEGACY_MODEL_INPUT_SIZE))
            else:
                model_config = checkpoint.get("model_config") or {}
                in_channels = int(model_config.get("in_channels", 1))
                num_classes = int(model_config.get("num_classes", max(issue_by_id.keys()) + 1))
                base_channels = int(model_config.get("base_channels", 32))
                model = UNet2D(
                    in_channels=in_channels,
                    num_classes=num_classes,
                    base_channels=base_channels,
                )
                encoder_name = None
                encoder_weights = None
                input_size = None
            model_version = str(checkpoint.get("model_version", ""))
            best_epoch = int(checkpoint.get("best_epoch", 0)) if checkpoint.get("best_epoch") is not None else None
            best_score = float(checkpoint.get("best_score", 0.0)) if checkpoint.get("best_score") is not None else None
        else:
            raise ValueError("Checkpoint payload is not supported.")

        model.load_state_dict(state_dict, strict=True)
        device = _resolve_model_device()
        if device is None:
            raise RuntimeError("Could not resolve model device.")
        model.to(device)
        model.eval()

        _MODEL_INFERER = model
        _MODEL_DEVICE = device
        _MODEL_META = {
            "path": str(MODEL_PATH),
            "model_version": model_version,
            "model_type": model_type,
            "format": checkpoint_info.get("format"),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "encoder_name": encoder_name,
            "encoder_weights": encoder_weights,
            "num_classes": max(issue_by_id.keys()) + 1,
            "base_channels": base_channels,
            "input_size": input_size,
            "device": str(device),
        }
        _MODEL_ERROR = None
        return _MODEL_INFERER
    except Exception as exc:  # pragma: no cover - runtime checkpoint errors
        _MODEL_ERROR = str(exc)
        return None


def _predict_issue_labels_model(volume_hu: np.ndarray) -> np.ndarray | None:
    model = _get_model_inferer()
    if model is None or torch is None or _MODEL_DEVICE is None:
        return None

    model_type = str(_MODEL_META.get("model_type", "")).strip().lower()
    if model_type == "legacy_vgg11_unet":
        z_dim, height, width = volume_hu.shape
        output = np.zeros((z_dim, height, width), dtype=np.uint8)
        input_size = int(_MODEL_META.get("input_size") or LEGACY_MODEL_INPUT_SIZE)

        with torch.no_grad():
            for start in range(0, z_dim, MODEL_BATCH_SIZE):
                end = min(z_dim, start + MODEL_BATCH_SIZE)
                batch_arrays: list[np.ndarray] = []
                for slice_hu in volume_hu[start:end]:
                    scaled = _normalize_slice_for_legacy_model(slice_hu)
                    resized = _resize_image_array(
                        scaled,
                        size_xy=(input_size, input_size),
                        resample=Image.Resampling.BILINEAR,
                    )
                    batch_arrays.append((resized.astype(np.float32) / 255.0) - 0.5)

                batch = torch.from_numpy(np.stack(batch_arrays, axis=0)[:, None, :, :]).to(_MODEL_DEVICE)
                logits = model(batch)
                predictions = torch.argmax(logits, dim=1).to("cpu").numpy().astype(np.uint8)
                for offset, predicted in enumerate(predictions):
                    output[start + offset] = _resize_image_array(
                        predicted,
                        size_xy=(width, height),
                        resample=Image.Resampling.NEAREST,
                    ).astype(np.uint8, copy=False)
        return output

    normalized = _normalize_volume_for_model(volume_hu)
    z_dim, height, width = normalized.shape
    output = np.zeros((z_dim, height, width), dtype=np.uint8)

    with torch.no_grad():
        for start in range(0, z_dim, MODEL_BATCH_SIZE):
            end = min(z_dim, start + MODEL_BATCH_SIZE)
            batch = torch.from_numpy(normalized[start:end, None, :, :]).to(_MODEL_DEVICE)
            logits = model(batch)
            predictions = torch.argmax(logits, dim=1).to("cpu").numpy().astype(np.uint8)
            output[start:end] = predictions
    return output


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


def _body_bounds(volume_hu: np.ndarray) -> tuple[slice, slice, slice]:
    body_mask = volume_hu > -900.0
    coords = np.argwhere(body_mask)
    if len(coords) == 0:
        shape = volume_hu.shape
        return slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return slice(int(mins[0]), int(maxs[0])), slice(int(mins[1]), int(maxs[1])), slice(int(mins[2]), int(maxs[2]))


def _threshold_lung_mask(volume_hu: np.ndarray) -> np.ndarray:
    z_slice, y_slice, x_slice = _body_bounds(volume_hu)
    mask = np.zeros_like(volume_hu, dtype=bool)
    body_crop = volume_hu[z_slice, y_slice, x_slice]
    lung_crop = (body_crop < -320.0) & (body_crop > -980.0)
    mask[z_slice, y_slice, x_slice] = lung_crop
    return mask


def _get_lungmask_inferer():
    global _LUNGMASK_INFERER
    if _SIMPLEITK is None or _LMInferer is None:
        return None
    if _LUNGMASK_INFERER is None:
        _LUNGMASK_INFERER = _LMInferer(modelname="R231")
    return _LUNGMASK_INFERER


def segment_lungs(volume_hu: np.ndarray) -> tuple[np.ndarray, str]:
    inferer = _get_lungmask_inferer()
    if inferer is not None and _SIMPLEITK is not None:
        try:
            image = _SIMPLEITK.GetImageFromArray(volume_hu.astype(np.int16, copy=False))
            prediction = inferer.apply(image)
            mask = np.asarray(prediction) > 0
            if mask.any():
                return mask.astype(bool), "lungmask"
        except Exception:  # pragma: no cover - backend fallback path
            pass
    return _threshold_lung_mask(volume_hu), "threshold"


def _segment_issues_threshold(volume_hu: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
    labels = np.zeros(volume_hu.shape, dtype=np.uint8)
    for issue in RULE_ISSUE_CLASSES:
        class_mask = (
            lung_mask
            & (volume_hu >= float(issue["hu_min"]))
            & (volume_hu < float(issue["hu_max"]))
        )
        labels[class_mask] = int(issue["id"])
    return labels


def segment_issues(volume_hu: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
    predicted = _predict_issue_labels_model(volume_hu)
    if predicted is not None:
        labels = predicted.astype(np.uint8, copy=False)
        valid_ids = set(_active_issue_by_id().keys())
        labels[~np.isin(labels, list(valid_ids))] = 0
        labels[~lung_mask] = 0
        if int((labels > 0).sum()) > 0:
            return labels
    if MODEL_PATH_EXPLICIT:
        return np.zeros(volume_hu.shape, dtype=np.uint8)
    if _active_issue_classes() == LEGACY_ISSUE_CLASSES:
        return np.zeros(volume_hu.shape, dtype=np.uint8)
    return _segment_issues_threshold(volume_hu, lung_mask)


def issue_volume_stats(
    labels: np.ndarray,
    lung_mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> list[dict[str, Any]]:
    lung_voxels = max(int(lung_mask.sum()), 1)
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    rows: list[dict[str, Any]] = []
    for issue in _active_issue_classes():
        issue_id = int(issue["id"])
        issue_voxels = int((labels == issue_id).sum())
        issue_volume_ml = (issue_voxels * voxel_volume_mm3) / 1000.0
        rows.append(
            {
                "id": issue_id,
                "issue_key": str(issue["key"]),
                "issue": str(issue["label"]),
                "color": issue["color"],
                "voxels": issue_voxels,
                "volume_ml": float(issue_volume_ml),
                "lung_percent": float((issue_voxels / lung_voxels) * 100.0),
            }
        )
    return rows


def issue_slice_stats(labels: np.ndarray, lung_mask: np.ndarray, slice_index: int) -> list[dict[str, Any]]:
    clamped_slice = int(np.clip(slice_index, 0, labels.shape[0] - 1))
    slice_labels = labels[clamped_slice]
    slice_lung = lung_mask[clamped_slice]
    lung_pixels = max(int(slice_lung.sum()), 1)
    rows: list[dict[str, Any]] = []
    for issue in _active_issue_classes():
        issue_id = int(issue["id"])
        issue_pixels = int((slice_labels == issue_id).sum())
        rows.append(
            {
                "id": issue_id,
                "issue_key": str(issue["key"]),
                "issue": str(issue["label"]),
                "color": issue["color"],
                "pixels": issue_pixels,
                "slice_percent": float((issue_pixels / lung_pixels) * 100.0),
            }
        )
    return rows


def window_slice(slice_hu: np.ndarray, preset: str) -> np.ndarray:
    level, width = WINDOW_PRESETS.get(preset, WINDOW_PRESETS["lung"])
    lower = level - width / 2.0
    upper = level + width / 2.0
    image = np.clip((slice_hu - lower) / max(width, 1.0), 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def render_segmentation_slice(
    volume_hu: np.ndarray,
    labels: np.ndarray,
    lung_mask: np.ndarray,
    slice_index: int,
    preset: str,
    selected_issues: list[str] | None = None,
    show_lung_layer: bool = True,
    show_damage_layer: bool = True,
    lung_alpha: float = DEFAULT_LUNG_ALPHA,
    damage_alpha: float = DEFAULT_DAMAGE_ALPHA,
) -> Image.Image:
    clamped_index = int(np.clip(slice_index, 0, volume_hu.shape[0] - 1))
    grayscale = window_slice(volume_hu[clamped_index], preset)
    rgb = np.stack([grayscale, grayscale, grayscale], axis=2).astype(np.float32)
    slice_labels = labels[clamped_index]
    slice_lung = lung_mask[clamped_index]

    if show_lung_layer:
        lung_alpha = float(np.clip(lung_alpha, 0.0, 1.0))
        lung_color = np.asarray(LUNG_LAYER_COLOR, dtype=np.float32)
        rgb[slice_lung] = rgb[slice_lung] * (1.0 - lung_alpha) + lung_color * lung_alpha

    if show_damage_layer:
        damage_alpha = float(np.clip(damage_alpha, 0.0, 1.0))
        selected_keys = set(selected_issues or [])
        for issue in _active_issue_classes():
            issue_key = str(issue["key"])
            if selected_keys and issue_key not in selected_keys:
                continue
            class_mask = slice_labels == int(issue["id"])
            if not class_mask.any():
                continue
            color = np.array(issue["color"], dtype=np.float32)
            rgb[class_mask] = rgb[class_mask] * (1.0 - damage_alpha) + color * damage_alpha
    return Image.fromarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8), mode="RGB")


def write_temp_image(image: Image.Image) -> str:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(handle.name)
    return handle.name


def write_temp_bundle(volume_hu: np.ndarray, labels: np.ndarray, lung_mask: np.ndarray) -> str:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
    np.savez_compressed(
        handle.name,
        volume_hu=volume_hu.astype(np.float32),
        labels=labels.astype(np.uint8),
        lung_mask=lung_mask.astype(np.uint8),
    )
    return handle.name


def read_temp_bundle(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path)
    return (
        payload["volume_hu"].astype(np.float32),
        payload["labels"].astype(np.uint8),
        payload["lung_mask"].astype(bool),
    )


def issue_rows_for_table(rows: list[dict[str, Any]]) -> list[list[Any]]:
    table_rows: list[list[Any]] = []
    for row in rows:
        table_rows.append(
            [
                row["issue"],
                round(float(row["lung_percent"]), 4),
                round(float(row["volume_ml"]), 4),
                int(row["voxels"]),
            ]
        )
    return table_rows


def slice_rows_for_table(rows: list[dict[str, Any]]) -> list[list[Any]]:
    table_rows: list[list[Any]] = []
    for row in rows:
        table_rows.append(
            [
                row["issue"],
                round(float(row["slice_percent"]), 4),
                int(row["pixels"]),
            ]
        )
    return table_rows


def blank_viewer_image() -> str:
    return write_temp_image(Image.new("RGB", (512, 512), "#111111"))


def load_study_from_zip_bytes(zip_bytes: bytes) -> LoadedStudy:
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
        qc_reasons.append("Study has too few slices for reliable segmentation.")

    slices = []
    for ds in selected:
        pixels = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(pixels * slope + intercept)

    volume_hu = np.stack(slices, axis=0).astype(np.float32)
    spacing = (spacing_z, spacing_y, spacing_x)

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
    )
