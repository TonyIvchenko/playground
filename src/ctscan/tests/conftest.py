from __future__ import annotations

from datetime import datetime
from pathlib import Path
import zipfile

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def _make_slice_dataset(
    pixel_array_hu: np.ndarray,
    output_path: Path,
    slice_index: int,
    z_mm: float,
    body_part: str,
    patient_id: str,
    study_uid: str,
    series_uid: str,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "CT"
    ds.BodyPartExamined = body_part
    ds.PatientID = patient_id
    ds.PatientName = ""
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyDate = datetime.utcnow().strftime("%Y%m%d")
    ds.StudyTime = datetime.utcnow().strftime("%H%M%S")
    ds.SeriesNumber = 1
    ds.InstanceNumber = slice_index + 1
    ds.ImagePositionPatient = [0.0, 0.0, float(z_mm)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.SliceThickness = 1.5
    ds.PixelSpacing = [1.0, 1.0]
    ds.Rows = int(pixel_array_hu.shape[0])
    ds.Columns = int(pixel_array_hu.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1
    raw_pixels = (pixel_array_hu - ds.RescaleIntercept).astype(np.int16)
    ds.PixelData = raw_pixels.tobytes()
    ds.save_as(output_path)


@pytest.fixture
def make_ct_zip(tmp_path: Path):
    def _build(
        body_part: str = "CHEST",
        patient_id: str = "demo-patient",
        n_slices: int = 24,
    ) -> Path:
        slices_dir = tmp_path / patient_id
        slices_dir.mkdir(parents=True, exist_ok=True)
        study_uid = generate_uid()
        series_uid = generate_uid()

        zz, yy, xx = np.indices((n_slices, 64, 64))
        volume = np.full((n_slices, 64, 64), -1000.0, dtype=np.float32)
        body = ((yy[0] - 32.0) ** 2) / (28.0**2) + ((xx[0] - 32.0) ** 2) / (24.0**2) <= 1.0
        volume[:, body] = 30.0

        left_lung = ((yy[0] - 32.0) ** 2) / (18.0**2) + ((xx[0] - 22.0) ** 2) / (10.0**2) <= 1.0
        right_lung = ((yy[0] - 32.0) ** 2) / (18.0**2) + ((xx[0] - 42.0) ** 2) / (10.0**2) <= 1.0
        lung_mask = left_lung | right_lung
        volume[:, lung_mask] = -920.0

        issue_specs = [
            ((n_slices // 2, 32, 20), 3, -980.0),  # emphysema
            ((n_slices // 2, 28, 25), 3, -800.0),  # fibrotic pattern
            ((n_slices // 2, 35, 40), 4, -560.0),  # ground-glass
            ((n_slices // 2 + 1, 32, 44), 3, -120.0),  # consolidation
        ]
        for center, radius, value in issue_specs:
            distance = np.sqrt(
                (zz - float(center[0])) ** 2
                + (yy - float(center[1])) ** 2
                + (xx - float(center[2])) ** 2
            )
            volume[distance <= float(radius)] = value

        for slice_index in range(n_slices):
            _make_slice_dataset(
                pixel_array_hu=volume[slice_index],
                output_path=slices_dir / f"slice_{slice_index:03d}.dcm",
                slice_index=slice_index,
                z_mm=slice_index * 1.5,
                body_part=body_part,
                patient_id=patient_id,
                study_uid=study_uid,
                series_uid=series_uid,
            )

        zip_path = tmp_path / f"{patient_id}.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for dicom_path in sorted(slices_dir.glob("*.dcm")):
                archive.write(dicom_path, arcname=dicom_path.name)
        return zip_path

    return _build
