from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

from src.ctscan.models.nodules import PATCH_SHAPE
from src.ctscan.scripts.nodules.download_data import (
    build_smoke_training_dataset,
    collect_real_annotation_manifest,
    fetch_series_uid,
    parse_lidc_xml_bytes,
    write_dataset_manifest,
)


def test_write_dataset_manifest_contains_expected_sources(tmp_path: Path):
    manifest_path = write_dataset_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = {item["name"] for item in payload}
    assert {"LIDC-IDRI", "LUNA16", "LNDb"} <= names


def test_build_smoke_training_dataset_shapes(tmp_path: Path):
    dataset_path = build_smoke_training_dataset(tmp_path, rows=12)
    bundle = np.load(dataset_path)
    assert bundle["patches"].shape == (12, 1, *PATCH_SHAPE)
    assert bundle["nodule_target"].shape == (12,)
    assert bundle["malignancy_target"].shape == (12,)
    assert bundle["malignancy_mask"].shape == (12,)
    assert bundle["nodule_weight"].shape == (12,)
    assert bundle["malignancy_weight"].shape == (12,)
    assert bundle["series_ids"].shape == (12,)


def test_fetch_series_uid_uses_largest_series(monkeypatch):
    monkeypatch.setattr(
        "src.ctscan.scripts.nodules.download_data._fetch_json",
        lambda *_args, **_kwargs: [
            {"SeriesInstanceUID": "small", "ImageCount": 24},
            {"SeriesInstanceUID": "large", "ImageCount": 133},
        ],
    )
    assert fetch_series_uid("LIDC-IDRI-0001") == "large"


def test_parse_lidc_xml_bytes_extracts_series_and_malignancy():
    xml_payload = b"""
    <LidcReadMessage xmlns="http://www.nih.gov">
      <ResponseHeader>
        <SeriesInstanceUid>1.2.3</SeriesInstanceUid>
        <StudyInstanceUID>4.5.6</StudyInstanceUID>
      </ResponseHeader>
      <readingSession>
        <servicingRadiologistID>reader-a</servicingRadiologistID>
        <unblindedReadNodule>
          <noduleID>n1</noduleID>
          <roi>
            <imageSOP_UID>sop-1</imageSOP_UID>
            <inclusion>TRUE</inclusion>
            <edgeMap>
              <xCoord>20</xCoord>
              <yCoord>30</yCoord>
            </edgeMap>
            <edgeMap>
              <xCoord>24</xCoord>
              <yCoord>34</yCoord>
            </edgeMap>
          </roi>
          <characteristics>
            <malignancy>4</malignancy>
          </characteristics>
        </unblindedReadNodule>
        <nonNodule>
          <nonNoduleID>nn-1</nonNoduleID>
          <imageSOP_UID>sop-2</imageSOP_UID>
          <locus>
            <xCoord>40</xCoord>
            <yCoord>50</yCoord>
          </locus>
        </nonNodule>
      </readingSession>
    </LidcReadMessage>
    """
    parsed = parse_lidc_xml_bytes(xml_payload)
    assert parsed is not None
    assert parsed["series_instance_uid"] == "1.2.3"
    assert parsed["study_instance_uid"] == "4.5.6"
    assert parsed["annotations"][0]["malignancy"] == 4
    assert parsed["annotations"][0]["sop_uids"] == ["sop-1", "sop-1"]
    assert parsed["non_nodules"][0]["sop_uid"] == "sop-2"
    assert parsed["non_nodules"][0]["x_coord"] == 40


def test_collect_real_annotation_manifest_merges_duplicate_series(tmp_path: Path):
    xml_payload = b"""
    <LidcReadMessage xmlns="http://www.nih.gov">
      <ResponseHeader>
        <SeriesInstanceUid>1.2.3</SeriesInstanceUid>
        <StudyInstanceUID>4.5.6</StudyInstanceUID>
      </ResponseHeader>
      <readingSession>
        <servicingRadiologistID>reader-a</servicingRadiologistID>
        <unblindedReadNodule>
          <noduleID>n1</noduleID>
          <roi>
            <imageSOP_UID>sop-1</imageSOP_UID>
            <inclusion>TRUE</inclusion>
            <edgeMap><xCoord>20</xCoord><yCoord>30</yCoord></edgeMap>
          </roi>
          <characteristics><malignancy>4</malignancy></characteristics>
        </unblindedReadNodule>
      </readingSession>
    </LidcReadMessage>
    """
    xml_payload_2 = b"""
    <LidcReadMessage xmlns="http://www.nih.gov">
      <ResponseHeader>
        <SeriesInstanceUid>1.2.3</SeriesInstanceUid>
        <StudyInstanceUID>4.5.6</StudyInstanceUID>
      </ResponseHeader>
      <readingSession>
        <servicingRadiologistID>reader-b</servicingRadiologistID>
        <nonNodule>
          <nonNoduleID>nn-1</nonNoduleID>
          <imageSOP_UID>sop-2</imageSOP_UID>
          <locus><xCoord>40</xCoord><yCoord>50</yCoord></locus>
        </nonNodule>
      </readingSession>
    </LidcReadMessage>
    """
    xml_path = tmp_path / "lidc.xml.zip"
    with zipfile.ZipFile(xml_path, "w") as archive:
        archive.writestr("a.xml", xml_payload)
        archive.writestr("b.xml", xml_payload_2)

    manifest = collect_real_annotation_manifest(xml_path)
    assert len(manifest) == 1
    assert len(manifest[0]["annotations"]) == 1
    assert len(manifest[0]["non_nodules"]) == 1
    assert manifest[0]["xml_names"] == ["a.xml", "b.xml"]
