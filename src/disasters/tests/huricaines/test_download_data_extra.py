from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

import src.disasters.scripts.huricaines.download_data as huricaines_download


def test_discover_latest_hurdat2_url(monkeypatch):
    html = b"<a href='hurdat2-1851-2023-040224.txt'>a</a><a href='hurdat2-1851-2024-051124.txt'>b</a>"

    class FakeResponse:
        def read(self) -> bytes:
            return html

    monkeypatch.setattr(huricaines_download, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    latest = huricaines_download.discover_latest_hurdat2_url("https://www.nhc.noaa.gov/data/hurdat/")
    assert latest.endswith("hurdat2-1851-2024-051124.txt")


def test_load_ibtracs_maps_fields(tmp_path: Path):
    path = tmp_path / "ibtracs.csv"
    csv = (
        "SID,ISO_TIME,LAT,LON,USA_ATCF_ID,USA_WIND,USA_PRES,WMO_WIND,WMO_PRES\n"
        "meta,meta,meta,meta,meta,meta,meta,meta,meta\n"
        "NA012000,2000-08-01 00:00:00,20.1,-60.2,AL012000,55,1000,54,1001\n"
    )
    path.write_text(csv, encoding="utf-8")

    df = huricaines_download.load_ibtracs(path)
    assert len(df) == 1
    assert df.iloc[0]["storm_id"] == "AL012000"
    assert float(df.iloc[0]["vmax_kt"]) == 55.0
    assert float(df.iloc[0]["min_pressure_mb"]) == 1000.0
