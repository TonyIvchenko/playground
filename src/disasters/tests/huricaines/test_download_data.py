from pathlib import Path

import pandas as pd

import src.disasters.scripts.huricaines.download_data as huricaines_download
from src.disasters.scripts.huricaines.download_data import load_hurdat2, merge_sources, parse_lat_lon


def test_parse_lat_lon():
    assert parse_lat_lon("28.0N") == 28.0
    assert parse_lat_lon("95.4W") == -95.4
    assert parse_lat_lon("12.5S") == -12.5
    assert parse_lat_lon("45.0E") == 45.0
    assert parse_lat_lon("") is None


def test_load_hurdat2_parses_atl_rows(tmp_path: Path):
    sample = "\n".join(
        [
            "AL011851,            UNNAMED,     2,",
            "18510625, 0000,  , HU, 28.0N,  94.8W,  80,  985, -999, -999",
            "18510625, 0600,  , HU, 28.0N,  95.4W,  80,  980, -999, -999",
        ]
    )
    path = tmp_path / "hurdat2.txt"
    path.write_text(sample, encoding="utf-8")

    df = load_hurdat2(path)

    assert len(df) == 2
    assert set(df["storm_id"].unique()) == {"AL011851"}
    assert set(df["source"].unique()) == {"hurdat2"}
    assert float(df["lat"].iloc[0]) == 28.0
    assert float(df["lon"].iloc[0]) == -94.8


def test_merge_sources_deduplicates_and_marks_merged():
    ibtracs_df = pd.DataFrame(
        [
            {
                "storm_id": "AL011851",
                "iso_time": pd.Timestamp("1851-06-25 00:00:00"),
                "lat": 28.0,
                "lon": -94.8,
                "vmax_kt": 80.0,
                "min_pressure_mb": 990.0,
                "source": "ibtracs",
            }
        ]
    )
    hurdat2_df = pd.DataFrame(
        [
            {
                "storm_id": "AL011851",
                "iso_time": pd.Timestamp("1851-06-25 00:00:00"),
                "lat": 28.0,
                "lon": -94.8,
                "vmax_kt": 80.0,
                "min_pressure_mb": 985.0,
                "source": "hurdat2",
            }
        ]
    )

    merged = merge_sources(ibtracs_df, hurdat2_df)

    assert len(merged) == 1
    assert merged.iloc[0]["source"] == "merged"


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
