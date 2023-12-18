from pathlib import Path

import pandas as pd

import src.disasters.scripts.wildfires.download_data as wildfires_download
from src.disasters.scripts.wildfires.download_data import (
    _normalize_us_fod_rows,
    _parse_algerian_line,
    fetch_us_fod_overlay_points,
    load_algerian_fires,
    load_forest_fires,
)


def test_parse_algerian_line_targets():
    fire_row = "06,06,2012,31,67,14,0,82.6,5.8,22.2,3.1,7,2.5,fire"
    not_fire_row = "05,06,2012,27,77,16,0,64.8,3,14.2,1.2,3.9,0.5,not fire"

    fire = _parse_algerian_line(fire_row)
    not_fire = _parse_algerian_line(not_fire_row)

    assert fire is not None
    assert not_fire is not None
    assert fire["ffmc"] == 82.6
    assert fire["dmc"] == 5.8
    assert fire["target"] == 1.0
    assert not_fire["target"] == 0.0


def test_load_forest_fires_maps_columns(tmp_path: Path):
    input_path = tmp_path / "forest.csv"
    pd.DataFrame(
        [
            {"temp": 20.5, "RH": 45, "wind": 3.2, "FFMC": 85.1, "DMC": 120.0, "DC": 200.0, "ISI": 5.2, "area": 0.0},
            {"temp": 31.2, "RH": 22, "wind": 9.1, "FFMC": 93.8, "DMC": 210.0, "DC": 500.0, "ISI": 14.1, "area": 4.4},
        ]
    ).to_csv(input_path, index=False)

    df = load_forest_fires(input_path)

    assert list(df.columns) == [
        "temp_c",
        "humidity_pct",
        "wind_kph",
        "ffmc",
        "dmc",
        "drought_code",
        "isi",
        "target",
        "source",
    ]
    assert len(df) == 2
    assert set(df["target"].tolist()) == {0.0, 1.0}
    assert set(df["source"].unique()) == {"uci_forestfires"}


def test_load_algerian_fires_filters_bad_lines(tmp_path: Path):
    path = tmp_path / "algerian.csv"
    path.write_text(
        "\n".join(
            [
                "Bejaia Region Dataset",
                "day,month,year,Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,FWI,Classes",
                "05,06,2012,27,77,16,0,64.8,3,14.2,1.2,3.9,0.5,not fire",
                "bad,row,that,should,be,ignored",
                "06,06,2012,31,67,14,0,82.6,5.8,22.2,3.1,7,2.5,fire",
            ]
        ),
        encoding="utf-8",
    )

    df = load_algerian_fires(path)
    assert len(df) == 2
    assert set(df["target"].tolist()) == {0.0, 1.0}


def test_normalize_us_fod_rows_filters_bounds():
    rows = [
        {
            "FIRE_YEAR": 2019,
            "DISCOVERY_DATE": 1561939200000,  # 2019-07-01
            "FIRE_SIZE": 120.0,
            "LATITUDE": 36.5,
            "LONGITUDE": -120.2,
            "STATE": "ca",
        },
        {
            "FIRE_YEAR": 2019,
            "DISCOVERY_DATE": 1561939200000,
            "FIRE_SIZE": 220.0,
            "LATITUDE": 64.0,  # out of CONUS filter
            "LONGITUDE": -149.0,
            "STATE": "ak",
        },
    ]
    df = _normalize_us_fod_rows(rows)
    assert len(df) == 1
    assert set(df.columns) == {"year", "month", "lat", "lon", "fire_size", "state", "source"}
    assert df.iloc[0]["state"] == "CA"
    assert df.iloc[0]["source"] == "usfs_fod6"


def test_fetch_us_fod_overlay_points_paginates(monkeypatch):
    pages = [
        {
            "features": [
                {
                    "attributes": {
                        "FIRE_YEAR": 2018,
                        "DISCOVERY_DATE": 1530403200000,
                        "FIRE_SIZE": 100.0,
                        "LATITUDE": 35.0,
                        "LONGITUDE": -100.0,
                        "STATE": "TX",
                    }
                }
            ],
            "exceededTransferLimit": True,
        },
        {
            "features": [
                {
                    "attributes": {
                        "FIRE_YEAR": 2019,
                        "DISCOVERY_DATE": 1561939200000,
                        "FIRE_SIZE": 200.0,
                        "LATITUDE": 40.0,
                        "LONGITUDE": -120.0,
                        "STATE": "CA",
                    }
                }
            ],
            "exceededTransferLimit": False,
        },
    ]

    calls = {"i": 0}

    def fake_query(_query_url: str, _params: dict[str, object]) -> dict[str, object]:
        i = calls["i"]
        calls["i"] += 1
        return pages[i] if i < len(pages) else {"features": [], "exceededTransferLimit": False}

    monkeypatch.setattr(wildfires_download, "_arcgis_query_json", fake_query)
    df = fetch_us_fod_overlay_points(
        query_url="https://example.com/query",
        min_year=2000,
        min_fire_size=50.0,
        page_size=1,
        max_rows=None,
    )
    assert len(df) == 2
    assert sorted(df["year"].astype(int).tolist()) == [2018, 2019]
