from pathlib import Path

import pandas as pd

from src.disasters.scripts.hurricane_download_data import load_hurdat2, merge_sources, parse_lat_lon


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
