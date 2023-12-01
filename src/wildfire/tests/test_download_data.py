from pathlib import Path

import pandas as pd

from src.wildfire.scripts.download_data import _parse_algerian_line, load_forest_fires


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
