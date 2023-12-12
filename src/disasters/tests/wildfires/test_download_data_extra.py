from __future__ import annotations

from pathlib import Path

from src.disasters.scripts.wildfires.download_data import load_algerian_fires


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
