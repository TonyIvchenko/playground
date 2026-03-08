"""Download and harmonize wildfires tabular data from multiple public sources."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


FOREST_FIRES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
ALGERIAN_FIRES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00547/Algerian_forest_fires_dataset_UPDATE.csv"
DISASTERS_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and harmonize wildfires training data.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DISASTERS_ROOT / "data" / "wildfires" / "raw",
        help="Directory where raw source files are stored.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DISASTERS_ROOT / "data" / "wildfires" / "raw" / "wildfires_training_merged.csv",
        help="Path for merged canonical wildfires training CSV.",
    )
    parser.add_argument(
        "--forest-fires-url",
        type=str,
        default=FOREST_FIRES_URL,
        help="UCI Forest Fires CSV URL.",
    )
    parser.add_argument(
        "--algerian-fires-url",
        type=str,
        default=ALGERIAN_FIRES_URL,
        help="UCI Algerian Forest Fires CSV URL.",
    )
    return parser.parse_args()


def load_forest_fires(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "temp_c": pd.to_numeric(df["temp"], errors="coerce"),
            "humidity_pct": pd.to_numeric(df["RH"], errors="coerce"),
            "wind_kph": pd.to_numeric(df["wind"], errors="coerce"),
            "ffmc": pd.to_numeric(df["FFMC"], errors="coerce"),
            "dmc": pd.to_numeric(df["DMC"], errors="coerce"),
            "drought_code": pd.to_numeric(df["DC"], errors="coerce"),
            "isi": pd.to_numeric(df["ISI"], errors="coerce"),
            "target": (pd.to_numeric(df["area"], errors="coerce") > 0.0).astype(float),
            "source": "uci_forestfires",
        }
    )
    return out.dropna(
        subset=["temp_c", "humidity_pct", "wind_kph", "ffmc", "dmc", "drought_code", "isi", "target"]
    ).reset_index(drop=True)


def _parse_algerian_line(line: str) -> dict[str, object] | None:
    text = line.strip()
    if not text:
        return None
    if "Region Dataset" in text:
        return None
    if text.lower().startswith("day,month,year"):
        return None

    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 14:
        return None

    try:
        temp_c = float(parts[3])
        humidity_pct = float(parts[4])
        wind_kph = float(parts[5])
        # Algerian columns: Rain, FFMC, DMC, DC, ISI, BUI, FWI
        ffmc = float(parts[7])
        dmc = float(parts[8])
        drought_code = float(parts[9])
        isi = float(parts[10])
    except ValueError:
        return None

    label = parts[13].strip().lower()
    if label == "fire":
        target = 1.0
    elif label == "not fire":
        target = 0.0
    else:
        return None

    return {
        "temp_c": temp_c,
        "humidity_pct": humidity_pct,
        "wind_kph": wind_kph,
        "ffmc": ffmc,
        "dmc": dmc,
        "drought_code": drought_code,
        "isi": isi,
        "target": target,
        "source": "uci_algerian",
    }


def load_algerian_fires(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_algerian_line(line)
            if parsed is not None:
                rows.append(parsed)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    forest_path = args.raw_dir / "forestfires_uci.csv"
    algerian_path = args.raw_dir / "algerian_forest_fires.csv"

    urlretrieve(args.forest_fires_url, forest_path)
    urlretrieve(args.algerian_fires_url, algerian_path)

    forest_df = load_forest_fires(forest_path)
    algerian_df = load_algerian_fires(algerian_path)
    merged_df = pd.concat([forest_df, algerian_df], ignore_index=True).reset_index(drop=True)
    merged_df.to_csv(args.output_path, index=False)

    print(f"Downloaded UCI Forest Fires to: {forest_path} ({len(forest_df)} rows)")
    print(f"Downloaded UCI Algerian Forest Fires to: {algerian_path} ({len(algerian_df)} rows)")
    print(f"Wrote merged canonical wildfires rows to: {args.output_path} ({len(merged_df)} rows)")


if __name__ == "__main__":
    main()
