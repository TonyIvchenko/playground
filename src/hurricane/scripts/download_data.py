"""Download and harmonize hurricane track data from multiple public sources."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen, urlretrieve

import pandas as pd


IBTRACS_NA_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
)
HURDAT_INDEX_URL = "https://www.nhc.noaa.gov/data/hurdat/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and harmonize hurricane training data.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("src/hurricane/data/raw"),
        help="Directory where raw source files are stored.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("src/hurricane/data/raw/hurricane_tracks_merged.csv"),
        help="Path for merged canonical hurricane tracks CSV.",
    )
    parser.add_argument(
        "--ibtracs-url",
        type=str,
        default=IBTRACS_NA_URL,
        help="IBTrACS CSV URL.",
    )
    parser.add_argument(
        "--hurdat2-url",
        type=str,
        default="",
        help="Optional explicit HURDAT2 URL. If empty, uses the latest listed file.",
    )
    return parser.parse_args()


def discover_latest_hurdat2_url(index_url: str) -> str:
    html = urlopen(index_url, timeout=60).read().decode("utf-8", errors="replace")
    filenames = sorted(set(re.findall(r"hurdat2-1851-\d{4}-\d+\.txt", html)))
    if not filenames:
        raise RuntimeError(f"Could not find any HURDAT2 files at: {index_url}")
    latest = filenames[-1]
    return urljoin(index_url, latest)


def parse_lat_lon(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    suffix = text[-1].upper()
    if suffix not in {"N", "S", "E", "W"}:
        return None
    try:
        magnitude = float(text[:-1])
    except ValueError:
        return None
    if suffix in {"S", "W"}:
        magnitude *= -1.0
    return magnitude


def load_ibtracs(path: Path) -> pd.DataFrame:
    usecols = ["SID", "ISO_TIME", "LAT", "LON", "USA_ATCF_ID", "USA_WIND", "USA_PRES", "WMO_WIND", "WMO_PRES"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False, skiprows=[1])

    storm_id = df["USA_ATCF_ID"].astype(str).str.strip()
    storm_id = storm_id.mask(storm_id == "", df["SID"].astype(str).str.strip())
    df["storm_id"] = storm_id
    df["iso_time"] = pd.to_datetime(df["ISO_TIME"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["lat"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["lon"] = pd.to_numeric(df["LON"], errors="coerce")

    usa_wind = pd.to_numeric(df["USA_WIND"], errors="coerce")
    wmo_wind = pd.to_numeric(df["WMO_WIND"], errors="coerce")
    usa_pres = pd.to_numeric(df["USA_PRES"], errors="coerce")
    wmo_pres = pd.to_numeric(df["WMO_PRES"], errors="coerce")
    df["vmax_kt"] = usa_wind.fillna(wmo_wind)
    df["min_pressure_mb"] = usa_pres.fillna(wmo_pres)

    out = df[["storm_id", "iso_time", "lat", "lon", "vmax_kt", "min_pressure_mb"]].copy()
    out["source"] = "ibtracs"
    out = out.dropna(subset=["storm_id", "iso_time", "lat", "lon", "vmax_kt"]).reset_index(drop=True)
    return out


def load_hurdat2(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_storm_id: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if not parts or not parts[0]:
                continue

            token = parts[0]
            if re.fullmatch(r"[A-Z]{2}\d{6}", token):
                # Keep Atlantic records to align with IBTrACS NA list.
                current_storm_id = token if token.startswith("AL") else None
                continue

            if current_storm_id is None:
                continue

            if len(parts) < 8:
                continue

            date_str = parts[0]
            time_str = parts[1].zfill(4)
            lat = parse_lat_lon(parts[4])
            lon = parse_lat_lon(parts[5])
            try:
                vmax_kt = float(parts[6])
            except ValueError:
                continue

            try:
                pressure = float(parts[7])
            except ValueError:
                pressure = float("nan")
            if pressure <= 0:
                pressure = float("nan")

            iso_time = pd.to_datetime(f"{date_str}{time_str}", format="%Y%m%d%H%M", errors="coerce")
            if pd.isna(iso_time) or lat is None or lon is None:
                continue

            rows.append(
                {
                    "storm_id": current_storm_id,
                    "iso_time": iso_time,
                    "lat": lat,
                    "lon": lon,
                    "vmax_kt": vmax_kt,
                    "min_pressure_mb": pressure,
                    "source": "hurdat2",
                }
            )

    return pd.DataFrame(rows)


def _first_valid(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.iloc[0])


def merge_sources(ibtracs_df: pd.DataFrame, hurdat2_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([hurdat2_df, ibtracs_df], ignore_index=True)
    combined["source_priority"] = combined["source"].map({"hurdat2": 0, "ibtracs": 1}).fillna(99)
    combined = combined.sort_values(["storm_id", "iso_time", "source_priority"]).reset_index(drop=True)

    grouped = combined.groupby(["storm_id", "iso_time"], as_index=False)
    merged = grouped.agg(
        lat=("lat", _first_valid),
        lon=("lon", _first_valid),
        vmax_kt=("vmax_kt", _first_valid),
        min_pressure_mb=("min_pressure_mb", _first_valid),
        source_count=("source", "nunique"),
        source=("source", "first"),
    )
    merged["source"] = merged["source"].where(merged["source_count"] == 1, "merged")
    merged = merged.drop(columns=["source_count"]).dropna(subset=["lat", "lon", "vmax_kt"]).reset_index(drop=True)
    return merged


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    ibtracs_path = args.raw_dir / "ibtracs_na.csv"
    hurdat2_path = args.raw_dir / "hurdat2_latest.txt"

    hurdat2_url = args.hurdat2_url.strip() or discover_latest_hurdat2_url(HURDAT_INDEX_URL)
    urlretrieve(args.ibtracs_url, ibtracs_path)
    urlretrieve(hurdat2_url, hurdat2_path)

    ibtracs_df = load_ibtracs(ibtracs_path)
    hurdat2_df = load_hurdat2(hurdat2_path)
    merged_df = merge_sources(ibtracs_df, hurdat2_df)
    merged_df.to_csv(args.output_path, index=False)

    print(f"Downloaded IBTrACS to: {ibtracs_path} ({len(ibtracs_df)} rows)")
    print(f"Downloaded HURDAT2 to: {hurdat2_path} ({len(hurdat2_df)} rows)")
    print(f"Wrote merged canonical tracks to: {args.output_path} ({len(merged_df)} rows)")


if __name__ == "__main__":
    main()
