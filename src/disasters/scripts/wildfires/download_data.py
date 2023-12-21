"""Download and harmonize wildfires tabular data from multiple public sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.request import urlretrieve

import pandas as pd
import requests


FOREST_FIRES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
ALGERIAN_FIRES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00547/Algerian_forest_fires_dataset_UPDATE.csv"
US_FOD_QUERY_URL = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_FireOccurrence6thEdition_01/MapServer/29/query"
DISASTERS_ROOT = Path(__file__).resolve().parents[2]

CONUS_LAT_MIN = 24.0
CONUS_LAT_MAX = 50.0
CONUS_LON_MIN = -125.0
CONUS_LON_MAX = -66.0


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
    parser.add_argument(
        "--us-overlay-path",
        type=Path,
        default=DISASTERS_ROOT / "data" / "wildfires" / "raw" / "wildfires_us_overlay.csv",
        help="Path for US wildfire overlay points CSV.",
    )
    parser.add_argument(
        "--us-fod-query-url",
        type=str,
        default=US_FOD_QUERY_URL,
        help="ArcGIS query endpoint for USFS FOD6 wildfire points.",
    )
    parser.add_argument("--us-min-year", type=int, default=2000)
    parser.add_argument("--us-min-fire-size", type=float, default=50.0)
    parser.add_argument("--us-page-size", type=int, default=2000)
    parser.add_argument("--us-max-rows", type=int, default=None)
    parser.add_argument("--us-timeout-sec", type=float, default=30.0)
    parser.add_argument("--us-retries", type=int, default=2)
    parser.add_argument("--us-backoff-sec", type=float, default=1.0)
    return parser.parse_args()


def _arcgis_query_json(
    query_url: str,
    params: dict[str, object],
    timeout_sec: float,
    retries: int,
    backoff_sec: float,
) -> dict[str, object]:
    last_error: Exception | None = None
    attempts = max(1, int(retries) + 1)
    for attempt in range(attempts):
        try:
            resp = requests.get(
                query_url,
                params=params,
                headers={"User-Agent": "playground-disasters/1.0"},
                timeout=timeout_sec,
            )
            payload = resp.text
            data = json.loads(payload)
            if "error" in data:
                raise RuntimeError(f"ArcGIS query failed: {data['error']}")
            return data
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(max(0.0, backoff_sec) * (attempt + 1))
                continue
            break
    raise RuntimeError(f"ArcGIS query failed after retries: {last_error}")


def _normalize_us_fod_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["year", "month", "lat", "lon", "fire_size", "state", "source"])

    dt = pd.to_datetime(pd.to_numeric(df["DISCOVERY_DATE"], errors="coerce"), unit="ms", errors="coerce")
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(df["FIRE_YEAR"], errors="coerce"),
            "month": dt.dt.month.fillna(7).astype(float),
            "lat": pd.to_numeric(df["LATITUDE"], errors="coerce"),
            "lon": pd.to_numeric(df["LONGITUDE"], errors="coerce"),
            "fire_size": pd.to_numeric(df["FIRE_SIZE"], errors="coerce"),
            "state": df["STATE"].astype(str).str.upper().str.strip(),
            "source": "usfs_fod6",
        }
    )
    out = out.dropna(subset=["year", "month", "lat", "lon", "fire_size"]).reset_index(drop=True)
    out = out[
        out["lat"].between(CONUS_LAT_MIN, CONUS_LAT_MAX) & out["lon"].between(CONUS_LON_MIN, CONUS_LON_MAX)
    ].reset_index(drop=True)
    out["month"] = out["month"].clip(1.0, 12.0)
    return out


def fetch_us_fod_overlay_points(
    query_url: str,
    min_year: int,
    min_fire_size: float,
    page_size: int,
    max_rows: int | None,
    timeout_sec: float,
    retries: int,
    backoff_sec: float,
) -> pd.DataFrame:
    out_fields = "OBJECTID,FIRE_YEAR,DISCOVERY_DATE,FIRE_SIZE,LATITUDE,LONGITUDE,STATE"

    print(f"Fetching US wildfire overlay points from ArcGIS ({min_year}-2020)...")
    rows: list[dict[str, object]] = []
    for year in range(int(min_year), 2021):
        print(f"  Year {year}: start")
        year_start_rows = len(rows)
        offset = 0
        page_idx = 0
        while True:
            page_idx += 1
            where = f"FIRE_YEAR = {year} AND FIRE_SIZE >= {float(min_fire_size)}"
            try:
                page = _arcgis_query_json(
                    query_url,
                    {
                        "f": "json",
                        "where": where,
                        "outFields": out_fields,
                        "returnGeometry": "false",
                        "orderByFields": "OBJECTID",
                        "resultOffset": offset,
                        "resultRecordCount": page_size,
                    },
                    timeout_sec=timeout_sec,
                    retries=retries,
                    backoff_sec=backoff_sec,
                )
            except RuntimeError as exc:
                # ArcGIS occasionally returns transient "Layer not found" under heavy load.
                # Continue with other years instead of failing the whole pipeline.
                if "Layer not found" in str(exc):
                    print(f"Warning: skipping year {year} at offset {offset} due to ArcGIS error: {exc}")
                    break
                raise
            features = page.get("features", [])
            if not features:
                break

            for feature in features:
                attrs = feature.get("attributes", {})
                if attrs:
                    rows.append(attrs)

            if max_rows is not None and len(rows) >= max_rows:
                rows = rows[:max_rows]
                print(f"Reached us-max-rows={max_rows}; stopping early.")
                return _normalize_us_fod_rows(rows)

            offset += len(features)
            print(f"    page {page_idx}: +{len(features)} rows (total={len(rows)})")
            if len(features) < page_size:
                break

        print(f"  Year {year}: +{len(rows) - year_start_rows} rows")

    print(f"Finished ArcGIS fetch. Raw rows collected: {len(rows)}")
    return _normalize_us_fod_rows(rows)


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
    args.us_overlay_path.parent.mkdir(parents=True, exist_ok=True)

    forest_path = args.raw_dir / "forestfires_uci.csv"
    algerian_path = args.raw_dir / "algerian_forest_fires.csv"

    urlretrieve(args.forest_fires_url, forest_path)
    urlretrieve(args.algerian_fires_url, algerian_path)

    forest_df = load_forest_fires(forest_path)
    algerian_df = load_algerian_fires(algerian_path)
    merged_df = pd.concat([forest_df, algerian_df], ignore_index=True).reset_index(drop=True)
    merged_df.to_csv(args.output_path, index=False)
    us_overlay_df = fetch_us_fod_overlay_points(
        query_url=args.us_fod_query_url,
        min_year=args.us_min_year,
        min_fire_size=args.us_min_fire_size,
        page_size=args.us_page_size,
        max_rows=args.us_max_rows,
        timeout_sec=args.us_timeout_sec,
        retries=args.us_retries,
        backoff_sec=args.us_backoff_sec,
    )
    us_overlay_df.to_csv(args.us_overlay_path, index=False)

    print(f"Downloaded UCI Forest Fires to: {forest_path} ({len(forest_df)} rows)")
    print(f"Downloaded UCI Algerian Forest Fires to: {algerian_path} ({len(algerian_df)} rows)")
    print(f"Wrote merged canonical wildfires rows to: {args.output_path} ({len(merged_df)} rows)")
    print(f"Wrote US wildfire overlay points to: {args.us_overlay_path} ({len(us_overlay_df)} rows)")


if __name__ == "__main__":
    main()
