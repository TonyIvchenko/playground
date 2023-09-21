"""Download real hurricane best-track data used for training."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


IBTRACS_NA_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download hurricane training data.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("src/hurricane/data/raw/ibtracs_na.csv"),
        help="Where to store the downloaded CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(IBTRACS_NA_URL, args.output_path)
    print(f"Downloaded hurricane dataset to: {args.output_path}")


if __name__ == "__main__":
    main()
