"""Download real wildfire dataset used for training."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


FOREST_FIRES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download wildfire training data.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("src/wildfire/data/raw/forestfires.csv"),
        help="Where to store the downloaded CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(FOREST_FIRES_URL, args.output_path)
    print(f"Downloaded wildfire dataset to: {args.output_path}")


if __name__ == "__main__":
    main()
