#!/usr/bin/env python3
"""Warm cached CT Scan viewer assets for sample studies."""

from __future__ import annotations

import argparse
import json

try:
    from main import warm_sample_viewer_cache
except ModuleNotFoundError:
    from src.ctscan.main import warm_sample_viewer_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm cached viewer assets for CT Scan sample studies."
    )
    parser.add_argument(
        "sample_ids",
        nargs="*",
        help="Optional sample ids to warm. Defaults to all configured samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Warm at most this many samples after selection.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the warmed cache summary as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warmed = warm_sample_viewer_cache(
        sample_ids=args.sample_ids or None, limit=args.limit
    )

    if args.json:
        print(json.dumps(warmed, indent=2))
        return

    if not warmed:
        print("No sample viewer caches were warmed.")
        return

    print(f"Warmed {len(warmed)} CT Scan sample viewer cache entries:")
    for row in warmed:
        print(
            f"- {row['sample_id']}: {row['viewer_url']} "
            f"(state: {row['viewer_state_path']})"
        )


if __name__ == "__main__":
    main()
