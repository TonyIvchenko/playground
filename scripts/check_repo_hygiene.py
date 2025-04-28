#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_LARGE_FILE_MB = 20
ALLOWED_LARGE_FILES = {
    "src/disasters/tiles/huricaines/overlay.npz",
    "src/disasters/tiles/wildfires/overlay.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check tracked and unignored files for common repo hygiene issues like "
            ".DS_Store, __pycache__, stray logs, and unexpectedly large files."
        )
    )
    parser.add_argument(
        "--max-large-file-mb",
        type=int,
        default=DEFAULT_MAX_LARGE_FILE_MB,
        help="Flag tracked or unignored files larger than this size in MiB.",
    )
    return parser.parse_args()


def iter_repo_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = [entry for entry in result.stdout.decode("utf-8").split("\0") if entry]
    return [ROOT / path for path in paths]


def category_for_path(path: Path, max_large_file_bytes: int) -> list[str]:
    categories: list[str] = []
    rel_path = path.relative_to(ROOT)
    rel_text = rel_path.as_posix()
    parts = rel_path.parts

    if path.name == ".DS_Store":
        categories.append(".DS_Store")
    if "__pycache__" in parts:
        categories.append("__pycache__")
    if path.suffix == ".log":
        categories.append("log file")
    if (
        path.is_file()
        and rel_text not in ALLOWED_LARGE_FILES
        and path.stat().st_size > max_large_file_bytes
    ):
        categories.append("large file")

    return categories


def main() -> int:
    args = parse_args()
    max_large_file_bytes = args.max_large_file_mb * 1024 * 1024

    issues: dict[str, list[str]] = defaultdict(list)
    for path in iter_repo_files():
        rel_path = path.relative_to(ROOT).as_posix()
        for category in category_for_path(path, max_large_file_bytes):
            issues[category].append(rel_path)

    if not issues:
        print(
            "No hygiene issues found in tracked or unignored files "
            f"(large file threshold: {args.max_large_file_mb} MiB)."
        )
        return 0

    print("Repo hygiene check failed:", file=sys.stderr)
    for category in sorted(issues):
        print(f"- {category}:", file=sys.stderr)
        for rel_path in sorted(issues[category]):
            print(f"  - {rel_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
