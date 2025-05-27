#!/usr/bin/env python3
from __future__ import annotations

from pathlib import PurePosixPath
import subprocess
import sys


TRACKED_JUNK_NAMES = {".DS_Store"}
TRACKED_JUNK_PARTS = {"__pycache__"}


def tracked_paths() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=False,
    )
    entries = completed.stdout.decode("utf-8").split("\0")
    return [entry for entry in entries if entry]


def junk_paths() -> list[str]:
    matches: list[str] = []
    for entry in tracked_paths():
        path = PurePosixPath(entry)
        if path.name in TRACKED_JUNK_NAMES or any(
            part in TRACKED_JUNK_PARTS for part in path.parts
        ):
            matches.append(entry)
    return sorted(matches)


def main() -> int:
    matches = junk_paths()
    if not matches:
        print("No tracked .DS_Store or __pycache__ paths found.")
        return 0

    print("Tracked junk file check failed:", file=sys.stderr)
    for path in matches:
        print(f"- {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
