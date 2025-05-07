#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FORMAT_TARGET_PATTERNS = [
    "scripts",
    "tests/test_static_app_smokes.py",
    "src/*/main.py",
    "src/ctscan/tests/test_ctscan_main.py",
    "src/disasters/tests/test_disasters_main.py",
    "src/voiceforge/tests/test_main.py",
]


def expand_targets(patterns: Iterable[str]) -> list[str]:
    targets: list[str] = []
    for pattern in patterns:
        matches = sorted(ROOT.glob(pattern))
        if matches:
            targets.extend(path.relative_to(ROOT).as_posix() for path in matches)
        else:
            targets.append(pattern)
    return targets


def main() -> int:
    format_targets = expand_targets(FORMAT_TARGET_PATTERNS)
    command = [sys.executable, "-m", "ruff", "format", *format_targets]
    print(
        "Running " + " ".join(command[:4]) + " " + " ".join(format_targets),
        flush=True,
    )
    completed = subprocess.run(command, cwd=ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
