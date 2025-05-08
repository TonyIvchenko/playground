#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LINT_TARGET_PATTERNS = [
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
    lint_targets = expand_targets(LINT_TARGET_PATTERNS)
    commands = [
        [sys.executable, "-m", "ruff", "check", *lint_targets],
        [sys.executable, "scripts/check_markdown_readmes.py"],
    ]
    for command in commands:
        if command[1:3] == ["-m", "ruff"]:
            print(
                "Running " + " ".join(command[:4]) + " " + " ".join(lint_targets),
                flush=True,
            )
        else:
            print("Running " + " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
