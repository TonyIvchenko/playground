#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pytest suite for a single Playground service."
    )
    parser.add_argument("service", help="Service name under src/<service>.")
    return parser.parse_args()


def discover_test_suites() -> dict[str, Path]:
    suites: dict[str, Path] = {}
    for path in sorted(SRC_DIR.iterdir()):
        if not path.is_dir() or path.name.startswith(".") or path.name == "__pycache__":
            continue
        tests_dir = path / "tests"
        if tests_dir.is_dir():
            suites[path.name] = tests_dir
    return suites


def main() -> int:
    args = parse_args()
    suites = discover_test_suites()
    tests_dir = suites.get(args.service)
    if tests_dir is None:
        available = ", ".join(sorted(suites)) or "(none)"
        print(
            f"No test suite found for service '{args.service}'. "
            f"Available test targets: {available}",
            file=sys.stderr,
        )
        return 1

    display_path = tests_dir.relative_to(ROOT).as_posix()
    command = [sys.executable, "-m", "pytest", "-q", str(tests_dir)]
    print(f"Running {' '.join(command[:4])} {display_path}", flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
