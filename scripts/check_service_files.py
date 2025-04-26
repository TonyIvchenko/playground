#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
REQUIRED_FILES = ("main.py", "README.md")


def iter_services() -> list[Path]:
    services: list[Path] = []
    for path in sorted(SRC_DIR.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith(".") or path.name == "__pycache__":
            continue
        services.append(path)
    return services


def main() -> int:
    missing: list[tuple[str, list[str]]] = []
    for service_dir in iter_services():
        absent = [name for name in REQUIRED_FILES if not (service_dir / name).exists()]
        if absent:
            missing.append((service_dir.name, absent))

    if not missing:
        print(
            f"All {len(iter_services())} services have the required files: "
            + ", ".join(REQUIRED_FILES)
        )
        return 0

    print("Service file check failed:", file=sys.stderr)
    for service, absent in missing:
        print(f"- {service}: missing {', '.join(absent)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
