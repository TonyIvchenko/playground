#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
STATIC_SERVICES = (
    "bert",
    "counterpoint",
    "debate",
    "facemesh",
    "manipulation",
    "memorypalace",
    "realitycheck",
    "realitymix",
    "vibedj",
)
REQUIRED_CLASSES = ("app-header", "app-kicker", "app-title", "app-subtitle")


def has_class(text: str, class_name: str) -> bool:
    for match in re.finditer(r'class="[^"]*"', text):
        class_names = match.group(0)[7:-1].split()
        if class_name in class_names:
            return True
    return False


def main() -> int:
    failures: list[str] = []

    for service in STATIC_SERVICES:
        path = ROOT / "src" / service / "index.html"
        text = path.read_text(encoding="utf-8")
        missing = [
            class_name
            for class_name in REQUIRED_CLASSES
            if not has_class(text, class_name)
        ]
        if missing:
            failures.append(
                f"{path.relative_to(ROOT)} is missing header classes: {', '.join(missing)}"
            )

    if failures:
        print("Browser app header check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Browser app header check passed for {len(STATIC_SERVICES)} static apps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
