#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
README_PATHS = [ROOT / "README.md", *sorted((ROOT / "src").glob("*/README.md"))]
HEADING_RE = re.compile(r"^(#{1,6}) (.+\S)\s*$")
FENCE_RE = re.compile(r"^```")


def lint_readme(path: Path) -> list[str]:
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_fence = False
    first_nonempty_line: tuple[int, str] | None = None
    h1_count = 0
    previous_heading_level = 0

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not in_fence and stripped and first_nonempty_line is None:
            first_nonempty_line = (line_number, line)

        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue

        if in_fence:
            continue

        if not line.startswith("#"):
            continue

        heading_match = HEADING_RE.match(line)
        if not heading_match:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} uses an invalid heading format"
            )
            continue

        level = len(heading_match.group(1))
        if level == 1:
            h1_count += 1

        if previous_heading_level and level > previous_heading_level + 1:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} jumps from heading level "
                f"{previous_heading_level} to {level}"
            )

        previous_heading_level = level

    if in_fence:
        errors.append(f"{path.relative_to(ROOT)} has an unclosed fenced code block")

    if first_nonempty_line is None:
        errors.append(f"{path.relative_to(ROOT)} is empty")
    else:
        line_number, line = first_nonempty_line
        if not line.startswith("# "):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} should start with a single H1 heading"
            )

    if h1_count != 1:
        errors.append(
            f"{path.relative_to(ROOT)} should contain exactly one H1 heading (found {h1_count})"
        )

    return errors


def main() -> int:
    errors: list[str] = []
    for path in README_PATHS:
        errors.extend(lint_readme(path))

    if errors:
        print("README markdown lint failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"README markdown lint passed for {len(README_PATHS)} README files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
