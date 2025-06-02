#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FRAGMENTS = {
    "shared/browser-tokens.css": (
        "--color-ink-muted: #5c6c7d;",
        "--color-warning: #9a4a08;",
        "--color-warning-soft: #fef3c7;",
    ),
    "src/counterpoint/index.html": ("--muted: #5c6c7d;",),
    "src/debate/index.html": ("--muted: #5c6c7d;",),
    "src/manipulation/index.html": ("--muted: #5c6c7d;",),
    "src/memorypalace/index.html": ("--muted: #5c6c7d;",),
    "src/realitycheck/index.html": ("--muted: #5c6c7d;",),
    "src/realitymix/index.html": ("--muted: #5c6c7d;",),
    "src/vibedj/index.html": ("--muted: #5c6c7d;",),
    "src/bert/index.html": (
        "color: #5c6c7d;",
        "background: #f4f6fa;",
    ),
    "src/facemesh/index.html": ("color: #5c6c7d;",),
}

CONTRAST_CASES = (
    ("shared muted text", "#5c6c7d", "#ffffff", 4.5),
    ("shared warning text", "#9a4a08", "#fef3c7", 4.5),
    ("counterpoint muted text", "#5c6c7d", "#ffffff", 4.5),
    ("debate muted text", "#5c6c7d", "#ffffff", 4.5),
    ("manipulation muted text", "#5c6c7d", "#ffffff", 4.5),
    ("memorypalace muted text", "#5c6c7d", "#ffffff", 4.5),
    ("realitycheck muted text", "#5c6c7d", "#ffffff", 4.5),
    ("realitymix muted text", "#5c6c7d", "#ffffff", 4.5),
    ("vibedj muted text", "#5c6c7d", "#ffffff", 4.5),
    ("bert disabled textarea text", "#5c6c7d", "#f4f6fa", 4.5),
    ("facemesh helper text", "#5c6c7d", "#eef3fa", 4.5),
)


def srgb_to_linear(channel: int) -> float:
    value = channel / 255
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return (
        0.2126 * srgb_to_linear(red)
        + 0.7152 * srgb_to_linear(green)
        + 0.0722 * srgb_to_linear(blue)
    )


def contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def main() -> int:
    failures: list[str] = []

    for relative_path, fragments in REQUIRED_FRAGMENTS.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in text:
                failures.append(
                    f"{relative_path} is missing contrast fragment {fragment!r}."
                )

    for label, foreground, background, minimum in CONTRAST_CASES:
        ratio = contrast_ratio(foreground, background)
        if ratio < minimum:
            failures.append(
                f"{label} contrast is {ratio:.2f}, below the required {minimum:.1f}."
            )

    if failures:
        print("Browser app contrast check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        f"Browser app contrast check passed for {len(CONTRAST_CASES)} checked color pairs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
