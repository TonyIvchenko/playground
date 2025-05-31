#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
STATIC_BROWSER_SERVICES = (
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
REQUIRED_SHARED_CSS_FRAGMENTS = (
    ".app-help-drawer {",
    ".app-help-toggle {",
    ".app-help-grid {",
    ".app-help-block {",
    ".app-help-kicker {",
    ".app-help-list {",
    ".app-keycap {",
)
REQUIRED_LABELS = (
    '<summary class="app-help-toggle">Help & Shortcuts</summary>',
    '<p class="app-help-kicker">How to use</p>',
    '<p class="app-help-kicker">Best results</p>',
    '<p class="app-help-kicker">Shortcuts</p>',
)
SERVICE_SPECIFIC_FRAGMENTS = {
    "bert": ("the score refreshes automatically",),
    "counterpoint": ("Specific claims and concrete audiences",),
    "debate": ("Shorter, focused docs usually outperform giant research dumps.",),
    "facemesh": ("Bright, even lighting and a mostly frontal pose",),
    "manipulation": ("Multi-sentence passages produce better pattern context",),
    "memorypalace": (
        "Blank lines create cleaner room boundaries",
        '<kbd class="app-keycap">W</kbd>',
    ),
    "realitycheck": (
        "direct image and video URL fields",
        '<kbd class="app-keycap">Enter</kbd>',
    ),
    "realitymix": ("Lower internal resolution is much faster",),
    "vibedj": ("Audio controls still require a click.",),
}


def main() -> int:
    failures: list[str] = []

    shared_css = (ROOT / "shared" / "browser-tokens.css").read_text(encoding="utf-8")
    for fragment in REQUIRED_SHARED_CSS_FRAGMENTS:
        if fragment not in shared_css:
            failures.append(
                f"shared/browser-tokens.css is missing help-drawer fragment {fragment!r}."
            )

    for service in STATIC_BROWSER_SERVICES:
        path = ROOT / "src" / service / "index.html"
        text = path.read_text(encoding="utf-8")

        if 'class="app-help-drawer app-panel"' not in text:
            failures.append(
                f"{path.relative_to(ROOT)} is missing the shared app-help-drawer app-panel shell."
            )

        for label in REQUIRED_LABELS:
            if label not in text:
                failures.append(
                    f"{path.relative_to(ROOT)} is missing help label fragment {label!r}."
                )

        for fragment in SERVICE_SPECIFIC_FRAGMENTS.get(service, ()):
            if fragment not in text:
                failures.append(
                    f"{path.relative_to(ROOT)} is missing app-specific help fragment {fragment!r}."
                )

    if failures:
        print("Browser app help-drawer check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        f"Browser app help-drawer check passed for {len(STATIC_BROWSER_SERVICES)} static apps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
