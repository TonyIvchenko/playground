#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
STATIC_APPS = (
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
    ".app-shell {",
    ".app-panel {",
    ".app-stack-md {",
    ".app-two-up-md {",
    "@media (max-width: 64rem) {",
    "@media (max-width: 48rem) {",
)
REQUIRED_APP_FRAGMENTS = {
    "bert": ("app-shell", "app-panel"),
    "counterpoint": ("app-shell", "app-panel", "app-stack-md"),
    "debate": ("app-shell", "app-panel", "app-stack-md"),
    "facemesh": ("app-shell", "app-panel", "app-stack-md"),
    "manipulation": ("app-shell", "app-panel", "app-stack-md"),
    "memorypalace": ("app-shell", "app-panel", "app-stack-md"),
    "realitycheck": ("app-shell", "app-panel", "app-stack-md"),
    "realitymix": ("app-shell", "app-panel", "app-stack-md"),
    "vibedj": ("app-shell", "app-panel", "app-stack-md", "app-two-up-md"),
}
PIXEL_BREAKPOINT_RE = re.compile(r"@media\s*\(max-width:\s*\d+px\)")


def main() -> int:
    failures: list[str] = []

    shared_css = (ROOT / "shared" / "browser-tokens.css").read_text(encoding="utf-8")
    for fragment in REQUIRED_SHARED_CSS_FRAGMENTS:
        if fragment not in shared_css:
            failures.append(
                f"shared/browser-tokens.css is missing mobile-layout fragment {fragment!r}."
            )

    for service in STATIC_APPS:
        path = ROOT / "src" / service / "index.html"
        text = path.read_text(encoding="utf-8")

        if PIXEL_BREAKPOINT_RE.search(text):
            failures.append(
                f"{path.relative_to(ROOT)} still uses a pixel-based max-width media query."
            )

        for fragment in REQUIRED_APP_FRAGMENTS[service]:
            if fragment not in text:
                failures.append(
                    f"{path.relative_to(ROOT)} is missing shared mobile fragment {fragment!r}."
                )

    if failures:
        print("Browser app mobile-layout check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Browser app mobile-layout check passed for {len(STATIC_APPS)} static apps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
