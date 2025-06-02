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
    ".app-starter-shell {",
    ".app-starter-surface {",
    ".app-starter-hero {",
    ".app-starter-panel {",
)


def main() -> int:
    failures: list[str] = []

    shared_css = (ROOT / "shared" / "browser-starter.css").read_text(encoding="utf-8")
    for fragment in REQUIRED_SHARED_CSS_FRAGMENTS:
        if fragment not in shared_css:
            failures.append(
                f"shared/browser-starter.css is missing starter fragment {fragment!r}."
            )

    for service in STATIC_BROWSER_SERVICES:
        path = ROOT / "src" / service / "index.html"
        text = path.read_text(encoding="utf-8")

        if '<link rel="stylesheet" href="/shared/browser-starter.css">' not in text:
            failures.append(
                f"{path.relative_to(ROOT)} is missing the shared browser starter stylesheet link."
            )

        if (
            'class="app-shell app-starter-shell"' not in text
            and 'class="app app-shell app-starter-shell"' not in text
        ):
            failures.append(
                f"{path.relative_to(ROOT)} is missing the shared app-starter-shell class on its main container."
            )

        if "app-starter-surface" not in text:
            failures.append(
                f"{path.relative_to(ROOT)} is missing a shared app-starter-surface section."
            )

        if "app-starter-panel" not in text and "app-starter-hero" not in text:
            failures.append(
                f"{path.relative_to(ROOT)} is missing shared app-starter panel or hero classes."
            )

    if failures:
        print("Browser app starter stylesheet check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        f"Browser app starter stylesheet check passed for {len(STATIC_BROWSER_SERVICES)} static apps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
