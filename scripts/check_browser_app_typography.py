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
    '@import url("https://fonts.googleapis.com/css2?family=Fraunces',
    "body {",
    "font-family: var(--font-body);",
    ".app-font-body {",
    ".app-font-display {",
    ".app-font-accent {",
    ".app-font-mono {",
)
BANNED_HTML_FRAGMENTS = (
    '@import url("https://fonts.googleapis.com',
    "font-family:",
)


def main() -> int:
    failures: list[str] = []

    shared_css = (ROOT / "shared" / "browser-tokens.css").read_text(encoding="utf-8")
    for fragment in REQUIRED_SHARED_CSS_FRAGMENTS:
        if fragment not in shared_css:
            failures.append(
                f"shared/browser-tokens.css is missing typography fragment {fragment!r}."
            )

    for service in STATIC_BROWSER_SERVICES:
        path = ROOT / "src" / service / "index.html"
        text = path.read_text(encoding="utf-8")

        if '<link rel="stylesheet" href="/shared/browser-tokens.css">' not in text:
            failures.append(
                f"{path.relative_to(ROOT)} is missing the shared browser token stylesheet link."
            )

        for fragment in BANNED_HTML_FRAGMENTS:
            if fragment in text:
                failures.append(
                    f"{path.relative_to(ROOT)} still defines local typography fragment {fragment!r}."
                )

    if failures:
        print("Browser app typography check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        f"Browser app typography check passed for {len(STATIC_BROWSER_SERVICES)} static apps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
