#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
BUTTON_SERVICES = (
    "counterpoint",
    "debate",
    "manipulation",
    "memorypalace",
    "realitycheck",
    "realitymix",
    "vibedj",
)
BUTTON_TAG_RE = re.compile(r"<button(?P<attrs>[^>]*)>", re.MULTILINE)
CLASS_ATTR_RE = re.compile(r'\bclass=["\'](?P<classes>[^"\']+)["\']')
REQUIRED_SHARED_CSS_FRAGMENTS = (
    ".app-button {",
    ".app-button:focus-visible {",
)
REQUIRED_SCRIPT_FRAGMENTS = {
    "memorypalace": ('button.className = "app-button chip";',),
}


def button_has_class(attrs: str, class_name: str) -> bool:
    match = CLASS_ATTR_RE.search(attrs)
    if not match:
        return False
    classes = set(match.group("classes").split())
    return class_name in classes


def main() -> int:
    failures: list[str] = []

    shared_css = (ROOT / "shared" / "browser-tokens.css").read_text(encoding="utf-8")
    for fragment in REQUIRED_SHARED_CSS_FRAGMENTS:
        if fragment not in shared_css:
            failures.append(
                f"shared/browser-tokens.css is missing button fragment {fragment!r}."
            )

    for service in BUTTON_SERVICES:
        path = ROOT / "src" / service / "index.html"
        text = path.read_text(encoding="utf-8")

        for index, match in enumerate(BUTTON_TAG_RE.finditer(text), start=1):
            if not button_has_class(match.group("attrs"), "app-button"):
                failures.append(
                    f"{path.relative_to(ROOT)} button #{index} is missing the shared app-button class."
                )

        for fragment in REQUIRED_SCRIPT_FRAGMENTS.get(service, ()):
            if fragment not in text:
                failures.append(
                    f"{path.relative_to(ROOT)} is missing dynamic button fragment {fragment!r}."
                )

    if failures:
        print("Browser app button check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        f"Browser app button check passed for {len(BUTTON_SERVICES)} static apps with buttons."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
