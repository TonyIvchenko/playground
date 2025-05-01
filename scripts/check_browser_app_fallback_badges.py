#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
FALLBACK_SERVICES = (
    "counterpoint",
    "manipulation",
    "realitycheck",
    "vibedj",
)
BANNED_PHRASES = {
    "Heuristic mode": "Use the shared fallback badge label instead.",
    "Model unavailable: heuristic fallback": "Use the shared fallback badge label instead.",
}
REQUIRED_FRAGMENTS = {
    "counterpoint": ("Fallback mode active",),
    "manipulation": ("Fallback mode active",),
    "realitycheck": ("Fallback mode active",),
    "vibedj": ("Fallback mode active",),
}
REQUIRED_CLASS_BY_ID = {
    "counterpoint": {"fallback-pill": ("app-pill", "is-fallback")},
    "manipulation": {"fallback-pill": ("app-pill", "is-fallback")},
    "realitycheck": {
        "image-fallback-pill": ("app-pill", "is-fallback"),
        "video-fallback-pill": ("app-pill", "is-fallback"),
    },
    "vibedj": {"fallback-pill": ("app-pill", "is-fallback")},
}


def element_has_classes(
    text: str, element_id: str, required_classes: tuple[str, ...]
) -> bool:
    pattern = re.compile(
        rf"<[a-zA-Z0-9]+(?P<attrs>[^>]*\bid=[\"']{re.escape(element_id)}[\"'][^>]*)>",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return False
    attrs = match.group("attrs")
    class_match = re.search(r'\bclass=["\']([^"\']+)["\']', attrs)
    if not class_match:
        return False
    classes = set(class_match.group(1).split())
    return all(class_name in classes for class_name in required_classes)


def main() -> int:
    failures: list[str] = []

    for service in FALLBACK_SERVICES:
        path = ROOT / "src" / service / "index.html"
        text = path.read_text(encoding="utf-8")

        for phrase, guidance in BANNED_PHRASES.items():
            if phrase in text:
                failures.append(
                    f"{path.relative_to(ROOT)} still uses {phrase!r}. {guidance}"
                )

        for fragment in REQUIRED_FRAGMENTS.get(service, ()):
            if fragment not in text:
                failures.append(
                    f"{path.relative_to(ROOT)} is missing fallback fragment {fragment!r}."
                )

        for element_id, required_classes in REQUIRED_CLASS_BY_ID.get(
            service, {}
        ).items():
            if not element_has_classes(text, element_id, required_classes):
                failures.append(
                    f"{path.relative_to(ROOT)} is missing {', '.join(required_classes)} on #{element_id}."
                )

    if failures:
        print("Browser app fallback-badge check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        f"Browser app fallback-badge check passed for {len(FALLBACK_SERVICES)} fallback-capable apps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
