#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
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
BANNED_PHRASES = {
    "No output yet.": "Use an action-oriented empty state instead.",
    "No palace built yet": "Use a ready-to-begin phrase instead.",
    "No mix yet": "Use a ready-to-begin phrase instead.",
    "No image loaded.": "Use an upload-to-begin phrase instead.",
    "No video loaded.": "Use an upload-to-begin phrase instead.",
    "No URL analyzed.": "Use an enter-a-URL-to-begin phrase instead.",
    "Enter text to analyze.": "Use a paste-text-to-begin phrase instead.",
    "Paste text to score manipulation-related patterns.": "Use a shorter paste-text-to-begin phrase instead.",
    "Build a palace to start walking.": "Use a build-a-palace-to-begin phrase instead.",
    "Generate a mix first to hear it.": "Use a generate-a-mix-to-begin phrase instead.",
    "Point extraction waits for both sides.": "Use an add-both-sides-to-begin phrase instead.",
}


def main() -> int:
    failures: list[str] = []

    for service in STATIC_SERVICES:
        path = ROOT / "src" / service / "index.html"
        text = path.read_text(encoding="utf-8")
        for phrase, guidance in BANNED_PHRASES.items():
            if phrase in text:
                failures.append(
                    f"{path.relative_to(ROOT)} still uses {phrase!r}. {guidance}"
                )

    if failures:
        print("Browser app empty-state check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        f"Browser app empty-state check passed for {len(STATIC_SERVICES)} static apps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
