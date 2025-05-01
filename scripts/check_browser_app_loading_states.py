#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_LOADING_SERVICES = (
    "bert",
    "counterpoint",
    "debate",
    "manipulation",
    "realitycheck",
    "realitymix",
    "vibedj",
)
BANNED_PHRASES = {
    "Loading model...": "Use a browser-specific loading phrase instead.",
    "Loading model…": "Use a browser-specific loading phrase instead.",
    "Loading flan-t5-small": "Use a shared browser generation-model loading phrase instead.",
    "Loading sentiment model": "Use a shared browser sentiment-model loading phrase instead.",
    "Loading text model...": "Use a shared browser text-model loading phrase instead.",
    "Loading text model…": "Use a shared browser text-model loading phrase instead.",
    "Loading neural style transfer model…": "Use a shared browser style-model loading phrase instead.",
}
REQUIRED_FRAGMENTS = {
    "bert": (
        'placeholder="Loading browser text model..."',
        "Loading browser text model…",
    ),
    "counterpoint": (
        'placeholder="Loading browser generation model..."',
        "Loading browser generation model…",
        "Downloading browser generation model…",
    ),
    "debate": (
        'placeholder="Loading browser generation model..."',
        "Loading browser generation model…",
    ),
    "manipulation": (
        'placeholder="Loading browser models..."',
        "Loading browser models…",
        "Toxicity model loading…",
        "Sentiment model loading…",
    ),
    "realitycheck": (
        'placeholder="Loading browser text model..."',
        "Loading browser text model…",
        "Loading browser image model…",
    ),
    "realitymix": ("Loading browser style model…",),
    "vibedj": (
        "Sentiment model loading",
        "Loading browser sentiment model…",
    ),
}
REQUIRED_CLASS_BY_ID = {
    "bert": {"status": ("app-status", "is-loading")},
    "counterpoint": {
        "model-pill": ("app-pill", "is-loading"),
        "status": ("app-status", "is-loading"),
    },
    "debate": {
        "model-pill": ("app-pill", "is-loading"),
        "status-text": ("app-status", "is-loading"),
    },
    "manipulation": {
        "status-text": ("app-status", "is-loading"),
        "toxicity-status": ("app-help", "is-loading"),
        "sentiment-status": ("app-help", "is-loading"),
    },
    "realitycheck": {"text-status": ("app-help", "is-loading")},
    "vibedj": {
        "model-pill": ("app-pill", "is-loading"),
        "status-text": ("app-status", "is-loading"),
    },
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

    for service in MODEL_LOADING_SERVICES:
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
                    f"{path.relative_to(ROOT)} is missing loading fragment {fragment!r}."
                )

        for element_id, required_classes in REQUIRED_CLASS_BY_ID.get(
            service, {}
        ).items():
            if not element_has_classes(text, element_id, required_classes):
                failures.append(
                    f"{path.relative_to(ROOT)} is missing {', '.join(required_classes)} on #{element_id}."
                )

    if failures:
        print("Browser app loading-state check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        f"Browser app loading-state check passed for {len(MODEL_LOADING_SERVICES)} model-loading apps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
