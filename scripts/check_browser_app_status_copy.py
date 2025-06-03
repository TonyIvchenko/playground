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
    "realitycheck",
    "realitymix",
    "vibedj",
)
GLOBAL_BANNED_FRAGMENTS = ("Failed to ",)
SERVICE_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "bert": {
        "required": (
            "Loading browser text model…",
            "Browser text model ready.",
            "Analysis ready.",
            "Couldn't load the browser text model.",
        ),
        "banned": (
            "Model ready.",
            "Analysis complete.",
            "Couldn't load the model.",
        ),
    },
    "counterpoint": {
        "required": (
            "Loading browser generation model…",
            "Downloading browser generation model…",
            "Loading browser generation model",
            "Downloading browser generation model",
            "Browser generation model ready",
            "Browser generation model unavailable",
            "Counter-side ready.",
            "Couldn't load the browser generation model.",
        ),
        "banned": (
            "Generation model loading",
            "Generation model downloading",
            "Counter-side ready from heuristic fallback.",
            "Couldn't load the browser model.",
        ),
    },
    "debate": {
        "required": (
            "Loading browser generation model…",
            "Loading browser generation model",
            "Browser generation model ready",
            "Couldn't load the browser generation model.",
        ),
        "banned": (
            "Generation model loading",
            "Model ready.",
            "Model unavailable",
            "Couldn't load the model.",
        ),
    },
    "facemesh": {
        "required": (
            "Loading camera…",
            "Loading reference image…",
            "Loading reference mesh…",
            "Camera ready. Loading face mesh",
            "Reference mesh ready.",
        ),
        "banned": (
            "Starting camera...",
            "Loading reference image...",
            "Detecting reference mesh...",
            "Loading face mesh...",
        ),
    },
    "manipulation": {
        "required": (
            "Loading browser analysis models…",
            "Loading browser toxicity model…",
            "Loading browser sentiment model…",
            "Browser toxicity model ready.",
            "Browser sentiment model ready.",
            "Browser analysis models ready. Paste text and analyze.",
            "Analysis ready.",
        ),
        "banned": (
            "Loading browser models…",
            "Toxicity model loading…",
            "Sentiment model loading…",
            "Toxicity model: ready",
            "Sentiment model: ready",
            "Models ready.",
            "Analysis complete.",
        ),
    },
    "realitycheck": {
        "required": (
            "Loading browser text model…",
            "Ready. The browser image model loads on the first image analysis.",
            "Browser image model ready",
            "Browser text model ready.",
            "Image ready",
            "Video ready",
        ),
        "banned": (
            "Image model will load on the first image analysis.",
            "Image model ready (",
            "Text model ready.",
            "Ready to analyze",
            "Couldn't load the text model.",
            "Couldn't load the image model (",
        ),
    },
    "realitymix": {
        "required": (
            "Ready. Upload a style image and start the camera to begin. The browser style model downloads on first use.",
            "Loading browser style model…",
            "Browser style model ready.",
            "Couldn't decode the style image.",
        ),
        "banned": (
            "Model ready. Start the camera or adjust style settings.",
            "Failed to decode style image.",
        ),
    },
    "vibedj": {
        "required": (
            "Loading browser sentiment model…",
            "Loading browser sentiment model",
            "Browser sentiment model ready",
            "Browser sentiment model unavailable",
            "Couldn't load the browser sentiment model.",
            "Mix ready.",
        ),
        "banned": (
            "Sentiment model loading",
            "Model ready.",
            "Couldn't load the model.",
        ),
    },
}


def main() -> int:
    failures: list[str] = []

    for service in STATIC_BROWSER_SERVICES:
        path = ROOT / "src" / service / "index.html"
        text = path.read_text(encoding="utf-8")
        rules = SERVICE_RULES[service]

        for fragment in rules.get("required", ()):
            if fragment not in text:
                failures.append(
                    f"{path.relative_to(ROOT)} is missing status-copy fragment {fragment!r}."
                )

        for fragment in GLOBAL_BANNED_FRAGMENTS + rules.get("banned", ()):
            if fragment in text:
                failures.append(
                    f"{path.relative_to(ROOT)} still contains drifted status-copy fragment {fragment!r}."
                )

    if failures:
        print("Browser app status-copy check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        f"Browser app status-copy check passed for {len(STATIC_BROWSER_SERVICES)} static apps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
