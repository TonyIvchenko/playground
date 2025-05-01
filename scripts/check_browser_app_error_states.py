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
    "Analysis failed.": "Use a consistent Couldn't-analyze phrasing instead.",
    "Failed to load model.": "Use a consistent Couldn't-load-the-model phrasing instead.",
    "The browser model failed to load, so Counterpoint will use heuristic counterargument templates for this run.": "Use the shared Couldn't-load wording and keep the fallback note concise.",
    "Model generation failed, so Counterpoint fell back to heuristic output.": "Use a Couldn't-build phrasing instead.",
    "Model load failed:": "Use a Couldn't-load phrasing instead.",
    "Model failed": "Use a clearer unavailable/error label instead of failed.",
    "Sparring failed:": "Use a Couldn't-run-sparring phrasing instead.",
    "Could not read Point of view A:": "Use a consistent Couldn't-read phrasing instead.",
    "Could not read Point of view B:": "Use a consistent Couldn't-read phrasing instead.",
    "Toxicity model: failed (": "Use unavailable instead of failed for model fallback lines.",
    "Sentiment model: failed (": "Use unavailable instead of failed for model fallback lines.",
    "Analysis failed:": "Use a consistent Couldn't-analyze phrasing instead.",
    "Stylization failed:": "Use a Couldn't-stylize phrasing instead.",
    "Camera start failed:": "Use a Couldn't-start-the-camera phrasing instead.",
    "Style image failed to load:": "Use a Couldn't-load-the-style-image phrasing instead.",
    "Model unavailable, using heuristics only:": "Use a Couldn't-load-the-model phrasing instead.",
    "Copy failed:": "Use a Couldn't-copy phrasing instead.",
    "Playback failed:": "Use a Couldn't-play phrasing instead.",
    "Remix failed:": "Use a Couldn't-remix phrasing instead.",
    "Load failed": "Use a clearer load-error label instead of failed.",
    "Fetch failed": "Use a clearer fetch-error label instead of failed.",
    "Failed to load image.": "Use a Couldn't-load-the-image phrasing instead.",
    "Failed to load image URL.": "Use a Couldn't-load-the-image-URL phrasing instead.",
    "Failed to load video.": "Use a Couldn't-load-the-video phrasing instead.",
    "Failed to load video URL.": "Use a Couldn't-load-the-video-URL phrasing instead.",
    "Image analysis failed.": "Use a Couldn't-analyze-the-image phrasing instead.",
    "Video analysis failed.": "Use a Couldn't-analyze-the-video phrasing instead.",
    "Text analysis failed.": "Use a Couldn't-analyze-the-text phrasing instead.",
    "URL analysis failed.": "Use a Couldn't-analyze-the-URL phrasing instead.",
    "Failed to load text model.": "Use a Couldn't-load-the-text-model phrasing instead.",
    "Reference image mesh failed.": "Use a Couldn't-map-the-reference-face phrasing instead.",
    "Reference image failed to load.": "Use a Couldn't-load-the-reference-image phrasing instead.",
    "Face mesh update failed.": "Use a Couldn't-update-the-face-mesh phrasing instead.",
    "Camera access failed.": "Use a Couldn't-access-the-camera phrasing instead.",
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
        print("Browser app error-state check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        f"Browser app error-state check passed for {len(STATIC_SERVICES)} static apps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
