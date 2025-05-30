#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SHARED_CSS_FRAGMENTS = (
    "@media (prefers-reduced-motion: reduce) {",
    "animation: none !important;",
    "transition: none !important;",
)
REQUIRED_APP_FRAGMENTS = {
    "vibedj": (
        'window.matchMedia("(prefers-reduced-motion: reduce)")',
        "function renderStageFrame(",
        "function refreshStageMotionPreference()",
        "renderStageFrame(performance.now(), { animate: false });",
    ),
    "memorypalace": (
        'window.matchMedia("(prefers-reduced-motion: reduce)")',
        "camera.position.copy(targetPosition);",
        "camera.position.y = reducedMotion ? 1.75",
    ),
    "realitymix": (
        'window.matchMedia("(prefers-reduced-motion: reduce)")',
        "let lastPreviewAt = 0;",
        "return prefersReducedMotion() ? Math.max(baseInterval, 1000) : baseInterval;",
        "function syncMotionPreference()",
    ),
    "facemesh": (
        'window.matchMedia("(prefers-reduced-motion: reduce)")',
        "const reducedMotionFrameIntervalMs = prefersReducedMotion() ? 120 : 0;",
        "Face mesh active in reduced-motion mode.",
    ),
}


def main() -> int:
    failures: list[str] = []

    shared_css = (ROOT / "shared" / "browser-tokens.css").read_text(encoding="utf-8")
    for fragment in REQUIRED_SHARED_CSS_FRAGMENTS:
        if fragment not in shared_css:
            failures.append(
                f"shared/browser-tokens.css is missing reduced-motion fragment {fragment!r}."
            )

    for service, fragments in REQUIRED_APP_FRAGMENTS.items():
        path = ROOT / "src" / service / "index.html"
        text = path.read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in text:
                failures.append(
                    f"{path.relative_to(ROOT)} is missing reduced-motion fragment {fragment!r}."
                )

    if failures:
        print("Browser app reduced-motion check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        f"Browser app reduced-motion check passed for {len(REQUIRED_APP_FRAGMENTS)} motion-heavy apps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
