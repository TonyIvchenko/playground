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
    ".app-footer {",
    ".app-footer-grid {",
    ".app-footer-block {",
    ".app-footer-kicker {",
    ".app-footer-copy {",
)
REQUIRED_LABELS = (
    '<p class="app-footer-kicker">Caveats</p>',
    '<p class="app-footer-kicker">Privacy</p>',
    '<p class="app-footer-kicker">Local only</p>',
)
SERVICE_SPECIFIC_FRAGMENTS = {
    "bert": ("Scores are directional moderation signals",),
    "counterpoint": ("Generated talking points can flatten nuance",),
    "debate": ("The sparring transcript is a prep heuristic",),
    "facemesh": ("Overlay quality drops with lighting, pose, occlusion",),
    "manipulation": ("Pattern scores are heuristics, not proof of intent",),
    "memorypalace": ("The palace is a mnemonic aid",),
    "realitycheck": ("URL mode asks the local server to fetch page text first.",),
    "realitymix": ("Style transfer favors mood over exact structure",),
    "vibedj": ("These recommendations are taste prompts",),
}


def main() -> int:
    failures: list[str] = []

    shared_css = (ROOT / "shared" / "browser-tokens.css").read_text(encoding="utf-8")
    for fragment in REQUIRED_SHARED_CSS_FRAGMENTS:
        if fragment not in shared_css:
            failures.append(
                f"shared/browser-tokens.css is missing footer fragment {fragment!r}."
            )

    for service in STATIC_BROWSER_SERVICES:
        path = ROOT / "src" / service / "index.html"
        text = path.read_text(encoding="utf-8")

        if 'class="app-footer app-panel"' not in text:
            failures.append(
                f"{path.relative_to(ROOT)} is missing the shared app-footer app-panel shell."
            )

        for label in REQUIRED_LABELS:
            if label not in text:
                failures.append(
                    f"{path.relative_to(ROOT)} is missing footer label fragment {label!r}."
                )

        for fragment in SERVICE_SPECIFIC_FRAGMENTS.get(service, ()):
            if fragment not in text:
                failures.append(
                    f"{path.relative_to(ROOT)} is missing app-specific footer copy fragment {fragment!r}."
                )

    if failures:
        print("Browser app footer check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        f"Browser app footer check passed for {len(STATIC_BROWSER_SERVICES)} static apps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
