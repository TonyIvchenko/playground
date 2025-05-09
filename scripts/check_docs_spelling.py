#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = sorted(ROOT.glob("*.md")) + sorted((ROOT / "src").glob("*/README.md"))
TYPO_FIXES = {
    "acheive": "achieve",
    "arguement": "argument",
    "definately": "definitely",
    "enviroment": "environment",
    "huricaines": "hurricanes",
    "mispelled": "misspelled",
    "occurence": "occurrence",
    "recieve": "receive",
    "seperate": "separate",
    "teh": "the",
}
FENCE_RE = re.compile(r"^```")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
URL_RE = re.compile(r"https?://\S+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z']+\b")


def prose_lines(path: Path) -> list[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_fence = False
    out: list[tuple[int, str]] = []

    for line_number, line in enumerate(lines, start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        line = URL_RE.sub(" ", line)
        line = MARKDOWN_LINK_RE.sub(r"\1", line)
        line = INLINE_CODE_RE.sub(" ", line)
        out.append((line_number, line))

    return out


def lint_doc(path: Path) -> list[str]:
    errors: list[str] = []
    for line_number, line in prose_lines(path):
        for match in WORD_RE.finditer(line):
            word = match.group(0)
            typo = word.lower()
            if typo not in TYPO_FIXES:
                continue
            suggestion = TYPO_FIXES[typo]
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} uses '{word}' "
                f"(prefer '{suggestion}')"
            )
    return errors


def main() -> int:
    errors: list[str] = []
    for path in DOC_PATHS:
        errors.extend(lint_doc(path))

    if errors:
        print("Docs spelling check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Docs spelling check passed for {len(DOC_PATHS)} markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
