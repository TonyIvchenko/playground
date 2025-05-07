#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
README_PATHS = [ROOT / "README.md", *sorted((ROOT / "src").glob("*/README.md"))]
CODE_BLOCK_PATTERN = re.compile(r"(?ms)^```(?P<info>[^\n`]*)\n(?P<body>.*?)^```$")
PATH_PATTERNS = (
    re.compile(r"(?<![\w./-])(src/[A-Za-z0-9_./-]+)"),
    re.compile(r"(?<![\w./-])(scripts/[A-Za-z0-9_./-]+)"),
    re.compile(r"(?<![\w./-])(tests/[A-Za-z0-9_./-]+)"),
    re.compile(r"(?<![\w./-])(notebooks/[A-Za-z0-9_./-]+)"),
    re.compile(r"(?<![\w./-])(main\.py)(?![\w./-])"),
    re.compile(r"(?<![\w./-])(index\.html)(?![\w./-])"),
    re.compile(r"(?<![\w./-])(Dockerfile)(?![\w./-])"),
    re.compile(r"(?<![\w./-])(requirements\.txt)(?![\w./-])"),
    re.compile(r"(?<![\w./-])(environment\.yml)(?![\w./-])"),
)


@dataclass(frozen=True)
class MissingPath:
    readme: Path
    line: int
    referenced_path: str
    resolved_path: Path


def extract_candidates(readme_path: Path) -> list[tuple[int, str]]:
    text = readme_path.read_text(encoding="utf-8")
    seen: set[tuple[int, str]] = set()
    candidates: list[tuple[int, str]] = []

    for block_match in CODE_BLOCK_PATTERN.finditer(text):
        body = block_match.group("body")
        block_offset = block_match.start("body")
        for pattern in PATH_PATTERNS:
            for path_match in pattern.finditer(body):
                candidate = path_match.group(1)
                candidate_offset = block_offset + path_match.start(1)
                line = text.count("\n", 0, candidate_offset) + 1
                key = (line, candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((line, candidate))

    return candidates


def resolve_candidate(readme_path: Path, candidate: str) -> Path:
    if candidate.startswith("src/") or candidate == "environment.yml":
        return ROOT / candidate
    return readme_path.parent / candidate


def find_missing_paths() -> tuple[list[MissingPath], int]:
    missing: list[MissingPath] = []
    checked = 0

    for readme_path in README_PATHS:
        for line, candidate in extract_candidates(readme_path):
            resolved = resolve_candidate(readme_path, candidate)
            checked += 1
            if resolved.exists():
                continue
            missing.append(
                MissingPath(
                    readme=readme_path.relative_to(ROOT),
                    line=line,
                    referenced_path=candidate,
                    resolved_path=resolved.relative_to(ROOT),
                )
            )

    return missing, checked


def main() -> int:
    missing, checked = find_missing_paths()
    if missing:
        print("README code block path check failed:", file=sys.stderr)
        for item in missing:
            print(
                f"- {item.readme}:{item.line} references '{item.referenced_path}' "
                f"but {item.resolved_path} does not exist",
                file=sys.stderr,
            )
        return 1

    print(
        "README code block path check passed for "
        f"{len(README_PATHS)} README files ({checked} paths checked)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
