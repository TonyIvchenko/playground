#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
README_PATHS = [ROOT / "README.md", *sorted((ROOT / "src").glob("*/README.md"))]
DOC_PATHS = sorted(ROOT.glob("*.md")) + sorted((ROOT / "src").glob("*/README.md"))
AVAILABLE_CHECKS = ("readme-markdown", "docs-spelling", "readme-code-paths")

HEADING_RE = re.compile(r"^(#{1,6}) (.+\S)\s*$")
FENCE_RE = re.compile(r"^```")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
URL_RE = re.compile(r"https?://\S+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z']+\b")
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


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MissingPath:
    readme: Path
    line: int
    referenced_path: str
    resolved_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run markdown, spelling, and README code-path checks from one command."
    )
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        help=f"Run only a named docs check. Available: {', '.join(AVAILABLE_CHECKS)}.",
    )
    return parser.parse_args()


def lint_readme_markdown(path: Path) -> list[str]:
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_fence = False
    first_nonempty_line: tuple[int, str] | None = None
    h1_count = 0
    previous_heading_level = 0

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not in_fence and stripped and first_nonempty_line is None:
            first_nonempty_line = (line_number, line)

        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue

        if in_fence or not line.startswith("#"):
            continue

        heading_match = HEADING_RE.match(line)
        if not heading_match:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} uses an invalid heading format"
            )
            continue

        level = len(heading_match.group(1))
        if level == 1:
            h1_count += 1

        if previous_heading_level and level > previous_heading_level + 1:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} jumps from heading level "
                f"{previous_heading_level} to {level}"
            )

        previous_heading_level = level

    if in_fence:
        errors.append(f"{path.relative_to(ROOT)} has an unclosed fenced code block")

    if first_nonempty_line is None:
        errors.append(f"{path.relative_to(ROOT)} is empty")
    else:
        line_number, line = first_nonempty_line
        if not line.startswith("# "):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} should start with a single H1 heading"
            )

    if h1_count != 1:
        errors.append(
            f"{path.relative_to(ROOT)} should contain exactly one H1 heading (found {h1_count})"
        )

    return errors


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


def lint_doc_spelling(path: Path) -> list[str]:
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


def extract_code_path_candidates(readme_path: Path) -> list[tuple[int, str]]:
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


def resolve_code_path_candidate(readme_path: Path, candidate: str) -> Path:
    if candidate.startswith("src/") or candidate == "environment.yml":
        return ROOT / candidate
    return readme_path.parent / candidate


def find_missing_code_paths() -> tuple[list[MissingPath], int]:
    missing: list[MissingPath] = []
    checked = 0

    for readme_path in README_PATHS:
        for line, candidate in extract_code_path_candidates(readme_path):
            resolved = resolve_code_path_candidate(readme_path, candidate)
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


def run_checks(selected_checks: list[str] | None = None) -> dict[str, Any]:
    selected = selected_checks or list(AVAILABLE_CHECKS)
    unknown = sorted(set(selected) - set(AVAILABLE_CHECKS))
    if unknown:
        raise ValidationError("Unknown docs checks: " + ", ".join(unknown))

    results: dict[str, Any] = {}

    if "readme-markdown" in selected:
        errors: list[str] = []
        for path in README_PATHS:
            errors.extend(lint_readme_markdown(path))
        results["readme-markdown"] = {
            "errors": errors,
            "readme_count": len(README_PATHS),
        }

    if "docs-spelling" in selected:
        errors = []
        for path in DOC_PATHS:
            errors.extend(lint_doc_spelling(path))
        results["docs-spelling"] = {
            "errors": errors,
            "doc_count": len(DOC_PATHS),
        }

    if "readme-code-paths" in selected:
        missing, checked = find_missing_code_paths()
        results["readme-code-paths"] = {
            "errors": [
                f"{item.readme}:{item.line} references '{item.referenced_path}' "
                f"but {item.resolved_path} does not exist"
                for item in missing
            ],
            "readme_count": len(README_PATHS),
            "checked_paths": checked,
        }

    return results


def main() -> int:
    args = parse_args()
    try:
        results = run_checks(args.check)
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    failures = {
        check_name: data["errors"]
        for check_name, data in results.items()
        if data["errors"]
    }
    if failures:
        print("Docs validation failed:", file=sys.stderr)
        for check_name, errors in failures.items():
            print(f"[{check_name}]", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
        return 1

    summaries: list[str] = []
    if "readme-markdown" in results:
        summaries.append(
            f"readme-markdown={results['readme-markdown']['readme_count']} README files"
        )
    if "docs-spelling" in results:
        summaries.append(
            f"docs-spelling={results['docs-spelling']['doc_count']} markdown files"
        )
    if "readme-code-paths" in results:
        summaries.append(
            "readme-code-paths="
            f"{results['readme-code-paths']['readme_count']} README files/"
            f"{results['readme-code-paths']['checked_paths']} paths"
        )

    print("Docs validation passed for " + ", ".join(summaries) + ".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
